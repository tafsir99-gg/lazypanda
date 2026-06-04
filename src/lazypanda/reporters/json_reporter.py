"""JSON Reporter.

WHAT IT DOES:
    Serializes the complete PipelineResult into a structured JSON file.

WHY JSON IN ADDITION TO MARKDOWN?
    Markdown is for humans. JSON is for machines. The JSON summary:
    - Can be parsed by CI/CD pipelines to gate on data quality
    - Can be ingested by logging tools (Datadog, Splunk, etc.)
    - Can be diffed between two runs to track data drift over time
    - Makes the tool composable with other tools via file-based I/O

    Example CI/CD use:
        quality = summary["summary"]["ai_quality_score"]
        critical = summary["summary"]["has_critical_issues"]
        if quality < 60 or critical:
            sys.exit(1)  # Fail the pipeline

JSON STRUCTURE:
    {
      "meta":       { tool, version, schema_version, generated_at, source_file, dry_run },
      "dataset":    { original/final shape, rows_removed, cols_removed },
      "summary":    { issues_found, cleaners_applied, has_critical_issues,
                      ai_available, ai_quality_score },   ← CI/CD shortcuts
      "analysis":   [ { analyzer_name, issues_found, severity, ... }, ... ],
      "cleaning":   [ { cleaner_name, applied, rows_before, ... }, ... ],
      "ai_insights":{ skipped, data_quality_score, overall_summary, ... } | null,
      "output_files":{ label: path }
    }

USAGE:
    from lazypanda.reporters.json_reporter import JSONReporter
    path = JSONReporter().write(result, Path("outputs/summary.json"), source_file="train.csv")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lazypanda.core.pipeline import PipelineResult

logger = logging.getLogger("lazypanda")

_TOOL_VERSION = "0.1.0"


class JSONReporter:
    """
    Generates a machine-readable JSON summary from a PipelineResult.

    The JSON is formatted with indentation for human readability,
    but compact enough for programmatic parsing.
    """

    def build_payload(
        self,
        result: PipelineResult,
        source_file: str = "unknown",
        output_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Build the complete JSON payload from a PipelineResult.

        Args:
            result:       The completed pipeline result.
            source_file:  Human-readable path/name of the original input file.
            output_files: Optional map of output files produced in this run.

        Returns:
            A plain dict ready for json.dumps().
        """
        now = datetime.now(timezone.utc).isoformat()

        ai = result.ai_insights
        ai_available = ai is not None and not ai.skipped

        return {
            "meta": {
                "tool":           "LazyPanda",
                "version":        _TOOL_VERSION,
                "schema_version": 1,           # increment when structure changes
                "generated_at":   now,
                "source_file":    source_file,
                "dry_run":        result.dry_run,
            },
            "dataset": {
                "original_rows": result.original_shape[0],
                "original_cols": result.original_shape[1],
                "final_rows":    result.final_shape[0],
                "final_cols":    result.final_shape[1],
                "rows_removed":  result.stats.get("rows_removed", 0),
                "cols_removed":  result.stats.get("cols_removed", 0),
            },
            # CI/CD-friendly summary — all key signals in one flat block
            "summary": {
                "issues_found":       result.stats.get("issues_found", 0),
                "cleaners_applied":   result.stats.get("cleaners_applied", 0),
                "cleaners_skipped":   result.stats.get("cleaners_skipped", 0),
                "has_critical_issues": any(
                    ar.severity == "critical" for ar in result.analysis_results
                ),
                # AI shortcuts — safe to read even without GEMINI_API_KEY
                "ai_available":    ai_available,
                "ai_quality_score": ai.data_quality_score if ai_available else None,
            },
            "analysis": [self._serialize_analysis(ar) for ar in result.analysis_results],
            "cleaning": [self._serialize_cleaning(cr) for cr in result.cleaning_results],
            "ai_insights": ai.to_dict() if ai is not None else None,
            "output_files": output_files or {},
        }

    def render(self, payload: dict[str, Any], indent: int = 2) -> str:
        """
        Serialize the payload dict to a JSON string.

        Args:
            payload: The dict from build_payload().
            indent:  JSON indentation spaces (default 2).

        Returns:
            JSON string.
        """
        return json.dumps(payload, indent=indent, default=str, ensure_ascii=False)

    def write(
        self,
        result: PipelineResult,
        output_path: Path,
        source_file: str = "unknown",
        output_files: dict[str, str] | None = None,
    ) -> Path:
        """
        Build and write the JSON report to disk.

        Args:
            result:       The completed pipeline result.
            output_path:  File path to write (e.g., outputs/summary.json).
            source_file:  Human-readable source file name for the meta section.
            output_files: Optional map of output label → path for the JSON footer.

        Returns:
            The path that was written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.build_payload(result, source_file=source_file, output_files=output_files)
        json_text = self.render(payload)

        output_path.write_text(json_text, encoding="utf-8")
        logger.info("JSONReporter: wrote summary to %s", output_path)
        return output_path

    # ── Private serializers ───────────────────────────────────────────────────

    @staticmethod
    def _serialize_analysis(ar) -> dict[str, Any]:
        """Serialize one AnalysisResult to a JSON-safe dict."""
        return {
            "analyzer_name": ar.analyzer_name,
            "display_name": ar.display_name,
            "issues_found": ar.issues_found,
            "severity": ar.severity,
            "affected_columns": ar.affected_columns,
            "summary": ar.summary,
            "recommendation": ar.recommendation,
        }

    @staticmethod
    def _serialize_cleaning(cr) -> dict[str, Any]:
        """Serialize one CleaningResult to a JSON-safe dict."""
        return {
            "cleaner_name": cr.cleaner_name,
            "display_name": cr.display_name,
            "applied": cr.applied,
            "rows_before": cr.rows_before,
            "rows_after": cr.rows_after,
            "rows_removed": cr.rows_removed,
            "cols_before": cr.cols_before,
            "cols_after": cr.cols_after,
            "cols_removed": cr.cols_removed,
            "summary": cr.summary,
            "operations": [
                {
                    "type": op.type,
                    "description": op.description,
                    "details": _make_json_safe(op.details),
                }
                for op in cr.operations
            ],
        }


def _make_json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy scalars and other non-serializable types
    to Python native types so json.dumps() doesn't crash.

    WHY: Pandas/NumPy return numpy.int64 and numpy.float64 — these are NOT
    serializable by default json.dumps(). We convert them here rather than
    using default=str which would wrap numbers in quotes.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
