"""
Markdown Reporter.

WHAT IT DOES:
    Renders a Jinja2 template (report.md.j2) into a human-readable
    Markdown file summarizing the full pipeline run.

WHY JINJA2 INSTEAD OF PYTHON F-STRINGS?
    F-strings get unreadable very fast for large documents. Jinja2:
    - Separates template logic from Python logic (Separation of Concerns)
    - Supports loops, conditionals, and filters inside the template
    - Easy to update the report's look without touching Python code
    - Industry standard for code/document generation

WHAT THE REPORT CONTAINS:
    1. Dataset overview (shape, memory, file)
    2. Analysis results — one section per analyzer
    3. Cleaning pipeline table — applied vs skipped
    4. Detailed operations per cleaner
    5. Before → After summary
    6. Output file paths

USAGE:
    from lazypanda.reporters.markdown_reporter import MarkdownReporter
    from lazypanda.core.pipeline import PipelineResult

    reporter = MarkdownReporter()
    path = reporter.write(result, output_path=Path("outputs/report.md"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from lazypanda.core.pipeline import PipelineResult

logger = logging.getLogger("lazypanda")

# Path to the templates directory (same package, templates/ subfolder)
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TOOL_VERSION = "0.1.0"


@dataclass
class ReportContext:
    """
    All data needed to render the Markdown report.

    WHY a dedicated dataclass?
        Jinja2 templates receive a flat dict. By preparing a ReportContext first,
        we can validate and transform all data in one place before rendering.
        This also makes the reporter unit-testable without a PipelineResult.
    """
    source_file: str
    generated_at: str
    tool_version: str
    dry_run: bool

    # Dataset shape
    original_rows: int
    original_cols: int
    final_rows: int
    final_cols: int
    rows_removed: int
    cols_removed: int
    memory_mb: float

    # Analysis
    analysis_results: list[dict[str, Any]]  # each is AnalysisResult.to_dict()

    # Cleaning
    cleaning_results: list[dict[str, Any]]  # each is CleaningResult.to_dict()
    cleaners_applied: int
    cleaners_total: int
    issues_found: int

    # Optional output file map for the report's footer
    output_files: dict[str, str]

    # AI insights (None if --no-ai or AI was skipped)
    ai_insights: Any  # AIInsights | None  — typed as Any to avoid circular import

    def to_template_vars(self) -> dict[str, Any]:
        """Convert to a flat dict for Jinja2 template rendering."""
        return {
            "source_file": self.source_file,
            "generated_at": self.generated_at,
            "tool_version": self.tool_version,
            "dry_run": self.dry_run,
            "original_rows": self.original_rows,
            "original_cols": self.original_cols,
            "final_rows": self.final_rows,
            "final_cols": self.final_cols,
            "rows_removed": self.rows_removed,
            "cols_removed": self.cols_removed,
            "memory_mb": self.memory_mb,
            "analysis_results": self.analysis_results,
            "cleaning_results": self.cleaning_results,
            "cleaners_applied": self.cleaners_applied,
            "cleaners_total": self.cleaners_total,
            "issues_found": self.issues_found,
            "output_files": self.output_files,
            # AI insights passed as the object directly (Jinja2 reads .attribute)
            "ai_insights": self.ai_insights,
        }


class MarkdownReporter:
    """
    Generates a human-readable Markdown report from a PipelineResult.

    The report is rendered from a Jinja2 template so its structure can
    be modified without changing any Python code.
    """

    def __init__(self, template_name: str = "report.md.j2"):
        self._template_name = template_name
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape([]),  # No HTML escaping — we're generating Markdown
            trim_blocks=True,    # Remove newline after block tags
            lstrip_blocks=True,  # Remove leading whitespace from block tags
        )

    def build_context(
        self,
        result: PipelineResult,
        source_file: str,
        output_files: dict[str, str] | None = None,
    ) -> ReportContext:
        """
        Build the ReportContext from a PipelineResult.

        Args:
            result:       The completed pipeline result.
            source_file:  Human-readable path/name of the original input file.
            output_files: Optional dict of label → path for the footer section.

        Returns:
            ReportContext ready to render.
        """
        stats = result.stats
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Serialize analysis results for the template
        analysis_dicts = []
        for ar in result.analysis_results:
            d = ar.to_dict()
            # Add display_name from the object directly (to_dict doesn't always include it)
            d["display_name"] = ar.display_name
            d["recommendation"] = ar.recommendation
            analysis_dicts.append(d)

        # Serialize cleaning results with operation detail
        cleaning_dicts = []
        for cr in result.cleaning_results:
            d = cr.to_dict()
            d["display_name"] = cr.display_name
            d["operations"] = [
                {"type": op.type, "description": op.description}
                for op in cr.operations
            ]
            cleaning_dicts.append(d)

        # Try to read memory from the first analysis details, or default to 0
        memory_mb = 0.0

        return ReportContext(
            source_file=source_file,
            generated_at=now,
            tool_version=_TOOL_VERSION,
            dry_run=result.dry_run,
            original_rows=result.original_shape[0],
            original_cols=result.original_shape[1],
            final_rows=result.final_shape[0],
            final_cols=result.final_shape[1],
            rows_removed=stats.get("rows_removed", 0),
            cols_removed=stats.get("cols_removed", 0),
            memory_mb=memory_mb,
            analysis_results=analysis_dicts,
            cleaning_results=cleaning_dicts,
            cleaners_applied=stats.get("cleaners_applied", 0),
            cleaners_total=len(result.cleaning_results),
            issues_found=stats.get("issues_found", 0),
            output_files=output_files or {},
            ai_insights=result.ai_insights,  # None if AI was not run
        )

    def render(self, context: ReportContext) -> str:
        """
        Render the Markdown report to a string.

        Args:
            context: A ReportContext built from build_context().

        Returns:
            The full Markdown report as a string.
        """
        template = self._env.get_template(self._template_name)
        return template.render(**context.to_template_vars())

    def write(
        self,
        result: PipelineResult,
        output_path: Path,
        source_file: str = "unknown",
        output_files: dict[str, str] | None = None,
    ) -> Path:
        """
        Render and write the Markdown report to disk.

        Args:
            result:       The completed pipeline result.
            output_path:  File path to write (e.g., outputs/report.md).
            source_file:  Human-readable source file name for the report header.
            output_files: Optional output file map for the footer section.

        Returns:
            The path that was written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        context = self.build_context(result, source_file=source_file, output_files=output_files)
        markdown_text = self.render(context)

        output_path.write_text(markdown_text, encoding="utf-8")
        logger.info("MarkdownReporter: wrote report to %s", output_path)
        return output_path
