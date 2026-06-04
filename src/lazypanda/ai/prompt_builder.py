"""
Prompt Builder — constructs the Gemini prompt from a PipelineResult.

WHAT IT DOES:
    Takes the full output of the cleaning pipeline and builds ONE compact,
    structured prompt asking Gemini for a natural language analysis and
    data quality score.

THE CARDINAL RULE: Never send raw data rows to Gemini.

    WHY?
    1. Privacy: CSV rows may contain personal data (names, emails, IDs)
    2. Cost:    Rows are the majority of tokens in a dataset — sending them
               would make every call expensive
    3. Utility: Gemini doesn't need to SEE the data to generate insights.
               It needs to know WHAT the analysis found, not the raw values.

WHAT THE PROMPT CONTAINS (safe metadata only):
    - Dataset shape summary (e.g. "21x12 → 18x14")
    - Column names with dtypes (compact "name:dtype" format)
    - Analysis findings: severity, affected columns, one-line summary
    - Cleaning decisions: what was applied and the summary
    - Response format instruction (compact one-liner, not a verbose schema)

WHAT THE PROMPT DOES NOT CONTAIN:
    - Any data rows or cell values
    - PII or file system paths (basename only)
    - Verbose descriptions or multi-line schema definitions

PROMPT FORMAT RATIONALE:
    We use a compact JSON structure to avoid redundant whitespace and
    verbose keys. This keeps the prompt well under 900 tokens for
    typical datasets (< 50 columns).

    The response format is described as a compact one-liner string, NOT
    as a verbose schema dict — this alone saves ~200 tokens.

TOKEN BUDGET TARGET: ≤ 900 input tokens for typical datasets.

USAGE:
    builder = PromptBuilder()
    prompt, estimated_tokens = builder.build(pipeline_result, source_file="train.csv")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from lazypanda.ai.token_budget import TokenBudgetManager
from lazypanda.core.pipeline import PipelineResult

logger = logging.getLogger("lazypanda")

# Limits to keep prompts token-efficient
_MAX_COLUMNS_IN_PROMPT = 12   # Show top N columns (compact format)
_MAX_ISSUES_IN_PROMPT  = 6    # Show top N issues by severity
_MAX_AFFECTED_COLS     = 5    # Per issue, cap affected columns list


class PromptBuilder:
    """
    Builds a compact Gemini prompt from a PipelineResult.

    Uses a compact JSON format (not verbose nested schema) to keep
    input tokens well under the budget.
    """

    # Compact schema instruction (JSON MIME type already enforces valid JSON —
    # this only guides the field names and constraints).
    _RESPONSE_SCHEMA = {
        "overall_summary":    "string (2-3 sentences, ML-focused)",
        "data_quality_score": "integer 0-100 (score BEFORE cleaning was applied)",
        "score_rationale":    "string (1-2 sentences explaining the score)",
        "column_insights":    "[{column:string, insight:string}] (1 sentence each)",
        "recommendations":    "[string] (actionable ML next steps)",
        "cleaning_assessment":"string (1-2 sentences on cleaning decision quality)",
    }

    def build(
        self,
        result: PipelineResult,
        source_file: str = "dataset.csv",
    ) -> tuple[str, int]:
        """
        Build the Gemini prompt from a PipelineResult.

        Args:
            result:       The completed pipeline result.
            source_file:  Basename of the source file (no PII).

        Returns:
            (prompt_text, estimated_tokens)
        """
        source_basename = Path(source_file).name

        # Compact column list: ["col_name:dtype", ...]
        df = result.cleaned_df
        all_cols = list(df.columns)[:_MAX_COLUMNS_IN_PROMPT]
        col_list = [f"{c}:{df[c].dtype}" for c in all_cols]
        truncated = len(df.columns) > _MAX_COLUMNS_IN_PROMPT

        payload = {
            "role": (
                "Senior data scientist reviewing an ML dataset cleaning report. "
                "Be concise, practical, and ML-focused."
            ),
            "output_schema": self._RESPONSE_SCHEMA,
            "dataset": {
                "file": source_basename,
                "shape_original": f"{result.original_shape[0]}x{result.original_shape[1]}",
                "shape_final": f"{result.final_shape[0]}x{result.final_shape[1]}",
                "rows_removed": result.original_shape[0] - result.final_shape[0],
                "cols": col_list,
                "cols_truncated": truncated,
            },
            "analysis": self._build_analysis_section(result),
            "cleaning": self._build_cleaning_section(result),
        }

        prompt = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        estimated_tokens = TokenBudgetManager.estimate_tokens(prompt)

        logger.debug(
            "PromptBuilder: built prompt — %d chars, ~%d tokens",
            len(prompt),
            estimated_tokens,
        )
        return prompt, estimated_tokens

    # ── Private section builders ──────────────────────────────────────────────

    def _build_analysis_section(self, result: PipelineResult) -> list[dict]:
        """Compact analysis findings — most severe first, capped at N."""
        _SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2, "ok": 3}

        sorted_results = sorted(
            result.analysis_results,
            key=lambda r: _SEVERITY_ORDER.get(r.severity, 4),
        )

        issues = []
        for ar in sorted_results[:_MAX_ISSUES_IN_PROMPT]:
            issues.append({
                "name":     ar.display_name,
                "severity": ar.severity,
                "cols":     ar.affected_columns[:_MAX_AFFECTED_COLS],
                "summary":  ar.summary,
            })
        return issues

    def _build_cleaning_section(self, result: PipelineResult) -> list[dict]:
        """Compact cleaning decisions list."""
        return [
            {
                "cleaner":  cr.display_name,
                "applied":  cr.applied,
                "rows_rm":  cr.rows_removed,
                "cols_rm":  cr.cols_removed,
                "summary":  cr.summary,
            }
            for cr in result.cleaning_results
        ]
