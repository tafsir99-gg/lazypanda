"""
Constant Column Cleaner.

WHAT IT DOES:
    Drops columns where every non-null value is identical (zero variance).

    WHY drop them?
        - They carry zero information for any ML model
        - sklearn's VarianceThreshold would remove them anyway
        - They waste memory and confuse feature importance analysis

    WHAT IT DOES NOT DO:
        - Drop near-constant columns (borderline — needs human review)
        - Drop high-cardinality columns (may be IDs — needs human review)
        - Those are flagged by CardinalityAnalyzer for the user to decide

OPERATION ORDER:
    Runs second in the pipeline, right after deduplication.
    Removing useless columns early speeds up all subsequent cleaners.
"""

from __future__ import annotations

import logging

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.cleaners.base import BaseCleaner, CleaningOperation, CleaningResult

logger = logging.getLogger("lazypanda")


class ConstantColumnCleaner(BaseCleaner):
    """Drops constant columns (zero variance) from a DataFrame."""

    @property
    def name(self) -> str:
        return "constant_columns"

    @property
    def display_name(self) -> str:
        return "Drop Constant Columns"

    def clean(
        self,
        df: pd.DataFrame,
        analysis_result: AnalysisResult | None = None,
    ) -> tuple[pd.DataFrame, CleaningResult]:

        # ── Get constant column list from analysis or detect fresh ────────────
        if analysis_result is not None and not analysis_result.issues_found:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No cardinality issues.")

        if analysis_result is not None:
            constant_cols = analysis_result.details.get("constant_columns", [])
        else:
            constant_cols = [
                col for col in df.columns
                if df[col].dropna().nunique() <= 1
            ]

        if not constant_cols:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No constant columns found.")

        # ── Apply transformation ───────────────────────────────────────────────
        df_clean = df.drop(columns=constant_cols)
        logger.info("ConstantColumnCleaner: dropped %d constant column(s): %s", len(constant_cols), constant_cols)

        ops = []
        for col in constant_cols:
            # Get the single unique value for the log
            unique_vals = df[col].dropna().unique()
            const_value = unique_vals[0] if len(unique_vals) > 0 else None
            ops.append(CleaningOperation(
                type="drop_column",
                description=f"Dropped constant column '{col}' (always = {const_value!r})",
                details={"column": col, "constant_value": str(const_value)},
            ))

        return df_clean, self._make_result(
            df_before=df,
            df_after=df_clean,
            operations=ops,
            summary=f"Dropped {len(constant_cols)} constant column(s): {', '.join(constant_cols[:3])}"
                    + ("..." if len(constant_cols) > 3 else ""),
        )
