"""
Duplicate Row Cleaner.

WHAT IT DOES:
    Drops exact duplicate rows from the DataFrame.
    Uses the analysis result from DuplicateAnalyzer to skip re-computation.

OPERATION ORDER NOTE:
    This runs FIRST in the pipeline because:
    - Fewer rows = faster imputation, faster outlier detection
    - Imputed values from duplicate rows would be wasted work
    - Keeps the dataset canonical before any other transformation

SAFETY:
    If analysis_result says no duplicates, returns the df unchanged.
    If analysis_result is None, runs a lightweight check itself.
"""

from __future__ import annotations

import logging

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.cleaners.base import BaseCleaner, CleaningOperation, CleaningResult

logger = logging.getLogger("lazypanda")


class DropDuplicatesCleaner(BaseCleaner):
    """Drops exact duplicate rows from a DataFrame."""

    @property
    def name(self) -> str:
        return "drop_duplicates"

    @property
    def display_name(self) -> str:
        return "Drop Duplicates"

    def clean(
        self,
        df: pd.DataFrame,
        analysis_result: AnalysisResult | None = None,
    ) -> tuple[pd.DataFrame, CleaningResult]:

        keep: str = self.config.get("keep", "first")
        pd_keep = False if keep == "none" else keep

        # ── Fast path: use pre-computed analysis if available ─────────────────
        if analysis_result is not None and not analysis_result.issues_found:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No duplicates detected.")

        # ── Detect or re-use duplicate count ─────────────────────────────────
        if analysis_result is not None:
            duplicate_count = analysis_result.details.get("duplicate_count", 0)
        else:
            duplicate_count = int(df.duplicated(keep=pd_keep).sum())

        if duplicate_count == 0:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No duplicates detected.")

        # ── Apply the transformation ──────────────────────────────────────────
        df_clean = df.drop_duplicates(keep=pd_keep)
        df_clean = df_clean.reset_index(drop=True)

        rows_dropped = len(df) - len(df_clean)
        logger.info("DropDuplicatesCleaner: removed %d duplicate row(s)", rows_dropped)

        ops = [
            CleaningOperation(
                type="drop_rows",
                description=f"Dropped {rows_dropped} duplicate row(s), keeping '{keep}' occurrence",
                details={"rows_dropped": rows_dropped, "keep_strategy": keep},
            )
        ]

        result = self._make_result(
            df_before=df,
            df_after=df_clean,
            operations=ops,
            summary=f"Removed {rows_dropped} duplicate row(s) — {len(df_clean):,} rows remain",
        )
        return df_clean, result
