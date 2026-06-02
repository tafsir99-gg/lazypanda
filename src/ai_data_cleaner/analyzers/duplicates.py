"""
Duplicate Row Analyzer.

WHAT IT DETECTS:
    - Exact duplicate rows (all column values identical)
    - Reports count, percentage, and sample row indices

WHY EXACT DUPLICATES ONLY IN V1:
    Near-duplicate detection (fuzzy matching) requires rapidfuzz or similar.
    That is scoped to v1.2. Exact duplicates are free with pandas and catch
    the most common Kaggle data problem: accidentally joined/appended twice.

OUTPUT DETAILS SCHEMA:
    {
        "duplicate_count": 42,
        "duplicate_pct": 0.047,
        "keep_strategy": "first",
        "sample_duplicate_indices": [8, 15, 23],  # up to 10 examples
        "rows_that_would_remain": 849,
    }
"""

from __future__ import annotations

import logging

import pandas as pd

from ai_data_cleaner.analyzers.base import AnalysisResult, BaseAnalyzer

logger = logging.getLogger("ai_data_cleaner")


class DuplicateAnalyzer(BaseAnalyzer):
    """Detects exact duplicate rows in a DataFrame."""

    @property
    def name(self) -> str:
        return "duplicates"

    @property
    def display_name(self) -> str:
        return "Duplicate Rows"

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        n_rows = len(df)
        keep: str = self.config.get("keep", "first")

        # pandas duplicated() marks rows that are duplicates.
        # keep="first" → marks all BUT the first occurrence as duplicate.
        # keep="none"  → marks ALL occurrences of any duplicated row.
        pd_keep = False if keep == "none" else keep  # pandas uses False for "none"
        duplicate_mask = df.duplicated(keep=pd_keep)
        duplicate_count = int(duplicate_mask.sum())

        if duplicate_count == 0:
            return AnalysisResult.ok(self.name, self.display_name, "No duplicate rows detected.")

        duplicate_pct = round(duplicate_count / n_rows, 4)
        sample_indices = duplicate_mask[duplicate_mask].index.tolist()[:10]
        rows_remaining = n_rows - duplicate_count

        severity = "critical" if duplicate_pct > 0.1 else "warning"

        return AnalysisResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            issues_found=True,
            severity=severity,
            affected_columns=[],  # Duplicates affect rows, not specific columns
            details={
                "duplicate_count": duplicate_count,
                "duplicate_pct": duplicate_pct,
                "keep_strategy": keep,
                "sample_duplicate_indices": sample_indices,
                "rows_that_would_remain": rows_remaining,
            },
            summary=(
                f"{duplicate_count:,} duplicate row(s) detected "
                f"({duplicate_pct:.1%} of dataset)"
            ),
            recommendation=(
                f"Drop {duplicate_count} duplicate row(s), keeping the '{keep}' occurrence. "
                f"{rows_remaining:,} rows would remain."
            ),
        )
