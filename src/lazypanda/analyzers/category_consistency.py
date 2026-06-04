"""
Category Consistency Analyzer.

WHAT IT DETECTS:
    Categorical columns where the SAME semantic value appears in multiple forms
    due to inconsistent data entry:

    Case inconsistency:    "Male" / "male" / "MALE" → all mean the same thing
    Whitespace issues:     "active " / "active" / " active" → same value
    Combined:              " Male " / "male" → same after strip + lowercase

    This is a near-universal problem in real Kaggle datasets. It silently
    creates extra categories that confuse models.

WHY PYTHON NOT GEMINI:
    This is pure string normalization math. After lowercasing + stripping,
    if the number of unique values shrinks, there are inconsistencies.
    No AI needed — the logic is deterministic.

APPROACH:
    For each categorical column:
    1. Compute raw unique count.
    2. Apply normalization (lowercase + strip based on config).
    3. Compute normalized unique count.
    4. If normalized < raw → there are inconsistencies.
    5. Group original values by their normalized form to show what clusters.

OUTPUT DETAILS SCHEMA:
    {
        "per_column": {
            "sex": {
                "raw_unique_count": 5,
                "normalized_unique_count": 2,
                "inconsistency_count": 3,
                "clusters": [
                    {"normalized": "male",   "raw_values": ["Male", "male", "MALE"]},
                    {"normalized": "female", "raw_values": ["Female", "female"]},
                ]
            }
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult, BaseAnalyzer

logger = logging.getLogger("lazypanda")


class CategoryConsistencyAnalyzer(BaseAnalyzer):
    """Detects case and whitespace inconsistencies in categorical columns."""

    @property
    def name(self) -> str:
        return "category_consistency"

    @property
    def display_name(self) -> str:
        return "Category Consistency"

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        normalize_case: bool = self.config.get("normalize_case", True)
        strip_ws: bool = self.config.get("strip_whitespace", True)

        if not normalize_case and not strip_ws:
            return AnalysisResult.ok(
                self.name,
                self.display_name,
                "Category normalization disabled in config.",
            )

        # Only check object/string dtype columns
        cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

        per_column: dict[str, Any] = {}

        for col in cat_cols:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            raw_uniques = series.unique().tolist()
            raw_count = len(raw_uniques)

            # Apply normalization
            normalized = series.copy()
            if strip_ws:
                normalized = normalized.str.strip()
            if normalize_case:
                normalized = normalized.str.lower()

            norm_uniques = normalized.unique()
            norm_count = len(norm_uniques)

            if norm_count >= raw_count:
                continue  # No inconsistencies found

            # Build clusters: group original values by their normalized form
            norm_to_raw: dict[str, list[str]] = {}
            for raw_val, norm_val in zip(series, normalized):
                norm_to_raw.setdefault(norm_val, [])
                if raw_val not in norm_to_raw[norm_val]:
                    norm_to_raw[norm_val].append(raw_val)

            # Only report clusters that actually have inconsistencies (>1 raw form)
            inconsistent_clusters = [
                {"normalized": norm, "raw_values": sorted(raws)}
                for norm, raws in norm_to_raw.items()
                if len(raws) > 1
            ]
            inconsistency_count = sum(len(c["raw_values"]) - 1 for c in inconsistent_clusters)

            per_column[col] = {
                "raw_unique_count": raw_count,
                "normalized_unique_count": norm_count,
                "inconsistency_count": inconsistency_count,
                "clusters": inconsistent_clusters,
            }

        if not per_column:
            return AnalysisResult.ok(
                self.name,
                self.display_name,
                "All categorical columns are consistent.",
            )

        affected = list(per_column.keys())
        total_inconsistencies = sum(v["inconsistency_count"] for v in per_column.values())

        ops = []
        if normalize_case:
            ops.append("lowercase")
        if strip_ws:
            ops.append("strip whitespace")

        return AnalysisResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            issues_found=True,
            severity="warning",
            affected_columns=affected,
            details={"per_column": per_column},
            summary=(
                f"{total_inconsistencies} category inconsistencies across "
                f"{len(affected)} column(s) (case/whitespace variants)"
            ),
            recommendation=(
                f"Normalize {len(affected)} column(s) by applying: {', '.join(ops)}. "
                f"This reduces spurious extra categories that confuse ML models."
            ),
        )
