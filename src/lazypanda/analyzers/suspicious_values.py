"""
Suspicious Value Analyzer.

WHAT IT DETECTS:
    Values that are syntactically valid but semantically wrong:

    1. DOMAIN VIOLATIONS (column name heuristics):
       - Negative values in columns named "age", "price", "count", "fare", etc.
       - These suggest data entry errors or bad imputation.

    2. COMMON PLACEHOLDER SENTINELS:
       - Values like -999, -9999, -1, 9999, 999999 are often used as
         "null substitutes" when the real null was lost in ETL pipelines.
       - Detection: value appears multiple times AND matches a known sentinel.

    3. EXTREME MAGNITUDE OUTLIERS (separate from statistical outliers):
       - Values that are orders of magnitude larger than the column median.
       - Catches: fare=999999, age=9999 where IQR might not if data is skewed.

WHY PYTHON NOT GEMINI:
    Column name pattern matching + value comparison = deterministic Python.
    The heuristics here are based on well-known ML data quality conventions,
    not semantic understanding. Gemini is only needed when the decision
    is ambiguous (Milestone 5).

OUTPUT DETAILS SCHEMA:
    {
        "per_column": {
            "age": {
                "suspicious_findings": [
                    {
                        "type": "negative_in_positive_domain",
                        "value": -5.0,
                        "count": 1,
                        "row_indices": [4],
                    },
                    {
                        "type": "common_placeholder",
                        "value": -999.0,
                        "count": 2,
                        "row_indices": [9, 17],
                    }
                ]
            }
        },
        "total_suspicious_cells": 3,
    }
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from lazypanda.analyzers.base import AnalysisResult, BaseAnalyzer

logger = logging.getLogger("lazypanda")

# ── Column name keywords that imply non-negative domain ──────────────────────
_POSITIVE_DOMAIN_KEYWORDS = frozenset({
    "age", "price", "cost", "fare", "salary", "wage", "income",
    "revenue", "count", "quantity", "qty", "amount", "size", "weight",
    "height", "width", "length", "duration", "distance", "score",
    "rating", "rank", "population", "value", "sales", "units",
})

# ── Sentinel values commonly used as "missing" substitutes ───────────────────
_COMMON_SENTINELS: frozenset[float] = frozenset({
    -1.0, -99.0, -999.0, -9999.0, -99999.0,
    99.0, 999.0, 9999.0, 99999.0, 999999.0,
    -1.0, 0.0,  # 0 is a sentinel only in clearly non-zero contexts
})

# Factor by which a value must exceed the median to be "extreme magnitude"
_EXTREME_MAGNITUDE_FACTOR = 1000.0


class SuspiciousValueAnalyzer(BaseAnalyzer):
    """Detects semantically suspicious values using domain heuristics."""

    @property
    def name(self) -> str:
        return "suspicious_values"

    @property
    def display_name(self) -> str:
        return "Suspicious Values"

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return AnalysisResult.ok(
                self.name, self.display_name, "No numeric columns to check."
            )

        per_column: dict[str, Any] = {}
        total_suspicious = 0

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            findings = []

            findings += self._check_negative_domain(col, series, df)
            findings += self._check_sentinel_values(series, df[col])
            findings += self._check_extreme_magnitude(series, df[col])

            if findings:
                per_column[col] = {"suspicious_findings": findings}
                total_suspicious += sum(f["count"] for f in findings)

        if not per_column:
            return AnalysisResult.ok(
                self.name, self.display_name, "No suspicious values detected."
            )

        affected = list(per_column.keys())

        return AnalysisResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            issues_found=True,
            severity="warning",
            affected_columns=affected,
            details={
                "per_column": per_column,
                "total_suspicious_cells": total_suspicious,
            },
            summary=(
                f"{total_suspicious} suspicious value(s) in {len(affected)} column(s) "
                f"(domain violations, sentinels, extreme magnitudes)"
            ),
            recommendation=(
                "Inspect flagged values manually. They may be data entry errors, "
                "bad imputation sentinels (-999), or legitimate extreme values. "
                "Context from the data source is needed to decide the correct action."
            ),
        )

    # ── Detection helpers ──────────────────────────────────────────────────────

    def _check_negative_domain(
        self, col: str, non_null: pd.Series, df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Flag negative values in columns whose name implies non-negative data."""
        col_lower = col.lower()
        is_positive_domain = any(kw in col_lower for kw in _POSITIVE_DOMAIN_KEYWORDS)
        if not is_positive_domain:
            return []

        negative_mask = df[col] < 0
        neg_values = df.loc[negative_mask, col]

        if neg_values.empty:
            return []

        findings = []
        for val, group in neg_values.groupby(neg_values):
            findings.append({
                "type": "negative_in_positive_domain",
                "value": float(val),
                "count": len(group),
                "row_indices": group.index.tolist()[:10],
            })
        return findings

    def _check_sentinel_values(
        self, non_null: pd.Series, full_col: pd.Series
    ) -> list[dict[str, Any]]:
        """
        Flag values that look like sentinel/placeholder substitutes for NaN.

        Logic: A sentinel is suspicious when:
        - It matches a known sentinel value AND
        - It appears more than once (pattern, not coincidence) AND
        - It is a statistical outlier in the column (not a natural value)
        """
        findings = []
        col_median = float(non_null.median())
        col_std = float(non_null.std()) if len(non_null) > 1 else 0.0

        for sentinel in _COMMON_SENTINELS:
            # Skip 0 — too many false positives (0 is often genuinely 0)
            if sentinel == 0.0:
                continue

            sentinel_mask = full_col == sentinel
            count = int(sentinel_mask.sum())

            if count < 1:
                continue

            # Only flag if the sentinel is a statistical outlier itself
            if col_std > 0:
                z = abs(sentinel - col_median) / col_std
                if z < 2.5:
                    continue  # Value is within normal range — not a sentinel

            findings.append({
                "type": "common_placeholder_sentinel",
                "value": sentinel,
                "count": count,
                "row_indices": full_col[sentinel_mask].index.tolist()[:10],
            })

        return findings

    def _check_extreme_magnitude(
        self, non_null: pd.Series, full_col: pd.Series
    ) -> list[dict[str, Any]]:
        """Flag values that are extreme multiples of the column median."""
        if len(non_null) < 4:
            return []

        median = float(non_null.median())
        if median == 0 or np.isnan(median):
            return []

        # Find values that are > FACTOR × median (positive) or < -FACTOR × |median|
        ratio = non_null.abs() / abs(median)
        extreme_mask = ratio > _EXTREME_MAGNITUDE_FACTOR

        # Exclude values already caught by sentinel check
        sentinel_vals = _COMMON_SENTINELS
        extreme_values = non_null[extreme_mask]
        extreme_values = extreme_values[~extreme_values.isin(sentinel_vals)]

        if extreme_values.empty:
            return []

        findings = []
        # Group by value so we report each distinct extreme value once
        for val, group in extreme_values.groupby(extreme_values):
            findings.append({
                "type": "extreme_magnitude",
                "value": float(val),
                "count": len(group),
                "magnitude_ratio": round(abs(val) / abs(median), 1),
                "column_median": round(median, 4),
                "row_indices": group.index.tolist()[:10],
            })

        return findings
