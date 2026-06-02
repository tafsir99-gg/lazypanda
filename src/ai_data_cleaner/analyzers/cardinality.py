"""
Cardinality Analyzer.

WHAT IT DETECTS:
    Three related but distinct problems:

    1. CONSTANT COLUMNS: Only one unique value across all rows.
       These columns carry zero information for ML — every row is identical.
       Example: A "version" column that is always "v1.0".

    2. NEAR-CONSTANT COLUMNS: One value dominates (>99% by default).
       These are almost useless for ML — the model can barely learn from them.
       Example: A "fraud" column where 99.5% of rows are False.

    3. HIGH-CARDINALITY COLUMNS: Almost every row has a unique value.
       These are often ID or free-text columns that shouldn't be one-hot encoded.
       Example: "customer_id", "name", "email", "comment".
       High-cardinality + categorical dtype = explosion in feature space.

WHY THIS MATTERS FOR ML:
    - Constant columns waste memory and can cause sklearn to throw errors.
    - Near-constant columns skew feature importance.
    - High-cardinality columns, if naively encoded, produce 10,000+ features.

OUTPUT DETAILS SCHEMA:
    {
        "constant_columns": ["version"],
        "near_constant_columns": {
            "fraud": {
                "dominant_value": false,
                "dominant_count": 995,
                "dominant_pct": 0.995,
                "unique_count": 2,
            }
        },
        "high_cardinality_columns": {
            "name": {
                "unique_count": 891,
                "unique_ratio": 1.0,
                "top_3_values": ["Alice", "Bob", "Carol"],
            }
        },
    }
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ai_data_cleaner.analyzers.base import AnalysisResult, BaseAnalyzer

logger = logging.getLogger("ai_data_cleaner")


class CardinalityAnalyzer(BaseAnalyzer):
    """Detects constant, near-constant, and high-cardinality columns."""

    @property
    def name(self) -> str:
        return "cardinality"

    @property
    def display_name(self) -> str:
        return "Cardinality"

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        high_card_thresh: float = self.config.get("high_cardinality_threshold", 0.95)
        near_const_thresh: float = self.config.get("near_constant_threshold", 0.99)
        n_rows = len(df)

        constant_columns: list[str] = []
        near_constant_columns: dict[str, Any] = {}
        high_cardinality_columns: dict[str, Any] = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            n_unique = series.nunique()
            unique_ratio = n_unique / n_rows

            # ── Constant: only 1 unique value ─────────────────────────────────
            if n_unique <= 1:
                constant_columns.append(col)
                continue  # No need to check further

            # ── Near-constant: one value dominates ────────────────────────────
            value_counts = series.value_counts()
            top_count = int(value_counts.iloc[0])
            top_pct = top_count / n_rows

            if top_pct >= near_const_thresh:
                near_constant_columns[col] = {
                    "dominant_value": _safe_json(value_counts.index[0]),
                    "dominant_count": top_count,
                    "dominant_pct": round(top_pct, 4),
                    "unique_count": n_unique,
                }
                continue  # Near-constant takes precedence over high-cardinality

            # ── High-cardinality: almost every row is unique ───────────────────
            if unique_ratio >= high_card_thresh:
                top_3 = [_safe_json(v) for v in value_counts.head(3).index.tolist()]
                high_cardinality_columns[col] = {
                    "unique_count": n_unique,
                    "unique_ratio": round(unique_ratio, 4),
                    "top_3_values": top_3,
                    "dtype": str(df[col].dtype),
                }

        issues_found = bool(constant_columns or near_constant_columns or high_cardinality_columns)

        if not issues_found:
            return AnalysisResult.ok(self.name, self.display_name, "No cardinality issues detected.")

        affected = (
            constant_columns
            + list(near_constant_columns.keys())
            + list(high_cardinality_columns.keys())
        )

        parts = []
        if constant_columns:
            parts.append(f"{len(constant_columns)} constant")
        if near_constant_columns:
            parts.append(f"{len(near_constant_columns)} near-constant")
        if high_cardinality_columns:
            parts.append(f"{len(high_cardinality_columns)} high-cardinality")

        summary = f"Cardinality issues: {', '.join(parts)} column(s)"

        severity = "critical" if constant_columns else "warning"

        recs = []
        if constant_columns:
            recs.append(f"Drop constant columns: {', '.join(constant_columns[:3])}")
        if near_constant_columns:
            recs.append(f"Review near-constant columns — low ML value")
        if high_cardinality_columns:
            recs.append(f"Avoid one-hot encoding high-cardinality columns; use target encoding")

        return AnalysisResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            issues_found=True,
            severity=severity,
            affected_columns=affected,
            details={
                "constant_columns": constant_columns,
                "near_constant_columns": near_constant_columns,
                "high_cardinality_columns": high_cardinality_columns,
            },
            summary=summary,
            recommendation="; ".join(recs),
        )


def _safe_json(value: Any) -> Any:
    """Convert a value to a JSON-serializable type."""
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)
