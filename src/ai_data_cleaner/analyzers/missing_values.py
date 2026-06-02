"""
Missing Value Analyzer.

WHAT IT DETECTS:
    - Columns with any missing values (NaN, None, pd.NA)
    - Columns above the drop_column_threshold (likely useless)
    - Rows where too many columns are missing simultaneously

WHY PYTHON NOT GEMINI:
    df.isnull().sum() is deterministic, exact, and free.
    Gemini cannot add value here — the math is the answer.

OUTPUT DETAILS SCHEMA:
    {
        "per_column": {
            "age": {
                "missing_count": 177,
                "missing_pct": 0.198,
                "recommended_action": "impute_median"  | "drop_column" | "ok"
            },
            ...
        },
        "columns_to_drop":   ["cabin"],       # above drop_column_threshold
        "columns_to_impute": {"age": "median", "embarked": "mode"},
        "rows_to_drop_count": 3,              # rows above drop_row_threshold
        "total_missing_cells": 866,
        "total_missing_pct": 0.077,
    }
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ai_data_cleaner.analyzers.base import AnalysisResult, BaseAnalyzer

logger = logging.getLogger("ai_data_cleaner")


class MissingValueAnalyzer(BaseAnalyzer):
    """Detects and characterizes missing values in a DataFrame."""

    @property
    def name(self) -> str:
        return "missing_values"

    @property
    def display_name(self) -> str:
        return "Missing Values"

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols

        drop_col_threshold: float = self.config.get("drop_column_threshold", 0.8)
        drop_row_threshold: float = self.config.get("drop_row_threshold", 0.5)
        numeric_impute: str = self.config.get("numeric_imputation", "median")
        cat_impute: str = self.config.get("categorical_imputation", "mode")

        # ── Per-column analysis ────────────────────────────────────────────────
        missing_counts = df.isnull().sum()
        columns_to_drop: list[str] = []
        columns_to_impute: dict[str, str] = {}
        per_column: dict[str, Any] = {}

        for col in df.columns:
            count = int(missing_counts[col])
            if count == 0:
                continue  # Skip clean columns entirely

            pct = count / n_rows
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            if pct >= drop_col_threshold:
                action = "drop_column"
                columns_to_drop.append(col)
            else:
                action = f"impute_{numeric_impute}" if is_numeric else f"impute_{cat_impute}"
                impute_strategy = numeric_impute if is_numeric else cat_impute
                if impute_strategy != "none":
                    columns_to_impute[col] = impute_strategy

            per_column[col] = {
                "missing_count": count,
                "missing_pct": round(pct, 4),
                "recommended_action": action,
            }

        # ── Row-level analysis ─────────────────────────────────────────────────
        row_missing_fractions = df.isnull().mean(axis=1)
        rows_to_drop_count = int((row_missing_fractions >= drop_row_threshold).sum())

        # ── Aggregate stats ────────────────────────────────────────────────────
        total_missing = int(missing_counts.sum())
        total_missing_pct = round(total_missing / total_cells, 4) if total_cells > 0 else 0.0
        affected_cols = list(per_column.keys())

        if not affected_cols:
            return AnalysisResult.ok(self.name, self.display_name, "No missing values detected.")

        # ── Determine severity ─────────────────────────────────────────────────
        if columns_to_drop or rows_to_drop_count > n_rows * 0.1:
            severity = "critical"
        elif total_missing_pct > 0.05:
            severity = "warning"
        else:
            severity = "info"

        summary = (
            f"{total_missing:,} missing cells across {len(affected_cols)} columns "
            f"({total_missing_pct:.1%} of dataset)"
        )
        recommendation = self._build_recommendation(columns_to_drop, columns_to_impute, rows_to_drop_count)

        return AnalysisResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            issues_found=True,
            severity=severity,
            affected_columns=affected_cols,
            details={
                "per_column": per_column,
                "columns_to_drop": columns_to_drop,
                "columns_to_impute": columns_to_impute,
                "rows_to_drop_count": rows_to_drop_count,
                "total_missing_cells": total_missing,
                "total_missing_pct": total_missing_pct,
            },
            summary=summary,
            recommendation=recommendation,
        )

    def _build_recommendation(
        self,
        columns_to_drop: list[str],
        columns_to_impute: dict[str, str],
        rows_to_drop_count: int,
    ) -> str:
        parts = []
        if columns_to_drop:
            parts.append(f"Drop {len(columns_to_drop)} column(s): {', '.join(columns_to_drop[:3])}"
                         + ("..." if len(columns_to_drop) > 3 else ""))
        if columns_to_impute:
            strategies = set(columns_to_impute.values())
            parts.append(f"Impute {len(columns_to_impute)} column(s) using: {', '.join(strategies)}")
        if rows_to_drop_count:
            parts.append(f"Drop {rows_to_drop_count} row(s) with excessive missing values")
        return "; ".join(parts) if parts else "Review missing value patterns."
