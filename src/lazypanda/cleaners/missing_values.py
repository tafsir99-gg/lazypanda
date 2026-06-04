"""
Missing Value Cleaner.

WHAT IT DOES:
    Executes the recommendations from MissingValueAnalyzer:
    1. Drops columns above the drop_column_threshold (e.g., >=80% missing)
    2. Drops rows above the drop_row_threshold (e.g., >=50% of columns missing)
    3. Imputes numeric columns with median / mean / zero / constant
    4. Imputes categorical columns with mode / constant

WHY MEDIAN IS DEFAULT FOR NUMERICS (not mean):
    The mean is sensitive to outliers. If "fare" has a sentinel value of
    -999, the mean is pulled negative, making it a terrible fill value.
    The median ignores outliers by design.

WHY MODE IS DEFAULT FOR CATEGORICALS:
    The most common value is the safest assumption when we have no other
    information. It preserves the existing distribution.

OPERATION ORDER:
    Runs 5th — after type casting (so dtypes are correct for statistics)
    and after category normalization (so mode computation is on normalized values).

    Order within this cleaner matters:
    1. Drop columns first (reduces the denominator for row missing fraction)
    2. Drop rows second (with updated column count)
    3. Impute remaining missing values
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.cleaners.base import BaseCleaner, CleaningOperation, CleaningResult

logger = logging.getLogger("lazypanda")

_IMPUTE_STRATEGIES = frozenset({"median", "mean", "mode", "zero", "constant"})


class MissingValueCleaner(BaseCleaner):
    """Drops and imputes missing values based on analyzer recommendations."""

    @property
    def name(self) -> str:
        return "missing_values"

    @property
    def display_name(self) -> str:
        return "Missing Value Handling"

    def clean(
        self,
        df: pd.DataFrame,
        analysis_result: AnalysisResult | None = None,
    ) -> tuple[pd.DataFrame, CleaningResult]:

        if analysis_result is not None and not analysis_result.issues_found:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No missing values detected.")

        # ── Extract decisions from analysis ────────────────────────────────────
        if analysis_result is not None:
            cols_to_drop: list[str] = analysis_result.details.get("columns_to_drop", [])
            cols_to_impute: dict[str, str] = analysis_result.details.get("columns_to_impute", {})
        else:
            cols_to_drop, cols_to_impute = self._compute_decisions(df)

        drop_row_threshold: float = self.config.get("drop_row_threshold", 0.5)
        constant_fill: Any = self.config.get("constant_fill_value", "UNKNOWN")

        df_clean = df.copy()
        ops: list[CleaningOperation] = []

        # ── Step 1: Drop high-missing columns ──────────────────────────────────
        valid_drop_cols = [c for c in cols_to_drop if c in df_clean.columns]
        if valid_drop_cols:
            df_clean = df_clean.drop(columns=valid_drop_cols)
            logger.info("MissingValueCleaner: dropped %d column(s): %s", len(valid_drop_cols), valid_drop_cols)
            for col in valid_drop_cols:
                ops.append(CleaningOperation(
                    type="drop_column",
                    description=f"Dropped '{col}' — too many missing values",
                    details={"column": col, "reason": "above_drop_threshold"},
                ))

        # ── Step 2: Drop rows with too many missing values ─────────────────────
        n_cols_now = len(df_clean.columns)
        # Keep row if strictly MORE than drop_row_threshold fraction is non-null.
        # E.g., drop_row_threshold=0.5 → drop if >=50% missing → keep if >50% present
        # thresh=N means "keep rows with at least N non-NaN values"
        # We need: at least floor(n_cols * (1 - threshold)) + 1 non-null values
        import math
        min_valid = math.floor(n_cols_now * (1.0 - drop_row_threshold)) + 1
        min_valid = min(min_valid, n_cols_now)  # Can't require more cols than exist

        df_before_row_drop = df_clean
        df_clean = df_clean.dropna(thresh=min_valid)

        rows_dropped = len(df_before_row_drop) - len(df_clean)
        if rows_dropped > 0:
            df_clean = df_clean.reset_index(drop=True)
            logger.info("MissingValueCleaner: dropped %d row(s) with excessive missing values", rows_dropped)
            ops.append(CleaningOperation(
                type="drop_rows",
                description=f"Dropped {rows_dropped} row(s) with >{drop_row_threshold:.0%} missing columns",
                details={"rows_dropped": rows_dropped, "threshold": drop_row_threshold},
            ))

        # ── Step 3: Impute remaining columns ───────────────────────────────────
        valid_impute = {c: s for c, s in cols_to_impute.items() if c in df_clean.columns}
        for col, strategy in valid_impute.items():
            series = df_clean[col]
            n_missing_before = int(series.isnull().sum())
            if n_missing_before == 0:
                continue

            fill_value, actual_strategy = self._compute_fill_value(
                series, strategy, constant_fill
            )
            if fill_value is None:
                continue

            df_clean[col] = series.fillna(fill_value)
            logger.info(
                "MissingValueCleaner: imputed '%s' with %s=%r (%d cells)",
                col, actual_strategy, fill_value, n_missing_before,
            )
            ops.append(CleaningOperation(
                type="impute_column",
                description=f"Imputed '{col}' with {actual_strategy} ({n_missing_before} cell(s) filled)",
                details={
                    "column": col,
                    "strategy": actual_strategy,
                    "fill_value": _safe_json(fill_value),
                    "cells_filled": n_missing_before,
                },
            ))

        if not ops:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No missing value changes applied.")

        return df_clean, self._make_result(
            df_before=df,
            df_after=df_clean,
            operations=ops,
            summary=(
                f"Missing values: dropped {len(valid_drop_cols)} col(s), "
                f"{rows_dropped} row(s), imputed {len(valid_impute)} col(s)"
            ),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_decisions(self, df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
        """Fallback: compute drop/impute decisions without an AnalysisResult."""
        drop_thresh = self.config.get("drop_column_threshold", 0.8)
        numeric_strategy = self.config.get("numeric_imputation", "median")
        cat_strategy = self.config.get("categorical_imputation", "mode")
        n_rows = len(df)

        cols_to_drop = []
        cols_to_impute = {}

        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            if n_missing == 0:
                continue
            pct = n_missing / n_rows
            if pct >= drop_thresh:
                cols_to_drop.append(col)
            else:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                cols_to_impute[col] = numeric_strategy if is_numeric else cat_strategy

        return cols_to_drop, cols_to_impute

    def _compute_fill_value(
        self,
        series: pd.Series,
        strategy: str,
        constant_fill: Any,
    ) -> tuple[Any, str]:
        """
        Compute the fill value for a given strategy.
        Returns (fill_value, actual_strategy_name).
        Returns (None, ...) if no fill value can be computed.
        """
        non_null = series.dropna()

        if len(non_null) == 0:
            return None, strategy  # All values missing — cannot impute

        if strategy == "median":
            if pd.api.types.is_numeric_dtype(series):
                return float(non_null.median()), "median"
            # Fallback to mode for non-numeric
            return non_null.mode().iloc[0] if not non_null.mode().empty else None, "mode"

        elif strategy == "mean":
            if pd.api.types.is_numeric_dtype(series):
                return float(non_null.mean()), "mean"
            return non_null.mode().iloc[0] if not non_null.mode().empty else None, "mode"

        elif strategy == "mode":
            mode_vals = non_null.mode()
            return (mode_vals.iloc[0], "mode") if not mode_vals.empty else (None, "mode")

        elif strategy == "zero":
            return 0, "zero"

        elif strategy == "constant":
            return constant_fill, "constant"

        else:
            logger.warning("Unknown imputation strategy '%s' for column '%s' — using mode", strategy, series.name)
            mode_vals = non_null.mode()
            return (mode_vals.iloc[0], "mode") if not mode_vals.empty else (None, "mode")


def _safe_json(value: Any) -> Any:
    """Convert numpy scalars to Python native types for JSON serialization."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)
