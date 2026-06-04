"""
Outlier Handler Cleaner.

WHAT IT DOES:
    Handles statistical outliers in numeric columns using one of three actions:

    "flag" (default): Add a boolean indicator column {col}_is_outlier.
        Best for: exploring the data, keeping outliers for now.
        Example: fare → fare_is_outlier (True/False)

    "cap" (Winsorization): Clip values to the IQR fence boundaries.
        Best for: ML models — extreme values become fence values, not dropped.
        Example: age=-5 → age=age_lower_fence, age=999 → age=upper_fence

    "drop": Drop rows containing any outlier in any flagged column.
        Best for: when outliers are confirmed data errors (not extreme reality).
        Warning: Can remove many rows — check result shape carefully.

WHY NOT AUTO-DECIDE WHICH TO USE:
    Whether age=150 is a typo or a valid super-centenarian requires domain
    knowledge. The cleaner respects the config setting and does not guess.
    The AI layer (Milestone 5) will add semantic reasoning here.

OPERATION ORDER:
    Runs LAST — after imputation fills all NaN values.
    Outlier stats (Q1, Q3, IQR) are meaningless with missing values present.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.cleaners.base import BaseCleaner, CleaningOperation, CleaningResult

logger = logging.getLogger("lazypanda")


class OutlierHandlerCleaner(BaseCleaner):
    """Handles outliers via flagging, capping, or dropping."""

    @property
    def name(self) -> str:
        return "outlier_handler"

    @property
    def display_name(self) -> str:
        return "Outlier Handling"

    def clean(
        self,
        df: pd.DataFrame,
        analysis_result: AnalysisResult | None = None,
    ) -> tuple[pd.DataFrame, CleaningResult]:

        action: str = self.config.get("action", "flag")
        iqr_mult: float = self.config.get("iqr_multiplier", 1.5)

        if analysis_result is not None and not analysis_result.issues_found:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No outliers detected.")

        # Get per-column outlier info from analysis
        if analysis_result is not None:
            per_column: dict[str, Any] = analysis_result.details.get("per_column", {})
        else:
            per_column = self._detect_all(df, iqr_mult)

        if not per_column:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No outlier data available.")

        df_clean = df.copy()
        ops: list[CleaningOperation] = []
        outlier_mask_any = pd.Series(False, index=df_clean.index)

        for col, col_info in per_column.items():
            if col not in df_clean.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue

            n_outliers = col_info.get("n_outliers", 0)
            if n_outliers == 0:
                continue

            # Rebuild the outlier mask for this column
            lower = col_info.get("lower_fence")
            upper = col_info.get("upper_fence")
            if lower is None or upper is None:
                # Recompute from data if not in analysis
                lower, upper = self._compute_fences(df_clean[col], iqr_mult)

            col_outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)

            if action == "flag":
                flag_col = f"{col}_is_outlier"
                df_clean[flag_col] = col_outlier_mask
                ops.append(CleaningOperation(
                    type="flag_outlier",
                    description=f"Added flag column '{flag_col}' ({n_outliers} outlier(s))",
                    details={"column": col, "flag_column": flag_col,
                             "n_outliers": n_outliers, "lower_fence": lower, "upper_fence": upper},
                ))

            elif action == "cap":
                df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                ops.append(CleaningOperation(
                    type="cap_outlier",
                    description=f"Capped '{col}' to [{lower:.4g}, {upper:.4g}] ({n_outliers} value(s) capped)",
                    details={"column": col, "lower_fence": lower, "upper_fence": upper,
                             "cells_capped": n_outliers},
                ))

            elif action == "drop":
                outlier_mask_any = outlier_mask_any | col_outlier_mask
                ops.append(CleaningOperation(
                    type="mark_for_drop_outlier",
                    description=f"Marked '{col}' outliers for row removal ({n_outliers} row(s))",
                    details={"column": col, "n_outliers": n_outliers},
                ))

        # For drop action: remove all rows where any column had an outlier
        if action == "drop" and outlier_mask_any.any():
            rows_to_drop = int(outlier_mask_any.sum())
            df_clean = df_clean[~outlier_mask_any].reset_index(drop=True)
            logger.info("OutlierHandlerCleaner: dropped %d row(s) containing outliers", rows_to_drop)
            ops.append(CleaningOperation(
                type="drop_rows",
                description=f"Dropped {rows_to_drop} row(s) containing outlier values",
                details={"rows_dropped": rows_to_drop},
            ))

        if not ops:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No outlier operations applied.")

        action_past = {"flag": "flagged", "cap": "capped", "drop": "removed rows for"}.get(action, action)
        affected_cols = [o.details.get("column", "") for o in ops if "column" in o.details]
        summary = f"Outlier action='{action}': {action_past} outliers in {len(set(affected_cols))} column(s)"

        return df_clean, self._make_result(df_before=df, df_after=df_clean, operations=ops, summary=summary)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_fences(self, series: pd.Series, multiplier: float) -> tuple[float, float]:
        """Compute IQR fences for a column."""
        non_null = series.dropna()
        q1 = float(non_null.quantile(0.25))
        q3 = float(non_null.quantile(0.75))
        iqr = q3 - q1
        return q1 - multiplier * iqr, q3 + multiplier * iqr

    def _detect_all(self, df: pd.DataFrame, multiplier: float) -> dict[str, Any]:
        """Fallback: detect outliers without an AnalysisResult."""
        result = {}
        for col in df.select_dtypes(include="number").columns:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            lower, upper = self._compute_fences(series, multiplier)
            mask = (series < lower) | (series > upper)
            n = int(mask.sum())
            if n > 0:
                result[col] = {
                    "n_outliers": n,
                    "lower_fence": lower,
                    "upper_fence": upper,
                }
        return result
