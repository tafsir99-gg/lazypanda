"""
Outlier Analyzer.

WHAT IT DETECTS:
    Statistical outliers in numeric columns using two methods:

    IQR METHOD (default, robust):
        Lower fence = Q1 - (multiplier × IQR)
        Upper fence = Q3 + (multiplier × IQR)
        Anything outside the fences is an outlier.
        WHY IQR? It is resistant to the outliers themselves. The median-based
        quartiles don't shift when there are extreme values, unlike the mean.

    Z-SCORE METHOD (assumes normal distribution):
        z = (value - mean) / std
        Outlier if |z| > threshold (default: 3.0)
        WHY Z-SCORE? Better for truly normal distributions. IQR can be too
        aggressive on skewed data. Use "both" mode for belt-and-suspenders.

IMPORTANT:
    Outlier detection is purely statistical. Whether an outlier is a REAL
    data error or a genuine extreme value is a SEMANTIC question → that is
    where Gemini adds value (Milestone 5). This analyzer only flags them.

OUTPUT DETAILS SCHEMA:
    {
        "per_column": {
            "age": {
                "method": "iqr",
                "n_outliers": 3,
                "outlier_pct": 0.034,
                "lower_fence": -1.5,
                "upper_fence": 72.5,
                "q1": 22.0, "q3": 57.0, "iqr": 35.0,
                "min_value": -5.0,
                "max_value": 999.0,
                "example_outlier_values": [-5.0, 999.0],
            }
        },
        "total_outlier_cells": 5,
        "method_used": "iqr",
    }
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ai_data_cleaner.analyzers.base import AnalysisResult, BaseAnalyzer

logger = logging.getLogger("ai_data_cleaner")


class OutlierAnalyzer(BaseAnalyzer):
    """Detects statistical outliers in numeric columns."""

    @property
    def name(self) -> str:
        return "outliers"

    @property
    def display_name(self) -> str:
        return "Outliers"

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        method: str = self.config.get("method", "iqr")
        iqr_mult: float = self.config.get("iqr_multiplier", 1.5)
        zscore_thresh: float = self.config.get("zscore_threshold", 3.0)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return AnalysisResult.ok(
                self.name, self.display_name, "No numeric columns to analyze for outliers."
            )

        per_column: dict[str, Any] = {}
        total_outlier_cells = 0

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:  # Need at least 4 points for meaningful stats
                continue

            col_result = self._analyze_column(col, series, method, iqr_mult, zscore_thresh)
            if col_result and col_result["n_outliers"] > 0:
                per_column[col] = col_result
                total_outlier_cells += col_result["n_outliers"]

        if not per_column:
            return AnalysisResult.ok(self.name, self.display_name, "No statistical outliers detected.")

        affected = list(per_column.keys())
        # Severity by proportion of columns affected
        outlier_col_pct = len(affected) / max(len(numeric_cols), 1)
        severity = "critical" if outlier_col_pct > 0.5 else "warning"

        return AnalysisResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            issues_found=True,
            severity=severity,
            affected_columns=affected,
            details={
                "per_column": per_column,
                "total_outlier_cells": total_outlier_cells,
                "method_used": method,
            },
            summary=(
                f"Outliers detected in {len(affected)} of {len(numeric_cols)} numeric column(s) "
                f"({total_outlier_cells:,} total outlier value(s))"
            ),
            recommendation=(
                f"Review {len(affected)} column(s) for outliers. "
                f"Current action: '{self.config.get('action', 'flag')}'. "
                f"Consider 'cap' for ML models or 'drop' if values are data errors."
            ),
        )

    def _analyze_column(
        self,
        col: str,
        series: pd.Series,
        method: str,
        iqr_mult: float,
        zscore_thresh: float,
    ) -> dict[str, Any] | None:
        """Run outlier detection on a single column. Returns None if <4 values."""
        result: dict[str, Any] = {}

        if method in ("iqr", "both"):
            iqr_result = self._iqr_outliers(series, iqr_mult)
            result.update(iqr_result)

        if method in ("zscore", "both"):
            zscore_result = self._zscore_outliers(series, zscore_thresh)
            if method == "both":
                # Combine: a value is an outlier if EITHER method flags it
                combined_mask = iqr_result.get("_mask", pd.Series(False, index=series.index))
                z_mask = zscore_result.get("_mask", pd.Series(False, index=series.index))
                combined = combined_mask | z_mask
                result["n_outliers"] = int(combined.sum())
                result["outlier_pct"] = round(result["n_outliers"] / len(series), 4)
                result["example_outlier_values"] = series[combined].head(5).tolist()
                result["method"] = "both"
            else:
                result.update(zscore_result)

        # Clean up internal mask keys before returning
        result.pop("_mask", None)

        result["min_value"] = float(series.min())
        result["max_value"] = float(series.max())
        result["mean_value"] = round(float(series.mean()), 4)
        result["std_value"] = round(float(series.std()), 4)

        return result if result.get("n_outliers", 0) > 0 else None

    def _iqr_outliers(self, series: pd.Series, multiplier: float) -> dict[str, Any]:
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = (series < lower) | (series > upper)
        n = int(mask.sum())
        return {
            "method": "iqr",
            "n_outliers": n,
            "outlier_pct": round(n / len(series), 4),
            "lower_fence": round(lower, 4),
            "upper_fence": round(upper, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "example_outlier_values": series[mask].head(5).tolist(),
            "_mask": mask,  # Internal — used for "both" mode, stripped before return
        }

    def _zscore_outliers(self, series: pd.Series, threshold: float) -> dict[str, Any]:
        z_scores = np.abs(stats.zscore(series))
        mask = z_scores > threshold
        n = int(mask.sum())
        return {
            "method": "zscore",
            "n_outliers": n,
            "outlier_pct": round(n / len(series), 4),
            "zscore_threshold": threshold,
            "max_zscore": round(float(z_scores.max()), 4),
            "example_outlier_values": series[mask].head(5).tolist(),
            "_mask": pd.Series(mask, index=series.index),
        }
