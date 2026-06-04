"""
Category Normalizer Cleaner.

WHAT IT DOES:
    Applies string normalization to categorical (string/object dtype) columns
    to eliminate inconsistencies caused by case and whitespace variations:
    - "Male" / "male" / "MALE"  →  "male"
    - "active " / " active"     →  "active"

WHY THIS RUNS BEFORE MISSING VALUE IMPUTATION:
    Mode imputation on a "sex" column with ["Male", "male", "MALE", "female"]
    would compute mode as one of the variants (all appear once), rather than
    the true mode "male" (appears 3 times after normalization).
    Normalizing first makes imputation more accurate.

IMPORTANT — WHAT IS NOT CHANGED:
    - Numeric columns are never touched
    - Normalization is only applied when the config explicitly enables it
    - The normalized values are always lowercase/stripped — never dropped

OPERATION ORDER:
    Runs 4th — after type casting, so numeric strings that became float64
    are not accidentally normalized as strings.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.cleaners.base import BaseCleaner, CleaningOperation, CleaningResult

logger = logging.getLogger("lazypanda")


class CategoryNormalizerCleaner(BaseCleaner):
    """Normalizes case and whitespace in categorical columns."""

    @property
    def name(self) -> str:
        return "category_normalizer"

    @property
    def display_name(self) -> str:
        return "Category Normalization"

    def clean(
        self,
        df: pd.DataFrame,
        analysis_result: AnalysisResult | None = None,
    ) -> tuple[pd.DataFrame, CleaningResult]:

        normalize_case: bool = self.config.get("normalize_case", True)
        strip_ws: bool = self.config.get("strip_whitespace", True)

        if not normalize_case and not strip_ws:
            return df, CleaningResult.skipped(
                self.name, self.display_name, df, "Normalization disabled in config."
            )

        if analysis_result is not None and not analysis_result.issues_found:
            return df, CleaningResult.skipped(
                self.name, self.display_name, df, "No category inconsistencies detected."
            )

        # Determine which columns to normalize
        if analysis_result is not None:
            cols_to_fix = list(analysis_result.details.get("per_column", {}).keys())
        else:
            # Fallback: normalize ALL object/string columns
            cols_to_fix = df.select_dtypes(include=["object", "string", "str"]).columns.tolist()

        if not cols_to_fix:
            return df, CleaningResult.skipped(
                self.name, self.display_name, df, "No categorical columns to normalize."
            )

        df_clean = df.copy()
        ops: list[CleaningOperation] = []

        for col in cols_to_fix:
            if col not in df_clean.columns:
                continue

            series = df_clean[col]
            if not pd.api.types.is_object_dtype(series) and str(series.dtype) not in ("str", "string"):
                continue  # Only normalize string-like columns

            raw_unique = series.nunique(dropna=True)
            normalized = series.copy()

            if strip_ws:
                normalized = normalized.str.strip()
            if normalize_case:
                normalized = normalized.str.lower()

            new_unique = normalized.nunique(dropna=True)
            n_cells_changed = int((series != normalized).sum())

            df_clean[col] = normalized
            logger.info(
                "CategoryNormalizerCleaner: '%s' unique values %d → %d (%d cells changed)",
                col, raw_unique, new_unique, n_cells_changed,
            )

            ops_applied = []
            if strip_ws:
                ops_applied.append("strip whitespace")
            if normalize_case:
                ops_applied.append("lowercase")

            ops.append(CleaningOperation(
                type="normalize_category",
                description=f"Normalized '{col}': {raw_unique} → {new_unique} unique values ({', '.join(ops_applied)})",
                details={
                    "column": col,
                    "raw_unique_count": raw_unique,
                    "normalized_unique_count": new_unique,
                    "cells_changed": n_cells_changed,
                    "operations_applied": ops_applied,
                },
            ))

        if not ops:
            return df, CleaningResult.skipped(
                self.name, self.display_name, df, "No normalization changes required."
            )

        return df_clean, self._make_result(
            df_before=df,
            df_after=df_clean,
            operations=ops,
            summary=f"Normalized {len(ops)} categorical column(s) (case + whitespace)",
        )
