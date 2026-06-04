"""
Type Caster Cleaner.

WHAT IT DOES:
    Converts columns from their incorrect stored dtype to their correct dtype:
    - Numeric strings  → float64 (via pd.to_numeric)
    - Date strings     → datetime64[ns] (via pd.to_datetime)
    - Boolean integers → bool (via .astype(bool))

WHY THIS RUNS BEFORE MISSING VALUE IMPUTATION:
    If "age" is stored as strings ["25", "30", None], we CANNOT compute
    the median for imputation until it's a numeric column. Type casting
    must happen before any statistical operations.

SAFETY — WHAT HAPPENS ON PARTIAL CONVERSION FAILURE:
    pd.to_numeric(errors='coerce') converts what it can and fills
    failures with NaN. This means a partially numeric column becomes
    properly numeric with NaN for the unconvertible cells. Those NaN
    values will then be handled by the MissingValueCleaner.
    We NEVER silently lose data.

OPERATION ORDER:
    Runs 3rd in the pipeline — after deduplication and constant column removal.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.cleaners.base import BaseCleaner, CleaningOperation, CleaningResult

logger = logging.getLogger("lazypanda")


class TypeCasterCleaner(BaseCleaner):
    """Converts column dtypes from their incorrect stored type to the correct type."""

    @property
    def name(self) -> str:
        return "type_caster"

    @property
    def display_name(self) -> str:
        return "Type Casting"

    def clean(
        self,
        df: pd.DataFrame,
        analysis_result: AnalysisResult | None = None,
    ) -> tuple[pd.DataFrame, CleaningResult]:

        if analysis_result is not None and not analysis_result.issues_found:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No type issues detected.")

        # Extract type issues from analysis or fallback to empty
        if analysis_result is not None:
            type_issues: dict[str, Any] = analysis_result.details.get("type_issues", {})
        else:
            type_issues = {}

        if not type_issues:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No type issues to fix.")

        df_clean = df.copy()
        ops: list[CleaningOperation] = []

        for col, issue in type_issues.items():
            if col not in df_clean.columns:
                continue

            issue_type = issue.get("issue_type", "")
            from_dtype = str(df_clean[col].dtype)
            cells_affected = 0

            try:
                if issue_type in ("numeric_string", "mixed_numeric_string"):
                    df_clean[col], cells_affected = self._cast_to_numeric(df_clean, col)
                    to_dtype = "float64"

                elif issue_type == "date_string":
                    df_clean[col], cells_affected = self._cast_to_datetime(df_clean, col)
                    to_dtype = "datetime64[ns]"

                elif issue_type == "boolean_as_integer":
                    df_clean[col] = df_clean[col].astype(bool)
                    cells_affected = df_clean[col].notna().sum()
                    to_dtype = "bool"

                else:
                    continue  # Unknown issue type — skip safely

                logger.info(
                    "TypeCasterCleaner: '%s' %s → %s (%d cells)", col, from_dtype, to_dtype, cells_affected
                )
                ops.append(CleaningOperation(
                    type="cast_column",
                    description=f"Converted '{col}' from {from_dtype} to {to_dtype}",
                    details={
                        "column": col,
                        "from_dtype": from_dtype,
                        "to_dtype": to_dtype,
                        "cells_converted": cells_affected,
                        "issue_type": issue_type,
                    },
                ))

            except Exception as e:
                logger.warning("TypeCasterCleaner: could not cast '%s': %s", col, e)
                # Do NOT raise — move on to the next column
                ops.append(CleaningOperation(
                    type="cast_column_failed",
                    description=f"Could not cast '{col}' — skipped ({e})",
                    details={"column": col, "error": str(e)},
                ))

        if not ops:
            return df, CleaningResult.skipped(self.name, self.display_name, df, "No type conversions applied.")

        summary = f"Converted {len([o for o in ops if o.type == 'cast_column'])} column dtype(s)"
        return df_clean, self._make_result(df_before=df, df_after=df_clean, operations=ops, summary=summary)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _cast_to_numeric(self, df: pd.DataFrame, col: str) -> tuple[pd.Series, int]:
        """Convert a string column to float64. Returns (series, n_cells_converted)."""
        stripped = df[col].astype(str).str.strip()
        converted = pd.to_numeric(stripped, errors="coerce")
        n_converted = int(converted.notna().sum())
        return converted, n_converted

    def _cast_to_datetime(self, df: pd.DataFrame, col: str) -> tuple[pd.Series, int]:
        """Convert a string column to datetime64. Returns (series, n_cells_converted)."""
        str_series = df[col].astype(str)
        try:
            converted = pd.to_datetime(str_series, errors="coerce", format="mixed")
        except Exception:
            converted = pd.to_datetime(str_series, errors="coerce")
        n_converted = int(converted.notna().sum())
        return converted, n_converted
