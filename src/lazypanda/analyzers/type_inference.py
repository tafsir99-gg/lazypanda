"""
Type Inference Analyzer.

WHAT IT DETECTS:
    Columns whose STORED dtype does not match their ACTUAL content:

    1. NUMERIC STRINGS: Column dtype is "object" but all non-null values
       parse successfully as numbers → should be float64.
       Example: age column stored as ["25", "30", "22"] instead of [25, 30, 22]

    2. DATE STRINGS: Column dtype is "object" and values match common date
       patterns (YYYY-MM-DD, MM/DD/YYYY, etc.) → should be datetime64.
       Example: "created_at" stored as "2024-01-15" strings.

    3. BOOLEAN-LIKE NUMERICS: Numeric column with ONLY {0, 1} values and
       meaningful column name → likely a boolean flag stored as int.
       Example: "is_active" column with values 0 and 1.

    4. MIXED TYPE COLUMNS: Object column where SOME values parse as numbers
       and some don't → genuinely mixed types, needs cleaning.

WHY PYTHON NOT GEMINI:
    pd.to_numeric(errors='coerce') and pd.to_datetime(errors='coerce') handle
    type detection deterministically and for free.

OUTPUT DETAILS SCHEMA:
    {
        "type_issues": {
            "age": {
                "current_dtype": "object",
                "suggested_dtype": "float64",
                "issue_type": "numeric_string",
                "convertible_pct": 1.0,
            },
            "signup_date": {
                "current_dtype": "object",
                "suggested_dtype": "datetime64[ns]",
                "issue_type": "date_string",
                "convertible_pct": 0.98,
            },
        }
    }
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult, BaseAnalyzer

logger = logging.getLogger("lazypanda")

# Common date-like column name keywords (heuristic for date detection)
_DATE_COLUMN_HINTS = frozenset({
    "date", "time", "datetime", "timestamp", "created", "updated",
    "modified", "at", "on", "day", "month", "year", "dob", "born",
})

# Boolean flag name keywords
_BOOL_COLUMN_HINTS = frozenset({
    "is_", "has_", "was_", "can_", "will_", "flag", "active",
    "enabled", "disabled", "deleted", "verified", "confirmed",
})


class TypeInferenceAnalyzer(BaseAnalyzer):
    """Detects columns where the stored dtype mismatches the actual content."""

    @property
    def name(self) -> str:
        return "type_inference"

    @property
    def display_name(self) -> str:
        return "Data Type Issues"

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        type_issues: dict[str, Any] = {}

        for col in df.columns:
            col_series = df[col]
            dtype_str = str(col_series.dtype)
            non_null = col_series.dropna()

            if len(non_null) == 0:
                continue  # All-null column — handled by MissingValueAnalyzer

            # pandas 3 uses 'str' for native StringDtype; pandas 2 uses 'object'
            is_string_like = dtype_str in ("object", "str") or dtype_str.startswith("string")

            issue = self._check_column(col, col_series, non_null, dtype_str, is_string_like)
            if issue:
                type_issues[col] = issue

        if not type_issues:
            return AnalysisResult.ok(self.name, self.display_name, "No data type issues detected.")

        severity = "warning" if len(type_issues) > 0 else "info"
        affected = list(type_issues.keys())

        return AnalysisResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            issues_found=True,
            severity=severity,
            affected_columns=affected,
            details={"type_issues": type_issues},
            summary=f"{len(type_issues)} column(s) with potential dtype mismatch",
            recommendation=(
                f"Convert {len(type_issues)} column(s) to their correct types "
                f"before training models. Wrong types cause silent errors in ML pipelines."
            ),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check_column(
        self,
        col: str,
        series: pd.Series,
        non_null: pd.Series,
        dtype_str: str,
        is_string_like: bool,
    ) -> dict[str, Any] | None:
        """Return an issue dict if the column has a type mismatch, else None."""

        # ── Check 1: Numeric strings (string-like dtype, numeric content) ────
        if is_string_like:
            numeric_issue = self._check_numeric_string(col, series, non_null)
            if numeric_issue:
                return numeric_issue

            date_issue = self._check_date_string(col, series, non_null)
            if date_issue:
                return date_issue

        # ── Check 2: Boolean-like integers ───────────────────────────────
        if pd.api.types.is_integer_dtype(series):
            bool_issue = self._check_boolean_like(col, series, non_null)
            if bool_issue:
                return bool_issue

        return None

    def _check_numeric_string(
        self, col: str, series: pd.Series, non_null: pd.Series
    ) -> dict[str, Any] | None:
        """Detect object columns whose content is actually numeric."""
        # Strip whitespace before testing
        stripped = non_null.astype(str).str.strip()
        converted = pd.to_numeric(stripped, errors="coerce")
        convertible_count = converted.notna().sum()
        convertible_pct = convertible_count / len(non_null)

        if convertible_pct >= 0.95:
            issue_type = "numeric_string" if convertible_pct == 1.0 else "mixed_numeric_string"
            return {
                "current_dtype": str(series.dtype),
                "suggested_dtype": "float64",
                "issue_type": issue_type,
                "convertible_pct": round(float(convertible_pct), 4),
                "non_convertible_count": int(len(non_null) - convertible_count),
            }
        return None

    def _check_date_string(
        self, col: str, series: pd.Series, non_null: pd.Series
    ) -> dict[str, Any] | None:
        """
        Detect string columns whose content looks like dates.
        Two signals required: date-like column name + parseable date values.
        """
        col_lower = col.lower()
        name_is_date_like = any(hint in col_lower for hint in _DATE_COLUMN_HINTS)
        if not name_is_date_like:
            return None

        sample = non_null.astype(str).head(100)
        try:
            # infer_datetime_format is deprecated in pandas 2.2+; use format='mixed'
            converted = pd.to_datetime(sample, errors="coerce", format="mixed")
        except Exception:
            try:
                converted = pd.to_datetime(sample, errors="coerce")
            except Exception:
                return None

        convertible_pct = converted.notna().sum() / len(sample)

        if convertible_pct >= 0.8:
            return {
                "current_dtype": str(series.dtype),
                "suggested_dtype": "datetime64[ns]",
                "issue_type": "date_string",
                "convertible_pct": round(float(convertible_pct), 4),
                "sample_values": sample.head(3).tolist(),
            }
        return None

    def _check_boolean_like(
        self, col: str, series: pd.Series, non_null: pd.Series
    ) -> dict[str, Any] | None:
        """Detect integer columns that are really boolean flags (only 0 and 1)."""
        unique_vals = set(non_null.unique())
        if unique_vals <= {0, 1}:
            col_lower = col.lower()
            name_suggests_bool = any(col_lower.startswith(h) or h in col_lower
                                     for h in _BOOL_COLUMN_HINTS)
            if name_suggests_bool or len(non_null) > 50:  # Large numeric-01 cols likely flags
                return {
                    "current_dtype": str(series.dtype),
                    "suggested_dtype": "bool",
                    "issue_type": "boolean_as_integer",
                    "unique_values": sorted(unique_vals),
                }
        return None
