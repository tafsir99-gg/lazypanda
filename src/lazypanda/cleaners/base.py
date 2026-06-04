"""
Base classes for all cleaners.

DESIGN PRINCIPLE: Analyzers report; Cleaners transform.
    Every cleaner receives:
        1. A DataFrame to transform (always a copy — never the original)
        2. Its config section (thresholds, strategies)
        3. The AnalysisResult from the matching analyzer (to reuse findings,
           avoiding re-running the same calculations)

    Every cleaner returns:
        1. A new DataFrame (the transformation applied)
        2. A CleaningResult (a structured log of what changed and why)

WHY return a new DataFrame instead of mutating in place?
    Immutability makes each cleaner independently testable.
    The pipeline threads the DataFrame through cleaners like a chain:
        df → Cleaner1 → df1 → Cleaner2 → df2 → ... → final_df

WHY receive the AnalysisResult from the matching Analyzer?
    Token efficiency + speed. The pipeline already ran analysis.
    There is zero reason to run df.duplicated() again in the cleaner
    when the DuplicateAnalyzer already computed it.

IMPORTANT: Cleaners MUST NOT call Gemini or any external API.
    They execute the decision made by the analyzer (or the AI in M5).
    They are pure, fast, deterministic DataFrame transformations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult

logger = logging.getLogger("lazypanda")


@dataclass
class CleaningOperation:
    """
    A single atomic operation performed by a cleaner.

    Examples:
        CleaningOperation(type="drop_rows", count=1, reason="duplicate")
        CleaningOperation(type="impute_column", column="age", strategy="median", value=29.0)
        CleaningOperation(type="drop_column", column="cabin", reason="high_missing_pct")
        CleaningOperation(type="cap_outlier", column="fare", lower=0.0, upper=300.0, cells=3)
        CleaningOperation(type="normalize_case", column="sex", old_unique=5, new_unique=2)
        CleaningOperation(type="cast_column", column="price", from_dtype="object", to_dtype="float64")
    """
    type: str                           # Machine-readable operation type
    description: str                    # Human-readable description for reports
    details: dict[str, Any] = field(default_factory=dict)  # Arbitrary metadata

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "description": self.description, "details": self.details}


@dataclass
class CleaningResult:
    """
    Structured output from a cleaner — records what was done and why.

    This is the audit trail. Every change to the DataFrame is logged here.
    The pipeline collects all CleaningResults to build the final report.

    Fields:
        cleaner_name:       Machine-readable snake_case identifier.
        display_name:       Human-readable name for reports.
        applied:            True if the cleaner actually changed the DataFrame.
                            False if it was skipped (no issues found by analyzer).
        rows_before:        Row count before this cleaner ran.
        rows_after:         Row count after this cleaner ran.
        cols_before:        Column count before this cleaner ran.
        cols_after:         Column count after this cleaner ran.
        operations:         List of CleaningOperation (the atomic changes made).
        summary:            One-line human summary of what was done.
    """
    cleaner_name: str
    display_name: str
    applied: bool
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    operations: list[CleaningOperation]
    summary: str

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def cols_removed(self) -> int:
        return self.cols_before - self.cols_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "cleaner": self.cleaner_name,
            "display_name": self.display_name,
            "applied": self.applied,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_removed,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
            "cols_removed": self.cols_removed,
            "operations": [op.to_dict() for op in self.operations],
            "summary": self.summary,
        }

    @classmethod
    def skipped(cls, cleaner_name: str, display_name: str, df: pd.DataFrame, reason: str = "No issues found.") -> CleaningResult:
        """
        Factory for a no-op result when the cleaner had nothing to do.

        WHY a factory? Keeps all 'skipped' results consistent and saves
        boilerplate in every cleaner's 'no issues' branch.
        """
        n_rows, n_cols = df.shape
        return cls(
            cleaner_name=cleaner_name,
            display_name=display_name,
            applied=False,
            rows_before=n_rows,
            rows_after=n_rows,
            cols_before=n_cols,
            cols_after=n_cols,
            operations=[],
            summary=f"Skipped — {reason}",
        )


class BaseCleaner(ABC):
    """
    Abstract base class for all cleaners.

    CONTRACT:
        Every subclass must implement:
            1. `name` property  — machine-readable snake_case identifier
            2. `display_name` property — human-readable name
            3. `clean(df, analysis_result)` — transforms the DataFrame

        Every subclass MUST:
            - Return a new DataFrame (not mutate the input)
            - Never fail silently — raise CleaningError for genuine failures
            - Log what it did at INFO level (one line), DEBUG for details
            - Skip gracefully when analysis_result.issues_found is False
            - Record every individual transformation in the CleaningResult

    WHAT CLEANERS MUST NOT DO:
        - Modify the original DataFrame passed in
        - Call external APIs
        - Re-run analysis (use the provided analysis_result instead)
        - Silently swallow exceptions
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable snake_case identifier. Used in JSON output."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for CLI reports."""
        ...

    @abstractmethod
    def clean(
        self,
        df: pd.DataFrame,
        analysis_result: AnalysisResult | None = None,
    ) -> tuple[pd.DataFrame, CleaningResult]:
        """
        Apply cleaning transformations to the DataFrame.

        Args:
            df: The working copy of the DataFrame to clean.
                ALWAYS a copy — do not call .copy() inside the cleaner.
            analysis_result: The result from the matching analyzer.
                             If None, the cleaner must run its own lightweight
                             detection (only for standalone use).

        Returns:
            (new_df, result) where:
                new_df: Transformed DataFrame (may be df itself if nothing changed)
                result: CleaningResult documenting what was done

        Raises:
            CleaningError: If the transformation fails unexpectedly.
        """
        ...

    def _make_result(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        operations: list[CleaningOperation],
        summary: str,
    ) -> CleaningResult:
        """
        Convenience builder for CleaningResult from before/after DataFrames.
        """
        return CleaningResult(
            cleaner_name=self.name,
            display_name=self.display_name,
            applied=True,
            rows_before=len(df_before),
            rows_after=len(df_after),
            cols_before=len(df_before.columns),
            cols_after=len(df_after.columns),
            operations=operations,
            summary=summary,
        )
