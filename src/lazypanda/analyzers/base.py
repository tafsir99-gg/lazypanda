"""
Base classes for all analyzers.

DESIGN PATTERN: Strategy + Template Method
    Every analyzer is a Strategy — a pluggable algorithm that does one thing.
    They all share the same interface (BaseAnalyzer) so the pipeline can
    treat them uniformly without knowing their internals.

    Adding a new analyzer in v2 = create one file + inherit BaseAnalyzer.
    You never touch existing code. This is the Open/Closed Principle.

IMPORTANT:
    Analyzers are READ-ONLY. They inspect a DataFrame and report findings.
    They NEVER modify the DataFrame. That is the job of Cleaners (Milestone 3).

USAGE:
    from lazypanda.analyzers.missing_values import MissingValueAnalyzer

    config = {"drop_column_threshold": 0.8, "drop_row_threshold": 0.5, ...}
    analyzer = MissingValueAnalyzer(config)
    result = analyzer.analyze(df)

    if result.issues_found:
        print(result.summary)
        print(result.recommendation)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger("lazypanda")

# Severity levels from most to least critical
SeverityLevel = Literal["critical", "warning", "info", "ok"]


@dataclass
class AnalysisResult:
    """
    Structured output from any analyzer.

    WHY a dataclass instead of a plain dict?
        - Type safety: you can't misspell a field name
        - Auto-completion in your editor
        - Consistent shape — the pipeline always knows what to expect
        - Easy to serialize to JSON for reports

    Fields:
        analyzer_name:    Machine-readable identifier for the analyzer.
        display_name:     Human-readable name for reports and CLI output.
        issues_found:     True if the analyzer found at least one problem.
        severity:         "critical" | "warning" | "info" | "ok"
                          ok = no issues found
        affected_columns: List of column names affected by the issue.
        details:          Flexible payload — each analyzer defines its own
                          structure. Always a dict so it serializes to JSON.
        summary:          One-line human-readable description of the finding.
        recommendation:   Suggested action. Shown in reports.
    """

    analyzer_name: str
    display_name: str
    issues_found: bool
    severity: SeverityLevel
    affected_columns: list[str]
    details: dict[str, Any]
    summary: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for JSON reports)."""
        return {
            "analyzer": self.analyzer_name,
            "display_name": self.display_name,
            "issues_found": self.issues_found,
            "severity": self.severity,
            "affected_columns": self.affected_columns,
            "details": self.details,
            "summary": self.summary,
            "recommendation": self.recommendation,
        }

    @classmethod
    def ok(cls, analyzer_name: str, display_name: str, message: str = "No issues detected.") -> AnalysisResult:
        """
        Factory method for a clean result (no issues found).

        WHY a factory method?
            Every analyzer needs to return a result even when nothing is wrong.
            This factory ensures all "clean" results look identical.
        """
        return cls(
            analyzer_name=analyzer_name,
            display_name=display_name,
            issues_found=False,
            severity="ok",
            affected_columns=[],
            details={},
            summary=message,
            recommendation="No action required.",
        )


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.

    CONTRACT:
        Every subclass must implement:
            1. `name` property  — machine-readable identifier (snake_case)
            2. `display_name` property — human-readable name
            3. `analyze(df)` method — returns an AnalysisResult

        Every subclass MUST:
            - Be stateless (no mutable state between calls)
            - Never modify the DataFrame passed to analyze()
            - Handle edge cases (empty df, all-null columns) gracefully
            - Log at DEBUG level for detailed tracing
            - Raise AnalysisError only for genuine unexpected failures

    WHAT ANALYZERS MUST NOT DO:
        - Call external APIs (no Gemini)
        - Write to disk
        - Modify the DataFrame
        - Raise exceptions for data quality issues (those are results, not errors)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: The relevant config section as a plain dict.
                    Use AppConfig.model_dump()['section_name'] to produce this.
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable snake_case identifier. Used in JSON output."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name. Used in CLI tables and reports."""
        ...

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """
        Inspect the DataFrame and return findings.

        Args:
            df: The DataFrame to analyze. DO NOT MODIFY IT.

        Returns:
            AnalysisResult with all findings. Return AnalysisResult.ok()
            if no issues are detected — never return None.

        Note:
            The df passed here is always a working COPY from the pipeline.
            Even so, analyzers must treat it as read-only by convention.
        """
        ...

    def _safe_analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """
        Wrapper that catches unexpected errors and returns a graceful result.

        The pipeline calls this instead of analyze() directly. This ensures
        one failing analyzer never crashes the entire pipeline.
        """
        from lazypanda.utils.exceptions import AnalysisError
        try:
            logger.debug("Running analyzer: %s", self.name)
            result = self.analyze(df)
            logger.debug(
                "Analyzer %s complete — issues_found=%s severity=%s",
                self.name,
                result.issues_found,
                result.severity,
            )
            return result
        except AnalysisError:
            raise  # Re-raise our own errors — they're intentional
        except Exception as e:
            logger.error("Analyzer %s failed unexpectedly: %s", self.name, e, exc_info=True)
            # Return a result that flags the failure rather than crashing
            return AnalysisResult(
                analyzer_name=self.name,
                display_name=self.display_name,
                issues_found=False,
                severity="warning",
                affected_columns=[],
                details={"error": str(e)},
                summary=f"Analyzer failed: {e}",
                recommendation="Check logs for details.",
            )
