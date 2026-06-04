"""
Cleaning Pipeline.

WHAT IT IS:
    The pipeline is the orchestrator. It:
    1. Runs all 7 analyzers on the original DataFrame (one read-only pass)
    2. Runs each cleaner in a fixed, correct order — passing each cleaner
       the matching analyzer's result so no calculation is repeated
    3. Collects all CleaningResults into a PipelineResult
    4. Saves the cleaned DataFrame to disk (if an output path is provided)

THE ORDER MATTERS:
    1. DropDuplicates       — fewer rows = less work for everything else
    2. ConstantColumns      — drop useless columns early
    3. TypeCaster           — correct dtypes before any statistics
    4. CategoryNormalizer   — correct categories before mode imputation
    5. MissingValues        — impute after correct types + normalized categories
    6. OutlierHandler       — outlier stats require complete (non-NaN) columns

    SuspiciousValues has NO cleaner in M3 — those decisions require AI context.

DRY RUN MODE:
    When dry_run=True, the pipeline runs all analyzers and SIMULATES each
    cleaner (reports what would change) but returns the ORIGINAL DataFrame.
    Use this to preview changes before committing.

TOKEN EFFICIENCY:
    Notice that every cleaner receives the matching AnalysisResult.
    This means the pipeline makes exactly ONE pass over each column per
    analyzer. The cleaners reuse those results — no redundant computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.analyzers.cardinality import CardinalityAnalyzer
from lazypanda.analyzers.category_consistency import CategoryConsistencyAnalyzer
from lazypanda.analyzers.duplicates import DuplicateAnalyzer
from lazypanda.analyzers.missing_values import MissingValueAnalyzer
from lazypanda.analyzers.outliers import OutlierAnalyzer
from lazypanda.analyzers.suspicious_values import SuspiciousValueAnalyzer
from lazypanda.analyzers.type_inference import TypeInferenceAnalyzer
from lazypanda.cleaners.base import CleaningResult
from lazypanda.cleaners.cardinality import ConstantColumnCleaner
from lazypanda.cleaners.category_normalizer import CategoryNormalizerCleaner
from lazypanda.cleaners.drop_duplicates import DropDuplicatesCleaner
from lazypanda.cleaners.missing_values import MissingValueCleaner
from lazypanda.cleaners.outlier_handler import OutlierHandlerCleaner
from lazypanda.cleaners.type_caster import TypeCasterCleaner
from lazypanda.core.config_manager import AppConfig

if TYPE_CHECKING:
    from lazypanda.ai.enricher import AIEnricher, AIInsights

logger = logging.getLogger("lazypanda")


@dataclass
class PipelineResult:
    """
    The complete output of one pipeline run.

    Contains both the analysis (what was found) and the cleaning audit trail
    (what was done and why), plus the final cleaned DataFrame.

    After Milestone 5, also contains ai_insights (if AI enrichment ran).
    """
    original_shape:   tuple[int, int]
    final_shape:      tuple[int, int]
    analysis_results: list[AnalysisResult]
    cleaning_results: list[CleaningResult]
    cleaned_df:       pd.DataFrame
    dry_run:          bool
    stats:            dict[str, Any] = field(default_factory=dict)
    ai_insights:      "AIInsights | None" = field(default=None)

    @property
    def rows_removed(self) -> int:
        return self.original_shape[0] - self.final_shape[0]

    @property
    def cols_removed(self) -> int:
        return self.original_shape[1] - self.final_shape[1]

    @property
    def issues_found(self) -> list[AnalysisResult]:
        return [r for r in self.analysis_results if r.issues_found]

    @property
    def cleaners_applied(self) -> list[CleaningResult]:
        return [r for r in self.cleaning_results if r.applied]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "rows_removed": self.rows_removed,
            "cols_removed": self.cols_removed,
            "dry_run": self.dry_run,
            "stats": self.stats,
            "analysis": [r.to_dict() for r in self.analysis_results],
            "cleaning": [r.to_dict() for r in self.cleaning_results],
        }


class CleaningPipeline:
    """
    Orchestrates the full analyze → clean cycle.

    Usage:
        config = ConfigManager().load()
        pipeline = CleaningPipeline(config)
        result = pipeline.run(df, dry_run=False)
        result.cleaned_df.to_csv("cleaned.csv", index=False)
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._cfg = config.model_dump()

    def run(
        self,
        df: pd.DataFrame,
        dry_run: bool = False,
        use_ai: bool = False,
        source_file: str = "dataset.csv",
        enricher: "AIEnricher | None" = None,
        model_override: str | None = None,
    ) -> PipelineResult:
        """
        Run the full pipeline: analyze → clean → (optionally) AI enrich.

        Args:
            df:             The raw DataFrame to process. WILL NOT BE MODIFIED.
                            The pipeline works on an internal copy.
            dry_run:        If True, simulate cleaners but return the original df.
            use_ai:         If True, call AIEnricher after cleaning (Phase 3).
                            Requires GEMINI_API_KEY to be set.
            source_file:    Basename used in the AI prompt (no PII sent).
            enricher:       Optional AIEnricher to inject (for testing/DI).
            model_override: If set, use this Gemini model instead of the default.
                            Passed through to AIEnricher.enrich().

        Returns:
            PipelineResult with everything — analysis, cleaning log, cleaned_df,
            and optionally ai_insights.
        """
        original_shape = df.shape
        working_df = df.copy()

        logger.info(
            "Pipeline starting: %d rows × %d cols (dry_run=%s)",
            *original_shape, dry_run,
        )

        # ── Phase 1: Run all analyzers ─────────────────────────────────────────
        analysis_results = self._run_analyzers(working_df)

        # Build a lookup dict: analyzer_name → result
        analysis_by_name: dict[str, AnalysisResult] = {
            r.analyzer_name: r for r in analysis_results
        }

        # ── Phase 2: Run cleaners in the correct order ─────────────────────────
        cleaning_results: list[CleaningResult] = []

        cleaner_steps = [
            # (Cleaner class, config_key, matching_analyzer_name)
            (DropDuplicatesCleaner,      self._cfg["duplicates"],      "duplicates"),
            (ConstantColumnCleaner,      self._cfg["cardinality"],     "cardinality"),
            (TypeCasterCleaner,          self._cfg.get("type_inference", {}), "type_inference"),
            (CategoryNormalizerCleaner,  self._cfg["categories"],      "category_consistency"),
            (MissingValueCleaner,        self._cfg["missing_values"],  "missing_values"),
            (OutlierHandlerCleaner,      self._cfg["outliers"],        "outliers"),
        ]

        for CleanerClass, cfg_section, analyzer_name in cleaner_steps:
            cleaner = CleanerClass(cfg_section)
            analysis = analysis_by_name.get(analyzer_name)

            if dry_run:
                # In dry-run: run the cleaner but discard the modified df
                _, result = cleaner.clean(working_df.copy(), analysis)
                result.summary = f"[DRY RUN] {result.summary}"
            else:
                working_df, result = cleaner.clean(working_df, analysis)

            cleaning_results.append(result)
            logger.info(
                "Cleaner '%s': applied=%s rows=%d→%d cols=%d→%d",
                cleaner.name, result.applied,
                result.rows_before, result.rows_after,
                result.cols_before, result.cols_after,
            )

        final_df = df if dry_run else working_df
        final_shape = final_df.shape

        stats = self._compute_stats(analysis_results, cleaning_results, original_shape, final_shape)

        logger.info(
            "Pipeline complete: %d→%d rows, %d→%d cols (%d cleaner(s) applied)",
            original_shape[0], final_shape[0],
            original_shape[1], final_shape[1],
            sum(1 for r in cleaning_results if r.applied),
        )

        result = PipelineResult(
            original_shape=original_shape,
            final_shape=final_shape,
            analysis_results=analysis_results,
            cleaning_results=cleaning_results,
            cleaned_df=final_df,
            dry_run=dry_run,
            stats=stats,
            ai_insights=None,
        )

        # ── Phase 3: AI enrichment (optional, never blocks the pipeline) ──────
        if use_ai and not dry_run:
            from lazypanda.ai.enricher import AIEnricher
            _enricher = enricher or AIEnricher()
            try:
                result.ai_insights = _enricher.enrich(
                    result,
                    config=self.config,
                    use_ai=True,
                    source_file=source_file,
                    model_override=model_override,
                )
            except Exception as e:
                # AI enrichment failure is NEVER fatal
                logger.error("AI enrichment raised an unexpected exception: %s", e, exc_info=True)
                from lazypanda.ai.enricher import AIInsights
                result.ai_insights = AIInsights.skipped_result(
                    f"Unexpected error during AI enrichment: {e}"
                )

        return result

    def save(self, result: PipelineResult, output_path: Path) -> Path:
        """
        Save the cleaned DataFrame to a CSV file.

        Args:
            result: The PipelineResult from run().
            output_path: File path or directory. If directory, creates
                         <original_stem>_cleaned.csv inside it.

        Returns:
            The actual file path written.
        """
        if result.dry_run:
            raise ValueError("Cannot save output from a dry-run pipeline.")

        output_path = Path(output_path)
        if output_path.is_dir():
            raise ValueError("output_path must be a file path, not a directory. Use pipeline.run() and pass the returned df.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = self.config.dataset.encoding or "utf-8"
        result.cleaned_df.to_csv(output_path, index=False, encoding=encoding)
        logger.info("Pipeline saved cleaned data to: %s", output_path)
        return output_path

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_analyzers(self, df: pd.DataFrame) -> list[AnalysisResult]:
        """Run all 7 analyzers and collect results."""
        cfg = self._cfg
        analyzers = [
            MissingValueAnalyzer(cfg["missing_values"]),
            DuplicateAnalyzer(cfg["duplicates"]),
            TypeInferenceAnalyzer(cfg.get("type_inference", {})),
            OutlierAnalyzer(cfg["outliers"]),
            CardinalityAnalyzer(cfg["cardinality"]),
            CategoryConsistencyAnalyzer(cfg["categories"]),
            SuspiciousValueAnalyzer(cfg.get("suspicious_values", {})),
        ]
        results = []
        for analyzer in analyzers:
            result = analyzer._safe_analyze(df)
            results.append(result)
        return results

    def _compute_stats(
        self,
        analysis_results: list[AnalysisResult],
        cleaning_results: list[CleaningResult],
        original_shape: tuple[int, int],
        final_shape: tuple[int, int],
    ) -> dict[str, Any]:
        """Compute summary statistics for the pipeline run."""
        return {
            "rows_original": original_shape[0],
            "rows_final": final_shape[0],
            "rows_removed": original_shape[0] - final_shape[0],
            "cols_original": original_shape[1],
            "cols_final": final_shape[1],
            "cols_removed": original_shape[1] - final_shape[1],
            "issues_found": sum(1 for r in analysis_results if r.issues_found),
            "cleaners_applied": sum(1 for r in cleaning_results if r.applied),
            "cleaners_skipped": sum(1 for r in cleaning_results if not r.applied),
        }
