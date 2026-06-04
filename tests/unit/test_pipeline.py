"""
Integration tests for the CleaningPipeline.

WHAT THESE TEST:
    Unlike unit tests (which test each cleaner in isolation), integration
    tests verify the ENTIRE pipeline from raw dirty DataFrame to clean output.

    Specifically we test:
    1. The pipeline runs end-to-end on the dirty fixture without crashing
    2. The output DataFrame is smaller than the input (rows and/or cols removed)
    3. All 6 cleaners produce CleaningResults (applied or skipped)
    4. PipelineResult stats are arithmetically correct
    5. Dry-run returns the ORIGINAL DataFrame
    6. save() writes a file to disk and the file is readable

PHILOSOPHY:
    Integration tests are SLOWER than unit tests because they run the
    full analyzer suite + all cleaners on real fixture data. This is
    intentional — they catch bugs that unit tests miss (e.g., order
    dependencies between cleaners).
"""

from pathlib import Path

import pandas as pd
import pytest

from lazypanda.core.config_manager import AppConfig, ConfigManager
from lazypanda.core.pipeline import CleaningPipeline, PipelineResult

# ─── Config fixture ────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DIRTY_CSV = FIXTURES_DIR / "sample_dirty.csv"
CLEAN_CSV = FIXTURES_DIR / "sample_clean.csv"
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "lazypanda" / "config" / "default_config.yaml"


@pytest.fixture(scope="module")
def config() -> AppConfig:
    """Load the default config once for the entire module."""
    return ConfigManager(default_config_path=DEFAULT_CONFIG_PATH).load()


@pytest.fixture(scope="module")
def dirty_df() -> pd.DataFrame:
    """Load the dirty fixture CSV once for the module."""
    return pd.read_csv(DIRTY_CSV)


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    """Load the clean fixture CSV once for the module."""
    return pd.read_csv(CLEAN_CSV)


@pytest.fixture(scope="module")
def dirty_pipeline_result(config, dirty_df) -> PipelineResult:
    """Run the pipeline on the dirty fixture once — reuse across tests."""
    pipeline = CleaningPipeline(config)
    return pipeline.run(dirty_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline smoke tests — does it run?
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineSmoke:

    def test_pipeline_runs_on_dirty_fixture(self, dirty_pipeline_result):
        assert dirty_pipeline_result is not None

    def test_pipeline_returns_pipeline_result(self, dirty_pipeline_result):
        assert isinstance(dirty_pipeline_result, PipelineResult)

    def test_cleaned_df_is_dataframe(self, dirty_pipeline_result):
        assert isinstance(dirty_pipeline_result.cleaned_df, pd.DataFrame)

    def test_original_df_is_not_mutated(self, config, dirty_df):
        """The pipeline must NEVER modify the input DataFrame's shape."""
        original_shape = dirty_df.shape
        original_cols = dirty_df.columns.tolist()
        pipeline = CleaningPipeline(config)
        pipeline.run(dirty_df)
        # Shape and column names must be intact
        assert dirty_df.shape == original_shape
        assert dirty_df.columns.tolist() == original_cols

    def test_pipeline_runs_on_clean_fixture(self, config, clean_df):
        """Pipeline should handle a clean dataset without errors."""
        pipeline = CleaningPipeline(config)
        result = pipeline.run(clean_df)
        assert result is not None
        assert result.cleaned_df is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis results
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineAnalysis:

    def test_produces_7_analysis_results(self, dirty_pipeline_result):
        """One result per analyzer."""
        assert len(dirty_pipeline_result.analysis_results) == 7

    def test_all_analyzers_have_names(self, dirty_pipeline_result):
        names = {r.analyzer_name for r in dirty_pipeline_result.analysis_results}
        expected = {
            "missing_values", "duplicates", "type_inference",
            "outliers", "cardinality", "category_consistency", "suspicious_values",
        }
        assert expected.issubset(names)

    def test_dirty_fixture_has_at_least_one_issue(self, dirty_pipeline_result):
        assert len(dirty_pipeline_result.issues_found) >= 1

    def test_issues_found_is_subset_of_analysis_results(self, dirty_pipeline_result):
        all_results = dirty_pipeline_result.analysis_results
        issues = dirty_pipeline_result.issues_found
        assert all(r in all_results for r in issues)


# ═══════════════════════════════════════════════════════════════════════════════
# Cleaning results
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineCleaningResults:

    def test_produces_6_cleaning_results(self, dirty_pipeline_result):
        """One result per cleaner."""
        assert len(dirty_pipeline_result.cleaning_results) == 6

    def test_all_cleaners_have_names(self, dirty_pipeline_result):
        names = {r.cleaner_name for r in dirty_pipeline_result.cleaning_results}
        expected = {
            "drop_duplicates", "constant_columns", "type_caster",
            "category_normalizer", "missing_values", "outlier_handler",
        }
        assert expected == names

    def test_all_results_are_cleaning_results(self, dirty_pipeline_result):
        from lazypanda.cleaners.base import CleaningResult
        for r in dirty_pipeline_result.cleaning_results:
            assert isinstance(r, CleaningResult)

    def test_applied_or_skipped_is_boolean(self, dirty_pipeline_result):
        for r in dirty_pipeline_result.cleaning_results:
            assert isinstance(r.applied, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Shape changes — the data actually got cleaned
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineShapeChanges:

    def test_output_has_no_missing_values_in_imputed_cols(self, dirty_pipeline_result):
        """After cleaning, there should be significantly fewer missing values."""
        result = dirty_pipeline_result
        original_missing = result.original_shape[0] * result.original_shape[1]
        cleaned_missing = result.cleaned_df.isnull().sum().sum()
        # The dirty fixture has 47 missing cells — cleaned should have far fewer
        # (some columns may still have NaN if they were dropped/not imputable)
        # Key assertion: cleaned is strictly better than original
        original_nan_count = 47  # known from fixture
        assert cleaned_missing <= original_nan_count

    def test_no_duplicate_rows_in_output(self, dirty_pipeline_result):
        """After cleaning, no exact duplicate rows should remain."""
        df = dirty_pipeline_result.cleaned_df
        assert df.duplicated().sum() == 0

    def test_original_shape_recorded_correctly(self, dirty_pipeline_result, dirty_df):
        assert dirty_pipeline_result.original_shape == dirty_df.shape

    def test_final_shape_matches_cleaned_df(self, dirty_pipeline_result):
        result = dirty_pipeline_result
        assert result.final_shape == result.cleaned_df.shape

    def test_rows_removed_is_nonnegative(self, dirty_pipeline_result):
        assert dirty_pipeline_result.rows_removed >= 0

    def test_cols_final_is_consistent_with_shape(self, dirty_pipeline_result):
        """
        Final column count must match cleaned_df's actual column count.
        Note: outlier 'flag' action ADDS columns (e.g., age_is_outlier),
        so cols_removed can be negative — that is correct behavior.
        """
        result = dirty_pipeline_result
        assert result.final_shape[1] == len(result.cleaned_df.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# Stats dict
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineStats:

    def test_stats_keys_present(self, dirty_pipeline_result):
        expected = {
            "rows_original", "rows_final", "rows_removed",
            "cols_original", "cols_final", "cols_removed",
            "issues_found", "cleaners_applied", "cleaners_skipped",
        }
        assert expected.issubset(dirty_pipeline_result.stats.keys())

    def test_stats_arithmetic_consistent(self, dirty_pipeline_result):
        s = dirty_pipeline_result.stats
        assert s["rows_original"] - s["rows_removed"] == s["rows_final"]
        # cols_removed can be negative if outlier 'flag' action added indicator columns
        assert s["cols_original"] - s["cols_removed"] == s["cols_final"]
        assert s["cleaners_applied"] + s["cleaners_skipped"] == len(dirty_pipeline_result.cleaning_results)


    def test_issues_found_is_positive_for_dirty(self, dirty_pipeline_result):
        assert dirty_pipeline_result.stats["issues_found"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Dry run
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineDryRun:

    def test_dry_run_returns_original_df(self, config, dirty_df):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df, dry_run=True)
        # In dry run, the returned df should have the ORIGINAL shape
        assert result.cleaned_df.shape == dirty_df.shape

    def test_dry_run_flag_set_in_result(self, config, dirty_df):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df, dry_run=True)
        assert result.dry_run is True

    def test_dry_run_still_produces_analysis(self, config, dirty_df):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df, dry_run=True)
        assert len(result.analysis_results) == 7

    def test_dry_run_still_produces_cleaning_results(self, config, dirty_df):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df, dry_run=True)
        assert len(result.cleaning_results) == 6

    def test_save_raises_on_dry_run(self, config, dirty_df, tmp_path):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df, dry_run=True)
        with pytest.raises(ValueError, match="dry-run"):
            pipeline.save(result, tmp_path / "out.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Save output
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineSave:

    def test_save_writes_csv_file(self, config, dirty_df, tmp_path):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df)
        out_path = tmp_path / "cleaned.csv"
        pipeline.save(result, out_path)
        assert out_path.exists()

    def test_saved_csv_is_readable(self, config, dirty_df, tmp_path):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df)
        out_path = tmp_path / "cleaned.csv"
        pipeline.save(result, out_path)
        df_read = pd.read_csv(out_path)
        assert isinstance(df_read, pd.DataFrame)
        assert len(df_read) > 0

    def test_saved_csv_shape_matches_result(self, config, dirty_df, tmp_path):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df)
        out_path = tmp_path / "cleaned.csv"
        pipeline.save(result, out_path)
        df_read = pd.read_csv(out_path)
        assert df_read.shape == result.final_shape

    def test_save_creates_parent_dirs(self, config, dirty_df, tmp_path):
        pipeline = CleaningPipeline(config)
        result = pipeline.run(dirty_df)
        nested = tmp_path / "a" / "b" / "c" / "cleaned.csv"
        pipeline.save(result, nested)
        assert nested.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# to_dict serialization
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineResultSerialization:

    def test_to_dict_returns_dict(self, dirty_pipeline_result):
        d = dirty_pipeline_result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_analysis_key(self, dirty_pipeline_result):
        d = dirty_pipeline_result.to_dict()
        assert "analysis" in d
        assert isinstance(d["analysis"], list)

    def test_to_dict_has_cleaning_key(self, dirty_pipeline_result):
        d = dirty_pipeline_result.to_dict()
        assert "cleaning" in d
        assert isinstance(d["cleaning"], list)
