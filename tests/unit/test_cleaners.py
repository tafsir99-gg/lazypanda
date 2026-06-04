"""
Unit tests for all Milestone 3 cleaners.

PHILOSOPHY:
    Each cleaner test follows the Arrange-Act-Assert pattern:
    1. ARRANGE — Create a minimal DataFrame with the exact issue to test
    2. ACT     — Call cleaner.clean(df, analysis_result=None) or with a mock result
    3. ASSERT  — Check the output DataFrame AND the CleaningResult metadata

    We test BOTH the transformed data AND the audit trail.
    If a cleaner transforms data correctly but logs wrong metadata,
    that's a bug — it would produce a misleading report.

    Every cleaner is tested for:
    1. Happy path — does the right thing when issues exist
    2. No-op path — does nothing and returns a 'Skipped' result when no issues

FIXTURE REUSE:
    We reuse the shared conftest.py fixtures (df_with_duplicates, etc.)
    for standard scenarios, and create inline DataFrames for edge cases.
"""

import pandas as pd
import pytest

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.cleaners.base import CleaningResult
from lazypanda.cleaners.cardinality import ConstantColumnCleaner
from lazypanda.cleaners.category_normalizer import CategoryNormalizerCleaner
from lazypanda.cleaners.drop_duplicates import DropDuplicatesCleaner
from lazypanda.cleaners.missing_values import MissingValueCleaner
from lazypanda.cleaners.outlier_handler import OutlierHandlerCleaner
from lazypanda.cleaners.type_caster import TypeCasterCleaner


# ── Helper to build a minimal AnalysisResult ─────────────────────────────────

def make_analysis(issues_found: bool, details: dict, analyzer_name: str = "test") -> AnalysisResult:
    """Build a minimal AnalysisResult for injecting into cleaners."""
    return AnalysisResult(
        analyzer_name=analyzer_name,
        display_name=analyzer_name.replace("_", " ").title(),
        issues_found=issues_found,
        affected_columns=list(details.get("per_column", {}).keys()),
        severity="warning" if issues_found else "ok",
        details=details,
        summary="Test analysis result",
        recommendation="Test recommendation.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DropDuplicatesCleaner
# ═══════════════════════════════════════════════════════════════════════════════

class TestDropDuplicatesCleaner:

    @pytest.fixture
    def cfg(self):
        return {"keep": "first"}

    def test_drops_duplicate_row(self, cfg, df_with_duplicates):
        analysis = make_analysis(True, {"duplicate_count": 1})
        df_out, result = DropDuplicatesCleaner(cfg).clean(df_with_duplicates, analysis)
        assert len(df_out) == len(df_with_duplicates) - 1
        assert df_out.duplicated().sum() == 0

    def test_result_is_applied_true(self, cfg, df_with_duplicates):
        analysis = make_analysis(True, {"duplicate_count": 1})
        _, result = DropDuplicatesCleaner(cfg).clean(df_with_duplicates, analysis)
        assert result.applied is True

    def test_rows_before_after_recorded(self, cfg, df_with_duplicates):
        analysis = make_analysis(True, {"duplicate_count": 1})
        _, result = DropDuplicatesCleaner(cfg).clean(df_with_duplicates, analysis)
        assert result.rows_before == 5
        assert result.rows_after == 4
        assert result.rows_removed == 1

    def test_operation_type_is_drop_rows(self, cfg, df_with_duplicates):
        analysis = make_analysis(True, {"duplicate_count": 1})
        _, result = DropDuplicatesCleaner(cfg).clean(df_with_duplicates, analysis)
        assert any(op.type == "drop_rows" for op in result.operations)

    def test_skips_when_no_duplicates(self, cfg):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        analysis = make_analysis(False, {"duplicate_count": 0})
        df_out, result = DropDuplicatesCleaner(cfg).clean(df, analysis)
        assert result.applied is False
        assert df_out.equals(df)

    def test_skips_when_analysis_says_no_issues(self, cfg, df_clean):
        analysis = make_analysis(False, {})
        _, result = DropDuplicatesCleaner(cfg).clean(df_clean, analysis)
        assert result.applied is False

    def test_keep_last_strategy(self, df_with_duplicates):
        cfg = {"keep": "last"}
        analysis = make_analysis(True, {"duplicate_count": 1})
        df_out, result = DropDuplicatesCleaner(cfg).clean(df_with_duplicates, analysis)
        # With keep='last': the second "Bob" is kept, first is dropped
        assert len(df_out) == 4

    def test_output_index_is_reset(self, cfg, df_with_duplicates):
        analysis = make_analysis(True, {"duplicate_count": 1})
        df_out, _ = DropDuplicatesCleaner(cfg).clean(df_with_duplicates, analysis)
        assert list(df_out.index) == list(range(len(df_out)))

    def test_works_without_analysis_result(self, cfg, df_with_duplicates):
        """Cleaner must detect and drop duplicates even without an AnalysisResult."""
        df_out, result = DropDuplicatesCleaner(cfg).clean(df_with_duplicates, analysis_result=None)
        assert result.applied is True
        assert df_out.duplicated().sum() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ConstantColumnCleaner
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstantColumnCleaner:

    @pytest.fixture
    def cfg(self):
        return {"high_cardinality_threshold": 0.95, "near_constant_threshold": 0.99}

    def test_drops_constant_column(self, cfg):
        df = pd.DataFrame({"a": [1, 2, 3], "const": ["X", "X", "X"]})
        analysis = make_analysis(True, {"constant_columns": ["const"]}, "cardinality")
        df_out, result = ConstantColumnCleaner(cfg).clean(df, analysis)
        assert "const" not in df_out.columns
        assert "a" in df_out.columns

    def test_result_applied_true(self, cfg):
        df = pd.DataFrame({"a": [1, 2, 3], "const": [0, 0, 0]})
        analysis = make_analysis(True, {"constant_columns": ["const"]}, "cardinality")
        _, result = ConstantColumnCleaner(cfg).clean(df, analysis)
        assert result.applied is True
        assert result.cols_removed == 1

    def test_skips_when_no_constant_cols(self, cfg, df_clean):
        analysis = make_analysis(False, {"constant_columns": []}, "cardinality")
        _, result = ConstantColumnCleaner(cfg).clean(df_clean, analysis)
        assert result.applied is False

    def test_drops_multiple_constant_columns(self, cfg):
        df = pd.DataFrame({
            "value": [1, 2, 3],
            "flag_a": [True, True, True],
            "flag_b": [0, 0, 0],
        })
        analysis = make_analysis(True, {"constant_columns": ["flag_a", "flag_b"]}, "cardinality")
        df_out, result = ConstantColumnCleaner(cfg).clean(df, analysis)
        assert "flag_a" not in df_out.columns
        assert "flag_b" not in df_out.columns
        assert result.cols_removed == 2

    def test_operation_records_constant_value(self, cfg):
        df = pd.DataFrame({"a": [1, 2], "status": ["active", "active"]})
        analysis = make_analysis(True, {"constant_columns": ["status"]}, "cardinality")
        _, result = ConstantColumnCleaner(cfg).clean(df, analysis)
        ops = [o for o in result.operations if o.type == "drop_column"]
        assert ops[0].details["constant_value"] == "active"

    def test_works_without_analysis(self, cfg):
        df = pd.DataFrame({"a": [1, 2, 3], "const": ["X", "X", "X"]})
        df_out, result = ConstantColumnCleaner(cfg).clean(df, analysis_result=None)
        assert "const" not in df_out.columns
        assert result.applied is True


# ═══════════════════════════════════════════════════════════════════════════════
# TypeCasterCleaner
# ═══════════════════════════════════════════════════════════════════════════════

class TestTypeCasterCleaner:

    @pytest.fixture
    def cfg(self):
        return {"min_numeric_ratio": 0.9, "min_date_ratio": 0.8}

    def test_casts_numeric_string_to_float64(self, cfg):
        df = pd.DataFrame({"age": pd.array(["25", "30", "35", "40"], dtype=object)})
        type_issues = {"age": {"issue_type": "numeric_string", "suggested_dtype": "float64"}}
        analysis = make_analysis(True, {"type_issues": type_issues}, "type_inference")
        df_out, result = TypeCasterCleaner(cfg).clean(df, analysis)
        # pd.to_numeric infers int64 for whole numbers — that's still numeric
        assert pd.api.types.is_numeric_dtype(df_out["age"])
        assert result.applied is True

    def test_casts_date_string_to_datetime(self, cfg):
        df = pd.DataFrame({
            "signup": pd.array(["2024-01-01", "2024-02-15", "2024-03-20"], dtype=object)
        })
        type_issues = {"signup": {"issue_type": "date_string", "suggested_dtype": "datetime64[ns]"}}
        analysis = make_analysis(True, {"type_issues": type_issues}, "type_inference")
        df_out, result = TypeCasterCleaner(cfg).clean(df, analysis)
        assert pd.api.types.is_datetime64_any_dtype(df_out["signup"])

    def test_cast_operation_logged(self, cfg):
        df = pd.DataFrame({"price": pd.array(["10.5", "20.0"], dtype=object)})
        type_issues = {"price": {"issue_type": "numeric_string", "suggested_dtype": "float64"}}
        analysis = make_analysis(True, {"type_issues": type_issues}, "type_inference")
        _, result = TypeCasterCleaner(cfg).clean(df, analysis)
        ops = [o for o in result.operations if o.type == "cast_column"]
        assert len(ops) == 1
        assert ops[0].details["to_dtype"] == "float64"
        # pandas 3 uses 'str', pandas 2 uses 'object' for string dtype
        assert ops[0].details["from_dtype"] in ("object", "str")

    def test_skips_when_no_type_issues(self, cfg, df_clean):
        analysis = make_analysis(False, {"type_issues": {}}, "type_inference")
        _, result = TypeCasterCleaner(cfg).clean(df_clean, analysis)
        assert result.applied is False

    def test_coerce_partial_numeric_to_nan(self, cfg):
        """Values that can't be cast become NaN — NOT silently dropped."""
        df = pd.DataFrame({"val": pd.array(["10", "abc", "30"], dtype=object)})
        type_issues = {"val": {"issue_type": "numeric_string", "suggested_dtype": "float64"}}
        analysis = make_analysis(True, {"type_issues": type_issues}, "type_inference")
        df_out, result = TypeCasterCleaner(cfg).clean(df, analysis)
        # "abc" → NaN, NOT dropped
        assert len(df_out) == 3
        assert pd.isna(df_out["val"].iloc[1])

    def test_skips_missing_column_gracefully(self, cfg):
        """If analysis references a column that doesn't exist, skip it."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        type_issues = {"nonexistent": {"issue_type": "numeric_string", "suggested_dtype": "float64"}}
        analysis = make_analysis(True, {"type_issues": type_issues}, "type_inference")
        df_out, result = TypeCasterCleaner(cfg).clean(df, analysis)
        # Should not raise, should return skipped
        assert "nonexistent" not in df_out.columns


# ═══════════════════════════════════════════════════════════════════════════════
# CategoryNormalizerCleaner
# ═══════════════════════════════════════════════════════════════════════════════

class TestCategoryNormalizerCleaner:

    @pytest.fixture
    def cfg(self):
        return {"normalize_case": True, "strip_whitespace": True}

    def test_lowercases_values(self, cfg, df_with_inconsistent_categories):
        per_col = {"gender": {}, "class": {}}
        analysis = make_analysis(True, {"per_column": per_col}, "category_consistency")
        df_out, result = CategoryNormalizerCleaner(cfg).clean(df_with_inconsistent_categories, analysis)
        assert all(v == v.lower() or pd.isna(v) for v in df_out["gender"])
        assert all(v == v.lower() or pd.isna(v) for v in df_out["class"])

    def test_strips_whitespace(self, cfg):
        df = pd.DataFrame({"status": [" active", "inactive ", "  pending  "]})
        per_col = {"status": {}}
        analysis = make_analysis(True, {"per_column": per_col}, "category_consistency")
        df_out, result = CategoryNormalizerCleaner(cfg).clean(df, analysis)
        assert all(v == v.strip() for v in df_out["status"])

    def test_reduces_unique_count(self, cfg, df_with_inconsistent_categories):
        per_col = {"gender": {}, "class": {}}
        analysis = make_analysis(True, {"per_column": per_col}, "category_consistency")
        df_out, result = CategoryNormalizerCleaner(cfg).clean(df_with_inconsistent_categories, analysis)
        assert df_out["gender"].nunique() < df_with_inconsistent_categories["gender"].nunique()

    def test_result_applied_true(self, cfg, df_with_inconsistent_categories):
        per_col = {"gender": {}, "class": {}}
        analysis = make_analysis(True, {"per_column": per_col}, "category_consistency")
        _, result = CategoryNormalizerCleaner(cfg).clean(df_with_inconsistent_categories, analysis)
        assert result.applied is True

    def test_operations_record_unique_counts(self, cfg, df_with_inconsistent_categories):
        per_col = {"gender": {}}
        analysis = make_analysis(True, {"per_column": per_col}, "category_consistency")
        _, result = CategoryNormalizerCleaner(cfg).clean(df_with_inconsistent_categories, analysis)
        ops = [o for o in result.operations if o.type == "normalize_category"]
        assert len(ops) == 1
        assert ops[0].details["normalized_unique_count"] < ops[0].details["raw_unique_count"]

    def test_skips_when_disabled_in_config(self):
        cfg = {"normalize_case": False, "strip_whitespace": False}
        df = pd.DataFrame({"status": ["Active", "ACTIVE"]})
        analysis = make_analysis(True, {"per_column": {"status": {}}})
        _, result = CategoryNormalizerCleaner(cfg).clean(df, analysis)
        assert result.applied is False

    def test_skips_when_no_issues(self, cfg, df_clean):
        analysis = make_analysis(False, {})
        _, result = CategoryNormalizerCleaner(cfg).clean(df_clean, analysis)
        assert result.applied is False

    def test_does_not_touch_numeric_columns(self, cfg):
        df = pd.DataFrame({"score": [10.5, 20.3, 15.1]})
        per_col = {"score": {}}
        analysis = make_analysis(True, {"per_column": per_col}, "category_consistency")
        df_out, result = CategoryNormalizerCleaner(cfg).clean(df, analysis)
        # Numeric column should be unchanged
        pd.testing.assert_series_equal(df_out["score"], df["score"])


# ═══════════════════════════════════════════════════════════════════════════════
# MissingValueCleaner
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingValueCleaner:

    @pytest.fixture
    def cfg(self):
        return {
            "drop_column_threshold": 0.8,
            "drop_row_threshold": 0.5,
            "numeric_imputation": "median",
            "categorical_imputation": "mode",
            "constant_fill_value": "UNKNOWN",
        }

    def test_imputes_missing_numeric_with_median(self, cfg, df_with_missing):
        analysis = make_analysis(
            True,
            {
                "columns_to_drop": [],
                "columns_to_impute": {"age": "median"},
            },
            "missing_values",
        )
        df_out, result = MissingValueCleaner(cfg).clean(df_with_missing, analysis)
        assert df_out["age"].isnull().sum() == 0
        # Median of [25, 30, 45] = 30
        assert df_out["age"].iloc[1] == pytest.approx(30.0)

    def test_imputes_missing_categorical_with_mode(self, cfg):
        # Three reds, one blue — mode must be 'red' unambiguously
        df = pd.DataFrame({"color": pd.array(["red", "red", None, "red", "blue"], dtype=object)})
        analysis = make_analysis(
            True,
            {"columns_to_drop": [], "columns_to_impute": {"color": "mode"}},
            "missing_values",
        )
        df_out, result = MissingValueCleaner(cfg).clean(df, analysis)
        assert df_out["color"].isnull().sum() == 0
        assert df_out["color"].iloc[2] == "red"   # mode of ["red","red","red","blue"]

    def test_drops_high_missing_column(self, cfg):
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "notes": [None, None, None],  # 100% missing → should be dropped
        })
        analysis = make_analysis(
            True,
            {"columns_to_drop": ["notes"], "columns_to_impute": {}},
            "missing_values",
        )
        df_out, result = MissingValueCleaner(cfg).clean(df, analysis)
        assert "notes" not in df_out.columns
        assert result.cols_removed == 1

    def test_drops_excessive_missing_rows(self, cfg):
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4, None, 6],
            "c": [7, None, 9],
        })
        # Row 1 is entirely missing → 100% missing → should be dropped
        analysis = make_analysis(
            True,
            {"columns_to_drop": [], "columns_to_impute": {}},
            "missing_values",
        )
        df_out, result = MissingValueCleaner(cfg).clean(df, analysis)
        assert len(df_out) < len(df)

    def test_result_applied_true_on_imputation(self, cfg, df_with_missing):
        analysis = make_analysis(
            True,
            {"columns_to_drop": [], "columns_to_impute": {"age": "median"}},
            "missing_values",
        )
        _, result = MissingValueCleaner(cfg).clean(df_with_missing, analysis)
        assert result.applied is True

    def test_operation_records_fill_value(self, cfg, df_with_missing):
        analysis = make_analysis(
            True,
            {"columns_to_drop": [], "columns_to_impute": {"age": "median"}},
            "missing_values",
        )
        _, result = MissingValueCleaner(cfg).clean(df_with_missing, analysis)
        impute_ops = [o for o in result.operations if o.type == "impute_column"]
        assert len(impute_ops) == 1
        assert impute_ops[0].details["column"] == "age"
        assert impute_ops[0].details["strategy"] == "median"

    def test_skips_when_no_missing(self, cfg, df_clean):
        analysis = make_analysis(False, {})
        _, result = MissingValueCleaner(cfg).clean(df_clean, analysis)
        assert result.applied is False

    def test_works_without_analysis(self, cfg, df_with_missing):
        """Falls back to own detection when no AnalysisResult provided."""
        df_out, result = MissingValueCleaner(cfg).clean(df_with_missing, analysis_result=None)
        # Should impute or at least not crash
        assert not (df_out.isnull().all().any())  # No fully-null columns remain

    def test_imputes_with_zero_strategy(self, cfg):
        """Explicit analysis result with 'zero' strategy must fill NaN with 0."""
        cfg_zero = {**cfg, "numeric_imputation": "zero"}
        # Three columns: 1 missing = 33% < 50% threshold, so the row is NOT dropped
        df = pd.DataFrame({
            "score": [10.0, None, 30.0],
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        analysis = make_analysis(
            True,
            {"columns_to_drop": [], "columns_to_impute": {"score": "zero"}},
        )
        df_out, _ = MissingValueCleaner(cfg_zero).clean(df, analysis)
        assert df_out["score"].iloc[1] == 0.0

    def test_index_reset_after_row_drop(self, cfg):
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4, None, 6],
            "c": [7, None, 9],
        })
        analysis = make_analysis(True, {"columns_to_drop": [], "columns_to_impute": {}})
        df_out, _ = MissingValueCleaner(cfg).clean(df, analysis)
        assert list(df_out.index) == list(range(len(df_out)))


# ═══════════════════════════════════════════════════════════════════════════════
# OutlierHandlerCleaner
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutlierHandlerCleaner:

    def _make_outlier_analysis(self, col: str, n_outliers: int, lower: float, upper: float) -> AnalysisResult:
        return make_analysis(
            True,
            {"per_column": {col: {"n_outliers": n_outliers, "lower_fence": lower, "upper_fence": upper}}},
            "outliers",
        )

    def test_flag_action_adds_indicator_column(self, df_with_outliers):
        cfg = {"action": "flag", "iqr_multiplier": 1.5}
        analysis = self._make_outlier_analysis("age", 2, -1.0, 55.0)
        df_out, result = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis)
        assert "age_is_outlier" in df_out.columns
        assert df_out["age_is_outlier"].dtype == bool

    def test_flag_marks_correct_rows(self, df_with_outliers):
        cfg = {"action": "flag", "iqr_multiplier": 1.5}
        analysis = self._make_outlier_analysis("age", 2, -1.0, 55.0)
        df_out, _ = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis)
        # age=-5 and age=999 should be flagged
        flagged = df_out.loc[df_out["age_is_outlier"], "age"].tolist()
        assert -5.0 in flagged
        assert 999.0 in flagged

    def test_flag_original_data_unchanged(self, df_with_outliers):
        cfg = {"action": "flag", "iqr_multiplier": 1.5}
        analysis = self._make_outlier_analysis("age", 2, -1.0, 55.0)
        df_out, _ = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis)
        # Original age values should be untouched
        assert -5.0 in df_out["age"].values
        assert 999.0 in df_out["age"].values

    def test_cap_action_clips_to_fence(self, df_with_outliers):
        cfg = {"action": "cap", "iqr_multiplier": 1.5}
        lower, upper = -1.0, 55.0
        analysis = self._make_outlier_analysis("age", 2, lower, upper)
        df_out, result = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis)
        assert df_out["age"].min() >= lower
        assert df_out["age"].max() <= upper
        assert result.applied is True

    def test_cap_does_not_remove_rows(self, df_with_outliers):
        cfg = {"action": "cap", "iqr_multiplier": 1.5}
        analysis = self._make_outlier_analysis("age", 2, -1.0, 55.0)
        df_out, result = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis)
        assert len(df_out) == len(df_with_outliers)

    def test_drop_action_removes_outlier_rows(self, df_with_outliers):
        cfg = {"action": "drop", "iqr_multiplier": 1.5}
        analysis = self._make_outlier_analysis("age", 2, -1.0, 55.0)
        df_out, result = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis)
        assert len(df_out) < len(df_with_outliers)
        # Neither -5 nor 999 should remain
        assert -5.0 not in df_out["age"].values
        assert 999.0 not in df_out["age"].values

    def test_drop_resets_index(self, df_with_outliers):
        cfg = {"action": "drop", "iqr_multiplier": 1.5}
        analysis = self._make_outlier_analysis("age", 2, -1.0, 55.0)
        df_out, _ = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis)
        assert list(df_out.index) == list(range(len(df_out)))

    def test_skips_when_no_outliers(self, df_clean):
        cfg = {"action": "flag", "iqr_multiplier": 1.5}
        analysis = make_analysis(False, {}, "outliers")
        _, result = OutlierHandlerCleaner(cfg).clean(df_clean, analysis)
        assert result.applied is False

    def test_works_without_analysis(self, df_with_outliers):
        cfg = {"action": "flag", "iqr_multiplier": 1.5}
        df_out, result = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis_result=None)
        # Should detect and flag without analysis
        assert result.applied is True

    def test_cap_operation_recorded_in_result(self, df_with_outliers):
        cfg = {"action": "cap", "iqr_multiplier": 1.5}
        analysis = self._make_outlier_analysis("age", 2, -1.0, 55.0)
        _, result = OutlierHandlerCleaner(cfg).clean(df_with_outliers, analysis)
        ops = [o for o in result.operations if o.type == "cap_outlier"]
        assert len(ops) == 1
        assert ops[0].details["column"] == "age"
