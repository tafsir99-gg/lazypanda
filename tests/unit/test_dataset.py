"""
Unit tests for the Dataset class.

WHAT WE'RE TESTING:
    1. Clean CSV loads correctly with accurate metadata
    2. Dirty CSV loads and flags expected issues in metadata
    3. Missing value counts and percentages are correct
    4. Column type classification is correct
    5. Duplicate row count is correct
    6. Error cases: file not found, empty file, encoding error
    7. .copy() returns a DataFrame, not the original
    8. __repr__ works correctly
"""

from pathlib import Path

import pandas as pd
import pytest

from ai_data_cleaner.core.dataset import Dataset, DatasetMetadata
from ai_data_cleaner.utils.exceptions import DatasetLoadError, DatasetValidationError


# ─── Load Success Tests ───────────────────────────────────────────────────────

class TestDatasetLoading:
    """Tests for successful CSV loading."""

    def test_load_clean_csv_returns_dataset(self, clean_dataset):
        """Clean fixture file loads without errors."""
        assert isinstance(clean_dataset, Dataset)

    def test_shape_matches_file(self, clean_dataset):
        """Shape should be (10 rows, 11 columns) for sample_clean.csv."""
        assert clean_dataset.shape == (10, 11)

    def test_column_names_are_correct(self, clean_dataset):
        """All column names from the CSV header are present."""
        expected = [
            "passenger_id", "survived", "pclass", "name", "sex",
            "age", "fare", "cabin", "embarked", "ticket_count", "status",
        ]
        assert clean_dataset.column_names == expected

    def test_df_is_a_dataframe(self, clean_dataset):
        """The .df property returns a pandas DataFrame."""
        assert isinstance(clean_dataset.df, pd.DataFrame)

    def test_meta_is_dataset_metadata(self, clean_dataset):
        """The .meta property returns a DatasetMetadata instance."""
        assert isinstance(clean_dataset.meta, DatasetMetadata)

    def test_source_path_is_resolved(self, clean_dataset):
        """Metadata stores the absolute (resolved) path."""
        assert clean_dataset.meta.source_path.is_absolute()
        assert clean_dataset.meta.source_path.name == "sample_clean.csv"

    def test_dirty_csv_loads(self, dirty_dataset):
        """Dirty fixture loads without errors — it has bad DATA not bad FORMAT."""
        assert isinstance(dirty_dataset, Dataset)

    def test_dirty_csv_row_count(self, dirty_dataset):
        """sample_dirty.csv has 21 data rows (including 1 duplicate)."""
        assert dirty_dataset.shape[0] == 21


# ─── Metadata Accuracy Tests ──────────────────────────────────────────────────

class TestDatasetMetadata:
    """Tests that precomputed metadata values are correct."""

    def test_no_missing_values_in_clean_dataset(self, clean_dataset):
        """Clean fixture has missing values in 'cabin' column only."""
        # cabin has 5 missing values in the clean fixture
        total_missing = sum(clean_dataset.missing_counts.values())
        assert clean_dataset.missing_counts["cabin"] == 5
        # All other columns are complete
        non_cabin_missing = total_missing - clean_dataset.missing_counts["cabin"]
        assert non_cabin_missing == 0

    def test_missing_pcts_sum_to_correct_total(self, clean_dataset):
        """Missing percentages are between 0.0 and 1.0."""
        for col, pct in clean_dataset.missing_pcts.items():
            assert 0.0 <= pct <= 1.0, f"Column '{col}' has invalid missing pct: {pct}"

    def test_missing_pct_matches_count(self, clean_dataset):
        """missing_pcts should equal missing_counts / n_rows."""
        n_rows = clean_dataset.shape[0]
        for col in clean_dataset.column_names:
            expected_pct = clean_dataset.missing_counts[col] / n_rows
            assert abs(clean_dataset.missing_pcts[col] - expected_pct) < 1e-9

    def test_numeric_columns_identified(self, clean_dataset):
        """Numeric columns are correctly classified."""
        numeric = clean_dataset.numeric_columns
        assert "age" in numeric
        assert "fare" in numeric
        assert "passenger_id" in numeric
        # Text columns must NOT be in numeric
        assert "name" not in numeric
        assert "sex" not in numeric

    def test_categorical_columns_identified(self, clean_dataset):
        """Object-dtype columns are classified as categorical."""
        categorical = clean_dataset.categorical_columns
        assert "name" in categorical
        assert "sex" in categorical
        # Numeric columns must NOT be in categorical
        assert "age" not in categorical

    def test_duplicate_count_in_clean_dataset_is_zero(self, clean_dataset):
        """Clean fixture has no duplicate rows."""
        assert clean_dataset.meta.duplicate_row_count == 0

    def test_duplicate_count_in_dirty_dataset(self, dirty_dataset):
        """Dirty fixture has exactly 1 duplicate row (row 8 appears twice)."""
        assert dirty_dataset.meta.duplicate_row_count == 1

    def test_memory_usage_is_positive(self, clean_dataset):
        """Memory usage must be greater than zero."""
        assert clean_dataset.meta.memory_usage_mb > 0.0

    def test_dtypes_dict_has_all_columns(self, clean_dataset):
        """dtypes dict contains an entry for every column."""
        assert set(clean_dataset.meta.dtypes.keys()) == set(clean_dataset.column_names)


# ─── Copy Tests ───────────────────────────────────────────────────────────────

class TestDatasetCopy:
    """Tests for the .copy() method."""

    def test_copy_returns_dataframe(self, clean_dataset):
        """copy() returns a pandas DataFrame."""
        result = clean_dataset.copy()
        assert isinstance(result, pd.DataFrame)

    def test_copy_is_independent_of_original(self, clean_dataset):
        """Modifying the copy does not affect the original."""
        original_shape = clean_dataset.df.shape
        copy = clean_dataset.copy()
        copy.drop(columns=["name"], inplace=True)
        # Original must be unchanged
        assert clean_dataset.df.shape == original_shape

    def test_copy_has_same_data_as_original(self, clean_dataset):
        """copy() contains the same data as the original."""
        copy = clean_dataset.copy()
        pd.testing.assert_frame_equal(copy, clean_dataset.df)


# ─── Error Case Tests ─────────────────────────────────────────────────────────

class TestDatasetLoadErrors:
    """Tests for file-level errors that must be caught cleanly."""

    def test_missing_file_raises_dataset_load_error(self):
        """Non-existent file raises DatasetLoadError (not FileNotFoundError)."""
        with pytest.raises(DatasetLoadError, match="File not found"):
            Dataset.from_csv("/nonexistent/path/data.csv")

    def test_unsupported_extension_raises_error(self, tmp_path):
        """A .json file raises DatasetLoadError about unsupported type."""
        fake = tmp_path / "data.json"
        fake.write_text('{"key": "value"}')

        with pytest.raises(DatasetLoadError, match="Unsupported file type"):
            Dataset.from_csv(fake)

    def test_empty_csv_raises_error(self, tmp_path):
        """A CSV with only a header and no rows raises DatasetValidationError."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("col1,col2,col3\n", encoding="utf-8")

        with pytest.raises(DatasetValidationError, match="empty"):
            Dataset.from_csv(empty_csv)

    def test_completely_empty_file_raises_error(self, tmp_path):
        """A completely empty file raises DatasetLoadError."""
        empty = tmp_path / "nothing.csv"
        empty.write_text("", encoding="utf-8")

        with pytest.raises(DatasetLoadError):
            Dataset.from_csv(empty)

    def test_wrong_encoding_raises_helpful_error(self, tmp_path):
        """UTF-16 file read as UTF-8 raises DatasetLoadError with encoding hint."""
        latin_csv = tmp_path / "latin.csv"
        # Write a file with a character that is invalid in UTF-8 when misread
        latin_csv.write_bytes(b"name,age\n\xff\xfeAlice,25\n")

        with pytest.raises(DatasetLoadError, match="Encoding error|Could not parse"):
            Dataset.from_csv(latin_csv, encoding="utf-8")

    def test_directory_path_raises_error(self, tmp_path):
        """Passing a directory instead of a file raises DatasetLoadError."""
        with pytest.raises(DatasetLoadError, match="not a file"):
            Dataset.from_csv(tmp_path)


# ─── Repr Tests ───────────────────────────────────────────────────────────────

class TestDatasetRepr:
    """Tests for __repr__ — useful for debugging."""

    def test_repr_contains_filename(self, clean_dataset):
        rep = repr(clean_dataset)
        assert "sample_clean.csv" in rep

    def test_repr_contains_shape(self, clean_dataset):
        rep = repr(clean_dataset)
        assert "10" in rep  # row count
        assert "11" in rep  # column count


# ─── Inline DataFrame Tests ───────────────────────────────────────────────────

class TestDatasetFromInlineData:
    """Tests using pytest's tmp_path fixture to create CSVs on the fly."""

    def test_single_missing_column(self, tmp_path):
        """Column with all NaN is properly counted."""
        csv = tmp_path / "all_nan.csv"
        csv.write_text("a,b\n1,\n2,\n3,\n", encoding="utf-8")

        dataset = Dataset.from_csv(csv)
        assert dataset.missing_counts["b"] == 3
        assert dataset.missing_pcts["b"] == 1.0

    def test_no_missing_values(self, tmp_path):
        """Dataset with no missing values has all counts at zero."""
        csv = tmp_path / "complete.csv"
        csv.write_text("x,y,z\n1,2,3\n4,5,6\n", encoding="utf-8")

        dataset = Dataset.from_csv(csv)
        assert all(v == 0 for v in dataset.missing_counts.values())

    def test_semicolon_delimiter(self, tmp_path):
        """Dataset with semicolon delimiter loads correctly."""
        csv = tmp_path / "semicolon.csv"
        csv.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")

        dataset = Dataset.from_csv(csv, delimiter=";")
        assert dataset.shape == (2, 3)
        assert dataset.column_names == ["a", "b", "c"]
