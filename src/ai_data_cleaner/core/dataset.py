"""
Dataset wrapper for AI Data Cleaner.

DESIGN DECISION:
    Why wrap pandas DataFrame in a custom class?

    1. METADATA: The raw DataFrame tells us nothing about WHERE it came from,
       WHAT it was called, or WHEN it was loaded. The Dataset class carries
       all of that context alongside the data.

    2. PRECOMPUTED STATS: Analyzers need shape, dtypes, missing counts, etc.
       Computing these once at load time (rather than per-analyzer) is faster
       and ensures all analyzers see the same baseline numbers.

    3. VALIDATION: We validate the file EXISTS, is readable, is a real CSV
       with columns and rows — BEFORE any analyzer runs. Fail early, fail loud.

    4. IMMUTABILITY OF ORIGINAL: The Dataset keeps a reference to the ORIGINAL
       dataframe. The CleaningPipeline works on a copy. This lets us compare
       "before" and "after" and compute what changed.

USAGE:
    from ai_data_cleaner.core.dataset import Dataset

    dataset = Dataset.from_csv("data/train.csv", encoding="utf-8")
    print(dataset.shape)           # (891, 12)
    print(dataset.column_names)    # ['PassengerId', 'Survived', ...]
    print(dataset.missing_counts)  # {'Age': 177, 'Cabin': 687, ...}
    print(dataset.numeric_columns) # ['Age', 'Fare', ...]
    print(dataset.df)              # The full pandas DataFrame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ai_data_cleaner.utils.exceptions import DatasetLoadError, DatasetValidationError

logger = logging.getLogger("ai_data_cleaner")


@dataclass
class DatasetMetadata:
    """
    Precomputed descriptive statistics for a dataset.

    Computed ONCE at load time. Analyzers read from here rather than
    recomputing the same stats repeatedly.
    """

    source_path: Path
    n_rows: int
    n_cols: int
    column_names: list[str]
    dtypes: dict[str, str]           # column_name -> dtype string (e.g., "int64", "object")
    missing_counts: dict[str, int]   # column_name -> count of NaN/None values
    missing_pcts: dict[str, float]   # column_name -> fraction missing (0.0–1.0)
    duplicate_row_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]   # dtype == "object" or low-cardinality
    datetime_columns: list[str]
    memory_usage_mb: float


class Dataset:
    """
    Immutable wrapper around a pandas DataFrame with precomputed metadata.

    The Dataset is ALWAYS built from a source file via Dataset.from_csv().
    Do not construct it directly — use the class method.

    Attributes:
        df:       The original loaded DataFrame. Read-only — do not modify.
        meta:     Precomputed DatasetMetadata.
        config:   The dataset-section config dict used during loading.
    """

    def __init__(self, df: pd.DataFrame, meta: DatasetMetadata):
        # Store originals — these are never modified after construction
        self._df = df
        self._meta = meta

    # ─── Properties (read-only access to internals) ───────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """The original DataFrame. Do not modify — work on .copy() instead."""
        return self._df

    @property
    def meta(self) -> DatasetMetadata:
        """Precomputed metadata — shape, types, missing counts, etc."""
        return self._meta

    # ─── Convenience pass-throughs (so callers can use dataset.shape etc.) ────

    @property
    def shape(self) -> tuple[int, int]:
        return self._meta.n_rows, self._meta.n_cols

    @property
    def column_names(self) -> list[str]:
        return self._meta.column_names

    @property
    def numeric_columns(self) -> list[str]:
        return self._meta.numeric_columns

    @property
    def categorical_columns(self) -> list[str]:
        return self._meta.categorical_columns

    @property
    def missing_counts(self) -> dict[str, int]:
        return self._meta.missing_counts

    @property
    def missing_pcts(self) -> dict[str, float]:
        return self._meta.missing_pcts

    def copy(self) -> pd.DataFrame:
        """
        Return a COPY of the underlying DataFrame for mutation.

        WHY: Analyzers and cleaners should never modify the original data.
        The pipeline always works on a copy so we can compare before/after.

        Usage:
            working_df = dataset.copy()  # mutate this freely
        """
        return self._df.copy()

    # ─── Factory Method ───────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        encoding: str = "utf-8",
        delimiter: str = ",",
        max_rows: int | None = None,
    ) -> Dataset:
        """
        Load a CSV file and return a validated Dataset instance.

        Args:
            path:      Absolute or relative path to the CSV file.
            encoding:  File encoding. Try "latin-1" if "utf-8" fails.
            delimiter: Column separator character.
            max_rows:  If set, only load this many rows (for performance testing).

        Returns:
            A Dataset instance with precomputed metadata.

        Raises:
            DatasetLoadError:       If the file cannot be read.
            DatasetValidationError: If the loaded data fails structural checks.
        """
        path = Path(path)
        logger.info("Loading dataset: %s", path.name)

        # ── Validate file existence BEFORE attempting to read ─────────────────
        if not path.exists():
            raise DatasetLoadError(
                f"File not found: {path}\n"
                f"  Tip: Check the path is correct and the file exists."
            )
        if not path.is_file():
            raise DatasetLoadError(f"Path is not a file: {path}")
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            raise DatasetLoadError(
                f"Unsupported file type: {path.suffix}\n"
                f"  Supported types: .csv, .tsv, .txt"
            )

        # ── Attempt to load with pandas ───────────────────────────────────────
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                sep=delimiter,
                nrows=max_rows,
                low_memory=False,   # Avoid mixed-type warnings on large files
            )
        except UnicodeDecodeError as e:
            raise DatasetLoadError(
                f"Encoding error reading {path.name} (tried '{encoding}').\n"
                f"  Tip: Try --encoding latin-1 for older Kaggle datasets.\n"
                f"  Detail: {e}"
            ) from e
        except pd.errors.EmptyDataError as e:
            raise DatasetLoadError(
                f"File is empty or has no data: {path.name}"
            ) from e
        except pd.errors.ParserError as e:
            raise DatasetLoadError(
                f"Could not parse {path.name} as CSV.\n"
                f"  Tip: Check the delimiter matches the file (default is comma).\n"
                f"  Detail: {e}"
            ) from e
        except Exception as e:
            raise DatasetLoadError(
                f"Unexpected error loading {path.name}: {e}"
            ) from e

        # ── Validate the loaded DataFrame ─────────────────────────────────────
        cls._validate_dataframe(df, path)

        # ── Compute metadata ──────────────────────────────────────────────────
        meta = cls._compute_metadata(df, path)

        logger.info(
            "Dataset loaded: %d rows × %d cols | %.1f MB | %d missing cells",
            meta.n_rows,
            meta.n_cols,
            meta.memory_usage_mb,
            sum(meta.missing_counts.values()),
        )

        return cls(df=df, meta=meta)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, path: Path) -> None:
        """
        Structural validation after loading.

        We check for the most common pathological cases that would cause
        confusing errors later in the pipeline.
        """
        if df.empty:
            raise DatasetValidationError(
                f"Dataset is empty (0 rows): {path.name}"
            )
        if len(df.columns) == 0:
            raise DatasetValidationError(
                f"Dataset has no columns: {path.name}"
            )
        if len(df.columns) == 1:
            # Likely a delimiter mismatch — entire row ended up in one column
            logger.warning(
                "Dataset has only 1 column. Is the delimiter correct? "
                "Try --delimiter ';' or '\\t' if the file uses those."
            )
        # Warn about duplicate column names (common in dirty Kaggle data)
        if len(df.columns) != len(set(df.columns)):
            dupes = [c for c in df.columns if list(df.columns).count(c) > 1]
            logger.warning("Duplicate column names detected: %s", dupes)

    @staticmethod
    def _compute_metadata(df: pd.DataFrame, path: Path) -> DatasetMetadata:
        """
        Precompute all statistics needed by analyzers.

        WHY precompute here?
            Each analyzer would otherwise recompute the same df.isnull().sum()
            or df.dtypes independently. Computing once is faster and guarantees
            all analyzers see the same baseline numbers.
        """
        n_rows, n_cols = df.shape
        column_names = df.columns.tolist()

        # Missing value counts and percentages per column
        missing_series = df.isnull().sum()
        missing_counts: dict[str, int] = missing_series.to_dict()
        missing_pcts: dict[str, float] = {
            col: (count / n_rows) if n_rows > 0 else 0.0
            for col, count in missing_counts.items()
        }

        # Duplicate row count
        duplicate_row_count = int(df.duplicated().sum())

        # Column type classification
        numeric_columns: list[str] = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        # Datetime columns — pandas auto-detect is off by default,
        # so we check object columns that look like dates separately
        datetime_columns: list[str] = df.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns.tolist()

        # Categorical = object dtype, pandas Categorical, or pandas StringDtype
        # NOTE: pandas 3 introduced a native 'string' dtype separate from 'object'.
        # We include both to work correctly on pandas 2 and 3.
        categorical_columns: list[str] = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        # dtype as string for serialization
        dtypes: dict[str, str] = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Memory usage in megabytes
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

        return DatasetMetadata(
            source_path=path.resolve(),
            n_rows=n_rows,
            n_cols=n_cols,
            column_names=column_names,
            dtypes=dtypes,
            missing_counts=missing_counts,
            missing_pcts=missing_pcts,
            duplicate_row_count=duplicate_row_count,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            memory_usage_mb=round(memory_usage_mb, 3),
        )

    def __repr__(self) -> str:
        return (
            f"Dataset(source='{self._meta.source_path.name}', "
            f"shape={self.shape}, "
            f"missing_cells={sum(self._meta.missing_counts.values())})"
        )
