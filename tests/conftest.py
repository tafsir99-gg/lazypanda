"""
Shared pytest fixtures for AI Data Cleaner tests.

WHAT IS A FIXTURE?
    A fixture is a function that provides a ready-to-use object to your tests.
    Instead of writing the same setup code in every test function, you define
    it once here and pytest injects it automatically.

    Example: Rather than loading the config in every test that needs it:
        def test_something():
            config = ConfigManager().load()   # repeated everywhere
            ...

    With a fixture:
        def test_something(default_config):   # injected automatically
            ...

HOW FIXTURES WORK:
    - A function decorated with @pytest.fixture becomes a fixture.
    - Tests declare which fixtures they need via function parameters.
    - pytest finds the matching fixture by name and calls it before the test.
    - The `scope` parameter controls how often the fixture is created:
        "function" (default): Fresh instance for every test — safest
        "session": Created once for the entire test run — faster but shared
"""

from pathlib import Path

import pandas as pd
import pytest

from lazypanda.core.config_manager import AppConfig, ConfigManager
from lazypanda.core.dataset import Dataset

# ─── Path helpers ─────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CLEAN_CSV = FIXTURES_DIR / "sample_clean.csv"
DIRTY_CSV = FIXTURES_DIR / "sample_dirty.csv"
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"


# ─── Config fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def default_config() -> AppConfig:
    """
    Load the default config once for the entire test session.

    Scope = "session": This is safe because we never MODIFY the config object —
    tests only READ from it. Loading once saves time.
    """
    return ConfigManager(default_config_path=DEFAULT_CONFIG_PATH).load()


@pytest.fixture
def config_dict() -> dict:
    """
    Return a plain dict of default config values.
    Useful when testing functions that accept raw dicts.
    """
    return {
        "dataset": {"encoding": "utf-8", "delimiter": ",", "max_rows_for_analysis": 100000},
        "missing_values": {
            "drop_column_threshold": 0.8,
            "drop_row_threshold": 0.5,
            "numeric_imputation": "median",
            "categorical_imputation": "mode",
            "constant_fill_value": "UNKNOWN",
        },
        "duplicates": {"keep": "first"},
        "outliers": {
            "method": "iqr",
            "iqr_multiplier": 1.5,
            "zscore_threshold": 3.0,
            "action": "flag",
        },
        "cardinality": {
            "high_cardinality_threshold": 0.95,
            "near_constant_threshold": 0.99,
        },
        "categories": {"normalize_case": True, "strip_whitespace": True},
        "ai": {
            "enabled": False,  # ALWAYS disabled in tests — no API calls
            "max_tokens_per_run": 2000,
            "use_cache": False,
            "cache_ttl_hours": 24,
        },
        "output": {
            "directory": "./outputs",
            "save_cleaned_csv": True,
            "save_markdown_report": True,
            "save_json_summary": True,
            "save_pipeline_script": True,
        },
    }


# ─── Dataset fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def clean_dataset() -> Dataset:
    """Load the clean fixture CSV once for the session."""
    return Dataset.from_csv(CLEAN_CSV)


@pytest.fixture(scope="session")
def dirty_dataset() -> Dataset:
    """Load the dirty fixture CSV once for the session."""
    return Dataset.from_csv(DIRTY_CSV)


# ─── DataFrame factories ───────────────────────────────────────────────────────
# These fixtures create specific DataFrames for targeted unit tests.
# Using factories (functions) instead of fixed DataFrames means each test
# gets a fresh copy and cannot accidentally affect other tests.

@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """DataFrame where 'age' has 40% missing values."""
    return pd.DataFrame({
        "age":  [25.0, None, 30.0, None, 45.0],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """DataFrame with 2 exact duplicate rows."""
    return pd.DataFrame({
        "id":   [1, 2, 2, 3, 4],
        "name": ["Alice", "Bob", "Bob", "Carol", "Dave"],
        "age":  [25, 30, 30, 35, 40],
    })


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """
    DataFrame with clear outliers in 'age' (IQR method will catch -5 and 999).
    Normal range: 20–45. Outliers: -5 and 999.
    """
    return pd.DataFrame({
        "age":  [25.0, 30.0, 35.0, 28.0, -5.0, 40.0, 32.0, 999.0, 27.0, 33.0],
        "fare": [10.0, 20.0, 30.0, 15.0,  8.0, 25.0, 18.0, 22.0, 12.0, 17.0],
    })


@pytest.fixture
def df_with_inconsistent_categories() -> pd.DataFrame:
    """DataFrame where 'gender' has case/whitespace inconsistencies."""
    return pd.DataFrame({
        "gender": ["Male", "male", "MALE", " female", "Female", "female"],
        "class":  ["First", "first", "FIRST", "Second", "second", "SECOND"],
    })


@pytest.fixture
def df_clean() -> pd.DataFrame:
    """A perfectly clean DataFrame — no issues expected."""
    return pd.DataFrame({
        "age":    [25, 30, 35, 40, 45],
        "name":   ["alice", "bob", "carol", "dave", "eve"],
        "active": [True, True, False, True, False],
    })
