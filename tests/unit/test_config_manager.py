"""
Unit tests for ConfigManager.

WHAT WE'RE TESTING:
    1. Defaults load correctly with valid values
    2. User overrides are applied (deep merge works)
    3. Missing user config keys fall back to defaults
    4. Invalid values raise ConfigValidationError with clear messages
    5. Missing files raise ConfigLoadError
    6. Invalid YAML syntax raises ConfigLoadError

TEST NAMING CONVENTION:
    test_<what>_<condition>_<expected result>
    Example: test_load_default_config_returns_valid_app_config
    This makes failing test names self-documenting.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from ai_data_cleaner.core.config_manager import AppConfig, ConfigManager
from ai_data_cleaner.utils.exceptions import ConfigLoadError, ConfigValidationError


# ─── Fixture for a temp config file ───────────────────────────────────────────

def _write_temp_yaml(data: dict) -> Path:
    """Helper: write a dict to a temp YAML file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


# ─── Happy Path Tests ─────────────────────────────────────────────────────────

class TestDefaultConfigLoading:
    """Tests for loading the bundled default config."""

    def test_load_returns_app_config_instance(self, default_config):
        """Default config loads and returns correct type."""
        assert isinstance(default_config, AppConfig)

    def test_default_encoding_is_utf8(self, default_config):
        assert default_config.dataset.encoding == "utf-8"

    def test_default_delimiter_is_comma(self, default_config):
        assert default_config.dataset.delimiter == ","

    def test_default_numeric_imputation_is_median(self, default_config):
        assert default_config.missing_values.numeric_imputation == "median"

    def test_default_outlier_action_is_flag(self, default_config):
        assert default_config.outliers.action == "flag"

    def test_default_iqr_multiplier_is_1_5(self, default_config):
        assert default_config.outliers.iqr_multiplier == 1.5

    def test_default_drop_column_threshold_is_0_8(self, default_config):
        assert default_config.missing_values.drop_column_threshold == 0.8

    def test_default_ai_enabled_is_true(self, default_config):
        assert default_config.ai.enabled is True

    def test_default_ai_max_tokens_is_2000(self, default_config):
        assert default_config.ai.max_tokens_per_run == 2000

    def test_default_normalize_case_is_true(self, default_config):
        assert default_config.categories.normalize_case is True


class TestDeepMerge:
    """Tests for the deep merge behaviour — user overrides only specified keys."""

    def test_single_key_override_preserves_other_keys(self):
        """Overriding one outlier key should leave others unchanged."""
        override = {"outliers": {"action": "drop"}}
        path = _write_temp_yaml(override)

        manager = ConfigManager()
        config = manager.load(user_config_path=path)

        # The override is applied
        assert config.outliers.action == "drop"
        # Other keys in the same section are preserved from defaults
        assert config.outliers.method == "iqr"
        assert config.outliers.iqr_multiplier == 1.5

        path.unlink()  # Cleanup temp file

    def test_nested_section_override(self):
        """Overriding a whole section replaces it but other sections survive."""
        override = {
            "missing_values": {
                "numeric_imputation": "mean",
                "drop_column_threshold": 0.9,
                "drop_row_threshold": 0.5,
                "categorical_imputation": "mode",
                "constant_fill_value": "UNKNOWN",
            }
        }
        path = _write_temp_yaml(override)

        config = ConfigManager().load(user_config_path=path)

        assert config.missing_values.numeric_imputation == "mean"
        assert config.missing_values.drop_column_threshold == 0.9
        # Sections not in override are default
        assert config.duplicates.keep == "first"

        path.unlink()

    def test_ai_disabled_in_override(self):
        """Can disable AI via user config override."""
        override = {"ai": {"enabled": False}}
        path = _write_temp_yaml(override)

        config = ConfigManager().load(user_config_path=path)
        assert config.ai.enabled is False

        path.unlink()


# ─── Validation Error Tests ───────────────────────────────────────────────────

class TestConfigValidation:
    """Tests that invalid values are caught and produce clear errors."""

    def test_invalid_threshold_above_1_raises_error(self):
        """drop_column_threshold must be between 0 and 1."""
        bad_data = {"missing_values": {"drop_column_threshold": 1.5}}
        path = _write_temp_yaml(bad_data)

        with pytest.raises(ConfigValidationError):
            ConfigManager().load(user_config_path=path)

        path.unlink()

    def test_negative_threshold_raises_error(self):
        """Negative thresholds are not valid."""
        bad_data = {"missing_values": {"drop_row_threshold": -0.1}}
        path = _write_temp_yaml(bad_data)

        with pytest.raises(ConfigValidationError):
            ConfigManager().load(user_config_path=path)

        path.unlink()

    def test_invalid_imputation_strategy_raises_error(self):
        """Only allowed imputation strategies are accepted."""
        bad_data = {"missing_values": {"numeric_imputation": "banana"}}
        path = _write_temp_yaml(bad_data)

        with pytest.raises(ConfigValidationError):
            ConfigManager().load(user_config_path=path)

        path.unlink()

    def test_invalid_outlier_action_raises_error(self):
        """Outlier action must be flag, cap, or drop."""
        bad_data = {"outliers": {"action": "delete"}}
        path = _write_temp_yaml(bad_data)

        with pytest.raises(ConfigValidationError):
            ConfigManager().load(user_config_path=path)

        path.unlink()

    def test_negative_iqr_multiplier_raises_error(self):
        """IQR multiplier must be positive."""
        bad_data = {"outliers": {"iqr_multiplier": -1.0}}
        path = _write_temp_yaml(bad_data)

        with pytest.raises(ConfigValidationError):
            ConfigManager().load(user_config_path=path)

        path.unlink()

    def test_zero_max_tokens_raises_error(self):
        """max_tokens_per_run must be at least 100."""
        bad_data = {"ai": {"max_tokens_per_run": 0}}
        path = _write_temp_yaml(bad_data)

        with pytest.raises(ConfigValidationError):
            ConfigManager().load(user_config_path=path)

        path.unlink()

    def test_invalid_duplicates_keep_raises_error(self):
        """duplicates.keep must be first, last, or none."""
        bad_data = {"duplicates": {"keep": "all"}}
        path = _write_temp_yaml(bad_data)

        with pytest.raises(ConfigValidationError):
            ConfigManager().load(user_config_path=path)

        path.unlink()


# ─── File Error Tests ─────────────────────────────────────────────────────────

class TestConfigFileErrors:
    """Tests for file-level errors (missing file, bad YAML)."""

    def test_missing_config_file_raises_config_load_error(self):
        """Non-existent file path raises ConfigLoadError, not FileNotFoundError."""
        with pytest.raises(ConfigLoadError, match="Config file not found"):
            ConfigManager().load(user_config_path="/nonexistent/path/config.yaml")

    def test_invalid_yaml_syntax_raises_config_load_error(self, tmp_path):
        """YAML with syntax errors raises ConfigLoadError with helpful message."""
        bad_yaml_file = tmp_path / "bad.yaml"
        bad_yaml_file.write_text("outliers:\n  action: [unclosed", encoding="utf-8")

        with pytest.raises(ConfigLoadError, match="invalid YAML syntax"):
            ConfigManager().load(user_config_path=bad_yaml_file)

    def test_non_mapping_yaml_raises_config_load_error(self, tmp_path):
        """YAML that is a list, not a mapping, raises ConfigLoadError."""
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ConfigLoadError, match="YAML mapping"):
            ConfigManager().load(user_config_path=list_yaml)
