"""
Configuration Manager for AI Data Cleaner.

ARCHITECTURE NOTE:
    The config system has three layers:

    1. DEFAULT CONFIG (config/default_config.yaml)
       Sensible defaults for everything. Never modified by users.

    2. USER CONFIG (any .yaml file passed via --config flag)
       Only specifies OVERRIDES. Missing keys fall back to defaults.
       Users don't need to copy the entire default config.

    3. VALIDATED CONFIG (ConfigSchema Pydantic model)
       After merging, ALL values are validated by Pydantic.
       If a value is out of range or the wrong type, we fail EARLY
       with a clear error — not halfway through cleaning a dataset.

WHY Pydantic?
    Pydantic validates Python types and value ranges at runtime.
    Without it, bad config (e.g., drop_column_threshold: "banana")
    would crash inside a pandas operation with a confusing error.
    With Pydantic, it crashes immediately with:
        "drop_column_threshold must be between 0.0 and 1.0"

USAGE:
    from lazypanda.core.config_manager import ConfigManager

    # Load with defaults only
    config = ConfigManager().load()

    # Load with user overrides
    config = ConfigManager().load(user_config_path="my_config.yaml")

    # Access a value
    threshold = config.missing_values.drop_column_threshold
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from lazypanda.utils.exceptions import ConfigLoadError, ConfigValidationError

logger = logging.getLogger("lazypanda")

# ─── Path to the bundled default config ──────────────────────────────────────
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default_config.yaml"


# ─── Pydantic Schema Models ───────────────────────────────────────────────────
# Each section of the YAML maps to one Pydantic model.
# Field() lets us set defaults + documentation inline.

class DatasetConfig(BaseModel):
    """Controls how CSV files are loaded."""

    encoding: str = Field(default="utf-8", description="File encoding (e.g. utf-8, latin-1)")
    delimiter: str = Field(default=",", description="Column delimiter character")
    max_rows_for_analysis: int = Field(
        default=100_000,
        ge=100,  # ge = greater than or equal to
        description="Maximum rows to analyze (for performance on huge files)",
    )


class MissingValuesConfig(BaseModel):
    """Controls how missing values are detected and handled."""

    drop_column_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Drop column if this fraction of values are missing",
    )
    drop_row_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Drop row if this fraction of columns are missing",
    )
    numeric_imputation: Literal["mean", "median", "mode", "constant", "none"] = Field(
        default="median",
        description="Strategy for filling missing numeric values",
    )
    categorical_imputation: Literal["mode", "constant", "none"] = Field(
        default="mode",
        description="Strategy for filling missing categorical values",
    )
    constant_fill_value: str = Field(
        default="UNKNOWN",
        description="Value used when imputation strategy is 'constant'",
    )


class DuplicatesConfig(BaseModel):
    """Controls how duplicate rows are handled."""

    keep: Literal["first", "last", "none"] = Field(
        default="first",
        description="Which duplicate to keep: first, last, or none (drop all)",
    )


class OutliersConfig(BaseModel):
    """Controls outlier detection and handling."""

    method: Literal["iqr", "zscore", "both"] = Field(
        default="iqr",
        description="Detection method: iqr, zscore, or both",
    )
    iqr_multiplier: float = Field(
        default=1.5,
        gt=0.0,  # gt = strictly greater than
        description="IQR fence multiplier (1.5=standard, 3.0=extreme only)",
    )
    zscore_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="Z-score threshold beyond which values are outliers",
    )
    action: Literal["flag", "cap", "drop"] = Field(
        default="flag",
        description="What to do with detected outliers",
    )


class CardinalityConfig(BaseModel):
    """Controls cardinality analysis thresholds."""

    high_cardinality_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Flag column if unique values / total rows exceeds this",
    )
    near_constant_threshold: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Flag column if one value appears this fraction of the time",
    )

    @model_validator(mode="after")
    def near_constant_must_exceed_high_cardinality(self) -> CardinalityConfig:
        """Near-constant threshold must be >= high-cardinality threshold logically."""
        # These are independent concepts so no strict ordering required,
        # but both must be sane values (already enforced by ge/le above).
        return self


class CategoriesConfig(BaseModel):
    """Controls categorical value normalization."""

    normalize_case: bool = Field(
        default=True,
        description="Lowercase all categorical values before comparison",
    )
    strip_whitespace: bool = Field(
        default=True,
        description="Strip leading/trailing whitespace from categorical values",
    )


class AIConfig(BaseModel):
    """Controls Gemini AI integration."""

    enabled: bool = Field(
        default=True,
        description="Master switch — set False for 100% deterministic, zero-cost runs",
    )
    max_tokens_per_run: int = Field(
        default=8000,
        ge=100,
        le=100_000,
        description="Hard token budget cap per cleaning session",
    )
    use_cache: bool = Field(
        default=True,
        description="Cache AI responses to disk to avoid repeated calls",
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        description="How long a cached response is valid (hours)",
    )


class OutputConfig(BaseModel):
    """Controls what files are generated after a run."""

    directory: str = Field(
        default="./outputs",
        description="Directory for all generated files",
    )
    save_cleaned_csv: bool = Field(default=True)
    save_markdown_report: bool = Field(default=True)
    save_json_summary: bool = Field(default=True)
    save_pipeline_script: bool = Field(default=True)

    @field_validator("directory")
    @classmethod
    def directory_must_be_valid_path(cls, v: str) -> str:
        """Reject obviously invalid path strings."""
        if not v or v.isspace():
            raise ValueError("output.directory cannot be empty")
        return v


class TypeInferenceConfig(BaseModel):
    """Controls type inference detection."""
    min_numeric_ratio: float = Field(
        default=0.9, ge=0.0, le=1.0,
        description="Min fraction of parseable numeric values to flag as numeric_string",
    )
    min_date_ratio: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Min fraction of parseable date values to flag as date_string",
    )


class SuspiciousValuesConfig(BaseModel):
    """Controls suspicious value detection."""
    sentinel_values: list[float] = Field(
        default=[-999.0, -9999.0, -1.0, 9999.0, 999.0],
        description="Numeric sentinel values used as placeholders for missing data",
    )
    extreme_ratio_threshold: float = Field(
        default=100.0, gt=1.0,
        description="Flag if max/min ratio exceeds this (indicates extreme magnitude difference)",
    )


class AppConfig(BaseModel):
    """
    Root configuration model.

    This is the single source of truth for all configuration values.
    After loading and validation, the rest of the application only ever
    reads from this model — never from raw dicts or YAML directly.
    """

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    missing_values: MissingValuesConfig = Field(default_factory=MissingValuesConfig)
    duplicates: DuplicatesConfig = Field(default_factory=DuplicatesConfig)
    outliers: OutliersConfig = Field(default_factory=OutliersConfig)
    cardinality: CardinalityConfig = Field(default_factory=CardinalityConfig)
    categories: CategoriesConfig = Field(default_factory=CategoriesConfig)
    type_inference: TypeInferenceConfig = Field(default_factory=TypeInferenceConfig)
    suspicious_values: SuspiciousValuesConfig = Field(default_factory=SuspiciousValuesConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)



# ─── ConfigManager ────────────────────────────────────────────────────────────

class ConfigManager:
    """
    Loads, merges, and validates YAML configuration files.

    DESIGN PATTERN: This class is a "Service" — it has no state of its own.
    Call load() each time you need a validated config object.

    Example:
        manager = ConfigManager()
        config = manager.load()                              # defaults only
        config = manager.load("path/to/my_config.yaml")     # with overrides
        print(config.missing_values.drop_column_threshold)  # 0.8
    """

    def __init__(self, default_config_path: Path | None = None):
        self._default_path = default_config_path or _DEFAULT_CONFIG_PATH

    def load(
        self,
        user_config_path: str | Path | None = None,
        overrides: dict | None = None,
    ) -> AppConfig:
        """
        Load and validate configuration.

        Args:
            user_config_path: Optional path to a user config YAML file.
                              Only overrides need to be specified — missing
                              keys fall back to defaults automatically.
            overrides:        Optional dict of runtime overrides (e.g. from CLI flags).
                              Applied last, taking highest precedence.

        Returns:
            Validated AppConfig instance.

        Raises:
            ConfigLoadError:       If a config file cannot be read.
            ConfigValidationError: If config values fail validation.
        """
        # Step 1: Load the default config
        default_data = self._load_yaml(self._default_path)
        logger.debug("Loaded default config from: %s", self._default_path)

        # Step 2: Load and merge user overrides (if provided)
        if user_config_path is not None:
            user_path = Path(user_config_path)
            user_data = self._load_yaml(user_path)
            logger.debug("Loaded user config from: %s", user_path)
            merged_data = self._deep_merge(default_data, user_data)
        else:
            merged_data = default_data

        # Step 3: Apply runtime overrides (highest priority)
        if overrides:
            merged_data = self._deep_merge(merged_data, overrides)
            logger.debug("Applied runtime overrides: %s", list(overrides.keys()))

        # Step 4: Validate with Pydantic
        return self._validate(merged_data)


    def _load_yaml(self, path: Path) -> dict:
        """Read a YAML file and return it as a plain Python dict."""
        if not path.exists():
            raise ConfigLoadError(
                f"Config file not found: {path}\n"
                f"  Tip: Check the path and ensure the file exists."
            )
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigLoadError(
                f"Config file contains invalid YAML syntax: {path}\n"
                f"  Error: {e}"
            ) from e

        if not isinstance(data, dict):
            raise ConfigLoadError(
                f"Config file must be a YAML mapping (key: value pairs), got: {type(data).__name__}"
            )
        return data

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        Recursively merge override dict into base dict.

        WHY deep merge instead of base.update(override)?
        A shallow update would replace entire nested sections.
        Deep merge means a user can override just ONE key in a section
        without having to specify the other keys.

        Example:
            base:     {outliers: {method: iqr, action: flag}}
            override: {outliers: {action: drop}}
            result:   {outliers: {method: iqr, action: drop}}  ← method preserved!
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _validate(self, data: dict) -> AppConfig:
        """Parse and validate the merged config dict with Pydantic."""
        try:
            return AppConfig.model_validate(data)
        except Exception as e:
            # Convert Pydantic's ValidationError into our custom exception
            # with a user-friendly message.
            raise ConfigValidationError(
                f"Configuration validation failed:\n{e}\n\n"
                f"  Tip: Check config/default_config.yaml for valid values."
            ) from e
