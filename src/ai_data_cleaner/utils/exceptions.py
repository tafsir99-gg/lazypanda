"""
Custom exception hierarchy for AI Data Cleaner.

WHY a custom hierarchy?
    When something goes wrong, we catch OUR exceptions at the CLI boundary
    and show the user a clean, helpful message — not a Python stack trace.

    The hierarchy lets us be specific: `except DatasetLoadError` is clearer
    than `except Exception`, and we can still catch everything with
    `except AIDataCleanerError` when needed.

RULE: Never use bare `except:` or `except Exception:` silently.
      Always log the error before handling it.
"""


class AIDataCleanerError(Exception):
    """
    Base exception for all AI Data Cleaner errors.

    Catch this in the CLI layer to handle all application errors uniformly.
    Never catch this inside the core logic — be specific there.
    """
    pass


# ─── Dataset Errors ──────────────────────────────────────────────────────────

class DatasetLoadError(AIDataCleanerError):
    """
    Raised when a dataset cannot be loaded or is invalid.

    Examples:
        - File does not exist
        - File is not a valid CSV
        - File encoding is wrong
        - File is empty
    """
    pass


class DatasetValidationError(AIDataCleanerError):
    """
    Raised when a loaded dataset fails structural validation.

    Examples:
        - Zero columns after loading
        - Required columns specified in config are missing
    """
    pass


# ─── Configuration Errors ─────────────────────────────────────────────────────

class ConfigLoadError(AIDataCleanerError):
    """
    Raised when a config file cannot be read from disk.

    Examples:
        - Config file path does not exist
        - Config file is not valid YAML syntax
    """
    pass


class ConfigValidationError(AIDataCleanerError):
    """
    Raised when a config file is syntactically valid YAML
    but contains invalid values for our expected schema.

    Examples:
        - numeric_imputation: "banana" (not a valid option)
        - drop_column_threshold: 1.5 (must be between 0 and 1)
    """
    pass


# ─── Analysis Errors ─────────────────────────────────────────────────────────

class AnalysisError(AIDataCleanerError):
    """
    Raised when an analyzer fails unexpectedly during execution.

    This should be rare — analyzers are designed to handle edge cases
    gracefully. If this fires, it indicates a bug in an analyzer.
    """
    pass


# ─── Cleaning Errors ─────────────────────────────────────────────────────────

class CleaningError(AIDataCleanerError):
    """
    Raised when a cleaner fails to apply a transformation.

    Examples:
        - Type conversion that fails on unexpected values
        - Imputation with incompatible data types
    """
    pass


# ─── AI / Gemini Errors ───────────────────────────────────────────────────────

class AIClientError(AIDataCleanerError):
    """
    Raised when the Gemini API client cannot be initialized.

    Examples:
        - GEMINI_API_KEY is missing from .env
        - Network connection failure
    """
    pass


class AIBudgetExceededError(AIDataCleanerError):
    """
    Raised when a planned API call would exceed the token budget.

    This is a SAFETY error — the tool refuses to proceed rather than
    silently exceeding the configured cost limit.
    """
    pass


class AIResponseError(AIDataCleanerError):
    """
    Raised when Gemini returns a response that cannot be parsed
    or used by the tool.

    Examples:
        - Empty response
        - Response in unexpected format
        - Blocked content
    """
    pass


# ─── Output Errors ────────────────────────────────────────────────────────────

class ReportGenerationError(AIDataCleanerError):
    """
    Raised when a report cannot be generated or written to disk.

    Examples:
        - Output directory is not writable
        - Template rendering failure
    """
    pass


class ExportError(AIDataCleanerError):
    """
    Raised when a pipeline script or config cannot be exported.
    """
    pass
