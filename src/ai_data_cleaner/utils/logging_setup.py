"""
Structured logging configuration for AI Data Cleaner.

WHY two handlers (console + file)?
    Console handler: The user sees INFO-level messages — clean, minimal.
    File handler:    Records DEBUG-level details — timestamps, function names,
                     line numbers. Used when you need to diagnose a bug.

HOW TO USE in any module:
    import logging
    logger = logging.getLogger("ai_data_cleaner")
    logger.info("Starting analysis...")
    logger.debug("Processing column: %s", col_name)   # Only in debug log
    logger.warning("High missing rate detected: %.1f%%", pct)
    logger.error("Failed to load file: %s", str(e))
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: Path | None = None) -> logging.Logger:
    """
    Configure and return the application logger.
    
    Args:
        log_level: Console log level. Options: DEBUG, INFO, WARNING, ERROR.
                   File always logs at DEBUG level.
        log_dir:   Directory to write log files. If None, file logging is skipped.
    
    Returns:
        Configured logger instance for "ai_data_cleaner".
    
    Note:
        Call this ONCE at application startup (in main.py).
        All other modules get the same logger via:
            logger = logging.getLogger("ai_data_cleaner")
    """
    logger = logging.getLogger("ai_data_cleaner")

    # Prevent adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Logger itself captures everything

    # ─── Console Handler (what the user sees) ─────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_formatter = logging.Formatter(
        fmt="%(levelname)-8s %(message)s",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ─── File Handler (detailed debug log) ────────────────────────────────────
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"adc_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(module)s.%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Log the file location so the user knows where to look for debug info
        logger.debug("Log file: %s", log_file)

    return logger


def get_logger() -> logging.Logger:
    """
    Get the application logger from any module.
    
    This is a convenience function. After setup_logging() is called once,
    every module can get the same pre-configured logger with this call.
    
    Usage:
        from ai_data_cleaner.utils.logging_setup import get_logger
        logger = get_logger()
        logger.info("My message")
    """
    return logging.getLogger("ai_data_cleaner")
