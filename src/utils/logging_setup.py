# src/utils/logging_setup.py
"""
Centralized logging configuration for Research Studio.

Usage:
    from src.utils.logging_setup import get_logger, setup_logging

    # At application start
    setup_logging()

    # In each module
    logger = get_logger(__name__)
    logger.info("Processing query")
    logger.debug("Details: %s", data)
    logger.warning("Something unexpected")
    logger.error("Failed to process", exc_info=True)
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

from src.core.config import get_config


# Custom formatter with colors for terminal
class ColorFormatter(logging.Formatter):
    """Formatter with color support for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


# Module-level state
_logging_configured = False


def setup_logging(
    level: Optional[str] = None,
    use_colors: bool = True,
    show_timestamps: bool = False,
) -> None:
    """
    Configure logging for the application.

    Should be called once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to config value.
        use_colors: Whether to use ANSI colors in terminal output.
        show_timestamps: Whether to include timestamps in log messages.
    """
    global _logging_configured

    if _logging_configured:
        return

    cfg = get_config()

    # Determine log level
    log_level = level or cfg.log.level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Build format string
    if show_timestamps or cfg.log.show_timestamps:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    else:
        fmt = "[%(levelname)s] %(name)s: %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)

    # Use color formatter if terminal supports it
    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    formatter = ColorFormatter(fmt, use_colors=use_colors and is_tty)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        Configured logger instance
    """
    # Ensure logging is set up
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """Change log level at runtime."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_level)


# Convenience functions for quick logging without getting a logger
def log_info(msg: str, *args, **kwargs) -> None:
    """Quick info log."""
    logging.getLogger("research").info(msg, *args, **kwargs)


def log_debug(msg: str, *args, **kwargs) -> None:
    """Quick debug log."""
    logging.getLogger("research").debug(msg, *args, **kwargs)


def log_warning(msg: str, *args, **kwargs) -> None:
    """Quick warning log."""
    logging.getLogger("research").warning(msg, *args, **kwargs)


def log_error(msg: str, *args, **kwargs) -> None:
    """Quick error log."""
    logging.getLogger("research").error(msg, *args, **kwargs)


__all__ = [
    "setup_logging",
    "get_logger",
    "set_log_level",
    "log_info",
    "log_debug",
    "log_warning",
    "log_error",
]
