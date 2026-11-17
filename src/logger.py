"""Application-wide logging configuration and logger access."""

from __future__ import annotations

import logging
import os
import sys
from typing import Final


_ROOT_LOGGER_NAME: Final = "counsel_agent"
_DEFAULT_LOG_FORMAT: Final = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DEFAULT_DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"
_configured = False


def _resolve_level(level: int | str | None) -> int:
    """Resolve an explicit or environment-provided log level."""
    configured_level = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    if isinstance(configured_level, int):
        return configured_level

    numeric_level = logging.getLevelNamesMapping().get(configured_level.upper())
    if numeric_level is None:
        raise ValueError(f"Invalid log level: {configured_level!r}")
    return numeric_level


def configure_logging(level: int | str | None = None) -> None:
    """Configure the application logger once.

    ``LOG_LEVEL`` controls the default level and accepts standard Python logging
    level names such as ``DEBUG``, ``INFO`` and ``WARNING``. Calling this
    function again only updates the level and does not add duplicate handlers.
    """
    global _configured

    app_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    app_logger.setLevel(_resolve_level(level))
    if _configured:
        return

    app_logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
    )
    app_logger.addHandler(handler)
    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the application's configured logger namespace."""
    configure_logging()
    logger_name = (
        _ROOT_LOGGER_NAME
        if not name
        else f"{_ROOT_LOGGER_NAME}.{name}"
    )
    return logging.getLogger(logger_name)
