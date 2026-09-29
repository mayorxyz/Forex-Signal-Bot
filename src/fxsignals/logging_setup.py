"""Console + rotating-file logging setup for fxsignals."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path = "logs/fxsignals.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure the root ``fxsignals`` logger with console and file handlers.

    Idempotent: calling it again replaces previous handlers instead of
    duplicating output.

    Args:
        level: Log level name (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        log_file: Path of the rotating log file.
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated files to keep.

    Returns:
        The configured ``fxsignals`` logger.

    Raises:
        ValueError: if ``level`` is not a valid logging level name.
    """
    numeric = getattr(logging, str(level).strip().upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level {level!r}")

    logger = logging.getLogger("fxsignals")
    logger.setLevel(numeric)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
