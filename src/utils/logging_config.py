"""Structured JSON logging without hidden global configuration."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

_STANDARD_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)


def configure_logging(
    *,
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure and return the project logger with JSON handlers."""
    logger = logging.getLogger("rsna_knee")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = JsonFormatter()
    stream_handler = logging.StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log a named operational event with structured fields."""
    logger.info(event, extra={"event": event, **fields})

