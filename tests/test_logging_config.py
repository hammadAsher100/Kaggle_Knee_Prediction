"""Tests for structured operational logs."""

from __future__ import annotations

import io
import json

from src.utils.logging_config import configure_logging, log_event


def test_structured_logging_emits_parseable_json() -> None:
    stream = io.StringIO()
    logger = configure_logging(stream=stream)

    log_event(logger, "experiment_started", fold=2, config_hash="abc123")

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "INFO"
    assert payload["message"] == "experiment_started"
    assert payload["event"] == "experiment_started"
    assert payload["fold"] == 2
    assert payload["config_hash"] == "abc123"
    assert payload["timestamp"].endswith("+00:00")

