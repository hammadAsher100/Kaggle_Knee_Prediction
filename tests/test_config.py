"""Tests for strict configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ConfigError, load_config, require_config_value


def test_load_config_deep_merges_and_expands_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.yaml"
    update = tmp_path / "update.yaml"
    base.write_text("paths:\n  root: ${TEST_DATA_ROOT}\nmodel:\n  size: 224\n", encoding="utf-8")
    update.write_text("model:\n  size: 320\n  name: tiny\n", encoding="utf-8")
    monkeypatch.setenv("TEST_DATA_ROOT", "data/location")

    config = load_config([base, update], overrides={"seed": 7})

    assert config == {
        "paths": {"root": "data/location"},
        "model": {"size": 320, "name": "tiny"},
        "seed": 7,
    }


def test_load_config_rejects_missing_environment_variable(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("path: ${VARIABLE_THAT_DOES_NOT_EXIST}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Undefined environment variable"):
        load_config(config_path)


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(config_path)


def test_require_config_value_distinguishes_missing_and_null() -> None:
    assert require_config_value({"model": {"name": "tiny"}}, "model.name") == "tiny"
    with pytest.raises(ConfigError, match="is null"):
        require_config_value({"model": {"name": None}}, "model.name")
    with pytest.raises(ConfigError, match="is missing"):
        require_config_value({}, "model.name")

