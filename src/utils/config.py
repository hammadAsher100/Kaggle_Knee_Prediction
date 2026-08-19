"""Strict YAML configuration loading and deterministic deep merging."""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or ambiguous."""


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Config:
    merged: Config = copy.deepcopy(dict(base))
    for key, value in update.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_environment(value: Any, source: Path) -> Any:
    if isinstance(value, str):
        missing = sorted({name for name in _ENV_PATTERN.findall(value) if name not in os.environ})
        if missing:
            names = ", ".join(missing)
            raise ConfigError(f"Undefined environment variable(s) in {source}: {names}")
        return _ENV_PATTERN.sub(lambda match: os.environ[match.group(1)], value)
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item, source) for item in value]
    return value


def _read_yaml(path: Path, expand_environment: bool) -> Config:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Unable to read configuration {path}: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")

    config = {str(key): value for key, value in loaded.items()}
    return _expand_environment(config, path) if expand_environment else config


def load_config(
    paths: str | Path | Sequence[str | Path],
    *,
    overrides: Mapping[str, Any] | None = None,
    expand_environment: bool = True,
) -> Config:
    """Load one or more YAML files, with later files and overrides taking precedence."""
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not raw_paths:
        raise ConfigError("At least one configuration path is required")

    merged: Config = {}
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()
        merged = _deep_merge(merged, _read_yaml(path, expand_environment))
    if overrides is not None:
        merged = _deep_merge(merged, overrides)
    return merged


def require_config_value(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Return a required nested value addressed by a dotted key."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Required configuration value is missing: {dotted_key}")
        current = current[part]
    if current is None:
        raise ConfigError(f"Required configuration value is null: {dotted_key}")
    return current

