"""Fail-closed local/Kaggle execution-mode discovery and validation."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.config import ConfigError

VALID_MODES = {"local", "kaggle"}
KAGGLE_REQUIRED_FILES = (
    "train.csv",
    "train_series.csv",
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
)
KAGGLE_REQUIRED_DIRECTORIES = ("train_series", "test_series")


@dataclass(frozen=True)
class EnvironmentReport:
    """Resolved execution environment without secret or recursive file inspection."""

    mode: str
    is_kaggle_runtime: bool
    competition_slug: str
    competition_root: str | None
    working_dir: str
    use_sample: bool
    max_studies: int | None
    internet_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_kaggle_runtime(environ: Mapping[str, str] | None = None) -> bool:
    """Detect Kaggle from its runtime environment markers."""
    values = os.environ if environ is None else environ
    markers = (
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_URL_BASE",
        "KAGGLE_DATA_PROXY_TOKEN",
    )
    return any(bool(values.get(marker)) for marker in markers)


def discover_competition_root(
    input_root: str | Path,
    competition_slug: str,
    *,
    required_files: Sequence[str] = KAGGLE_REQUIRED_FILES,
    required_directories: Sequence[str] = KAGGLE_REQUIRED_DIRECTORIES,
) -> Path:
    """Find one mounted competition directory by shallow inspection only."""
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise ConfigError(f"Kaggle input root is unavailable: {root}")
    competition_group = root / "competitions"
    candidates = [root / competition_slug, competition_group / competition_slug]
    candidates.extend(
        path
        for path in root.iterdir()
        if path.is_dir() and competition_slug in path.name and path not in candidates
    )
    if competition_group.is_dir():
        candidates.extend(
            path
            for path in competition_group.iterdir()
            if path.is_dir() and competition_slug in path.name and path not in candidates
        )
    valid = [
        candidate
        for candidate in candidates
        if candidate.is_dir()
        and all((candidate / name).is_file() for name in required_files)
        and all((candidate / name).is_dir() for name in required_directories)
    ]
    if not valid:
        raise ConfigError(
            "Competition mount was not found with the expected top-level schema under "
            f"{root}; attach {competition_slug} to the Kaggle notebook"
        )
    resolved = sorted(set(valid), key=lambda path: path.as_posix())
    if len(resolved) != 1:
        joined = ", ".join(str(path) for path in resolved)
        raise ConfigError(f"Competition mount is ambiguous: {joined}")
    return resolved[0]


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section must be a mapping: {key}")
    return value


def prepare_runtime_config(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], EnvironmentReport]:
    """Validate mode constraints and resolve mounted Kaggle paths at runtime."""
    resolved = copy.deepcopy(dict(config))
    environment = dict(_require_mapping(resolved, "environment"))
    data = dict(_require_mapping(resolved, "data"))
    paths = dict(_require_mapping(resolved, "paths"))
    runtime = dict(_require_mapping(resolved, "runtime"))

    mode = str(environment.get("mode", "")).lower()
    if mode not in VALID_MODES:
        raise ConfigError(f"environment.mode must be one of {sorted(VALID_MODES)}")
    slug = str(environment.get("competition_slug", "")).strip()
    if not slug:
        raise ConfigError("environment.competition_slug must be non-empty")
    detected_kaggle = is_kaggle_runtime(environ)
    require_kaggle = bool(environment.get("require_kaggle_runtime", False))
    use_sample = bool(data.get("use_sample", False))
    max_studies_raw = data.get("max_studies")
    max_studies = int(max_studies_raw) if max_studies_raw is not None else None
    maximum = float(runtime.get("max_seconds", 0))
    reserve = float(runtime.get("safety_reserve_seconds", 0))
    if maximum <= 0 or reserve < 0 or reserve >= maximum:
        raise ConfigError("Runtime maximum and safety reserve are inconsistent")
    if maximum > 9 * 60 * 60:
        raise ConfigError("Configured runtime exceeds the verified 9-hour competition limit")

    competition_root: Path | None = None
    if mode == "local":
        if require_kaggle:
            raise ConfigError("Local mode cannot require a Kaggle runtime")
        if not use_sample:
            raise ConfigError("Local mode requires data.use_sample=true")
        if max_studies is None or not 1 <= max_studies <= 20:
            raise ConfigError("Local mode requires data.max_studies between 1 and 20")
        working_dir = Path(paths.get("working_dir", "artifacts/local")).expanduser().resolve()
    else:
        if require_kaggle and not detected_kaggle:
            raise ConfigError("Kaggle mode requires an actual Kaggle runtime")
        if bool(environment.get("internet_enabled", False)):
            raise ConfigError("Kaggle competition execution must keep internet disabled")
        input_root = paths.get("kaggle_input_root", "/kaggle/input")
        competition_root = discover_competition_root(input_root, slug)
        working_dir = Path(paths.get("working_dir", "/kaggle/working")).resolve()
        if not working_dir.is_dir():
            raise ConfigError(f"Kaggle working directory is unavailable: {working_dir}")
        paths.update(
            {
                "competition_root": str(competition_root),
                "train_csv": str(competition_root / "train.csv"),
                "sample_submission_csv": str(competition_root / "sample_submission.csv"),
                "train_dicom_root": str(competition_root / "train_series"),
                "test_dicom_root": str(competition_root / "test_series"),
                "metadata_files": [
                    str(competition_root / "train_series.csv"),
                    str(competition_root / "test_series.csv"),
                ],
            }
        )

    resolved["environment"] = environment
    resolved["data"] = data
    resolved["paths"] = paths
    report = EnvironmentReport(
        mode=mode,
        is_kaggle_runtime=detected_kaggle,
        competition_slug=slug,
        competition_root=str(competition_root) if competition_root else None,
        working_dir=str(working_dir),
        use_sample=use_sample,
        max_studies=max_studies,
        internet_enabled=bool(environment.get("internet_enabled", False)),
    )
    return resolved, report


def deterministic_study_sample(
    study_ids: Sequence[str],
    *,
    max_studies: int,
) -> list[str]:
    """Select a stable small study set for software verification."""
    if not 1 <= max_studies <= 20:
        raise ValueError("max_studies must be between 1 and 20")
    unique = sorted({str(study_id) for study_id in study_ids if str(study_id).strip()})
    return unique[:max_studies]
