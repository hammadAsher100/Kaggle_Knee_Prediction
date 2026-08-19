"""Tests for fail-closed local/Kaggle execution separation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ConfigError
from src.utils.environment import (
    deterministic_study_sample,
    discover_competition_root,
    prepare_runtime_config,
)


def _base_config(tmp_path: Path) -> dict[str, object]:
    return {
        "environment": {
            "mode": "local",
            "competition_slug": "rsna-knee-abnormality-detection",
            "require_kaggle_runtime": False,
            "internet_enabled": True,
        },
        "data": {"use_sample": True, "max_studies": 10},
        "paths": {"working_dir": str(tmp_path / "working")},
        "runtime": {"max_seconds": 3600, "safety_reserve_seconds": 300},
    }


def _make_competition_mount(input_root: Path) -> Path:
    competition = input_root / "rsna-knee-abnormality-detection"
    competition.mkdir(parents=True)
    for filename in (
        "train.csv",
        "train_series.csv",
        "test.csv",
        "test_series.csv",
        "sample_submission.csv",
    ):
        (competition / filename).write_text("header\n", encoding="utf-8")
    (competition / "train_series").mkdir()
    (competition / "test_series").mkdir()
    return competition


def test_local_mode_requires_a_bounded_sample(tmp_path: Path) -> None:
    config = _base_config(tmp_path)

    resolved, report = prepare_runtime_config(config, environ={})

    assert resolved["environment"]["mode"] == "local"
    assert report.use_sample is True
    assert report.max_studies == 10
    assert report.competition_root is None

    config["data"] = {"use_sample": False, "max_studies": None}
    with pytest.raises(ConfigError, match="use_sample"):
        prepare_runtime_config(config, environ={})


def test_kaggle_mode_discovers_and_resolves_the_attached_mount(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    competition = _make_competition_mount(input_root)
    working = tmp_path / "working"
    working.mkdir()
    config = _base_config(tmp_path)
    config["environment"] = {
        "mode": "kaggle",
        "competition_slug": "rsna-knee-abnormality-detection",
        "require_kaggle_runtime": True,
        "internet_enabled": False,
    }
    config["data"] = {"use_sample": False, "max_studies": None}
    config["paths"] = {
        "kaggle_input_root": str(input_root),
        "working_dir": str(working),
    }
    config["runtime"] = {
        "max_seconds": 9 * 60 * 60,
        "safety_reserve_seconds": 20 * 60,
    }

    resolved, report = prepare_runtime_config(
        config,
        environ={"KAGGLE_KERNEL_RUN_TYPE": "Interactive"},
    )

    assert report.is_kaggle_runtime is True
    assert report.competition_root == str(competition.resolve())
    assert resolved["paths"]["train_csv"] == str(competition / "train.csv")
    assert resolved["paths"]["train_dicom_root"] == str(competition / "train_series")


def test_kaggle_mode_supports_grouped_competition_mounts(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    competition = _make_competition_mount(input_root / "competitions")

    discovered = discover_competition_root(
        input_root,
        "rsna-knee-abnormality-detection",
    )

    assert discovered == competition.resolve()


def test_kaggle_mode_fails_without_runtime_or_complete_mount(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["environment"] = {
        "mode": "kaggle",
        "competition_slug": "rsna-knee-abnormality-detection",
        "require_kaggle_runtime": True,
        "internet_enabled": False,
    }
    config["data"] = {"use_sample": False, "max_studies": None}
    config["paths"] = {
        "kaggle_input_root": str(tmp_path / "input"),
        "working_dir": str(tmp_path / "working"),
    }

    with pytest.raises(ConfigError, match="actual Kaggle runtime"):
        prepare_runtime_config(config, environ={})

    (tmp_path / "input").mkdir()
    (tmp_path / "working").mkdir()
    with pytest.raises(ConfigError, match="mount was not found"):
        prepare_runtime_config(
            config,
            environ={"KAGGLE_KERNEL_RUN_TYPE": "Interactive"},
        )


def test_discovery_does_not_accept_incomplete_schema(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    competition = input_root / "rsna-knee-abnormality-detection"
    competition.mkdir(parents=True)
    (competition / "train.csv").write_text("header\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="expected top-level schema"):
        discover_competition_root(input_root, competition.name)


def test_deterministic_study_sample_is_sorted_unique_and_bounded() -> None:
    studies = ["study-c", "study-a", "study-b", "study-a", ""]

    assert deterministic_study_sample(studies, max_studies=2) == ["study-a", "study-b"]
    with pytest.raises(ValueError, match="between 1 and 20"):
        deterministic_study_sample(studies, max_studies=21)
