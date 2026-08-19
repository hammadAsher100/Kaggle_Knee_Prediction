"""Deterministic multilabel stratification with indivisible leakage groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldQuality:
    """Optimization diagnostics for one generated fold assignment."""

    objective: float
    restart: int
    n_splits: int
    group_count: int
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_inputs(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_columns: Sequence[str],
    n_splits: int,
) -> None:
    required = {group_column, *target_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fold input is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Fold input must not be empty")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if frame[group_column].isna().any():
        raise ValueError(f"Grouping column contains missing values: {group_column}")
    if frame[group_column].nunique() < n_splits:
        raise ValueError("Number of unique groups must be at least n_splits")
    if not target_columns:
        raise ValueError("At least one target column is required")
    for target in target_columns:
        values = set(frame[target].dropna().unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Target {target} is not binary: {sorted(values, key=str)}")


def _group_matrices(
    frame: pd.DataFrame,
    group_column: str,
    target_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby(group_column, sort=True, dropna=False)
    target_frame = frame.loc[:, target_columns]
    positives = target_frame.eq(1).astype(float)
    observed = target_frame.notna().astype(float)
    balance_frame = pd.concat(
        [
            positives.add_prefix("positive::"),
            observed.add_prefix("observed::"),
        ],
        axis=1,
    )
    labels = balance_frame.groupby(frame[group_column], sort=True).sum().astype(float)
    sizes = grouped.size().reindex(labels.index).to_numpy(dtype=float)
    groups = labels.index.to_numpy()
    return groups, labels.to_numpy(dtype=float), sizes


def _objective(
    fold_labels: np.ndarray,
    fold_sizes: np.ndarray,
    total_labels: np.ndarray,
    total_size: float,
) -> float:
    expected_labels = total_labels / fold_labels.shape[0]
    label_scale = np.maximum(expected_labels, 1.0)
    label_error = np.mean(((fold_labels - expected_labels) / label_scale) ** 2)
    expected_size = total_size / fold_sizes.shape[0]
    size_error = np.mean(((fold_sizes - expected_size) / max(expected_size, 1.0)) ** 2)
    empty_penalty = float(np.count_nonzero(fold_sizes == 0)) * 1000.0
    return float(label_error + 0.15 * size_error + empty_penalty)


def _assign_once(
    labels: np.ndarray,
    sizes: np.ndarray,
    *,
    n_splits: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    group_count, target_count = labels.shape
    total_labels = labels.sum(axis=0)
    rarity = 1.0 / np.maximum(total_labels, 1.0)
    difficulty = labels.dot(rarity) + sizes / max(sizes.sum(), 1.0) * 0.01
    jitter = rng.random(group_count) * 1e-7
    order = np.argsort(-(difficulty + jitter), kind="stable")

    assignments = np.full(group_count, -1, dtype=np.int16)
    fold_labels = np.zeros((n_splits, target_count), dtype=float)
    fold_sizes = np.zeros(n_splits, dtype=float)
    seeded_folds = rng.permutation(n_splits)

    for order_index, group_index in enumerate(order):
        if order_index < n_splits:
            chosen = int(seeded_folds[order_index])
        else:
            candidate_scores: list[tuple[float, float, int]] = []
            for fold in range(n_splits):
                candidate_labels = fold_labels.copy()
                candidate_sizes = fold_sizes.copy()
                candidate_labels[fold] += labels[group_index]
                candidate_sizes[fold] += sizes[group_index]
                score = _objective(
                    candidate_labels,
                    candidate_sizes,
                    total_labels,
                    float(sizes.sum()),
                )
                candidate_scores.append((score, candidate_sizes[fold], fold))
            chosen = min(candidate_scores)[2]
        assignments[group_index] = chosen
        fold_labels[chosen] += labels[group_index]
        fold_sizes[chosen] += sizes[group_index]

    score = _objective(fold_labels, fold_sizes, total_labels, float(sizes.sum()))
    return assignments, score


def make_multilabel_group_folds(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_columns: Sequence[str],
    n_splits: int = 5,
    seed: int = 20260812,
    restarts: int = 64,
    fold_column: str = "fold",
) -> tuple[pd.DataFrame, FoldQuality]:
    """Assign all rows in a group to one fold while balancing multilabel positives."""
    _validate_inputs(
        frame,
        group_column=group_column,
        target_columns=target_columns,
        n_splits=n_splits,
    )
    if restarts < 1:
        raise ValueError("restarts must be positive")
    groups, labels, sizes = _group_matrices(frame, group_column, target_columns)
    best_assignment: np.ndarray | None = None
    best_score = float("inf")
    best_restart = -1
    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        assignment, score = _assign_once(labels, sizes, n_splits=n_splits, rng=rng)
        if score < best_score:
            best_assignment = assignment
            best_score = score
            best_restart = restart
    if best_assignment is None:
        raise RuntimeError("Fold assignment failed unexpectedly")

    group_to_fold = dict(zip(groups.tolist(), best_assignment.tolist(), strict=True))
    result = frame.copy()
    result[fold_column] = result[group_column].map(group_to_fold).astype("int16")
    assert_group_disjoint(result, group_column=group_column, fold_column=fold_column)
    quality = FoldQuality(
        objective=best_score,
        restart=best_restart,
        n_splits=n_splits,
        group_count=len(groups),
        row_count=len(result),
    )
    return result, quality


def assert_group_disjoint(
    frame: pd.DataFrame,
    *,
    group_column: str,
    fold_column: str = "fold",
) -> None:
    """Raise if any leakage group occurs in more than one fold."""
    fold_counts = frame.groupby(group_column, dropna=False)[fold_column].nunique()
    leaking = fold_counts[fold_counts > 1]
    if not leaking.empty:
        examples = ", ".join(str(value) for value in leaking.index[:5])
        raise AssertionError(f"Groups span validation folds: {examples}")


def fold_audit(
    frame: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    study_column: str,
    patient_column: str | None = None,
    subgroup_columns: Sequence[str] = (),
    fold_column: str = "fold",
) -> dict[str, Any]:
    """Return counts, prevalences, and subgroup distributions for each fold."""
    required = {fold_column, study_column, *target_columns, *subgroup_columns}
    if patient_column is not None:
        required.add(patient_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fold audit is missing columns: {', '.join(missing)}")

    folds: dict[str, Any] = {}
    for fold, subset in frame.groupby(fold_column, sort=True):
        target_summary = {
            target: {
                "known_count": int(subset[target].notna().sum()),
                "positive_count": int((subset[target] == 1).sum()),
                "negative_count": int((subset[target] == 0).sum()),
                "prevalence_among_known": (
                    float(subset[target].dropna().eq(1).mean())
                    if subset[target].notna().any()
                    else None
                ),
            }
            for target in target_columns
        }
        subgroups = {
            column: {
                str(key): int(value)
                for key, value in subset[column].fillna("<missing>").value_counts().items()
            }
            for column in subgroup_columns
        }
        folds[str(int(fold))] = {
            "row_count": int(len(subset)),
            "study_count": int(subset[study_column].nunique(dropna=True)),
            "patient_count": (
                int(subset[patient_column].nunique(dropna=True))
                if patient_column is not None
                else None
            ),
            "targets": target_summary,
            "subgroups": subgroups,
        }
    return {"folds": folds}


def assert_train_valid_disjoint(
    frame: pd.DataFrame,
    *,
    validation_fold: int,
    study_column: str,
    patient_column: str | None = None,
    fold_column: str = "fold",
) -> None:
    """Assert explicit study and optional patient disjointness for one split."""
    train = frame.loc[frame[fold_column] != validation_fold]
    valid = frame.loc[frame[fold_column] == validation_fold]
    train_studies = set(train[study_column].dropna())
    valid_studies = set(valid[study_column].dropna())
    if not train_studies.isdisjoint(valid_studies):
        raise AssertionError("Training and validation study IDs overlap")
    if patient_column is not None:
        train_patients = set(train[patient_column].dropna())
        valid_patients = set(valid[patient_column].dropna())
        if not train_patients.isdisjoint(valid_patients):
            raise AssertionError("Training and validation patient IDs overlap")
