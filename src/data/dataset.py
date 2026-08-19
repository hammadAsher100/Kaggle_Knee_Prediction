"""Study-level 2.5D DICOM dataset for training and inference."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dicom_io import DicomDecodeError, decode_dicom
from src.data.preprocessing import normalize_mri
from src.data.series_selector import select_series
from src.data.slice_sampler import (
    neighborhood_indices,
    random_uniform_indices,
    uniform_indices,
)


def _as_path_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError("ordered_dicom_paths must be a list-like Parquet value")


class KneeStudyDataset:
    """Load a fixed number of geometry-ordered 2.5D stacks per study."""

    def __init__(
        self,
        studies: pd.DataFrame,
        series_manifest: pd.DataFrame,
        dicom_root: str | Path,
        *,
        target_names: Sequence[str] = (),
        slices_per_series: int = 8,
        max_series: int = 3,
        image_size: int = 224,
        training: bool = False,
        seed: int = 20260812,
    ) -> None:
        import torch

        self.torch = torch
        self.studies = studies.reset_index(drop=True).copy()
        self.root = Path(dicom_root).expanduser().resolve()
        self.target_names = tuple(target_names)
        self.slices_per_series = int(slices_per_series)
        self.max_series = int(max_series)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        if min(self.slices_per_series, self.max_series, self.image_size) < 1:
            raise ValueError("sampling and image-size parameters must be positive")
        required = {"StudyInstanceUID", "SeriesInstanceUID", "ordered_dicom_paths"}
        missing = required.difference(series_manifest.columns)
        if missing:
            raise ValueError(f"Series manifest is missing columns: {sorted(missing)}")
        self.series_by_study = {
            str(study): group.to_dict("records")
            for study, group in series_manifest.groupby("StudyInstanceUID", sort=False)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.studies)

    def _resolve_dicom(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"DICOM path escapes configured root: {relative}") from error
        return candidate

    def _stack(self, paths: list[str], center: int) -> tuple[Any, bool]:
        from torch.nn import functional as F

        channels: list[np.ndarray] = []
        valid = True
        for index in neighborhood_indices(center, len(paths)):
            try:
                image, _ = decode_dicom(self._resolve_dicom(paths[int(index)]))
                channels.append(normalize_mri(image))
            except (DicomDecodeError, FileNotFoundError, OSError):
                valid = False
                channels.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
        if len({channel.shape for channel in channels}) != 1:
            minimum_h = min(channel.shape[0] for channel in channels)
            minimum_w = min(channel.shape[1] for channel in channels)
            channels = [
                channel[
                    (channel.shape[0] - minimum_h) // 2 : (channel.shape[0] + minimum_h) // 2,
                    (channel.shape[1] - minimum_w) // 2 : (channel.shape[1] + minimum_w) // 2,
                ]
                for channel in channels
            ]
        tensor = self.torch.from_numpy(np.stack(channels)).float().unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std, valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.studies.iloc[index]
        study_id = str(row["StudyInstanceUID"])
        rows = self.series_by_study.get(study_id, [])
        selected_uids = {
            item.series_uid for item in select_series(rows, max_series=self.max_series)
        }
        selected = [row for row in rows if str(row["SeriesInstanceUID"]) in selected_uids]
        selected.sort(key=lambda item: str(item["SeriesInstanceUID"]))
        stacks: list[Any] = []
        mask: list[bool] = []
        plane_ids: list[int] = []
        plane_map = {"Sagittal": 1, "Coronal": 2, "Axial": 3}
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
        for series in selected[: self.max_series]:
            paths = _as_path_list(series["ordered_dicom_paths"])
            if not paths:
                continue
            centers = (
                random_uniform_indices(len(paths), self.slices_per_series, rng)
                if self.training
                else uniform_indices(len(paths), self.slices_per_series)
            )
            for center in centers:
                stack, valid = self._stack(paths, int(center))
                stacks.append(stack)
                mask.append(valid)
                plane_ids.append(plane_map.get(str(series.get("Anatomical_Plane")), 0))
        expected = self.max_series * self.slices_per_series
        while len(stacks) < expected:
            stacks.append(
                self.torch.zeros((3, self.image_size, self.image_size), dtype=self.torch.float32)
            )
            mask.append(False)
            plane_ids.append(0)
        if not any(mask):
            mask[0] = True
        result: dict[str, Any] = {
            "study_id": study_id,
            "images": self.torch.stack(stacks[:expected]),
            "slice_mask": self.torch.tensor(mask[:expected], dtype=self.torch.bool),
            "plane_ids": self.torch.tensor(plane_ids[:expected], dtype=self.torch.long),
        }
        if self.target_names:
            result["targets"] = self.torch.tensor(
                [float(row[f"{target}__train"]) for target in self.target_names],
                dtype=self.torch.float32,
            )
            result["gold_targets"] = self.torch.tensor(
                [
                    float(row[f"{target}__gold"])
                    if pd.notna(row[f"{target}__gold"])
                    else float("nan")
                    for target in self.target_names
                ],
                dtype=self.torch.float32,
            )
            result["gold_mask"] = self.torch.tensor(
                [bool(row[f"{target}__gold_mask"]) for target in self.target_names],
                dtype=self.torch.bool,
            )
        return result


class FrozenFeatureDataset:
    """Join cached per-study image features to soft/gold training targets."""

    def __init__(
        self,
        studies: pd.DataFrame,
        feature_path: str | Path,
        *,
        target_names: Sequence[str] = (),
    ) -> None:
        import torch

        self.torch = torch
        self.studies = studies.reset_index(drop=True).copy()
        archive = np.load(feature_path, allow_pickle=False)
        feature_ids = archive["study_ids"].astype(str)
        if len(set(feature_ids.tolist())) != len(feature_ids):
            raise ValueError("feature archive contains duplicate study IDs")
        index_by_id = {value: index for index, value in enumerate(feature_ids)}
        try:
            order = [index_by_id[str(value)] for value in self.studies["StudyInstanceUID"]]
        except KeyError as error:
            raise ValueError(f"feature archive is missing study {error.args[0]}") from error
        self.features = archive["features"][order].astype(np.float32)
        self.masks = archive["slice_mask"][order].astype(bool)
        self.plane_ids = archive["plane_ids"][order].astype(np.int64)
        self.target_names = tuple(target_names)

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.studies.iloc[index]
        result: dict[str, Any] = {
            "study_id": str(row["StudyInstanceUID"]),
            "features": self.torch.from_numpy(self.features[index]),
            "slice_mask": self.torch.from_numpy(self.masks[index]),
            "plane_ids": self.torch.from_numpy(self.plane_ids[index]),
        }
        if self.target_names:
            result["targets"] = self.torch.tensor(
                [float(row[f"{target}__train"]) for target in self.target_names],
                dtype=self.torch.float32,
            )
            result["gold_targets"] = self.torch.tensor(
                [
                    float(row[f"{target}__gold"])
                    if pd.notna(row[f"{target}__gold"])
                    else float("nan")
                    for target in self.target_names
                ],
                dtype=self.torch.float32,
            )
            result["gold_mask"] = self.torch.tensor(
                [bool(row[f"{target}__gold_mask"]) for target in self.target_names],
                dtype=self.torch.bool,
            )
        return result
