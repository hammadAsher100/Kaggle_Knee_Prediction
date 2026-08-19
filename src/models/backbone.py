"""Offline-loadable image backbones."""

from __future__ import annotations

from pathlib import Path

from torch import nn


class DinoV2Backbone(nn.Module):
    """Thin wrapper around a locally mounted Hugging Face DINOv2 encoder."""

    def __init__(self, model_path: str | Path, *, trainable_blocks: int = 2) -> None:
        super().__init__()
        from transformers import AutoModel

        path = Path(model_path).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"DINOv2 model directory is invalid: {path}")
        self.model = AutoModel.from_pretrained(path, local_files_only=True)
        self.feature_dim = int(self.model.config.hidden_size)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(self.model, "encoder", None), "layer", [])
        for layer in list(layers)[-max(int(trainable_blocks), 0) :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def forward(self, images):
        return self.model(pixel_values=images).last_hidden_state[:, 0]
