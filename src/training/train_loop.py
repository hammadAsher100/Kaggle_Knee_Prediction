"""Time-guarded mixed-precision training loop."""

from __future__ import annotations

import time
from typing import Any

import torch

from src.training.losses import weighted_soft_bce


def train_one_epoch(
    model,
    loader,
    optimizer,
    *,
    device: torch.device,
    scaler,
    scheduler=None,
    accumulation_steps: int = 1,
    gold_weight: float = 4.0,
    time_guard=None,
) -> dict[str, Any]:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    example_count = 0
    step_count = 0
    for step, batch in enumerate(loader):
        started = time.monotonic()
        if time_guard is not None:
            time_guard.check(training_state={"batch": step})
        input_key = "features" if "features" in batch else "images"
        images = batch[input_key].to(device, non_blocking=True)
        mask = batch["slice_mask"].to(device, non_blocking=True)
        plane_ids = batch.get("plane_ids")
        if plane_ids is not None:
            plane_ids = plane_ids.to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        gold_mask = batch["gold_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images, mask, plane_ids)["logits"]
            loss = weighted_soft_bce(
                logits,
                targets,
                gold_mask,
                gold_weight=gold_weight,
            )
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            step_count += 1
        batch_size = int(images.shape[0])
        running_loss += float(loss.detach()) * batch_size
        example_count += batch_size
        if time_guard is not None:
            time_guard.record_operation(time.monotonic() - started)
    return {
        "loss": running_loss / max(example_count, 1),
        "examples": example_count,
        "optimizer_steps": step_count,
    }
