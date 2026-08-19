"""Tests for checkpoint-first wall-clock termination."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from src.utils.time_guard import TimeGuard, TimeLimitReached


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_time_guard_finishes_operation_then_checkpoints_latest_state() -> None:
    clock = FakeClock()
    checkpoints: list[Mapping[str, Any]] = []
    training_state = {"epoch": 3, "global_step": 99, "best_score": 0.72}
    guard = TimeGuard(
        max_runtime_seconds=10,
        safety_reserve_seconds=2,
        checkpoint_callback=checkpoints.append,
        clock=clock,
        wall_clock=lambda: datetime(2026, 8, 12, tzinfo=UTC),
    )

    def operation() -> str:
        clock.advance(9)
        training_state["global_step"] = 100
        return "finished"

    with pytest.raises(TimeLimitReached, match="checkpoint"):
        guard.run_safe_operation(operation, state_provider=lambda: training_state)

    assert training_state["global_step"] == 100
    assert len(checkpoints) == 1
    assert checkpoints[0]["training_state"]["global_step"] == 100
    assert checkpoints[0]["time_guard"]["stop_requested"] is True


def test_time_guard_refuses_operation_that_cannot_fit_safely() -> None:
    clock = FakeClock()
    ran = False
    guard = TimeGuard(max_runtime_seconds=20, safety_reserve_seconds=5, clock=clock)
    clock.advance(12)

    def operation() -> None:
        nonlocal ran
        ran = True

    with pytest.raises(TimeLimitReached):
        guard.run_safe_operation(operation, estimated_seconds=4)

    assert ran is False


@pytest.mark.parametrize(
    ("maximum", "reserve"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_time_guard_rejects_invalid_budget(maximum: float, reserve: float) -> None:
    with pytest.raises(ValueError):
        TimeGuard(max_runtime_seconds=maximum, safety_reserve_seconds=reserve)

