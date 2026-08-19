"""Wall-clock safety guard for checkpointed Kaggle workloads."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

T = TypeVar("T")
Clock = Callable[[], float]
CheckpointCallback = Callable[[Mapping[str, Any]], None]


class TimeLimitReached(RuntimeError):
    """Raised after a time-guard checkpoint completes successfully."""


@dataclass(frozen=True)
class TimeGuardState:
    session_started_at: str
    max_runtime_seconds: float
    safety_reserve_seconds: float
    elapsed_seconds: float
    remaining_runtime_seconds: float
    remaining_safe_seconds: float
    estimated_operation_seconds: float | None
    stop_requested: bool
    completed_operations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TimeGuard:
    """Permit safe units of work and checkpoint before the hard runtime limit.

    A training loop should wrap each atomic batch/epoch boundary with
    :meth:`run_safe_operation`. If the safety boundary is crossed while the
    operation runs, that operation finishes, the callback is invoked with the
    latest state, handlers are flushed, and :class:`TimeLimitReached` is raised.
    The callback owns serialization of model, optimizer, scheduler, scaler,
    epoch, best score, and random state because those objects belong to the
    future training layer.
    """

    def __init__(
        self,
        *,
        max_runtime_seconds: float,
        safety_reserve_seconds: float,
        checkpoint_callback: CheckpointCallback | None = None,
        logger: logging.Logger | None = None,
        operation_ema_alpha: float = 0.25,
        clock: Clock = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if safety_reserve_seconds < 0:
            raise ValueError("safety_reserve_seconds must be non-negative")
        if safety_reserve_seconds >= max_runtime_seconds:
            raise ValueError("safety_reserve_seconds must be smaller than max_runtime_seconds")
        if not 0 < operation_ema_alpha <= 1:
            raise ValueError("operation_ema_alpha must be in (0, 1]")

        self.max_runtime_seconds = float(max_runtime_seconds)
        self.safety_reserve_seconds = float(safety_reserve_seconds)
        self.checkpoint_callback = checkpoint_callback
        self.logger = logger or logging.getLogger("rsna_knee.time_guard")
        self.operation_ema_alpha = operation_ema_alpha
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(tz=UTC))
        self._started_monotonic = self._clock()
        self._started_at = self._wall_clock().isoformat()
        self._estimated_operation_seconds: float | None = None
        self._completed_operations = 0
        self._stop_requested = False
        self._checkpoint_completed = False

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_monotonic)

    @property
    def remaining_runtime_seconds(self) -> float:
        return max(0.0, self.max_runtime_seconds - self.elapsed_seconds)

    @property
    def remaining_safe_seconds(self) -> float:
        return max(0.0, self.remaining_runtime_seconds - self.safety_reserve_seconds)

    def state(self) -> TimeGuardState:
        return TimeGuardState(
            session_started_at=self._started_at,
            max_runtime_seconds=self.max_runtime_seconds,
            safety_reserve_seconds=self.safety_reserve_seconds,
            elapsed_seconds=self.elapsed_seconds,
            remaining_runtime_seconds=self.remaining_runtime_seconds,
            remaining_safe_seconds=self.remaining_safe_seconds,
            estimated_operation_seconds=self._estimated_operation_seconds,
            stop_requested=self._stop_requested,
            completed_operations=self._completed_operations,
        )

    def record_operation(self, duration_seconds: float) -> None:
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        previous = self._estimated_operation_seconds
        alpha = self.operation_ema_alpha
        if previous is None:
            self._estimated_operation_seconds = duration_seconds
        else:
            self._estimated_operation_seconds = (
                alpha * duration_seconds + (1 - alpha) * previous
            )
        self._completed_operations += 1

    def should_stop(self, *, estimated_next_operation_seconds: float | None = None) -> bool:
        estimate = estimated_next_operation_seconds
        if estimate is None:
            estimate = self._estimated_operation_seconds or 0.0
        if estimate < 0:
            raise ValueError("estimated_next_operation_seconds must be non-negative")
        return self.remaining_safe_seconds <= estimate

    def checkpoint_and_stop(self, *, training_state: Mapping[str, Any] | None = None) -> None:
        """Checkpoint exactly once, flush logs, and raise a clean stop signal."""
        self._stop_requested = True
        guard_state = self.state().to_dict()
        payload: dict[str, Any] = {
            "time_guard": guard_state,
            "training_state": dict(training_state or {}),
        }
        if not self._checkpoint_completed:
            self.logger.warning(
                "time_guard_stop",
                extra={"event": "time_guard_stop", **guard_state},
            )
            if self.checkpoint_callback is not None:
                self.checkpoint_callback(payload)
            self._checkpoint_completed = True
            logging.shutdown()
        raise TimeLimitReached(
            "Wall-clock safety boundary reached after checkpoint; exit the training process cleanly"
        )

    def check(
        self,
        *,
        training_state: Mapping[str, Any] | None = None,
        estimated_next_operation_seconds: float | None = None,
    ) -> None:
        """Checkpoint and stop when another safe operation should not begin."""
        if self.should_stop(estimated_next_operation_seconds=estimated_next_operation_seconds):
            self.checkpoint_and_stop(training_state=training_state)

    def run_safe_operation(
        self,
        operation: Callable[[], T],
        *,
        state_provider: Callable[[], Mapping[str, Any]] | None = None,
        estimated_seconds: float | None = None,
    ) -> T:
        """Run one atomic operation, then enforce the safety boundary."""
        state_before = state_provider() if state_provider is not None else None
        self.check(
            training_state=state_before,
            estimated_next_operation_seconds=estimated_seconds,
        )
        operation_started = self._clock()
        result = operation()
        self.record_operation(max(0.0, self._clock() - operation_started))
        state_after = state_provider() if state_provider is not None else None
        self.check(training_state=state_after)
        return result
