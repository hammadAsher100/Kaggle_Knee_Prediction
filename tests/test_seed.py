"""Tests for deterministic random seeds without importing the training stack."""

from __future__ import annotations

import random

import numpy as np

from src.utils.seed import set_global_seed


def test_global_seed_repeats_python_and_numpy_sequences() -> None:
    first_report = set_global_seed(42, include_torch=False)
    first = (random.random(), np.random.random())
    second_report = set_global_seed(42, include_torch=False)
    second = (random.random(), np.random.random())

    assert first == second
    assert first_report == second_report
    assert first_report.python is True
    assert first_report.numpy is True
    assert first_report.torch is False

