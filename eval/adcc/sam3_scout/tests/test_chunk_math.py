"""Unit tests for SAM3 scout chunk planning (no CUDA)."""
from __future__ import annotations

import pytest

from eval.adcc.sam3_scout.chunk_math import (
    DEFAULT_CHUNK_SECONDS,
    plan_chunks,
    chunk_threshold_seconds,
)


def test_threshold_is_1_5x_chunk():
    assert chunk_threshold_seconds(30) == 45.0
    assert chunk_threshold_seconds(20) == 30.0


def test_short_clip_not_chunked():
    plan = plan_chunks(40.0, chunk_seconds=30.0)
    assert plan.chunked is False
    assert plan.n_chunks == 1
    assert plan.threshold_seconds == 45.0
    assert plan.concurrency == 1
    assert plan.chunk_boundaries == [(0.0, 40.0)]


def test_exactly_threshold_not_chunked():
    plan = plan_chunks(45.0, chunk_seconds=30.0)
    assert plan.chunked is False
    assert plan.n_chunks == 1


def test_just_over_threshold_chunks():
    plan = plan_chunks(45.1, chunk_seconds=30.0)
    assert plan.chunked is True
    assert plan.n_chunks == 2
    bounds = plan.chunk_boundaries
    assert bounds[0] == (0.0, 30.0)
    assert bounds[1][0] == 30.0
    assert abs(bounds[1][1] - 45.1) < 1e-9


def test_70s_makes_three_chunks():
    plan = plan_chunks(70.0, chunk_seconds=30.0)
    assert plan.chunked is True
    assert plan.n_chunks == 3
    assert len(plan.chunk_boundaries) == 3


def test_concurrency_must_stay_one():
    with pytest.raises(ValueError, match="concurrency must be 1"):
        plan_chunks(10.0, concurrency=2)


def test_default_chunk_seconds():
    plan = plan_chunks(10.0)
    assert plan.chunk_seconds == DEFAULT_CHUNK_SECONDS


def test_negative_duration_raises():
    with pytest.raises(ValueError):
        plan_chunks(-1.0)
