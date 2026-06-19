"""Unit tests for G4/G7 cost-and-abuse guardrails (service.guardrails)."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from service import guardrails
from service.config import ServiceConfig


@pytest.fixture(autouse=True)
def _reset_counter():
    guardrails.reset_daily_counter()
    yield
    guardrails.reset_daily_counter()


def _cfg(**overrides) -> ServiceConfig:
    base = dict(
        max_concurrent_jobs=1,
        max_queued_jobs=1,
        analysis_disabled=False,
        max_daily_analyses=50,
    )
    base.update(overrides)
    return ServiceConfig(**base)


# --- G7 kill switch ---------------------------------------------------------


def test_kill_switch_off_is_noop():
    guardrails.check_kill_switch(_cfg(analysis_disabled=False))  # no raise


def test_kill_switch_on_raises_503():
    with pytest.raises(HTTPException) as ei:
        guardrails.check_kill_switch(_cfg(analysis_disabled=True))
    assert ei.value.status_code == 503


# --- G4 admission (queue cap) ----------------------------------------------


def test_admit_within_capacity_increments_counter():
    cfg = _cfg(max_concurrent_jobs=1, max_queued_jobs=1)
    assert guardrails.daily_count() == 0
    guardrails.admit_new_analysis(cfg, in_flight=0)  # running slot
    guardrails.admit_new_analysis(cfg, in_flight=1)  # queue slot
    assert guardrails.daily_count() == 2


def test_admit_over_capacity_raises_429_and_does_not_count():
    cfg = _cfg(max_concurrent_jobs=1, max_queued_jobs=1)  # capacity = 2
    with pytest.raises(HTTPException) as ei:
        guardrails.admit_new_analysis(cfg, in_flight=2)
    assert ei.value.status_code == 429
    assert "busy" in ei.value.detail.lower()
    assert guardrails.daily_count() == 0  # rejected request must not count


def test_zero_queue_rejects_as_soon_as_gpu_busy():
    cfg = _cfg(max_concurrent_jobs=1, max_queued_jobs=0)  # capacity = 1
    guardrails.admit_new_analysis(cfg, in_flight=0)  # the running job is admitted
    with pytest.raises(HTTPException) as ei:
        guardrails.admit_new_analysis(cfg, in_flight=1)  # no queue → reject
    assert ei.value.status_code == 429


# --- G7 daily cap -----------------------------------------------------------


def test_daily_cap_blocks_after_limit():
    cfg = _cfg(max_concurrent_jobs=99, max_queued_jobs=99, max_daily_analyses=3)
    for _ in range(3):
        guardrails.admit_new_analysis(cfg, in_flight=0)
    with pytest.raises(HTTPException) as ei:
        guardrails.admit_new_analysis(cfg, in_flight=0)
    assert ei.value.status_code == 429
    assert "daily" in ei.value.detail.lower()


def test_daily_cap_zero_blocks_everything():
    cfg = _cfg(max_daily_analyses=0)
    with pytest.raises(HTTPException) as ei:
        guardrails.admit_new_analysis(cfg, in_flight=0)
    assert ei.value.status_code == 429


def test_capacity_checked_before_daily_cap():
    """A busy engine reports 'busy' even when also over the daily cap, so the
    caller gets the most actionable retry signal."""
    cfg = _cfg(max_concurrent_jobs=1, max_queued_jobs=0, max_daily_analyses=0)
    with pytest.raises(HTTPException) as ei:
        guardrails.admit_new_analysis(cfg, in_flight=1)
    assert ei.value.status_code == 429
    assert "busy" in ei.value.detail.lower()


# --- config defaults --------------------------------------------------------


def test_config_defaults_are_single_gpu_safe():
    cfg = ServiceConfig()
    assert cfg.max_concurrent_jobs == 1  # single GB10
    assert cfg.max_queued_jobs == 1
    assert cfg.analysis_disabled is False
    assert cfg.max_daily_analyses == 50


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("BJJ_MAX_CONCURRENT_JOBS", "2")
    monkeypatch.setenv("BJJ_MAX_QUEUED_JOBS", "5")
    monkeypatch.setenv("BJJ_ANALYSIS_DISABLED", "true")
    monkeypatch.setenv("BJJ_MAX_DAILY_ANALYSES", "10")
    cfg = ServiceConfig()
    assert cfg.max_concurrent_jobs == 2
    assert cfg.max_queued_jobs == 5
    assert cfg.analysis_disabled is True
    assert cfg.max_daily_analyses == 10
