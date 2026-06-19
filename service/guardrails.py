"""Cost / abuse guardrails for the single-GPU vision engine.

These guardrails protect a SINGLE NVIDIA GB10 GPU that runs the
YOLO -> SAM2 -> upscale -> Gemini pipeline behind open public signup, where
the engine is both a cost driver (Gemini calls) and a DoS bottleneck (one job
pins the GPU; concurrent jobs starve each other or OOM).

Two gates live here:

* G4 — GPU concurrency admission. The actual one-at-a-time serialization is
  enforced by ``service.routes.state._job_semaphore`` (acquired around the
  GPU-bound ``run_job`` in ``service.routes.scheduling._run_with_semaphore``).
  This module only decides *admission*: whether a new request is allowed to
  enter the bounded queue or must be rejected with 429.

* G7 — daily analysis cap + kill switch. A per-UTC-day in-memory counter of
  NEW analyses, plus a hard ``analysis_disabled`` flag that returns 503.

LIMITATION (intentional, documented): the daily counter is in-memory and resets
on process restart. That is acceptable for the initial public launch; a durable
per-day counter (e.g. a Keyspaces row keyed by UTC date) is the follow-up.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from fastapi import HTTPException

from service.config import ServiceConfig


# --- G7 daily-cap counter (in-memory, per UTC day) ---------------------------

_lock = threading.Lock()
_daily_count: dict[str, int] = {}


def _utc_day_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def daily_count(day_key: str | None = None) -> int:
    """Return the number of NEW analyses started on the given UTC day."""
    key = day_key or _utc_day_key()
    with _lock:
        return _daily_count.get(key, 0)


def reset_daily_counter() -> None:
    """Test helper: clear the in-memory daily counter."""
    with _lock:
        _daily_count.clear()


def _increment_daily() -> int:
    key = _utc_day_key()
    with _lock:
        # Drop stale days so the dict cannot grow unbounded across a long uptime.
        for stale in [k for k in _daily_count if k != key]:
            del _daily_count[stale]
        _daily_count[key] = _daily_count.get(key, 0) + 1
        return _daily_count[key]


# --- Gate checks -------------------------------------------------------------


def check_kill_switch(config: ServiceConfig) -> None:
    """G7 kill switch. Raise 503 when analysis is disabled.

    Applied to BOTH new-analysis and resume entrypoints: the panic button must
    stop ALL GPU/Gemini spend, including in-flight corrections.
    """
    if config.analysis_disabled:
        raise HTTPException(
            503,
            "Analysis is temporarily disabled by the operator. Try again later.",
        )


def admit_new_analysis(config: ServiceConfig, in_flight: int) -> None:
    """Admission check for a NEW analysis job (G4 queue cap + G7 daily cap).

    ``in_flight`` is the count of analysis jobs already scheduled (running on the
    GPU + queued waiting for the semaphore). Call this AFTER ``check_kill_switch``
    and BEFORE scheduling the job. On success it increments the daily counter.

    Order of checks: kill switch (caller) -> capacity (429) -> daily cap (429).
    Capacity is checked before the daily cap so a busy engine fails fast with the
    most actionable message.
    """
    capacity = config.max_concurrent_jobs + config.max_queued_jobs
    if in_flight >= capacity:
        raise HTTPException(
            429,
            (
                f"Engine busy: {in_flight} analysis job(s) running or queued "
                f"(capacity {capacity}). Try again later."
            ),
        )

    if daily_count() >= config.max_daily_analyses:
        raise HTTPException(
            429,
            (
                f"Daily analysis limit reached ({config.max_daily_analyses} per UTC "
                "day). Try again tomorrow."
            ),
        )

    _increment_daily()
