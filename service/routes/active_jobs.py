"""GET /jobs/active — list jobs in non-terminal states.

The release pipeline GPU-guard calls this endpoint before any deploy or
rollback to confirm no GPU job is in flight.  An empty list means it is
safe to restart the engine.

Terminal states (JobStatus values that represent a finished/stopped job):
  COMPLETED, FAILED, CANCELLED

Every other JobStatus value (all working/in-flight states) is considered
active.  The set is derived as the COMPLEMENT of the terminal set so that
any new state added to JobStatus is active-by-default (fail-safe).  We do
NOT hardcode a working-state list — that approach silently becomes stale
every time a new state is added.

Read-only.  No state mutation.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from service.models import JobStatus
from service.routes import state as route_state

# ---------------------------------------------------------------------------
# Terminal states — the only states where a job is definitively done.
# Everything outside this set is considered active (fail-safe default).
# ---------------------------------------------------------------------------
_TERMINAL_STATES: frozenset[str] = frozenset(
    s.value for s in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    )
)

# The complement — computed once at import time for O(1) membership tests and
# so tests can assert the active set == all_states - terminal_states.
_ALL_JOB_STATUS_VALUES: frozenset[str] = frozenset(s.value for s in JobStatus)
_ACTIVE_STATES: frozenset[str] = _ALL_JOB_STATUS_VALUES - _TERMINAL_STATES


class ActiveJobEntry(BaseModel):
    """One active (non-terminal) job."""
    job_id: str
    state: str
    started_at: Optional[str] = None


class ActiveJobsResponse(BaseModel):
    """Response body for GET /jobs/active."""
    active: list[ActiveJobEntry]


async def get_active_jobs() -> ActiveJobsResponse:
    """Return jobs currently in a non-terminal state.

    Checks the in-memory job store only — the canonical source for
    jobs this process started.  Keyspaces is the durable source for
    cross-restart queries, but the GPU-guard only needs to know about
    jobs that are **right now consuming the semaphore/GPU** in this
    process, which is exactly what _active_tasks tracks.

    Design notes:
    - Read-only: no writes, no side effects.
    - Fail-closed contract: if _job_store is None (uninitialised process),
      we return an empty list (safe: no jobs possible without a store).
    - Active = complement of {COMPLETED, FAILED, CANCELLED} over the full
      JobStatus enum.  Any unknown/new status value is treated as active
      (fail-safe: unknown state should block a restart, not allow it).
    """
    if route_state._job_store is None:
        return ActiveJobsResponse(active=[])

    # Snapshot the in-memory job dictionary under the async lock.
    async with route_state._job_store._lock:
        snapshot = dict(route_state._job_store._jobs)

    active: list[ActiveJobEntry] = []
    for job_id, job in snapshot.items():
        # Use .value to get the raw string (e.g. "tracking"), not the
        # full enum repr "JobStatus.TRACKING".
        raw = job.status.value if hasattr(job.status, "value") else str(job.status)
        # A job is active unless its status is in the explicit terminal set.
        # Unknown/unrecognised status values are treated as active (fail-safe).
        if raw not in _TERMINAL_STATES:
            active.append(
                ActiveJobEntry(
                    job_id=job_id,
                    state=raw,
                    started_at=job.created_at,
                )
            )

    return ActiveJobsResponse(active=active)
