"""GET /jobs/active — list jobs in non-terminal states.

The release pipeline GPU-guard calls this endpoint before any deploy or
rollback to confirm no GPU job is in flight.  An empty list means it is
safe to restart the engine.

Non-terminal states (matches JobState enum + JobStatus.PENDING/RUNNING/
AWAITING_CORRECTION/INTERRUPTED):
  PENDING, RUNNING, AWAITING_CORRECTION, INTERRUPTED

Read-only.  No state mutation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from service.analysis_keyspaces_enums import JobState
from service.routes import state as route_state

# The JobStatus string enum in models.py uses lowercase values; the Keyspaces
# JobState enum (analysis_keyspaces_enums) uses UPPERCASE.  The in-memory
# _job_store stores JobStatus (lowercase) while Keyspaces (via _jobs_store)
# uses JobState (UPPERCASE).  We normalise to UPPERCASE for the comparison.

_NON_TERMINAL: frozenset[str] = frozenset(
    {
        JobState.PENDING.upper(),       # "PENDING"
        JobState.RUNNING.upper(),       # "RUNNING"
        JobState.AWAITING_CORRECTION.upper(),  # "AWAITING_CORRECTION"
        JobState.INTERRUPTED.upper(),   # "INTERRUPTED"
    }
)

# JobStatus (models.py) also uses lowercase values — add them so we can match
# either representation.
_NON_TERMINAL_LOWER: frozenset[str] = frozenset(s.lower() for s in _NON_TERMINAL)


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
    - State is sourced from _job_store (in-memory, per-process).
    """
    if route_state._job_store is None:
        return ActiveJobsResponse(active=[])

    # Snapshot the in-memory job dictionary under the async lock.
    async with route_state._job_store._lock:
        snapshot = dict(route_state._job_store._jobs)

    active: list[ActiveJobEntry] = []
    for job_id, job in snapshot.items():
        # Use .value to get the raw string (e.g. "pending"), not the
        # full enum repr "JobStatus.PENDING".
        raw = job.status.value if hasattr(job.status, "value") else str(job.status)
        status_upper = raw.upper()
        status_lower = raw.lower()
        if status_upper in _NON_TERMINAL or status_lower in _NON_TERMINAL_LOWER:
            active.append(
                ActiveJobEntry(
                    job_id=job_id,
                    state=raw,
                    started_at=job.created_at,
                )
            )

    return ActiveJobsResponse(active=active)
