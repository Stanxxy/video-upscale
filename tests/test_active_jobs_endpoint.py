"""Unit tests for GET /jobs/active — the release GPU-guard endpoint.

Verifies:
1. Empty list when no jobs are in the store.
2. Every non-terminal JobStatus (PENDING, DOWNLOADING, DETECTING,
   WAITING_FOR_DETECTION, TRACKING, UPSCALING, UPLOADING, PUBLISHING,
   AWAITING_CORRECTION, INTERRUPTED) appears in /jobs/active.
3. Terminal jobs (COMPLETED, FAILED, CANCELLED) are excluded.
4. Multiple non-terminal jobs are all returned.
5. The endpoint is read-only — no state mutation after the call.
6. Response schema: HTTP 200, body has .active list with id/state/started_at.
7. REGRESSION (C1): the active set == complement of the terminal set.
   TRACKING and all GPU-working states must be reported active, not silently
   dropped by a hardcoded non-terminal allowlist that drifts from the enum.
"""
from __future__ import annotations

import pytest

from service.models import JobStatus
from service.routes.active_jobs import _ACTIVE_STATES, _TERMINAL_STATES


# ---------------------------------------------------------------------------
# Derived sets used throughout (single source of truth)
# ---------------------------------------------------------------------------
_TERMINAL: frozenset[JobStatus] = frozenset({
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
})

_NON_TERMINAL: frozenset[JobStatus] = frozenset(JobStatus) - _TERMINAL


# ---------------------------------------------------------------------------
# Shared stub helpers
# ---------------------------------------------------------------------------

def _patch_track_deps(monkeypatch):
    """Monkeypatch POST /track side effects so tests don't launch real tasks.

    _cleanup_orphaned_tasks is awaited in create_track_job (it is an async
    function), so the stub must be a coroutine, not a plain lambda.
    _schedule_job is called without await, so a plain function is fine.
    """
    from service import routes as routes_mod

    async def _noop_cleanup():
        pass

    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, req: None)
    monkeypatch.setattr(routes_mod, "_cleanup_orphaned_tasks", _noop_cleanup)


# ---------------------------------------------------------------------------
# 1. Empty store → empty list
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_active_jobs_empty_store(service_client):
    resp = await service_client.get("/jobs/active")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] == []


# ---------------------------------------------------------------------------
# 2. PENDING job appears in active list (via POST /track)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_active_jobs_pending_appears(service_client, service_components, monkeypatch):
    _, job_store, _ = service_components
    _patch_track_deps(monkeypatch)

    payload = {"bucket": "b", "key": "k.mp4"}
    create_resp = await service_client.post("/track", json=payload)
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]

    resp = await service_client.get("/jobs/active")
    assert resp.status_code == 200
    active = resp.json()["active"]
    ids = [e["job_id"] for e in active]
    assert job_id in ids
    entry = next(e for e in active if e["job_id"] == job_id)
    assert entry["state"] == JobStatus.PENDING.value
    assert entry["started_at"] is not None


# ---------------------------------------------------------------------------
# 3. Terminal states are excluded
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_active_jobs_terminal_excluded(service_client, service_components, monkeypatch):
    """Jobs in COMPLETED / FAILED / CANCELLED do NOT appear in /jobs/active."""
    _, job_store, _ = service_components
    _patch_track_deps(monkeypatch)

    payload = {"bucket": "b", "key": "k.mp4"}

    # Create three jobs and immediately flip them to terminal states.
    job_ids_terminal = []
    for terminal_status in _TERMINAL:
        create_resp = await service_client.post("/track", json=payload)
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]
        await job_store.update_job(job_id, status=terminal_status)
        job_ids_terminal.append(job_id)

    resp = await service_client.get("/jobs/active")
    assert resp.status_code == 200
    active_ids = {e["job_id"] for e in resp.json()["active"]}
    for job_id in job_ids_terminal:
        assert job_id not in active_ids, (
            f"Terminal job {job_id} must not appear in /jobs/active"
        )


# ---------------------------------------------------------------------------
# 4. Every non-terminal JobStatus is reported active
#
# C1 REGRESSION: Previously _NON_TERMINAL only checked
# {PENDING, RUNNING, AWAITING_CORRECTION, INTERRUPTED} — missing
# DOWNLOADING, DETECTING, WAITING_FOR_DETECTION, TRACKING, UPSCALING,
# UPLOADING, PUBLISHING.  A TRACKING job returned active=[] and the
# GPU-guard reported the box safe, allowing engine restart mid-GPU-job.
# This test asserts ALL non-terminal states are reported active.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_every_non_terminal_status_is_active(service_client, service_components):
    """Every non-terminal JobStatus must appear in /jobs/active.

    This is the C1 regression: the implementation must not use a hardcoded
    allowlist of "working" states.  Instead it uses the complement of the
    terminal set, so each state below is guaranteed to be covered even if
    new states are added to JobStatus later.
    """
    _, job_store, _ = service_components
    from service.models import TrackRequest

    # Create one job per non-terminal status, set it directly in the store.
    status_to_job_id: dict[JobStatus, str] = {}
    for status in _NON_TERMINAL:
        request = TrackRequest(bucket="b", key="k.mp4")
        job = await job_store.create_job(request)
        await job_store.update_job(job.job_id, status=status)
        status_to_job_id[status] = job.job_id

    resp = await service_client.get("/jobs/active")
    assert resp.status_code == 200
    active_ids = {e["job_id"] for e in resp.json()["active"]}

    for status, job_id in status_to_job_id.items():
        assert job_id in active_ids, (
            f"Non-terminal status {status.value!r} (job {job_id}) must appear "
            f"in /jobs/active — C1 regression: was silently dropped by "
            f"hardcoded allowlist"
        )


# ---------------------------------------------------------------------------
# 4b. Explicit spot-check: TRACKING, UPSCALING, DETECTING all active
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gpu_working_states_are_active(service_client, service_components):
    """TRACKING, UPSCALING, DETECTING must all appear in /jobs/active.

    These are the GPU-consuming states.  If any of them is silently excluded
    the GPU-guard gives a false 'safe' signal → catastrophic mid-job restart.
    """
    _, job_store, _ = service_components
    from service.models import TrackRequest

    gpu_states = [
        JobStatus.TRACKING,
        JobStatus.UPSCALING,
        JobStatus.DETECTING,
        JobStatus.DOWNLOADING,
        JobStatus.WAITING_FOR_DETECTION,
        JobStatus.UPLOADING,
        JobStatus.PUBLISHING,
    ]

    status_to_job_id: dict[JobStatus, str] = {}
    for status in gpu_states:
        request = TrackRequest(bucket="b", key="k.mp4")
        job = await job_store.create_job(request)
        await job_store.update_job(job.job_id, status=status)
        status_to_job_id[status] = job.job_id

    resp = await service_client.get("/jobs/active")
    assert resp.status_code == 200
    active_ids = {e["job_id"] for e in resp.json()["active"]}

    for status, job_id in status_to_job_id.items():
        assert job_id in active_ids, (
            f"GPU-working status {status.value!r} must appear in /jobs/active"
        )


# ---------------------------------------------------------------------------
# 5. REGRESSION (complement invariant): active_set == all_states - terminal
#
# This test asserts the implementation's module-level constants are wired
# correctly: _ACTIVE_STATES must equal the set-theoretic complement of
# _TERMINAL_STATES over all JobStatus values.
# ---------------------------------------------------------------------------
def test_active_states_equals_complement_of_terminal():
    """_ACTIVE_STATES must be exactly the complement of _TERMINAL_STATES.

    Implementation must derive active = all_states - terminal_states so new
    states are active-by-default.  A hardcoded working-state list that drifts
    would cause this test to fail immediately when the enum grows.
    """
    all_values = frozenset(s.value for s in JobStatus)
    expected_active = all_values - _TERMINAL_STATES
    assert _ACTIVE_STATES == expected_active, (
        f"_ACTIVE_STATES does not equal complement of _TERMINAL_STATES.\n"
        f"  Extra (in _ACTIVE_STATES but not in complement): {_ACTIVE_STATES - expected_active}\n"
        f"  Missing (in complement but not in _ACTIVE_STATES): {expected_active - _ACTIVE_STATES}"
    )

    # Sanity: terminal states must not appear in active states
    assert _ACTIVE_STATES.isdisjoint(_TERMINAL_STATES), (
        f"Terminal states leaked into _ACTIVE_STATES: "
        f"{_ACTIVE_STATES & _TERMINAL_STATES}"
    )

    # Verify specific GPU states are in the active set
    for gpu_status in (
        JobStatus.TRACKING,
        JobStatus.UPSCALING,
        JobStatus.DETECTING,
        JobStatus.DOWNLOADING,
        JobStatus.WAITING_FOR_DETECTION,
        JobStatus.UPLOADING,
        JobStatus.PUBLISHING,
        JobStatus.AWAITING_CORRECTION,
        JobStatus.INTERRUPTED,
        JobStatus.PENDING,
    ):
        assert gpu_status.value in _ACTIVE_STATES, (
            f"{gpu_status.value!r} must be in _ACTIVE_STATES (fail-safe)"
        )


# ---------------------------------------------------------------------------
# 6. Multiple non-terminal jobs are all returned
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_active_jobs_multiple_non_terminal(service_client, service_components, monkeypatch):
    """Create jobs directly in the store (bypassing the semaphore) so multiple
    non-terminal states can coexist and all appear in /jobs/active."""
    _, job_store, _ = service_components
    from service.models import TrackRequest

    created_ids = set()
    for non_terminal in (
        JobStatus.PENDING,
        JobStatus.AWAITING_CORRECTION,
        JobStatus.INTERRUPTED,
        JobStatus.TRACKING,
        JobStatus.UPSCALING,
    ):
        request = TrackRequest(bucket="b", key="k.mp4")
        job = await job_store.create_job(request)
        await job_store.update_job(job.job_id, status=non_terminal)
        created_ids.add(job.job_id)

    resp = await service_client.get("/jobs/active")
    assert resp.status_code == 200
    active_ids = {e["job_id"] for e in resp.json()["active"]}

    for job_id in created_ids:
        assert job_id in active_ids, (
            f"Non-terminal job {job_id} must appear in /jobs/active"
        )


# ---------------------------------------------------------------------------
# 7. Endpoint is read-only — calling it does not mutate job state
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_active_jobs_is_read_only(service_client, service_components, monkeypatch):
    _, job_store, _ = service_components
    _patch_track_deps(monkeypatch)

    payload = {"bucket": "b", "key": "k.mp4"}
    create_resp = await service_client.post("/track", json=payload)
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]

    state_before = (await job_store.get_job(job_id)).status

    # Call GET /jobs/active once (and again)
    await service_client.get("/jobs/active")
    await service_client.get("/jobs/active")

    state_after = (await job_store.get_job(job_id)).status
    assert state_before == state_after, "GET /jobs/active must not mutate job state"


# ---------------------------------------------------------------------------
# 8. Schema: response has "active" key, each entry has job_id/state/started_at
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_active_jobs_response_schema(service_client, service_components, monkeypatch):
    _, job_store, _ = service_components
    _patch_track_deps(monkeypatch)

    payload = {"bucket": "b", "key": "k.mp4"}
    create_resp = await service_client.post("/track", json=payload)
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]

    resp = await service_client.get("/jobs/active")
    assert resp.status_code == 200
    body = resp.json()
    assert "active" in body
    assert isinstance(body["active"], list)
    entry = next((e for e in body["active"] if e["job_id"] == job_id), None)
    assert entry is not None
    assert "job_id" in entry
    assert "state" in entry
    assert "started_at" in entry
