"""Unit tests for GET /jobs/active — the release GPU-guard endpoint.

Verifies:
1. Empty list when no jobs are in the store.
2. Non-terminal jobs (PENDING, RUNNING, AWAITING_CORRECTION, INTERRUPTED) appear.
3. Terminal jobs (COMPLETED, FAILED, CANCELLED) are excluded.
4. Multiple non-terminal jobs are all returned.
5. The endpoint is read-only — no state mutation after the call.
6. Response schema: HTTP 200, body has .active list with id/state/started_at.
"""
from __future__ import annotations

import pytest

from service.models import JobStatus


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
# 2. PENDING job appears in active list
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
    # status from in-memory store is lowercase "pending"
    assert entry["state"].upper() in ("PENDING", "RUNNING", "AWAITING_CORRECTION", "INTERRUPTED")
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
    for terminal_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
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
# 4. Multiple non-terminal jobs are all returned
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
# 5. Endpoint is read-only — calling it does not mutate job state
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
# 6. Schema: response has "active" key, each entry has job_id/state/started_at
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
