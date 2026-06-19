"""Characterization tests for POST /track, GET /job/{id}, and GET /health."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from service import guardrails
from service import routes as routes_mod
from service.routes import state as route_state
from service.analysis_keyspaces_enums import JobState
from service.models import TrackRequest


@pytest.fixture(autouse=True)
def _reset_guardrail_state():
    """Each test starts with a clean G7 daily counter and empty active-task map
    (both are module-global and would otherwise leak across tests)."""
    guardrails.reset_daily_counter()
    route_state._active_tasks.clear()
    yield
    guardrails.reset_daily_counter()
    route_state._active_tasks.clear()


@pytest.fixture()
def scheduled_jobs(monkeypatch):
    scheduled: list[tuple[str, TrackRequest]] = []

    def _schedule(job_id: str, request: TrackRequest) -> None:
        scheduled.append((job_id, request))

    async def _noop_cleanup() -> None:
        # create_track_job awaits _cleanup_orphaned_tasks(); the stub must be a
        # coroutine function so the await is valid.
        return None

    monkeypatch.setattr(routes_mod, "_schedule_job", _schedule)
    monkeypatch.setattr(routes_mod, "_cleanup_orphaned_tasks", _noop_cleanup)
    return scheduled


@pytest.mark.asyncio
async def test_post_track_creates_lifecycle_and_schedules_worker(
    service_client, service_components, scheduled_jobs,
):
    _, _, jobs_store = service_components
    payload = {
        "bucket": "test-bucket",
        "key": "videos/match.mp4",
        "video_id": "vid-123",
        "user_id": "user-456",
    }

    resp = await service_client.post("/track", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    job_id = body["job_id"]
    assert job_id

    lifecycle = await jobs_store.get_lifecycle(job_id)
    assert lifecycle is not None
    assert lifecycle["video_id"] == "vid-123"
    assert lifecycle["job_state"] == "PENDING"

    request_json = await jobs_store.get_request(job_id)
    assert request_json is not None
    assert "videos/match.mp4" in request_json

    assert jobs_store._latest["vid-123"]["job_id"] == job_id
    assert len(scheduled_jobs) == 1
    assert scheduled_jobs[0][0] == job_id


@pytest.mark.asyncio
async def test_get_job_reads_keyspaces_lifecycle(service_client, service_components):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-abc", "vid", "user", progress_percent=42.5)
    await jobs_store.set_state("job-abc", JobState.RUNNING)

    resp = await service_client.get("/job/job-abc")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-abc"
    assert body["status"] == "running"
    assert body["progress_percent"] == 42.5


@pytest.mark.asyncio
async def test_get_health_returns_ok(service_client):
    resp = await service_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_post_track_returns_429_when_at_capacity(
    service_client, service_components, scheduled_jobs, monkeypatch,
):
    """G4: with capacity = max_concurrent(1) + max_queued(1) = 2, a 3rd in-flight
    job (running + queued) is rejected with 429 (engine busy)."""
    config, _, _ = service_components
    assert config.max_concurrent_jobs + config.max_queued_jobs == 2

    # Simulate two jobs already scheduled (one on the GPU, one queued).
    monkeypatch.setitem(route_state._active_tasks, "running-job", MagicMock())
    monkeypatch.setitem(route_state._active_tasks, "queued-job", MagicMock())

    resp = await service_client.post(
        "/track",
        json={"bucket": "b", "key": "v.mp4"},
    )
    assert resp.status_code == 429
    assert "busy" in resp.json()["detail"].lower()
    # The rejected request must NOT have been scheduled.
    assert scheduled_jobs == []


@pytest.mark.asyncio
async def test_post_track_queues_second_job_within_capacity(
    service_client, service_components, scheduled_jobs, monkeypatch,
):
    """G4: with one job already on the GPU, a second NEW job is admitted (queued)
    rather than rejected — bounded queue depth of 1."""
    monkeypatch.setitem(route_state._active_tasks, "running-job", MagicMock())

    resp = await service_client.post("/track", json={"bucket": "b", "key": "v.mp4"})
    assert resp.status_code == 200
    assert len(scheduled_jobs) == 1


@pytest.mark.asyncio
async def test_post_track_503_when_analysis_disabled(
    service_client, service_components, scheduled_jobs, monkeypatch,
):
    """G7 kill switch: BJJ_ANALYSIS_DISABLED=true → 503, nothing scheduled."""
    config, _, _ = service_components
    monkeypatch.setattr(config, "analysis_disabled", True)

    resp = await service_client.post("/track", json={"bucket": "b", "key": "v.mp4"})
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"].lower()
    assert scheduled_jobs == []


@pytest.mark.asyncio
async def test_post_track_429_over_daily_cap(
    service_client, service_components, scheduled_jobs, monkeypatch,
):
    """G7 daily cap: the (cap+1)-th NEW analysis of the UTC day → 429."""
    config, _, _ = service_components
    monkeypatch.setattr(config, "max_daily_analyses", 2)

    for _ in range(2):
        ok = await service_client.post("/track", json={"bucket": "b", "key": "v.mp4"})
        assert ok.status_code == 200

    resp = await service_client.post("/track", json={"bucket": "b", "key": "v.mp4"})
    assert resp.status_code == 429
    assert "daily" in resp.json()["detail"].lower()
    assert len(scheduled_jobs) == 2  # only the two under-cap jobs scheduled


@pytest.mark.asyncio
async def test_daily_cap_zero_blocks_all(
    service_client, service_components, scheduled_jobs, monkeypatch,
):
    """G7: max_daily_analyses=0 blocks every new analysis immediately."""
    config, _, _ = service_components
    monkeypatch.setattr(config, "max_daily_analyses", 0)

    resp = await service_client.post("/track", json={"bucket": "b", "key": "v.mp4"})
    assert resp.status_code == 429
    assert scheduled_jobs == []
