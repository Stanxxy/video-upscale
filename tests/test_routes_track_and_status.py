"""Characterization tests for POST /track, GET /job/{id}, and GET /health."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from service import routes as routes_mod
from service.analysis_keyspaces_enums import JobState
from service.models import TrackRequest


@pytest.fixture()
def scheduled_jobs(monkeypatch):
    scheduled: list[tuple[str, TrackRequest]] = []

    def _schedule(job_id: str, request: TrackRequest) -> None:
        scheduled.append((job_id, request))

    monkeypatch.setattr(routes_mod, "_schedule_job", _schedule)
    monkeypatch.setattr(routes_mod, "_cleanup_orphaned_tasks", lambda: None)
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
    service_client, monkeypatch,
):
    sem = MagicMock()
    sem.locked.return_value = True
    monkeypatch.setattr(routes_mod, "_job_semaphore", sem)

    resp = await service_client.post(
        "/track",
        json={"bucket": "b", "key": "v.mp4"},
    )
    assert resp.status_code == 429
    assert "capacity" in resp.json()["detail"].lower()
