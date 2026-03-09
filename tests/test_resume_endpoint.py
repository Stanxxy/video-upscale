"""Tests for POST /jobs/{job_id}/resume endpoint."""

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.models import JobStatus, TrackRequest
from service.routes import init_routes, router
from service.ws_manager import WSManager

from fastapi import FastAPI


@pytest.fixture()
def _components():
    """Create fresh config / job_store / ws_manager for each test."""
    config = ServiceConfig(
        detection_timeout=86400.0,
        s3_endpoint_url="http://localhost:4566",
    )
    job_store = InMemoryJobStore()
    ws_manager = WSManager()
    init_routes(config, job_store, ws_manager)
    return config, job_store, ws_manager


@pytest.fixture()
def app(_components):
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest_asyncio.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture()
async def waiting_job(_components):
    """Create a job that is in WAITING_FOR_DETECTION state."""
    _, job_store, ws_manager = _components
    req = TrackRequest(bucket="test-bucket", key="test/video.mp4")
    job = await job_store.create_job(req)
    # Simulate the pipeline reaching the waiting state
    await job_store.update_job(job.job_id, status=JobStatus.WAITING_FOR_DETECTION)
    # Register a fake detection event so deliver_detection can unblock
    ws_manager._detection_events[job.job_id] = asyncio.Event()
    ws_manager._detection_responses[job.job_id] = None
    return job.job_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_delivers_detection(client, waiting_job, _components):
    """POST /jobs/{id}/resume delivers boxes and returns 200."""
    _, job_store, ws_manager = _components
    job_id = waiting_job

    resp = await client.post(
        f"/jobs/{job_id}/resume",
        json={
            "box_a": [10, 20, 100, 200],
            "box_b": [300, 20, 100, 200],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resumed"
    assert body["job_id"] == job_id

    # The detection event should be set
    assert ws_manager._detection_events[job_id].is_set()

    # The stored response should contain the boxes
    stored = ws_manager._detection_responses[job_id]
    assert stored["box_a"] == [10, 20, 100, 200]
    assert stored["box_b"] == [300, 20, 100, 200]


@pytest.mark.asyncio
async def test_resume_with_player_mapping(client, waiting_job, _components):
    """Player mapping is forwarded in the payload."""
    _, _, ws_manager = _components
    job_id = waiting_job

    resp = await client.post(
        f"/jobs/{job_id}/resume",
        json={
            "box_a": [10, 20, 100, 200],
            "box_b": [300, 20, 100, 200],
            "player_mapping": {"athlete_a": "Gordon Ryan", "athlete_b": "Felipe Pena"},
        },
    )
    assert resp.status_code == 200

    stored = ws_manager._detection_responses[job_id]
    assert stored["player_mapping"]["athlete_a"] == "Gordon Ryan"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_not_found(client):
    """404 when job_id does not exist."""
    resp = await client.post(
        "/jobs/nonexistent-id/resume",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_resume_completed_job(client, _components):
    """409 when job is already completed."""
    _, job_store, _ = _components
    req = TrackRequest(bucket="b", key="k")
    job = await job_store.create_job(req)
    await job_store.update_job(job.job_id, status=JobStatus.COMPLETED)

    resp = await client.post(
        f"/jobs/{job.job_id}/resume",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 409
    assert "terminal" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_resume_failed_job(client, _components):
    """409 when job is failed."""
    _, job_store, _ = _components
    req = TrackRequest(bucket="b", key="k")
    job = await job_store.create_job(req)
    await job_store.update_job(job.job_id, status=JobStatus.FAILED)

    resp = await client.post(
        f"/jobs/{job.job_id}/resume",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_resume_tracking_state(client, _components):
    """409 when job is in tracking (not waiting_for_detection)."""
    _, job_store, _ = _components
    req = TrackRequest(bucket="b", key="k")
    job = await job_store.create_job(req)
    await job_store.update_job(job.job_id, status=JobStatus.TRACKING)

    resp = await client.post(
        f"/jobs/{job.job_id}/resume",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 409
    assert "waiting_for_detection" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_resume_cancelled_job(client, _components):
    """409 when job is cancelled."""
    _, job_store, _ = _components
    req = TrackRequest(bucket="b", key="k")
    job = await job_store.create_job(req)
    await job_store.update_job(job.job_id, status=JobStatus.CANCELLED)

    resp = await client.post(
        f"/jobs/{job.job_id}/resume",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_resume_unblocks_waiter(client, waiting_job, _components):
    """The wait_for_detection coroutine is unblocked by the resume call."""
    _, _, ws_manager = _components
    job_id = waiting_job

    # Start a waiter in background
    waiter = asyncio.create_task(ws_manager.wait_for_detection(job_id, timeout=5.0))

    # Small yield so the waiter task starts
    await asyncio.sleep(0.05)

    # Call resume via REST
    resp = await client.post(
        f"/jobs/{job_id}/resume",
        json={"box_a": [1, 2, 3, 4], "box_b": [5, 6, 7, 8]},
    )
    assert resp.status_code == 200

    # Waiter should now return the detection payload
    result = await asyncio.wait_for(waiter, timeout=2.0)
    assert result is not None
    assert result["box_a"] == [1, 2, 3, 4]
    assert result["box_b"] == [5, 6, 7, 8]
