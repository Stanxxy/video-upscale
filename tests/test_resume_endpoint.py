"""Tests for POST /jobs/{job_id}/resume and /jobs/{job_id}/detection_response.

These tests validate the Keyspaces-backed suspend/resume flow:
  - POST /jobs/{id}/resume delegates to detection_response
  - POST /jobs/{id}/detection_response creates a new job from checkpoint
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.models import JobStatus, TrackRequest
from service.routes import init_routes, router

from fastapi import FastAPI


def _make_mock_jobs_store():
    """Create a mock JobsStore that simulates Keyspaces operations in memory."""
    store = MagicMock()
    _lifecycles = {}
    _requests = {}
    _checkpoints = {}
    _latest = {}

    async def create_lifecycle(job_id, video_id="", user_id="", origin_job_id="", owner_instance_id=""):
        _lifecycles[job_id] = {
            "job_id": job_id, "video_id": video_id, "user_id": user_id,
            "origin_job_id": origin_job_id, "job_state": "PENDING",
            "stage": "", "progress_percent": 0.0,
            "current_frame": 0, "total_frames": 0,
            "stage_message": "", "error_message": "",
            "owner_instance_id": owner_instance_id,
        }

    async def get_lifecycle(job_id):
        return _lifecycles.get(job_id)

    async def set_state(job_id, state, error_message=""):
        if job_id in _lifecycles:
            _lifecycles[job_id]["job_state"] = (
                state.value if hasattr(state, "value") else state
            )

    async def save_request(job_id, request_json):
        _requests[job_id] = request_json

    async def get_request(job_id):
        return _requests.get(job_id)

    async def write_checkpoint(job_id, stage_name, completed, data):
        sn = stage_name.value if hasattr(stage_name, "value") else stage_name
        _checkpoints[(job_id, sn)] = {
            "stage_name": sn,
            "completed": completed,
            "checkpoint_data": data,
        }

    async def get_checkpoint(job_id, stage_name):
        sn = stage_name.value if hasattr(stage_name, "value") else stage_name
        return _checkpoints.get((job_id, sn))

    async def get_all_checkpoints(job_id):
        return [v for (jid, _), v in _checkpoints.items() if jid == job_id]

    async def set_latest(video_id, job_id, state):
        st = state.value if hasattr(state, "value") else state
        _latest[video_id] = {"job_id": job_id, "job_state": st}

    async def update_progress(*args, **kwargs):
        pass

    store.create_lifecycle = create_lifecycle
    store.get_lifecycle = get_lifecycle
    store.set_state = set_state
    store.save_request = save_request
    store.get_request = get_request
    store.write_checkpoint = write_checkpoint
    store.get_checkpoint = get_checkpoint
    store.get_all_checkpoints = get_all_checkpoints
    store.set_latest = set_latest
    store.update_progress = update_progress
    store.register_owned_job = MagicMock()
    store.unregister_owned_job = MagicMock()
    store.owned_jobs = set()
    store._lifecycles = _lifecycles
    store._requests = _requests
    store._checkpoints = _checkpoints
    return store


@pytest.fixture()
def _components():
    config = ServiceConfig(
        detection_timeout=86400.0,
        s3_endpoint_url="http://localhost:4566",
    )
    job_store = InMemoryJobStore()
    jobs_store = _make_mock_jobs_store()
    init_routes(config, job_store, jobs_store)
    return config, job_store, jobs_store


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
async def awaiting_job(_components):
    """Create a job in AWAITING_CORRECTION state with checkpoint data."""
    _, job_store, jobs_store = _components
    req = TrackRequest(bucket="test-bucket", key="test/video.mp4")
    job = await job_store.create_job(req)
    job_id = job.job_id

    # Simulate job reaching AWAITING_CORRECTION
    await jobs_store.create_lifecycle(job_id, "test-video-id", "test-user")
    await jobs_store.save_request(job_id, req.model_dump_json())
    await jobs_store.set_state(job_id, JobState.AWAITING_CORRECTION)
    await jobs_store.write_checkpoint(job_id, PipelineStage.DETECT, False, {
        "pending_detection": {
            "frame_idx": 0,
            "candidates": [{"candidate_id": 0, "box": [10, 20, 100, 200]}],
            "reason": "initial",
        }
    })
    return job_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detection_response_creates_resume_job(client, awaiting_job, _components):
    """POST /jobs/{id}/detection_response creates a new job and returns its ID."""
    _, job_store, jobs_store = _components
    job_id = awaiting_job

    resp = await client.post(
        f"/jobs/{job_id}/detection_response",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resumed"
    assert body["origin_job_id"] == job_id
    new_job_id = body["job_id"]
    assert new_job_id != job_id

    # New job should exist in Keyspaces
    new_lifecycle = await jobs_store.get_lifecycle(new_job_id)
    assert new_lifecycle is not None
    assert new_lifecycle["origin_job_id"] == job_id

    # Old job should be CANCELLED
    old_lifecycle = await jobs_store.get_lifecycle(job_id)
    assert old_lifecycle["job_state"] == "CANCELLED"


@pytest.mark.asyncio
async def test_resume_delegates_to_detection_response(client, awaiting_job):
    """POST /jobs/{id}/resume delegates to detection_response logic."""
    job_id = awaiting_job

    resp = await client.post(
        f"/jobs/{job_id}/resume",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resumed"
    assert body["job_id"] != job_id


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detection_response_not_found(client):
    """404 when job_id does not exist in Keyspaces."""
    resp = await client.post(
        "/jobs/nonexistent-id/detection_response",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_detection_response_wrong_state(client, _components):
    """409 when job is not in AWAITING_CORRECTION state."""
    _, job_store, jobs_store = _components
    req = TrackRequest(bucket="b", key="k")
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "", "")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    # State is PENDING, not AWAITING_CORRECTION

    resp = await client.post(
        f"/jobs/{job.job_id}/detection_response",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 409
