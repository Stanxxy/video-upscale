"""Tests for POST /jobs/{job_id}/resume and /jobs/{job_id}/detection_response.

These tests validate the Keyspaces-backed suspend/resume flow:
  - POST /jobs/{id}/resume delegates to detection_response
  - POST /jobs/{id}/detection_response creates a new job from checkpoint

Shared fixtures live in ``tests/conftest.py``:
  - ``mock_jobs_store`` — fresh in-memory mock per test.
  - ``service_components`` — ``(config, job_store, jobs_store)`` triple.
  - ``service_client`` — ASGI client wired to the routes.
"""

import json

import pytest
import pytest_asyncio

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.models import TrackRequest


@pytest_asyncio.fixture()
async def awaiting_job(service_components):
    """Create a job in AWAITING_CORRECTION state with checkpoint data."""
    _, job_store, jobs_store = service_components
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
async def test_detection_response_creates_resume_job(
    service_client, awaiting_job, service_components,
):
    """POST /jobs/{id}/detection_response creates a new job and returns its ID."""
    _, _, jobs_store = service_components
    job_id = awaiting_job

    resp = await service_client.post(
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
    assert new_lifecycle["parent_job_id"] == job_id
    assert jobs_store._latest["test-video-id"]["job_id"] == new_job_id

    # Old job should be CANCELLED
    old_lifecycle = await jobs_store.get_lifecycle(job_id)
    assert old_lifecycle["job_state"] == "CANCELLED"
    assert old_lifecycle["replacement_job_id"] == new_job_id


@pytest.mark.asyncio
async def test_detection_response_rejects_duplicate_resume(
    service_client, awaiting_job, service_components,
):
    """409 when an awaiting job already has a replacement job."""
    _, _, jobs_store = service_components
    await jobs_store.set_replacement(awaiting_job, "replacement-job-id")

    resp = await service_client.post(
        f"/jobs/{awaiting_job}/detection_response",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )

    assert resp.status_code == 409
    assert "already has replacement" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_detection_response_rejects_lost_replacement_claim(
    service_client, awaiting_job, service_components,
):
    """409 when another worker wins the replacement claim race."""
    _, _, jobs_store = service_components
    jobs_store._claim_replacement_result = False

    resp = await service_client.post(
        f"/jobs/{awaiting_job}/detection_response",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )

    assert resp.status_code == 409
    assert "replacement claim" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_detection_response_uses_track_resume_cursor(
    service_client, awaiting_job, service_components,
):
    """Mid-track correction resumes from the next unprocessed frame."""
    _, _, jobs_store = service_components
    await jobs_store.write_checkpoint(awaiting_job, PipelineStage.TRACK, False, {
        "frame_count": 50,
        "resume_cursor": {"frame_idx": 51},
        "partial_tracking_s3_key": "checkpoints/original/partial_tracking.json",
        "pending_detection": {
            "frame_idx": 51,
            "candidates": [{"candidate_id": 0, "box": [10, 20, 100, 200]}],
            "reason": "identity_lost",
        },
    })

    resp = await service_client.post(
        f"/jobs/{awaiting_job}/resume",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )

    assert resp.status_code == 200
    new_job_id = resp.json()["job_id"]
    request_json = await jobs_store.get_request(new_job_id)
    resumed_request = json.loads(request_json)
    assert resumed_request["resume_from_job_id"] == awaiting_job
    assert resumed_request["resume_tracking_s3_key"] == (
        "checkpoints/original/partial_tracking.json"
    )
    assert resumed_request["resume_from_frame"] == 51


@pytest.mark.asyncio
async def test_resume_delegates_to_detection_response(service_client, awaiting_job):
    """POST /jobs/{id}/resume delegates to detection_response logic."""
    job_id = awaiting_job

    resp = await service_client.post(
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
async def test_detection_response_not_found(service_client):
    """404 when job_id does not exist in Keyspaces."""
    resp = await service_client.post(
        "/jobs/nonexistent-id/detection_response",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_detection_response_wrong_state(service_client, service_components):
    """409 when job is not in AWAITING_CORRECTION state."""
    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="k")
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "", "")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    # State is PENDING, not AWAITING_CORRECTION

    resp = await service_client.post(
        f"/jobs/{job.job_id}/detection_response",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 409
