"""Tests for DELETE /job/{job_id} cancellation semantics.

Shared fixtures live in ``tests/conftest.py``.
"""

import pytest

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.models import JobStatus, TrackRequest


@pytest.mark.asyncio
async def test_cancel_active_in_memory_job(service_client, service_components):
    _, job_store, jobs_store = service_components
    request = TrackRequest(bucket="test-bucket", key="video.mp4")
    job = await job_store.create_job(request)
    await jobs_store.create_lifecycle(job.job_id, "video-id", "user-id")

    resp = await service_client.delete(f"/job/{job.job_id}")

    assert resp.status_code == 200
    assert resp.json() == {"status": "cancelled", "job_id": job.job_id}
    in_memory = await job_store.get_job(job.job_id)
    assert in_memory.status == JobStatus.CANCELLED
    lifecycle = await jobs_store.get_lifecycle(job.job_id)
    assert lifecycle["job_state"] == JobState.CANCELLED.value
    checkpoint = jobs_store._checkpoints[(job.job_id, PipelineStage.TRACK.value)]
    assert checkpoint["checkpoint_data"]["reason"] == "user_cancelled"
    assert checkpoint["checkpoint_data"]["resume_cursor"]["frame_idx"] == 0


@pytest.mark.asyncio
async def test_cancel_keyspaces_only_job(service_client, service_components):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("keyspaces-only-job", "video-id", "user-id")

    resp = await service_client.delete("/job/keyspaces-only-job")

    assert resp.status_code == 200
    lifecycle = await jobs_store.get_lifecycle("keyspaces-only-job")
    assert lifecycle["job_state"] == JobState.CANCELLED.value
    checkpoint = jobs_store._checkpoints[
        ("keyspaces-only-job", PipelineStage.TRACK.value)
    ]
    assert checkpoint["checkpoint_data"]["reason"] == "user_cancelled"


@pytest.mark.asyncio
async def test_cancel_replaced_job_returns_conflict(
    service_client, service_components,
):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle(
        "old-job",
        "video-id",
        "user-id",
        replacement_job_id="new-job",
    )

    resp = await service_client.delete("/job/old-job")

    assert resp.status_code == 409
    assert "replacement" in resp.json()["detail"]
