"""Tests for Keyspaces job store persistence semantics."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from service.analysis_keyspaces_enums import JobState
from service.jobs_store import JobsStore


class FakeKeyspacesClient:
    keyspace = "video_analysis"

    def __init__(self, write_results, rows=None):
        self.write_results = list(write_results)
        self.rows = rows or []

    async def execute_write(self, query, params=None):
        return self.write_results.pop(0)

    async def execute(self, query, params=None):
        return self.rows


def _lifecycle_row(job_state=JobState.RUNNING.value):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        job_id="job-id",
        video_id="video-id",
        user_id="user-id",
        origin_job_id="",
        parent_job_id="",
        replacement_job_id="",
        job_state=job_state,
        stage="",
        progress_percent=0.0,
        current_frame=0,
        total_frames=0,
        stage_message="",
        error_message="",
        owner_instance_id="worker",
        last_heartbeat_at=now,
        started_at=now,
        updated_at=now,
        completed_at=None,
    )


@pytest.mark.asyncio
async def test_create_lifecycle_propagates_recovery_index_failure():
    store = JobsStore(FakeKeyspacesClient([True, False]))

    ok = await store.create_lifecycle("job-id", "video-id", "user-id")

    assert ok is False


@pytest.mark.asyncio
async def test_heartbeat_propagates_recovery_index_failure():
    store = JobsStore(FakeKeyspacesClient([True, False], [_lifecycle_row()]))

    ok = await store.heartbeat("job-id", "worker")

    assert ok is False
