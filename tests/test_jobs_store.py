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
        self.write_calls = []

    async def execute_write(self, query, params=None):
        self.write_calls.append((query, params))
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


# ---------------------------------------------------------------------------
# pipeline_kind additive column. Rows with no pipeline_kind read back as None
# (no silent "tracking" default).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_lifecycle_absent_pipeline_kind_reads_as_none():
    store = JobsStore(FakeKeyspacesClient([], [_lifecycle_row()]))

    lifecycle = await store.get_lifecycle("job-id")

    assert lifecycle["pipeline_kind"] is None


# ---------------------------------------------------------------------------
# Rows left by the retired local tracking pipeline are read explicitly.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_lifecycle_reads_legacy_awaiting_correction_as_failed_with_reason():
    store = JobsStore(FakeKeyspacesClient([], [_lifecycle_row(job_state="AWAITING_CORRECTION")]))

    lifecycle = await store.get_lifecycle("job-id")

    assert lifecycle["job_state"] == JobState.FAILED.value
    assert "retired local tracking pipeline" in lifecycle["error_message"]


@pytest.mark.asyncio
async def test_get_lifecycle_reads_retired_tracking_stage_explicitly():
    row = _lifecycle_row(job_state=JobState.COMPLETED.value)
    row.stage = "upscale_analyze"
    store = JobsStore(FakeKeyspacesClient([], [row]))

    lifecycle = await store.get_lifecycle("job-id")

    assert lifecycle["job_state"] == JobState.COMPLETED.value
    assert lifecycle["stage"] == ""
    assert lifecycle["stage_message"] == "retired tracking stage 'upscale_analyze'"
    assert lifecycle["error_message"] == ""


@pytest.mark.asyncio
async def test_get_lifecycle_unknown_stage_still_raises():
    row = _lifecycle_row()
    row.stage = "bogus"
    store = JobsStore(FakeKeyspacesClient([], [row]))

    with pytest.raises(ValueError, match="invalid Keyspaces lifecycle.stage"):
        await store.get_lifecycle("job-id")


@pytest.mark.asyncio
async def test_get_lifecycle_preserves_explicit_highlight_v2_pipeline_kind():
    row = _lifecycle_row()
    row.pipeline_kind = "highlight_v2"
    store = JobsStore(FakeKeyspacesClient([], [row]))

    lifecycle = await store.get_lifecycle("job-id")

    assert lifecycle["pipeline_kind"] == "highlight_v2"


@pytest.mark.asyncio
async def test_update_highlight_progress_writes_v2_fields():
    from service.analysis_keyspaces_enums import PipelineStage

    store = JobsStore(FakeKeyspacesClient([True]))

    ok = await store.update_highlight_progress(
        "job-id", PipelineStage.HIGHLIGHT_CHUNK,
        55.0, chunk_index=2, chunks_total=5, highlights_found_so_far=9,
        attribution_metrics_json='{"p1": 3}',
    )

    assert ok is True


@pytest.mark.asyncio
async def test_get_lifecycle_defaults_v2_progress_fields_to_none_when_absent():
    store = JobsStore(FakeKeyspacesClient([], [_lifecycle_row()]))

    lifecycle = await store.get_lifecycle("job-id")

    assert lifecycle["chunk_index"] is None
    assert lifecycle["chunks_total"] is None
    assert lifecycle["highlights_found_so_far"] is None
    assert lifecycle["attribution_metrics_json"] is None


@pytest.mark.asyncio
async def test_get_lifecycle_reads_back_v2_progress_fields_when_set():
    row = _lifecycle_row()
    row.chunk_index = 3
    row.chunks_total = 6
    row.highlights_found_so_far = 12
    row.attribution_metrics_json = '{"p1": 3}'
    store = JobsStore(FakeKeyspacesClient([], [row]))

    lifecycle = await store.get_lifecycle("job-id")

    assert lifecycle["chunk_index"] == 3
    assert lifecycle["chunks_total"] == 6
    assert lifecycle["highlights_found_so_far"] == 12
    assert lifecycle["attribution_metrics_json"] == '{"p1": 3}'


@pytest.mark.asyncio
async def test_update_highlight_progress_is_a_thin_raw_persistence_boundary():
    from service.analysis_keyspaces_enums import PipelineStage

    row = _lifecycle_row()
    row.progress_percent = 64.0
    row.highlights_found_so_far = 7
    client = FakeKeyspacesClient([True], [row])
    store = JobsStore(client)

    ok = await store.update_highlight_progress(
        "job-id",
        PipelineStage.HIGHLIGHT_CHUNK,
        20.0,
        highlights_found_so_far=3,
        stage_message="detecting",
    )

    assert ok is True
    params = client.write_calls[-1][1]
    assert params[1] == 20.0
    assert params[5] == 3


@pytest.mark.asyncio
async def test_complete_highlight_publishes_terminal_snapshot_in_one_lifecycle_update():
    from service.analysis_keyspaces_enums import PipelineStage

    row = _lifecycle_row(JobState.COMPLETED.value)
    row.progress_percent = 100.0
    row.stage = PipelineStage.HIGHLIGHT_PUBLISH.value
    row.stage_message = "completed"
    client = FakeKeyspacesClient([True, True, True], [row])
    store = JobsStore(client)

    assert await store.complete_highlight(
        "job-id",
        PipelineStage.HIGHLIGHT_PUBLISH,
        highlights_found_so_far=4,
        attribution_metrics_json='{"total_published": 4}',
    )

    query, params = client.write_calls[0]
    assert "job_state = %s" in query
    assert "stage = %s" in query
    assert "progress_percent = %s" in query
    assert params[:6] == [
        JobState.COMPLETED.value,
        PipelineStage.HIGHLIGHT_PUBLISH.value,
        100.0,
        0,
        0,
        "completed",
    ]
    assert params[8] == 4
    assert params[9] == '{"total_published": 4}'


@pytest.mark.asyncio
async def test_fail_highlight_publishes_error_snapshot_in_one_lifecycle_update():
    from service.analysis_keyspaces_enums import PipelineStage

    row = _lifecycle_row(JobState.FAILED.value)
    row.progress_percent = 47.0
    row.stage = PipelineStage.HIGHLIGHT_CHUNK.value
    row.stage_message = "error"
    client = FakeKeyspacesClient([True, True, True], [row])
    store = JobsStore(client)

    assert await store.fail_highlight(
        "job-id",
        PipelineStage.HIGHLIGHT_CHUNK,
        progress_percent=47.0,
        error_message="Gemini unavailable",
        highlights_found_so_far=2,
    )

    query, params = client.write_calls[0]
    assert "job_state = %s" in query
    assert params[:6] == [
        JobState.FAILED.value,
        PipelineStage.HIGHLIGHT_CHUNK.value,
        47.0,
        0,
        0,
        "error",
    ]
    assert params[9] is None
    assert params[10] == "Gemini unavailable"


@pytest.mark.asyncio
async def test_create_lifecycle_accepts_pipeline_kind_kwarg():
    """Callers (create_track_job) can pass pipeline_kind="highlight_v2" — the
    method must accept it without raising (real DDL/param-count coverage is
    a live-Keyspaces concern; this test only guards the Python call
    signature against accidental removal)."""
    store = JobsStore(FakeKeyspacesClient([True, True]))

    ok = await store.create_lifecycle(
        "job-id", "video-id", "user-id", pipeline_kind="highlight_v2",
    )

    assert ok is True


@pytest.mark.asyncio
async def test_heartbeat_propagates_recovery_index_failure():
    store = JobsStore(FakeKeyspacesClient([True, False], [_lifecycle_row()]))

    ok = await store.heartbeat("job-id", "worker")

    assert ok is False


@pytest.mark.asyncio
async def test_list_active_recovery_index_rows_newest_first():
    now = datetime.now(timezone.utc)
    r1 = SimpleNamespace(
        job_id="j1",
        video_id="v1",
        job_state="PENDING",
        owner_instance_id="o1",
        last_heartbeat_at=now,
    )
    fake = FakeKeyspacesClient([], [r1])
    store = JobsStore(fake)

    rows = await store.list_active_recovery_index_rows_newest_first(
        ["2026010112"],
        limit_per_bucket=50,
    )

    assert len(rows) == 1
    assert rows[0]["job_id"] == "j1"
    assert rows[0]["heartbeat_bucket"] == "2026010112"


@pytest.mark.asyncio
async def test_claim_pending_job_takeover_reads_applied_flag():
    class LwtRow:
        applied = True

    fake = FakeKeyspacesClient([], [LwtRow()])
    store = JobsStore(fake)

    ok = await store.claim_pending_job_takeover(
        "job-id",
        "new-owner",
        expected_owner_instance_id="old-owner",
    )

    assert ok is True
