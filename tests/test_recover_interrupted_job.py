"""Tests for ``recover_interrupted_job``: a ``highlight_v2`` job resumes under the
same job_id; any other pipeline_kind (the NULL/"tracking" rows left by the
retired local tracking pipeline) is failed with an explicit reason.

Shared fixtures live in ``tests/conftest.py``.
"""

import pytest

from service import routes as routes_mod
from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.analysis_settings import resolve_analysis_settings
from service.models import TrackRequest


@pytest.fixture(autouse=True)
def _noop_schedule_job(monkeypatch):
    """Recovery schedules the worker which touches S3 (ensure_bucket); CI may
    have no LocalStack. These tests assert persisted state + the scheduled id."""
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, request: None)


@pytest.mark.asyncio
async def test_recover_interrupted_job_highlight_v2_resumes_same_job_id_no_replacement(
    service_components, monkeypatch,
):
    """A pipeline_kind="highlight_v2" interrupted job resumes the SAME job_id:
    no new job_id, no replacement_job_id written on the old row."""
    from service.routes import recover_interrupted_job

    _, job_store, jobs_store = service_components
    req = resolve_analysis_settings(TrackRequest(bucket="b", key="folder/v.mp4"))
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "vid-v2", "u", pipeline_kind="highlight_v2")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    await jobs_store.set_state(job.job_id, JobState.INTERRUPTED)
    await jobs_store.write_checkpoint(
        job.job_id, PipelineStage.HIGHLIGHT_CHUNK, True,
        {
            "schema_version": 1,
            "artifacts": {"gemini_file_uri": "files/abc"},
            "chunk_index": 1, "chunks_total": 3,
            "worker_state": {
                "progress_percent": 33.0, "current_frame": 0,
                "total_frames": 0, "stage_progress_fraction": 1.0,
            },
        },
    )

    scheduled = []
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, request: scheduled.append(job_id))

    lifecycle = await jobs_store.get_lifecycle(job.job_id)
    assert lifecycle["pipeline_kind"] == "highlight_v2"
    await recover_interrupted_job(lifecycle)

    assert scheduled == [job.job_id]  # SAME job_id, no replacement

    lc_after = await jobs_store.get_lifecycle(job.job_id)
    assert not lc_after.get("replacement_job_id")  # no replacement-job chain
    assert lc_after["job_state"] == JobState.INTERRUPTED.value  # unchanged — resume owns the transition


@pytest.mark.asyncio
@pytest.mark.parametrize("pipeline_kind", [None, "tracking"])
async def test_recover_interrupted_job_retired_tracking_row_is_failed_not_resumed(
    service_components, monkeypatch, pipeline_kind,
):
    from service.routes import recover_interrupted_job

    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="folder/v.mp4")
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "vid-tracking", "u", pipeline_kind=pipeline_kind)
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    await jobs_store.set_state(job.job_id, JobState.INTERRUPTED)

    scheduled = []
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, request: scheduled.append(job_id))

    await recover_interrupted_job(await jobs_store.get_lifecycle(job.job_id))

    assert scheduled == []
    lc = await jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.FAILED.value
    assert "retired local tracking pipeline" in lc["error_message"]
    assert not lc.get("replacement_job_id")


@pytest.mark.asyncio
async def test_recover_interrupted_job_highlight_v2_without_valid_snapshot_is_failed(
    service_components, monkeypatch,
):
    from service.routes import recover_interrupted_job

    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="folder/v.mp4")  # not admitted: no analysis settings snapshot
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "vid-v2", "u", pipeline_kind="highlight_v2")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    await jobs_store.set_state(job.job_id, JobState.INTERRUPTED)

    scheduled = []
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, request: scheduled.append(job_id))

    with pytest.raises(RuntimeError, match="effective analysis settings snapshot"):
        await recover_interrupted_job(await jobs_store.get_lifecycle(job.job_id))

    assert scheduled == []
    lc = await jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.FAILED.value
