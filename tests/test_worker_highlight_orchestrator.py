"""S12 Phase 1b — service/worker/highlight_orchestrator.py::run_highlight_job
(item 11, the centerpiece test file).

Every Gemini/S3/SNS boundary is mocked at this level — the pipeline's OWN
Gemini-calling logic is already covered by test_pipelines_executors.py/
test_pipelines_highlight_v2.py; this file only proves the ADAPTER (outer
chunk loop, checkpointing, resume, publish/ditch/error dispatch).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.models import TrackRequest
from service.worker import highlight_orchestrator
from service.worker.stages.highlight_ingest import HighlightIngestResult


class FakeJobsStore:
    def __init__(self, checkpoints=None):
        self._checkpoints = list(checkpoints or [])
        self.written: list[dict] = []
        self.states: list[tuple] = []
        self.progress_writes: list[tuple] = []

    async def set_state(self, job_id, state, error_message=""):
        self.states.append((state, error_message))
        return True

    async def get_all_checkpoints(self, job_id):
        return list(self._checkpoints)

    async def write_checkpoint(self, job_id, stage_name, completed, data):
        record = {"stage_name": stage_name.value, "completed": completed, "checkpoint_data": data}
        self._checkpoints.append(record)
        self.written.append(record)
        return True

    async def update_highlight_chunk_progress(self, job_id, stage, pct, **kwargs):
        self.progress_writes.append((stage, pct, kwargs))
        return True


def _ingest_result(duration_sec=1500.0) -> HighlightIngestResult:
    return HighlightIngestResult(
        video_path="/tmp/video.mp4",
        gemini_file_uri="https://.../files/abc",
        gemini_file_name="files/abc",
        gemini_file_mime_type="video/mp4",
        gemini_file_expiration=datetime.now(timezone.utc) + timedelta(hours=40),
        video_duration_sec=duration_sec,
        player_references=[],
    )


def _config():
    return ServiceConfig(
        outer_chunk_scope_sec=720, highlight_pipeline_budget_cap=60,
        sns_topic_arn="arn:aws:sns:us-east-1:000:topic",
    )


def _request(**kwargs):
    return TrackRequest(bucket="src-bucket", key="videos/match.mp4", output_bucket="out-bucket", **kwargs)


def _analyzed_event(highlight_index=1, player_id=None, actor_sentinel=None):
    return {
        "type": "highlight_result", "highlight_index": highlight_index,
        "status": "analyzed", "ditch_reason": None,
        "clips": [{
            "start_s": 10.0, "end_s": 20.0,
            "position": "mount", "action_class": "submission_arm_lock", "outcome": "successful",
            "player_id": player_id, "player_name": "Alice" if player_id else None,
            "identity_uncertain": False, "actor_sentinel": actor_sentinel,
            "notes": "n",
        }],
    }


def _ditched_event(highlight_index=1):
    return {
        "type": "highlight_result", "highlight_index": highlight_index,
        "status": "ditched", "ditch_reason": "nothing happened", "clips": [],
    }


@pytest.fixture(autouse=True)
def _mock_boundaries(monkeypatch):
    """Mock every external boundary (Gemini Files API client construction,
    S3, SNS, Gemini cleanup) — the ADAPTER logic under test never actually
    talks to any of them."""
    monkeypatch.setattr(highlight_orchestrator, "genai", MagicMock())
    monkeypatch.setattr(highlight_orchestrator.gemini_upload, "delete_gemini_file", AsyncMock())

    fake_s3 = MagicMock()
    fake_s3.upload_json.return_value = "s3://out-bucket/videos/match_v2_events.json"
    monkeypatch.setattr(highlight_orchestrator, "_make_s3", lambda config: fake_s3)

    fake_sns = MagicMock()
    monkeypatch.setattr(highlight_orchestrator, "SNSPublisher", MagicMock(return_value=fake_sns))

    def _fake_clip_to_axis_only_event(clip, video_id):
        return SimpleNamespace(model_dump=lambda mode="json": {"clip": dict(clip)})

    monkeypatch.setattr(highlight_orchestrator, "clip_to_axis_only_event", _fake_clip_to_axis_only_event)
    return {"s3": fake_s3, "sns": fake_sns}


def _patch_ingest(monkeypatch, result: HighlightIngestResult):
    monkeypatch.setattr(highlight_orchestrator, "run_highlight_ingest_stage", AsyncMock(return_value=result))


def _patch_run_pipeline(monkeypatch, events_per_chunk: list[list[dict]]):
    """``events_per_chunk[i]`` is the event list yielded for the i-th CALL
    to run_pipeline (i.e. the i-th chunk actually processed, in call order —
    NOT indexed by chunk_index, so a resumed run's first call gets
    events_per_chunk[0])."""
    call_count = {"n": 0}

    async def _fake_run_pipeline(pipeline, ctx, planned, budget_cap=None):
        idx = call_count["n"]
        call_count["n"] += 1
        events = events_per_chunk[idx] if idx < len(events_per_chunk) else []
        for event in events:
            yield event

    monkeypatch.setattr(highlight_orchestrator.executors, "run_pipeline", _fake_run_pipeline)
    return call_count


@pytest.mark.asyncio
async def test_multi_chunk_job_completes_with_per_chunk_checkpoints_in_order(monkeypatch, _mock_boundaries):
    # 1500s / 720s -> 3 chunks: [0,720), [720,1440), [1440,1500).
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=1500.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(player_id="p1")],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(player_id="p2")],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    chunk_checkpoints = [
        w for w in jobs_store.written if w["stage_name"] == PipelineStage.HIGHLIGHT_CHUNK.value
    ]
    assert [cp["checkpoint_data"]["chunk_index"] for cp in chunk_checkpoints] == [0, 1, 2]
    assert all(cp["completed"] for cp in chunk_checkpoints)
    publish_checkpoints = [
        w for w in jobs_store.written if w["stage_name"] == PipelineStage.HIGHLIGHT_PUBLISH.value
    ]
    assert len(publish_checkpoints) == 1
    assert publish_checkpoints[0]["checkpoint_data"]["artifacts"]["sns_event_count"] == 2

    final_job = await job_store.get_job(job.job_id)
    assert final_job.status.value == "completed"
    assert jobs_store.states[-1][0] == JobState.COMPLETED


@pytest.mark.asyncio
async def test_ditched_highlight_never_published(monkeypatch, _mock_boundaries):
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))  # 1 chunk
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}, {"index": 2}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _ditched_event(highlight_index=1),
         _analyzed_event(highlight_index=2, player_id="p1")],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 1
    chunk_cp = next(w for w in jobs_store.written if w["stage_name"] == PipelineStage.HIGHLIGHT_CHUNK.value)
    assert chunk_cp["checkpoint_data"]["highlights_ditched"] == 1
    assert chunk_cp["checkpoint_data"]["highlights_analyzed"] == 1
    assert chunk_cp["checkpoint_data"]["highlights_published"] == 1


@pytest.mark.asyncio
async def test_error_event_does_not_abort_chunk_or_job(monkeypatch, _mock_boundaries):
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}, {"index": 2}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         {"type": "error", "stage_id": "highlight_analyze", "message": "highlight 1: actor call: boom",
          "highlight_index": 1},
         _analyzed_event(highlight_index=2, player_id="p1")],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 1  # highlight 2 still published
    final_job = await job_store.get_job(job.job_id)
    assert final_job.status.value == "completed"  # job never aborted on the error event
    assert jobs_store.states[-1][0] == JobState.COMPLETED


@pytest.mark.asyncio
async def test_resume_skips_completed_chunks_and_does_not_reupload_non_expired_file(monkeypatch, _mock_boundaries):
    """A prior HIGHLIGHT_CHUNK checkpoint for chunk 0 -> resume starts at
    chunk 1. run_highlight_ingest_stage itself owns the reuse-vs-reupload
    decision (tested directly in test_worker_highlight_ingest.py) — this
    test proves run_highlight_job actually SKIPS chunk 0's processing."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=1500.0))  # 3 chunks
    call_count = _patch_run_pipeline(monkeypatch, [
        # First call corresponds to chunk 1 (chunk 0 skipped via resume).
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
    ])

    prior_chunk0 = {
        "stage_name": PipelineStage.HIGHLIGHT_CHUNK.value,
        "completed": True,
        "checkpoint_data": {"chunk_index": 0, "chunks_total": 3},
    }
    jobs_store = FakeJobsStore(checkpoints=[prior_chunk0])
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert call_count["n"] == 2  # only chunks 1 and 2 processed, never chunk 0 again
    new_chunk_checkpoints = [
        w for w in jobs_store.written if w["stage_name"] == PipelineStage.HIGHLIGHT_CHUNK.value
    ]
    assert [cp["checkpoint_data"]["chunk_index"] for cp in new_chunk_checkpoints] == [1, 2]


@pytest.mark.asyncio
async def test_analysis_complete_published_once_after_last_chunk(monkeypatch, _mock_boundaries):
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(player_id="p1")],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    _mock_boundaries["sns"].publish_analysis_complete.assert_called_once()
    call = _mock_boundaries["sns"].publish_analysis_complete.call_args
    assert call.kwargs["total_event_count"] == 1
    assert call.kwargs["result_s3_uri"] == "s3://out-bucket/videos/match_v2_events.json"


@pytest.mark.asyncio
async def test_attribution_metrics_accumulate_player_id_and_sentinel_counts(monkeypatch, _mock_boundaries):
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}, {"index": 2}, {"index": 3}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(highlight_index=1, player_id="p1"),
         _analyzed_event(highlight_index=2, player_id="p1"),
         _analyzed_event(highlight_index=3, actor_sentinel="contested")],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    final_progress = jobs_store.progress_writes[-1]
    metrics_json = final_progress[2]["attribution_metrics_json"]
    import json
    metrics = json.loads(metrics_json)
    assert metrics["player_id_counts"] == {"p1": 2}
    assert metrics["sentinel_count"] == 1
    assert metrics["total_published"] == 3


@pytest.mark.asyncio
async def test_no_sns_topic_configured_analyzes_but_does_not_publish(monkeypatch, _mock_boundaries):
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(player_id="p1")],
    ])

    config = ServiceConfig(outer_chunk_scope_sec=720, sns_topic_arn="")
    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request(sns_topic_arn=None))

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(sns_topic_arn=None), config, job_store, jobs_store)

    _mock_boundaries["sns"].publish_axis_only_event.assert_not_called()
    final_job = await job_store.get_job(job.job_id)
    assert final_job.status.value == "completed"  # still completes — no fallback publish path


@pytest.mark.asyncio
async def test_ingest_failure_marks_job_failed_never_swallowed(monkeypatch, _mock_boundaries):
    monkeypatch.setattr(
        highlight_orchestrator, "run_highlight_ingest_stage",
        AsyncMock(side_effect=RuntimeError("Gemini Files API upload failed: state=FAILED")),
    )

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    final_job = await job_store.get_job(job.job_id)
    assert final_job.status.value == "failed"
    assert "FAILED" in final_job.error_message
    assert jobs_store.states[-1][0] == JobState.FAILED


@pytest.mark.asyncio
async def test_gemini_file_cleanup_called_in_finally(monkeypatch, _mock_boundaries):
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))
    _patch_run_pipeline(monkeypatch, [[]])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    highlight_orchestrator.gemini_upload.delete_gemini_file.assert_awaited_once()
