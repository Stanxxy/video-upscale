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


def _analyzed_event(
    highlight_index=1, player_id=None, actor_sentinel=None,
    start_s=10.0, end_s=20.0, action_class="submission_arm_lock",
):
    return {
        "type": "highlight_result", "highlight_index": highlight_index,
        "status": "analyzed", "ditch_reason": None,
        "clips": [{
            "start_s": start_s, "end_s": end_s,
            "position": "mount", "action_class": action_class, "outcome": "successful",
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
    # 2026-07-26 batched-publish re-scope: HIGHLIGHT_PUBLISH now carries a
    # per-candidate progress row (completed=False) per published candidate,
    # PLUS the one terminal row (completed=True) — not just the terminal
    # marker alone.
    publish_checkpoints = [
        w for w in jobs_store.written if w["stage_name"] == PipelineStage.HIGHLIGHT_PUBLISH.value
    ]
    progress_rows = [cp for cp in publish_checkpoints if not cp["completed"]]
    terminal_rows = [cp for cp in publish_checkpoints if cp["completed"]]
    assert len(progress_rows) == 2  # one per published candidate (p1's + p2's highlight)
    assert len(terminal_rows) == 1
    assert terminal_rows[0]["checkpoint_data"]["artifacts"]["sns_event_count"] == 2

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
    # 2026-07-26 batched-publish re-scope: publish no longer happens per-
    # chunk at all — highlights_published is always 0 at chunk-checkpoint
    # time now (see build_highlight_chunk_completed's own docstring); the
    # real publish count lives on the TERMINAL HIGHLIGHT_PUBLISH checkpoint.
    assert chunk_cp["checkpoint_data"]["highlights_published"] == 0
    assert chunk_cp["checkpoint_data"]["artifacts"]["clips"][0]["_highlight_index"] == 2


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
async def test_attribution_metrics_reflect_majority_vote_reconciled_counts(monkeypatch, _mock_boundaries):
    """2026-07-26 CEO batched-publish re-scope: attribution_metrics now
    counts the RECONCILED identity, not the raw per-highlight guess —
    p1 is the match's only real vote (2/2), so majority vote applies it to
    EVERY clip, including the one that raw-resolved to a sentinel (Brooks
    §2a: "make per-player stats reflect the reconciled player")."""
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
    assert metrics["player_id_counts"] == {"p1": 3}  # reconciled — the sentinel clip flipped to p1
    assert metrics["sentinel_count"] == 0
    assert metrics["total_published"] == 3
    assert metrics["cross_highlight_flip_rate"] == round(1 / 3, 4)  # 1 of 3 clips disagreed with the winner
    assert metrics["seam_duplicates_dropped"] == 0


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


# =============================================================================== #
# 2026-07-26 re-scope AC5 — _outer_chunks 45s backward-only overlap.
# =============================================================================== #
def test_outer_chunks_backward_only_45s_overlap_after_first_chunk():
    # 1500s / 720s -> 3 chunks: [0,720), [675,1440) (720-45), [1395,1500) (1440-45).
    chunks = highlight_orchestrator._outer_chunks(1500.0, 720)
    assert chunks == [(0.0, 720.0), (675.0, 1440.0), (1395.0, 1500.0)]


def test_outer_chunks_first_chunk_never_extends_before_zero():
    """The first chunk has no predecessor to overlap into — its start stays
    exactly at 0, never negative."""
    chunks = highlight_orchestrator._outer_chunks(100.0, 720)
    assert chunks == [(0.0, 100.0)]


def test_outer_chunks_second_chunk_clamps_overlap_to_zero_floor():
    """A chunk grid small enough that nominal_start - overlap_s would go
    negative must clamp to 0, never a negative offset."""
    chunks = highlight_orchestrator._outer_chunks(50.0, 20, overlap_s=45.0)
    # nominal boundaries: [0,20), [20,40), [40,50) -> overlapped starts:
    # chunk1: max(0, 20-45)=0; chunk2: max(0, 40-45)=0.
    assert chunks == [(0.0, 20.0), (0.0, 40.0), (0.0, 50.0)]


def test_outer_chunks_ends_stay_at_nominal_grid_boundary_no_forward_overlap():
    """OQ5: backward-only — every chunk's END is exactly the nominal grid
    boundary, never extended forward."""
    chunks = highlight_orchestrator._outer_chunks(2160.0, 720)
    assert [end for _, end in chunks] == [720.0, 1440.0, 2160.0]


def test_outer_chunks_overlap_s_is_configurable_not_hardcoded():
    chunks = highlight_orchestrator._outer_chunks(200.0, 100, overlap_s=10.0)
    assert chunks == [(0.0, 100.0), (90.0, 200.0)]


# =============================================================================== #
# 2026-07-26 re-scope AC6 — seam dedup wired into run_highlight_job: a
# highlight already published by chunk k-1 near the seam is NOT re-published
# when chunk k's 45s backward-extended read re-discovers it.
# =============================================================================== #
@pytest.mark.asyncio
async def test_seam_duplicate_highlight_suppressed_not_republished(monkeypatch, _mock_boundaries):
    """1500s job -> chunks (0,720),(675,1440),(1395,1500) — seam band with
    chunk 0 is [675,720]. Chunk 0's trailing highlight [700,715] and chunk
    1's leading highlight [703,716] are class-compatible (submission family)
    and temporally close -> chunk 1's copy must be suppressed."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=1500.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(start_s=700.0, end_s=715.0, action_class="submission_choke")],
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(start_s=703.0, end_s=716.0, action_class="submission_arm_lock")],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 1  # chunk 1's duplicate suppressed
    terminal_publish_cp = next(
        w for w in jobs_store.written
        if w["stage_name"] == PipelineStage.HIGHLIGHT_PUBLISH.value and w["completed"]
    )
    assert terminal_publish_cp["checkpoint_data"]["artifacts"]["sns_event_count"] == 1


@pytest.mark.asyncio
async def test_seam_highlights_outside_class_compat_both_publish(monkeypatch, _mock_boundaries):
    """Same seam-band positions, but the action classes are NOT compatible
    (not exact match, not both in the submission family) -> both must
    publish — never merged/suppressed just for being temporally close."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=1500.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(start_s=700.0, end_s=715.0, action_class="guard_pass")],
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(start_s=703.0, end_s=716.0, action_class="takedown_attempt")],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 2


@pytest.mark.asyncio
async def test_seam_highlights_outside_the_band_both_publish_no_false_positive_dedup(monkeypatch, _mock_boundaries):
    """Two class-compatible, temporally-close-to-EACH-OTHER highlights that
    are NOT near the chunk seam (both deep inside chunk 0's own [0,720)
    interior) must never be cross-chunk-deduped — the seam-band pre-filter
    means dedup is never even attempted outside the overlap zone."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=1500.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}, {"index": 2}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(highlight_index=1, start_s=100.0, end_s=115.0, action_class="submission_choke"),
         _analyzed_event(highlight_index=2, start_s=103.0, end_s=116.0, action_class="submission_choke")],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 2


@pytest.mark.asyncio
async def test_seam_duplicate_excluded_from_attribution_metrics(monkeypatch, _mock_boundaries):
    """A suppressed seam duplicate must not double-count attribution metrics
    for what is really ONE event."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=1500.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(start_s=700.0, end_s=715.0, action_class="submission_choke", player_id="p1")],
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(start_s=703.0, end_s=716.0, action_class="submission_arm_lock", player_id="p1")],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    import json
    metrics = json.loads(jobs_store.progress_writes[-1][2]["attribution_metrics_json"])
    assert metrics["player_id_counts"] == {"p1": 1}  # not 2 — the duplicate never counted
    assert metrics["total_published"] == 1


# =============================================================================== #
# Evaluator condition (2026-07-26, PASS-WITH-CONDITIONS on commits 1-3):
# regression-pin the resume x seam-dedup interaction against the NEW
# (whole-match, checkpoint-reconstructed) dedup shape landed by the
# 2026-07-26 CEO batched-publish re-scope. Under the OLD streaming design
# (99d1a1a), `prior_seam_clips` reset to `[]` on every resume — a duplicate
# straddling the resume boundary would NOT have been caught (documented,
# accepted limitation at the time). The batched-publish redesign CLOSES
# this gap as a side effect: dedup now runs ONCE, post-hoc, over clips
# reconstructed from EVERY completed HIGHLIGHT_CHUNK checkpoint (including
# ones from before this resume) — so seam-dedup correctness no longer
# depends on which chunks THIS process happened to re-run.
# =============================================================================== #
def _checkpointed_chunk0_with_trailing_seam_clip(action_class: str) -> dict:
    return {
        "stage_name": PipelineStage.HIGHLIGHT_CHUNK.value,
        "completed": True,
        "checkpoint_data": {
            "chunk_index": 0, "chunks_total": 3,
            "artifacts": {
                "clips": [{
                    "start_s": 700.0, "end_s": 715.0, "action_class": action_class,
                    "position": "mount", "outcome": "successful",
                    "player_id": None, "player_name": None,
                    "identity_uncertain": None, "actor_sentinel": None, "notes": "n",
                    "_chunk_index": 0, "_highlight_index": 1, "_candidate_key": "0:1",
                }],
            },
        },
    }


@pytest.mark.asyncio
async def test_resume_seam_dedup_correctly_suppresses_duplicate_using_checkpointed_prior_chunk_clips(
    monkeypatch, _mock_boundaries,
):
    """Resuming from chunk_index=1 (chunk 0 already completed on a PRIOR
    run, never re-analyzed THIS run) with a genuine seam duplicate in the
    resumed chunk's leading band must still correctly suppress it — the
    gap the old streaming design had (prior_seam_clips empty post-resume)
    is closed because dedup now reads chunk 0's clips back from its own
    durable checkpoint, not from this run's in-memory state."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=1500.0))
    _patch_run_pipeline(monkeypatch, [
        # First call THIS run corresponds to chunk 1 (chunk 0 skipped via resume).
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(start_s=703.0, end_s=716.0, action_class="submission_arm_lock")],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
    ])

    jobs_store = FakeJobsStore(checkpoints=[_checkpointed_chunk0_with_trailing_seam_clip("submission_choke")])
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    # Only chunk 0's checkpointed clip is published — chunk 1's rediscovery
    # is correctly suppressed as a seam duplicate, even though THIS process
    # never re-processed chunk 0 at all.
    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 1


@pytest.mark.asyncio
async def test_resume_seam_dedup_correctly_publishes_non_duplicate_across_resume_boundary(
    monkeypatch, _mock_boundaries,
):
    """Same resume shape, but chunk 1's leading-seam highlight is NOT a
    duplicate (class-incompatible) — both chunk 0's (checkpointed, prior
    run) and chunk 1's (this run) highlights must publish; the resume
    boundary must never cause an over-eager false-positive suppression
    either."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=1500.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(start_s=703.0, end_s=716.0, action_class="takedown_attempt")],
        [{"type": "highlight_map", "highlights": []},
         {"type": "stage_complete", "stage_type": "highlight_scan"}],
    ])

    jobs_store = FakeJobsStore(checkpoints=[_checkpointed_chunk0_with_trailing_seam_clip("submission_choke")])
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 2


# =============================================================================== #
# 2026-07-26 CEO batched-publish re-scope (AC8-11) — publish-batching tests:
# nothing published mid-run, everything published at finalize, a crash-mid-
# finalize resume publishes only the remainder.
# =============================================================================== #
@pytest.mark.asyncio
async def test_nothing_published_during_the_per_chunk_loop_only_at_finalize(monkeypatch, _mock_boundaries):
    """Direct proof of the batched-publish redesign: publish_axis_only_event
    is NEVER called while a chunk's own run_pipeline is still streaming
    events — only after the full per-chunk analyze loop completes."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))

    call_count = {"n": 0}

    async def _fake_run_pipeline(pipeline, ctx, planned, budget_cap=None):
        yield {"type": "highlight_map", "highlights": [{"index": 1}]}
        yield {"type": "stage_complete", "stage_type": "highlight_scan"}
        assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 0
        yield _analyzed_event(player_id="p1")
        # Still not published — this highlight_result was JUST yielded; the
        # orchestrator only COLLECTS it now (Phase 1), publish is Phase 6.
        assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 0
        call_count["n"] += 1

    monkeypatch.setattr(highlight_orchestrator.executors, "run_pipeline", _fake_run_pipeline)

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert call_count["n"] == 1
    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 1  # published only at finalize


@pytest.mark.asyncio
async def test_all_candidates_published_exactly_once_at_finalize_on_a_full_run(monkeypatch, _mock_boundaries):
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))
    _patch_run_pipeline(monkeypatch, [
        [{"type": "highlight_map", "highlights": [{"index": 1}, {"index": 2}]},
         {"type": "stage_complete", "stage_type": "highlight_scan"},
         _analyzed_event(highlight_index=1, start_s=10.0, end_s=20.0, player_id="p1"),
         _analyzed_event(highlight_index=2, start_s=100.0, end_s=110.0, player_id="p1")],
    ])

    jobs_store = FakeJobsStore()
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 2
    progress_rows = [
        w for w in jobs_store.written
        if w["stage_name"] == PipelineStage.HIGHLIGHT_PUBLISH.value and not w["completed"]
    ]
    assert {r["checkpoint_data"]["artifacts"]["candidate_key"] for r in progress_rows} == {"0:1", "0:2"}


@pytest.mark.asyncio
async def test_resume_mid_finalize_publishes_only_the_remainder(monkeypatch, _mock_boundaries):
    """Brooks's named new requirement (§2a): a crash mid-finalize-batch
    (e.g. after 1 of 2 candidates published) must resume publishing ONLY
    the remaining candidate(s) — zero double-sends."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))
    # All chunks already complete — the per-chunk loop is a no-op this run.
    checkpoints = [
        {
            "stage_name": PipelineStage.HIGHLIGHT_CHUNK.value,
            "completed": True,
            "checkpoint_data": {
                "chunk_index": 0, "chunks_total": 1,
                "artifacts": {"clips": [
                    {
                        "start_s": 10.0, "end_s": 20.0, "action_class": "guard_pass",
                        "position": "mount", "outcome": "successful",
                        "player_id": "p1", "player_name": "Alice",
                        "identity_uncertain": False, "actor_sentinel": None, "notes": "n",
                        "_chunk_index": 0, "_highlight_index": 1, "_candidate_key": "0:1",
                    },
                    {
                        "start_s": 100.0, "end_s": 110.0, "action_class": "sweep",
                        "position": "half_guard", "outcome": "successful",
                        "player_id": "p1", "player_name": "Alice",
                        "identity_uncertain": False, "actor_sentinel": None, "notes": "n",
                        "_chunk_index": 0, "_highlight_index": 2, "_candidate_key": "0:2",
                    },
                ]},
            },
        },
        # Candidate 0:1 already published on a PRIOR (crashed-mid-finalize) run.
        {
            "stage_name": PipelineStage.HIGHLIGHT_PUBLISH.value,
            "completed": False,
            "checkpoint_data": {
                "reason": "highlight_publish_candidate",
                "artifacts": {"candidate_key": "0:1", "event_index": 1},
            },
        },
    ]
    _patch_run_pipeline(monkeypatch, [])  # resume_from_chunk_index == chunks_total -> loop never runs

    jobs_store = FakeJobsStore(checkpoints=checkpoints)
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    # Only candidate 0:2 gets a real SNS call this run — 0:1 is skipped
    # (already published on the prior run), never double-sent.
    assert _mock_boundaries["sns"].publish_axis_only_event.call_count == 1
    terminal_cp = next(
        w for w in jobs_store.written
        if w["stage_name"] == PipelineStage.HIGHLIGHT_PUBLISH.value and w["completed"]
    )
    # The audit view (sns_event_count) still counts BOTH candidates — the
    # already-published one is rebuilt into the audit record, just never
    # re-sent over the wire.
    assert terminal_cp["checkpoint_data"]["artifacts"]["sns_event_count"] == 2


@pytest.mark.asyncio
async def test_resume_after_publish_terminal_already_done_never_resends_analysis_complete(
    monkeypatch, _mock_boundaries,
):
    """A resume that lands after the terminal HIGHLIGHT_PUBLISH checkpoint
    was already written (a narrow crash window between that write and the
    job being marked COMPLETED) must never re-publish ANY candidate and
    must never resend the analysis_complete notification (product memory:
    users get a completion email — a duplicate would be a real bug)."""
    _patch_ingest(monkeypatch, _ingest_result(duration_sec=600.0))
    checkpoints = [
        {
            "stage_name": PipelineStage.HIGHLIGHT_CHUNK.value,
            "completed": True,
            "checkpoint_data": {
                "chunk_index": 0, "chunks_total": 1,
                "artifacts": {"clips": [{
                    "start_s": 10.0, "end_s": 20.0, "action_class": "guard_pass",
                    "position": "mount", "outcome": "successful",
                    "player_id": "p1", "player_name": "Alice",
                    "identity_uncertain": False, "actor_sentinel": None, "notes": "n",
                    "_chunk_index": 0, "_highlight_index": 1, "_candidate_key": "0:1",
                }]},
            },
        },
        {
            "stage_name": PipelineStage.HIGHLIGHT_PUBLISH.value,
            "completed": True,
            "checkpoint_data": {
                "reason": "highlight_publish_completed",
                "artifacts": {
                    "sns_topic_arn": "arn:aws:sns:x", "sns_event_count": 1,
                    "sns_completion_sent": True, "result_s3_uri": "s3://out-bucket/videos/match_v2_events.json",
                },
            },
        },
    ]
    _patch_run_pipeline(monkeypatch, [])

    jobs_store = FakeJobsStore(checkpoints=checkpoints)
    job_store = InMemoryJobStore()
    job = await job_store.create_job(_request())

    await highlight_orchestrator.run_highlight_job(job.job_id, _request(), _config(), job_store, jobs_store)

    _mock_boundaries["sns"].publish_axis_only_event.assert_not_called()
    _mock_boundaries["sns"].publish_analysis_complete.assert_not_called()
    _mock_boundaries["s3"].upload_json.assert_not_called()
    # No NEW terminal HIGHLIGHT_PUBLISH checkpoint written this run either.
    new_terminal_rows = [
        w for w in jobs_store.written
        if w["stage_name"] == PipelineStage.HIGHLIGHT_PUBLISH.value and w["completed"]
    ]
    assert new_terminal_rows == []
    final_job = await job_store.get_job(job.job_id)
    assert final_job.status.value == "completed"
