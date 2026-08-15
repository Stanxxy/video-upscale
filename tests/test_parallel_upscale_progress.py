"""Tests for the parallel-upscale progress aggregator.

Design context (post 2026-05-25 refactor):
- Tracking is **always** sequential (see
  ``contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md``
  §``stage_name == track`` and ``JOB_ROTATION_HANDOFF_AND_RESUME.md`` §4.1
  for why mid-track loss requires the V1 ``track`` checkpoint shape +
  AWAITING_CORRECTION + HumanVerificationSuspend, which only the
  sequential path honors end-to-end).
- The UPSCALE_ANALYZE stage is parallelized via
  ``service.worker._run_parallel_upscale``. Each segment runs in its own
  thread executor with its own ``_run_upscale_analysis`` invocation; each
  segment's per-frame ``progress_cb`` reports a per-segment fraction to a
  shared aggregator dict; a background asyncio task aggregates the sum
  every ``LIFECYCLE_HEARTBEAT_INTERVAL`` and writes
  ``PipelineStage.UPSCALE_ANALYZE`` lifecycle progress in the 55%-80%
  band so SSE clients see the upscale stage advance.

These tests guard the aggregator path: lifecycle writes must (a) be
present, (b) be in the 55%-80% band, (c) be monotonic, and (d) only
write ``UPSCALE_ANALYZE`` progress (no ``TRACK`` writes; tracking is
already complete by the time this helper runs).
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import patch

import pytest

from service.analysis_keyspaces_enums import PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.models import TrackRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request() -> TrackRequest:
    return TrackRequest(
        bucket="bjj-video-analysis",
        key="input/video.mp4",
        output_bucket="bjj-video-analysis",
        box_a=[1.0, 2.0, 3.0, 4.0],
        box_b=[5.0, 6.0, 7.0, 8.0],
    )


def _patch_segment_runner_to(n_total: int, segment_length: int = 100):
    """Patch ``compute_segment_ranges`` to return synthetic segments so we
    don't need a real video on disk to determine clip length."""
    ranges = [
        (i * segment_length, (i + 1) * segment_length) for i in range(n_total)
    ]
    return patch(
        "service.segment_runner.compute_segment_ranges",
        return_value=ranges,
    )


async def _seed_job_store(job_id: str) -> InMemoryJobStore:
    """Build an in-memory job store with ``job_id`` already hydrated."""
    js = InMemoryJobStore()
    await js.hydrate_job(job_id, _make_request())
    return js


def _write_merged_tracking_json(
    tmp_path,
    *,
    clip_start_frame: int,
    clip_end_frame: int,
    n_frames: int,
    fps: float = 30.0,
) -> str:
    """Write a minimal merged tracking JSON spanning ``clip_start_frame``..
    ``clip_end_frame`` with ``n_frames`` evenly spaced entries. Returns the
    on-disk path the production code would have produced."""
    track_dir = os.path.join(str(tmp_path), "tracking")
    os.makedirs(track_dir, exist_ok=True)
    path = os.path.join(track_dir, "tracking.json")
    step = max(1, (clip_end_frame - clip_start_frame) // max(n_frames, 1))
    frames = [
        {
            "frame_idx": clip_start_frame + i * step,
            "athletes": [],
        }
        for i in range(n_frames)
    ]
    with open(path, "w") as f:
        json.dump(
            {
                "video": "fake.mp4",
                "fps": fps,
                "start_frame": clip_start_frame,
                "end_frame": clip_end_frame,
                "frames": frames,
            },
            f,
        )
    return path


# ---------------------------------------------------------------------------
# Test 1: aggregator writes monotonic upscale-stage progress in 55-80 band
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_upscale_aggregator_writes_monotonic_55_to_80(
    mock_jobs_store, tmp_path,
):
    """As per-segment upscale ``progress_cb`` fractions advance, the
    aggregator must write a monotonically non-decreasing percent value to
    ``jobs_store.update_progress``, and every value must stay inside the
    upscale band ``[55.0, 80.0]``."""

    progress_calls: list[dict] = []

    async def _capture_update_progress(
        job_id, stage, pct, *,
        current_frame=None, total_frames=None, stage_message=None,
    ):
        progress_calls.append(
            {
                "stage": stage,
                "pct": pct,
                "current_frame": current_frame,
                "total_frames": total_frames,
            },
        )

    mock_jobs_store.update_progress = _capture_update_progress

    from service import worker as worker_mod

    n_total = 3
    seg_len = 100
    clip_total_frames = n_total * seg_len

    def _fake_run_upscale_analysis(*args, **kwargs):
        # Drive the per-segment progress_cb a few times to simulate
        # frames being processed. Each tick advances the per-segment
        # fraction so the aggregator picks it up.
        progress_cb = kwargs["progress_cb"]
        if progress_cb is not None:
            for frac in (0.1, 0.5, 1.0):
                progress_cb(frac)
                # Yield briefly so the aggregator task (running on the
                # main loop at LIFECYCLE_HEARTBEAT_INTERVAL=0.01s in
                # this test) has a chance to wake and read frames_done.
                import time as _time
                _time.sleep(0.02)
        # Include at least one clip per segment so merge_analysis_results
        # returns a non-None merged dict.
        return (
            {
                "match_summary": "stub",
                "clips": [{"start_frame": 0, "end_frame": 10, "category": "x"}],
                "fps": 30.0,
            },
            30.0,
        )

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        standard_segments=n_total,
    )
    job_store = await _seed_job_store("job-upscale")

    merged_tracking_json_path = _write_merged_tracking_json(
        tmp_path,
        clip_start_frame=0,
        clip_end_frame=clip_total_frames,
        n_frames=clip_total_frames,  # one per frame so all segments are non-empty
    )

    with _patch_segment_runner_to(n_total, seg_len), \
         patch("service.worker.stages.parallel_upscale._run_upscale_analysis",
               side_effect=_fake_run_upscale_analysis), \
         patch("service.worker.stages.parallel_upscale.LIFECYCLE_HEARTBEAT_INTERVAL", 0.01):
        request = _make_request()
        loop = asyncio.get_event_loop()
        result, fps = await worker_mod._run_parallel_upscale(
            job_id="job-upscale",
            video_path=str(tmp_path / "video.mp4"),
            merged_tracking_json_path=merged_tracking_json_path,
            clip_start_frame=0,
            clip_end_resolved=clip_total_frames,
            clip_total_frames=clip_total_frames,
            config=config,
            request=request,
            work_dir=str(tmp_path),
            job_store=job_store,
            jobs_store=mock_jobs_store,
            loop=loop,
            tracking_s3_key="some/key.json",
            progress_floor=0.0,
        )

    upscale_writes = [
        c for c in progress_calls if c["stage"] == PipelineStage.UPSCALE_ANALYZE
    ]
    assert upscale_writes, (
        f"Aggregator wrote no upscale-stage progress. Calls: {progress_calls}"
    )

    # No TRACK writes from the aggregator — tracking is complete before this
    # helper runs.
    track_writes = [c for c in progress_calls if c["stage"] == PipelineStage.TRACK]
    assert not track_writes, (
        f"Aggregator unexpectedly wrote TRACK progress: {track_writes}"
    )

    # Every write must stay inside the 55-80 band.
    for w in upscale_writes:
        assert 55.0 <= w["pct"] <= 80.0, (
            f"Upscale progress {w['pct']} outside 55-80 band"
        )

    # Monotonic non-decreasing percent across all upscale writes.
    pct_seq = [w["pct"] for w in upscale_writes]
    assert all(b >= a for a, b in zip(pct_seq, pct_seq[1:])), (
        f"Upscale progress went backwards: {pct_seq}"
    )

    assert fps == 30.0
    assert result is not None
    assert "clips" in result


# ---------------------------------------------------------------------------
# Test 2: aggregator never exceeds 80% even when all segments report done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_upscale_aggregator_clamps_at_80(
    mock_jobs_store, tmp_path,
):
    """When every segment reports fraction=1.0 the aggregator must clamp
    progress at 80.0 (the upper edge of the upscale band) so the
    stage-complete write at 80% is the only thing that hits the boundary
    cleanly."""
    progress_calls: list[dict] = []

    async def _capture(job_id, stage, pct, *, current_frame=None,
                       total_frames=None, stage_message=None):
        progress_calls.append({"stage": stage, "pct": pct})

    mock_jobs_store.update_progress = _capture

    from service import worker as worker_mod

    n_total = 2
    seg_len = 50
    clip_total_frames = n_total * seg_len

    def _fake_upscale(*args, **kwargs):
        cb = kwargs["progress_cb"]
        if cb is not None:
            cb(1.0)  # immediately report done
            import time as _time
            _time.sleep(0.05)  # give aggregator at least one tick
        return ({}, 30.0)

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        standard_segments=n_total,
    )
    job_store = await _seed_job_store("job-upscale")
    merged_tracking_json_path = _write_merged_tracking_json(
        tmp_path,
        clip_start_frame=0,
        clip_end_frame=clip_total_frames,
        n_frames=clip_total_frames,
    )

    with _patch_segment_runner_to(n_total, seg_len), \
         patch("service.worker.stages.parallel_upscale._run_upscale_analysis",
               side_effect=_fake_upscale), \
         patch("service.worker.stages.parallel_upscale.LIFECYCLE_HEARTBEAT_INTERVAL", 0.01):
        request = _make_request()
        loop = asyncio.get_event_loop()
        await worker_mod._run_parallel_upscale(
            job_id="job-upscale",
            video_path=str(tmp_path / "video.mp4"),
            merged_tracking_json_path=merged_tracking_json_path,
            clip_start_frame=0,
            clip_end_resolved=clip_total_frames,
            clip_total_frames=clip_total_frames,
            config=config,
            request=request,
            work_dir=str(tmp_path),
            job_store=job_store,
            jobs_store=mock_jobs_store,
            loop=loop,
            tracking_s3_key="some/key.json",
            progress_floor=0.0,
        )

    upscale_writes = [
        c for c in progress_calls if c["stage"] == PipelineStage.UPSCALE_ANALYZE
    ]
    assert upscale_writes, "expected at least one upscale-stage write"
    for w in upscale_writes:
        assert w["pct"] <= 80.0, (
            f"Upscale aggregator wrote past 80%: {w['pct']}"
        )
    # Final aggregated write after segments all reported 1.0 should be at
    # the band ceiling.
    assert upscale_writes[-1]["pct"] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Test 3: aggregator does not write checkpoint rows or change job_state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_upscale_aggregator_does_not_write_checkpoints(
    mock_jobs_store, tmp_path,
):
    """The aggregator must only call ``update_progress``. It must NOT write
    V1 checkpoint rows (those are owned by ``_run_upscale_analysis`` per the
    addendum) or transition job state. This guards against regressions where
    the aggregator accidentally clobbers per-stage checkpoint state."""
    set_state_calls: list = []
    write_checkpoint_calls: list = []

    async def _capture_set_state(job_id, state, **kwargs):
        set_state_calls.append((job_id, state, kwargs))
        return True

    async def _capture_write_checkpoint(job_id, stage, completed, data):
        write_checkpoint_calls.append((job_id, stage, completed, data))
        return True

    mock_jobs_store.set_state = _capture_set_state
    mock_jobs_store.write_checkpoint = _capture_write_checkpoint

    from service import worker as worker_mod

    n_total = 2
    seg_len = 50
    clip_total_frames = n_total * seg_len

    def _fake_upscale(*args, **kwargs):
        cb = kwargs["progress_cb"]
        if cb is not None:
            cb(0.25)
            cb(0.75)
        return ({}, 30.0)

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        standard_segments=n_total,
    )
    job_store = await _seed_job_store("job-upscale")
    merged_tracking_json_path = _write_merged_tracking_json(
        tmp_path,
        clip_start_frame=0,
        clip_end_frame=clip_total_frames,
        n_frames=clip_total_frames,
    )

    with _patch_segment_runner_to(n_total, seg_len), \
         patch("service.worker.stages.parallel_upscale._run_upscale_analysis",
               side_effect=_fake_upscale), \
         patch("service.worker.stages.parallel_upscale.LIFECYCLE_HEARTBEAT_INTERVAL", 0.01):
        request = _make_request()
        loop = asyncio.get_event_loop()
        await worker_mod._run_parallel_upscale(
            job_id="job-upscale",
            video_path=str(tmp_path / "video.mp4"),
            merged_tracking_json_path=merged_tracking_json_path,
            clip_start_frame=0,
            clip_end_resolved=clip_total_frames,
            clip_total_frames=clip_total_frames,
            config=config,
            request=request,
            work_dir=str(tmp_path),
            job_store=job_store,
            jobs_store=mock_jobs_store,
            loop=loop,
            tracking_s3_key="some/key.json",
            progress_floor=0.0,
        )

    # The aggregator must not change job_state (the upscale stage's normal
    # success path keeps state RUNNING; only set_state callers outside this
    # helper transition state).
    assert not set_state_calls, (
        f"Aggregator unexpectedly called set_state: {set_state_calls}"
    )

    # The aggregator must not write checkpoint rows. (_run_upscale_analysis
    # is mocked out in this test so the only place that could write
    # checkpoints is the aggregator itself.)
    assert not write_checkpoint_calls, (
        f"Aggregator unexpectedly wrote checkpoints: {write_checkpoint_calls}"
    )


# ---------------------------------------------------------------------------
# Test 4: aggregator task is cancelled cleanly on segment failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_upscale_aggregator_clean_shutdown_on_failure(
    mock_jobs_store, tmp_path,
):
    """If a segment raises, the aggregator task must not leak (no orphan
    task) and the exception must propagate to the caller."""
    from service import worker as worker_mod

    def _failing_upscale(*args, **kwargs):
        raise RuntimeError("simulated upscale failure")

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        standard_segments=1,
    )
    job_store = await _seed_job_store("job-upscale")
    merged_tracking_json_path = _write_merged_tracking_json(
        tmp_path,
        clip_start_frame=0,
        clip_end_frame=100,
        n_frames=100,
    )

    before = {t for t in asyncio.all_tasks()}

    with _patch_segment_runner_to(1, 100), \
         patch("service.worker.stages.parallel_upscale._run_upscale_analysis",
               side_effect=_failing_upscale), \
         patch("service.worker.stages.parallel_upscale.LIFECYCLE_HEARTBEAT_INTERVAL", 0.01):
        request = _make_request()
        loop = asyncio.get_event_loop()
        with pytest.raises(RuntimeError, match="simulated upscale failure"):
            await worker_mod._run_parallel_upscale(
                job_id="job-upscale",
                video_path=str(tmp_path / "video.mp4"),
                merged_tracking_json_path=merged_tracking_json_path,
                clip_start_frame=0,
                clip_end_resolved=100,
                clip_total_frames=100,
                config=config,
                request=request,
                work_dir=str(tmp_path),
                job_store=job_store,
                jobs_store=mock_jobs_store,
                loop=loop,
                tracking_s3_key="some/key.json",
                progress_floor=0.0,
            )

    # Yield once so any pending cancellations finish.
    await asyncio.sleep(0)
    current = asyncio.current_task()
    after = {t for t in asyncio.all_tasks() if not t.done() and t is not current}
    leaked = after - before
    assert not leaked, f"Aggregator task leaked: {leaked}"
