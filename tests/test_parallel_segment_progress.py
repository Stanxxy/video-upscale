"""Tests for the parallel-segment tracking progress aggregator.

Bug context:
- Before the fix in ``fix/parallel-segment-progress-and-yolo-thrash``,
  ``service.worker._run_parallel_segments`` launched each segment with
  ``progress_cb=None``. As a result the lifecycle row stayed frozen at the
  seeded ``current_frame=1, progress_percent=15.0`` for the entire duration
  of parallel tracking, and SSE clients (the UI progress bar) saw no
  movement.

These tests target the new aggregator path: per-segment ``progress_cb``
closures write into a shared ``frames_done`` dict, and a background asyncio
task periodically aggregates and writes the lifecycle row via
``jobs_store.update_progress`` + ``job_store.update_job``.
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


class _FakeS3:
    """Minimal S3 stub: records uploads, ignores reads."""

    def __init__(self):
        self.uploaded: list[tuple[str, str, object]] = []

    def ensure_bucket(self, *_a, **_kw):
        return None

    def upload_json(self, body, bucket, key):
        self.uploaded.append((bucket, key, body))

    def upload_file(self, *_a, **_kw):
        return None

    def put_object(self, *_a, **_kw):
        return None


async def _seed_job_store(job_id: str) -> InMemoryJobStore:
    """Build an in-memory job store with ``job_id`` already hydrated."""
    js = InMemoryJobStore()
    await js.hydrate_job(job_id, _make_request())
    return js


# ---------------------------------------------------------------------------
# Test 1: aggregator writes lifecycle as segments make progress
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_aggregator_writes_monotonic_progress(
    mock_jobs_store, tmp_path,
):
    """As per-segment ``progress_cb`` closures bump ``frames_done``, the
    aggregator must write a monotonically non-decreasing ``current_frame`` to
    ``jobs_store.update_progress``."""

    progress_calls: list[dict] = []

    async def _capture_update_progress(job_id, stage, pct, *,
                                       current_frame=None, total_frames=None,
                                       stage_message=None):
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

    def _fake_run_tracking_job(**kwargs):
        progress_cb = kwargs["progress_cb"]
        start = kwargs["start_frame"]
        end = kwargs["end_frame"]
        local_total = end - start
        # Drive progress_cb a few times to simulate tracking ticks.
        for done in (10, 50, local_total):
            progress_cb(done, local_total, start + done - 1)
        # Write a stub tracking JSON for the merge step.
        out_path = os.path.join(kwargs["output_dir"], "tracking.json")
        with open(out_path, "w") as f:
            json.dump({"fps": 30.0, "frames": []}, f)
        return out_path

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        standard_segments=n_total,
    )
    job_store = await _seed_job_store("job-parallel")

    with _patch_segment_runner_to(n_total, seg_len), \
         patch("service.worker._make_s3", return_value=_FakeS3()), \
         patch("service.tracking_runner.run_tracking_job",
               side_effect=_fake_run_tracking_job), \
         patch.object(worker_mod, "LIFECYCLE_HEARTBEAT_INTERVAL", 0.01), \
         patch("service.segment_runner.stitch_segment_identity",
               side_effect=lambda prev, curr, overlap: curr), \
         patch("service.segment_runner.merge_tracking_results",
               return_value={"frames": []}), \
         patch.object(worker_mod, "_run_upscale_analysis",
                      return_value=({}, 30.0)):
        request = _make_request()
        loop = asyncio.get_event_loop()
        await worker_mod._run_parallel_segments(
            job_id="job-parallel",
            video_path=str(tmp_path / "video.mp4"),
            box_a=request.box_a,
            box_b=request.box_b,
            clip_start_frame=0,
            clip_end_resolved=clip_total_frames,
            clip_total_frames=clip_total_frames,
            config=config,
            request=request,
            work_dir=str(tmp_path),
            job_store=job_store,
            jobs_store=mock_jobs_store,
            loop=loop,
            eff_step_size=60,
            eff_max_history=8,
            eff_max_missing_frames=15,
            eff_frame_stride=1,
            eff_prop_stride=1,
            progress_floor=0.0,
        )

    track_writes = [c for c in progress_calls if c["stage"] == PipelineStage.TRACK]
    assert track_writes, (
        f"Aggregator wrote no tracking-stage progress. All calls: {progress_calls}"
    )
    # Last tracking write should reflect the final aggregated total.
    last = track_writes[-1]
    assert last["current_frame"] == clip_total_frames
    assert last["total_frames"] == clip_total_frames
    # Monotonic non-decreasing current_frame across all tracking writes.
    cf_seq = [c["current_frame"] for c in track_writes]
    assert all(b >= a for a, b in zip(cf_seq, cf_seq[1:])), (
        f"current_frame went backwards: {cf_seq}"
    )


# ---------------------------------------------------------------------------
# Test 2: aggregator never regresses below the seed value of 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_aggregator_never_regresses_below_seed(
    mock_jobs_store, tmp_path,
):
    """When the aggregator runs before any segment reports progress, it must
    still write clip_done >= 1 (matching the seed at run_job)."""

    progress_calls: list[dict] = []

    async def _capture(job_id, stage, pct, *, current_frame=None,
                       total_frames=None, stage_message=None):
        progress_calls.append({"stage": stage, "current_frame": current_frame})

    mock_jobs_store.update_progress = _capture

    from service import worker as worker_mod

    def _slow_tracking(**kwargs):
        # Sleep so the aggregator gets at least one tick with frames_done=0.
        import time as _time
        _time.sleep(0.05)
        out_path = os.path.join(kwargs["output_dir"], "tracking.json")
        with open(out_path, "w") as f:
            json.dump({"fps": 30.0, "frames": []}, f)
        return out_path

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        standard_segments=1,
    )
    job_store = await _seed_job_store("job-parallel")

    with _patch_segment_runner_to(1, 100), \
         patch("service.worker._make_s3", return_value=_FakeS3()), \
         patch("service.tracking_runner.run_tracking_job",
               side_effect=_slow_tracking), \
         patch.object(worker_mod, "LIFECYCLE_HEARTBEAT_INTERVAL", 0.01), \
         patch("service.segment_runner.stitch_segment_identity",
               side_effect=lambda prev, curr, overlap: curr), \
         patch("service.segment_runner.merge_tracking_results",
               return_value={"frames": []}), \
         patch.object(worker_mod, "_run_upscale_analysis",
                      return_value=({}, 30.0)):
        request = _make_request()
        loop = asyncio.get_event_loop()
        await worker_mod._run_parallel_segments(
            job_id="job-parallel",
            video_path=str(tmp_path / "video.mp4"),
            box_a=request.box_a, box_b=request.box_b,
            clip_start_frame=0,
            clip_end_resolved=100,
            clip_total_frames=100,
            config=config,
            request=request,
            work_dir=str(tmp_path),
            job_store=job_store,
            jobs_store=mock_jobs_store,
            loop=loop,
            eff_step_size=60, eff_max_history=8,
            eff_max_missing_frames=15,
            eff_frame_stride=1, eff_prop_stride=1,
            progress_floor=0.0,
        )

    track_writes = [c for c in progress_calls if c["stage"] == PipelineStage.TRACK]
    assert track_writes, "expected at least one heartbeat tracking-stage write"
    for c in track_writes:
        assert c["current_frame"] is not None and c["current_frame"] >= 1, (
            f"Aggregator regressed below seed: {c}"
        )


# ---------------------------------------------------------------------------
# Test 3: aggregator task is cancelled cleanly on segment failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_aggregator_clean_shutdown_on_segment_failure(
    mock_jobs_store, tmp_path,
):
    """If a segment raises, the aggregator task must not leak (no orphan
    task) and the exception must propagate to the caller."""

    from service import worker as worker_mod

    def _failing_tracking(**kwargs):
        raise RuntimeError("simulated tracking failure")

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        standard_segments=1,
    )
    job_store = await _seed_job_store("job-parallel")

    before = {t for t in asyncio.all_tasks()}

    with _patch_segment_runner_to(1, 100), \
         patch("service.worker._make_s3", return_value=_FakeS3()), \
         patch("service.tracking_runner.run_tracking_job",
               side_effect=_failing_tracking), \
         patch.object(worker_mod, "LIFECYCLE_HEARTBEAT_INTERVAL", 0.01):
        request = _make_request()
        loop = asyncio.get_event_loop()
        with pytest.raises(RuntimeError, match="simulated tracking failure"):
            await worker_mod._run_parallel_segments(
                job_id="job-parallel",
                video_path=str(tmp_path / "video.mp4"),
                box_a=request.box_a, box_b=request.box_b,
                clip_start_frame=0,
                clip_end_resolved=100,
                clip_total_frames=100,
                config=config,
                request=request,
                work_dir=str(tmp_path),
                job_store=job_store,
                jobs_store=mock_jobs_store,
                loop=loop,
                eff_step_size=60, eff_max_history=8,
                eff_max_missing_frames=15,
                eff_frame_stride=1, eff_prop_stride=1,
                progress_floor=0.0,
            )

    # Yield once so any pending cancellations finish.
    await asyncio.sleep(0)
    current = asyncio.current_task()
    after = {t for t in asyncio.all_tasks() if not t.done() and t is not current}
    leaked = after - before
    assert not leaked, f"Aggregator task leaked: {leaked}"
