"""Progress math and background coroutine helpers."""
import asyncio
import logging
import os

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import WorkerStateSnapshot, build_track_progress
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import TrackRequest
from service.s3 import S3Client

from service.worker.gpu import _load_partial_tracking_dict

logger = logging.getLogger("service.worker")

PARTIAL_UPLOAD_INTERVAL = 30.0
LIFECYCLE_HEARTBEAT_INTERVAL = 1.0


def _log_progress_future_failure(future, *, context: str = "progress write"):
    """``add_done_callback`` for fire-and-forget ``run_coroutine_threadsafe``.

    Best-effort heartbeats (lifecycle update_progress, upscale progress) are
    scheduled from worker threads via ``run_coroutine_threadsafe`` and we do
    NOT block on ``future.result()`` for them — but the resulting Future will
    silently swallow any coroutine exception (Keyspaces timeout, serialization
    error, network blip) if nobody inspects it. Attach this callback so the
    exception lands in the engine log without raising.
    """
    try:
        exc = future.exception()
    except Exception:
        # Future was cancelled, or .exception() itself failed; nothing to log.
        return
    if exc is not None:
        logger.error(
            "background %s coroutine failed: %s",
            context,
            exc,
            exc_info=exc,
        )


def _schedule_background_coro(coro, loop, *, context: str):
    """Schedule a fire-and-forget coroutine and log any failure.

    Wraps the common pattern ``asyncio.run_coroutine_threadsafe(coro, loop)``
    followed by ``add_done_callback(_log_progress_future_failure)`` so callers
    can't forget the failure-logging hook.
    """
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    future.add_done_callback(
        lambda f, _ctx=context: _log_progress_future_failure(f, context=_ctx),
    )
    return future


def _tracking_progress_flags(
    now: float,
    last_ks_write: float,
    last_partial_upload: float,
    *,
    ks_interval: float = LIFECYCLE_HEARTBEAT_INTERVAL,
    partial_interval: float = PARTIAL_UPLOAD_INTERVAL,
) -> tuple[bool, bool]:
    """Decide whether the tracking-progress callback should write to the
    lifecycle row (1s cadence) and/or upload a partial-tracking checkpoint
    (30s cadence). Pure function — easy to unit test without a fake clock."""
    return (
        (now - last_ks_write) >= ks_interval,
        (now - last_partial_upload) >= partial_interval,
    )


def _make_worker_state(
    *,
    progress_percent: float,
    current_frame: int = 0,
    total_frames: int = 0,
    stage_progress_fraction: float = 0.0,
) -> WorkerStateSnapshot:
    """Snapshot the in-memory worker progress for a checkpoint write."""
    return WorkerStateSnapshot(
        progress_percent=progress_percent,
        current_frame=current_frame,
        total_frames=total_frames,
        stage_progress_fraction=stage_progress_fraction,
    )


def _pct_at_least(pct: float, floor: float) -> float:
    """Replacement jobs seed lifecycle progress — never regress below ``floor``."""
    return max(pct, floor)


def _video_frame_cap(video_path: str) -> int:
    """Total frames reported by OpenCV (at least 1)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        return max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 1)
    finally:
        cap.release()


def _resolved_clip_end_and_total(
    clip_start_frame: int,
    end_frame: int | None,
    video_frame_cap: int,
) -> tuple[int, int]:
    """Return ``(clip_end_resolved, clip_total_frames)`` for the requested clip."""
    raw_end = end_frame if end_frame is not None else video_frame_cap
    clip_end = min(max(raw_end, clip_start_frame + 1), video_frame_cap)
    clip_total = max(clip_end - clip_start_frame, 1)
    return clip_end, clip_total


def _clip_done_inclusive_through_global(
    global_idx: int,
    clip_start_frame: int,
    clip_total_frames: int,
) -> int:
    """1-based count of clip frames from ``clip_start`` through ``global_idx`` inclusive."""
    return min(max(global_idx - clip_start_frame + 1, 0), clip_total_frames)


def _tracking_pct_from_clip_done(
    clip_done: int,
    clip_total_frames: int,
    progress_floor: float,
) -> float:
    """Tracking-stage percent (15-55% band) from a 1-based clip-relative done count.

    Used by the sequential tracking progress callback (via global frame
    index, see ``_tracking_progress_pct_clip``). Tracking always runs
    sequentially; the upscale stage has its own band (55%-80%) and uses a
    different aggregator (``_run_parallel_upscale._aggregate_and_write_lifecycle``).
    """
    frac = clip_done / max(clip_total_frames, 1)
    return _pct_at_least(15.0 + frac * 40.0, progress_floor)


def _tracking_progress_pct_clip(
    global_idx: int,
    clip_start_frame: int,
    clip_total_frames: int,
    progress_floor: float,
) -> tuple[int, float]:
    """Return (clip_done_1based, tracking-stage percent) for Keyspaces/UI."""
    done = _clip_done_inclusive_through_global(
        global_idx, clip_start_frame, clip_total_frames,
    )
    pct = _tracking_pct_from_clip_done(done, clip_total_frames, progress_floor)
    return done, pct


def _last_global_frame_idx_from_tracking(track_data: dict) -> int | None:
    frames = track_data.get("frames") or []
    if not frames:
        return None
    last = frames[-1]
    if "frame_idx" in last:
        return int(last["frame_idx"])
    if "frame" in last:
        return int(last["frame"])
    return None


def _track_completed_clip_worker_state(
    track_data: dict,
    clip_start_frame: int,
    clip_total_frames: int,
    *,
    progress_percent: float,
) -> WorkerStateSnapshot:
    last_g = _last_global_frame_idx_from_tracking(track_data)
    if last_g is not None:
        done = _clip_done_inclusive_through_global(
            last_g, clip_start_frame, clip_total_frames,
        )
    else:
        done = clip_total_frames
    return _make_worker_state(
        progress_percent=progress_percent,
        current_frame=done,
        total_frames=clip_total_frames,
        stage_progress_fraction=(done / max(clip_total_frames, 1)),
    )


async def _update_tracking_progress(
    job_id: str,
    clip_done: int,
    clip_total: int,
    pct: float,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
    write_ks: bool = True,
):
    await job_store.update_job(
        job_id,
        progress_percent=pct,
        current_frame=clip_done,
        total_frames=clip_total,
    )
    if write_ks:
        await jobs_store.update_progress(
            job_id, PipelineStage.TRACK, pct,
            current_frame=clip_done, total_frames=clip_total,
        )


async def _update_tracking_progress_with_partial(
    job_id: str,
    clip_done: int,
    clip_total: int,
    pct: float,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
    request: TrackRequest,
    work_dir: str,
    s3: S3Client,
    *,
    resume_next_global: int,
    write_lifecycle: bool,
    upload_partial: bool,
):
    """Lifecycle progress + (every 30s) partial-tracking S3 upload + V1 checkpoint.

    The two cadences are independent: ``write_lifecycle`` is the 1-second
    Keyspaces lifecycle heartbeat; ``upload_partial`` is the 30-second
    partial-tracking durable checkpoint.

    ``clip_done`` / ``clip_total`` are 1-based position within the requested clip
    (same semantics as ``job_lifecycle.current_frame`` / ``total_frames``).
    ``resume_next_global`` is the absolute video frame index to resume SAM2 at
    (written into checkpoint artifacts).
    """
    if write_lifecycle:
        await _update_tracking_progress(
            job_id, clip_done, clip_total, pct, job_store, jobs_store, write_ks=True,
        )
    if not upload_partial:
        return
    tracking_json_path = os.path.join(work_dir, "tracking", "tracking.json")
    if not os.path.isfile(tracking_json_path):
        return
    partial_key = f"checkpoints/{job_id}/partial_tracking.json"
    upload_bucket = request.output_bucket or request.bucket
    try:
        partial_data = _load_partial_tracking_dict(tracking_json_path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, s3.upload_json, partial_data, upload_bucket, partial_key,
        )
    except Exception as e:
        logger.warning("Periodic partial-tracking upload failed: %s", e)
        return
    ws = _make_worker_state(
        progress_percent=pct,
        current_frame=clip_done,
        total_frames=clip_total,
        stage_progress_fraction=(clip_done / max(clip_total, 1)),
    )
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.TRACK, False,
        build_track_progress(
            partial_tracking_s3_key=partial_key,
            resume_from_frame=resume_next_global,
            worker_state=ws,
        ),
    )
