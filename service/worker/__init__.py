"""
Unified pipeline worker: detect → verify → (track → detect)[inloop] → upscale → upload → publish.

Package re-exports preserve ``from service.worker import …`` and
``from service import worker`` compatibility for tests and callers.
"""
from service.sns import SNSPublisher

from service.worker.callbacks.analysis_checkpoint import _flush_analysis_checkpoint
from service.worker.callbacks.detection import _make_detection_cb
from service.worker.gpu import _ensure_models_released, _load_partial_tracking_dict
from service.worker.helpers import _is_cancelled, _make_s3
from service.worker.orchestrator import run_job
from service.worker.progress import (
    LIFECYCLE_HEARTBEAT_INTERVAL,
    PARTIAL_UPLOAD_INTERVAL,
    _clip_done_inclusive_through_global,
    _last_global_frame_idx_from_tracking,
    _log_progress_future_failure,
    _make_worker_state,
    _pct_at_least,
    _resolved_clip_end_and_total,
    _schedule_background_coro,
    _track_completed_clip_worker_state,
    _tracking_pct_from_clip_done,
    _tracking_progress_flags,
    _tracking_progress_pct_clip,
    _update_tracking_progress,
    _update_tracking_progress_with_partial,
    _video_frame_cap,
)
from service.worker.stages.parallel_upscale import _run_parallel_upscale
from service.worker.stages.upscale import _parse_time_range, _run_upscale_analysis

__all__ = [
    "SNSPublisher",
    "run_job",
    "PARTIAL_UPLOAD_INTERVAL",
    "LIFECYCLE_HEARTBEAT_INTERVAL",
    "_make_s3",
    "_is_cancelled",
    "_parse_time_range",
    "_run_upscale_analysis",
    "_run_parallel_upscale",
    "_make_detection_cb",
    "_flush_analysis_checkpoint",
    "_update_tracking_progress_with_partial",
    "_update_tracking_progress",
    "_ensure_models_released",
    "_load_partial_tracking_dict",
    "_log_progress_future_failure",
    "_schedule_background_coro",
    "_tracking_progress_flags",
    "_make_worker_state",
    "_pct_at_least",
    "_video_frame_cap",
    "_resolved_clip_end_and_total",
    "_clip_done_inclusive_through_global",
    "_tracking_pct_from_clip_done",
    "_tracking_progress_pct_clip",
    "_last_global_frame_idx_from_tracking",
    "_track_completed_clip_worker_state",
]
