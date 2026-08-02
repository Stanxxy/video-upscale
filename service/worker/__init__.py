"""Lazy compatibility exports for the dormant legacy tracking worker.

The default service schedules only ``highlight_orchestrator.run_highlight_job``.
Legacy worker symbols remain available for the optional GPU profile without
loading that profile during Gemini-native service startup.
"""
from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "SNSPublisher": ("service.sns", "SNSPublisher"),
    "run_job": ("service.worker.orchestrator", "run_job"),
    "run_highlight_job": ("service.worker.highlight_orchestrator", "run_highlight_job"),
    "_flush_analysis_checkpoint": ("service.worker.callbacks.analysis_checkpoint", "_flush_analysis_checkpoint"),
    "_make_detection_cb": ("service.worker.callbacks.detection", "_make_detection_cb"),
    "_ensure_models_released": ("service.worker.gpu", "_ensure_models_released"),
    "_load_partial_tracking_dict": ("service.worker.gpu", "_load_partial_tracking_dict"),
    "_make_s3": ("service.worker.helpers", "_make_s3"),
    "_is_cancelled": ("service.worker.helpers", "_is_cancelled"),
    "_run_parallel_upscale": ("service.worker.stages.parallel_upscale", "_run_parallel_upscale"),
    "_parse_time_range": ("service.worker.stages.upscale", "_parse_time_range"),
    "_run_upscale_analysis": ("service.worker.stages.upscale", "_run_upscale_analysis"),
    "LIFECYCLE_HEARTBEAT_INTERVAL": ("service.worker.progress", "LIFECYCLE_HEARTBEAT_INTERVAL"),
    "PARTIAL_UPLOAD_INTERVAL": ("service.worker.progress", "PARTIAL_UPLOAD_INTERVAL"),
    "_clip_done_inclusive_through_global": ("service.worker.progress", "_clip_done_inclusive_through_global"),
    "_last_global_frame_idx_from_tracking": ("service.worker.progress", "_last_global_frame_idx_from_tracking"),
    "_log_progress_future_failure": ("service.worker.progress", "_log_progress_future_failure"),
    "_make_worker_state": ("service.worker.progress", "_make_worker_state"),
    "_pct_at_least": ("service.worker.progress", "_pct_at_least"),
    "_resolved_clip_end_and_total": ("service.worker.progress", "_resolved_clip_end_and_total"),
    "_schedule_background_coro": ("service.worker.progress", "_schedule_background_coro"),
    "_track_completed_clip_worker_state": ("service.worker.progress", "_track_completed_clip_worker_state"),
    "_tracking_pct_from_clip_done": ("service.worker.progress", "_tracking_pct_from_clip_done"),
    "_tracking_progress_flags": ("service.worker.progress", "_tracking_progress_flags"),
    "_tracking_progress_pct_clip": ("service.worker.progress", "_tracking_progress_pct_clip"),
    "_update_tracking_progress": ("service.worker.progress", "_update_tracking_progress"),
    "_update_tracking_progress_with_partial": ("service.worker.progress", "_update_tracking_progress_with_partial"),
    "_video_frame_cap": ("service.worker.progress", "_video_frame_cap"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
