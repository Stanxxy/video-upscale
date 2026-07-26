"""Checkpoint helpers for durable job resume decisions.

All builders emit the V1 envelope:
    {
        "schema_version": 1,
        "pending_detection": null | dict,
        "artifacts": dict,
        "worker_state": {...},
        ... stage-specific scalars ...
    }
"""

from service.checkpoints.builders import (
    build_annotate_completed,
    build_cancellation_checkpoint,
    build_detect_initial_pending,
    build_download_completed,
    build_highlight_chunk_completed,
    build_highlight_ingest_completed,
    build_highlight_publish_completed,
    build_highlight_publish_progress,
    build_publish_completed,
    build_replaced_by_new_job,
    build_track_completed,
    build_track_mid_loss,
    build_track_progress,
    build_tracking_started,
    build_upload_incremental,
    build_upscale_started,
    build_upscale_window_progress,
    build_verified_boxes_checkpoint,
    should_flush_analysis,
)
from service.checkpoints.constants import END_OF_TRACKING_SENTINEL, HIGHLIGHT_STAGE_ORDER, STAGE_ORDER
from service.checkpoints.highlight_resume import build_highlight_resume_plan
from service.checkpoints.envelope import WorkerStateSnapshot, make_envelope
from service.checkpoints.query import (
    checkpoint_by_stage,
    latest_checkpoint_data_by_stage,
    next_unprocessed_frame,
    select_correction_checkpoint,
    worker_state_from,
)
from service.checkpoints.resume import (
    ResumePlan,
    build_resume_overrides,
    build_resume_plan,
    resume_plan_to_request_fields,
)

__all__ = [
    "STAGE_ORDER",
    "HIGHLIGHT_STAGE_ORDER",
    "END_OF_TRACKING_SENTINEL",
    "WorkerStateSnapshot",
    "make_envelope",
    "build_download_completed",
    "build_detect_initial_pending",
    "build_tracking_started",
    "build_track_progress",
    "build_track_mid_loss",
    "build_track_completed",
    "build_upscale_started",
    "build_upscale_window_progress",
    "build_annotate_completed",
    "build_upload_incremental",
    "build_publish_completed",
    "should_flush_analysis",
    "build_replaced_by_new_job",
    "build_verified_boxes_checkpoint",
    "build_cancellation_checkpoint",
    "build_highlight_ingest_completed",
    "build_highlight_chunk_completed",
    "build_highlight_publish_completed",
    "build_highlight_publish_progress",
    "build_highlight_resume_plan",
    "latest_checkpoint_data_by_stage",
    "checkpoint_by_stage",
    "ResumePlan",
    "build_resume_plan",
    "resume_plan_to_request_fields",
    "select_correction_checkpoint",
    "next_unprocessed_frame",
    "worker_state_from",
    "build_resume_overrides",
]
