"""Checkpoint read/query helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from service.analysis_keyspaces_enums import PipelineStage

from service.checkpoints.constants import STAGE_ORDER
from service.checkpoints.envelope import WorkerStateSnapshot


def _checkpoint_ts(cp: dict[str, Any]) -> datetime | None:
    """Parse updated_at for ordering duplicate stage rows."""
    ts = cp.get("updated_at")
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    return None


def latest_checkpoint_data_by_stage(
    checkpoints: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pick the newest checkpoint row per ``stage_name`` by ``updated_at``.

    Keyspaces may contain multiple INSERTs per (job_id, stage_name); arbitrary
    iteration order would pick the wrong row. Rows without ``updated_at`` sort
    first so explicit timestamps win.
    """
    def sort_key(cp: dict[str, Any]) -> tuple[float, int]:
        ts = _checkpoint_ts(cp)
        if ts is None:
            return (0.0, id(cp))
        return (ts.timestamp(), id(cp))

    sorted_cps = sorted(checkpoints, key=sort_key)
    out: dict[str, dict[str, Any]] = {}
    for cp in sorted_cps:
        out[cp["stage_name"]] = cp.get("checkpoint_data") or {}
    return out


def checkpoint_by_stage(checkpoints: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return checkpoint data keyed by stage name (latest row per stage)."""
    return latest_checkpoint_data_by_stage(checkpoints)


def next_unprocessed_frame(checkpoint_data: dict[str, Any]) -> int:
    """Resolve the resume frame cursor; prefers V1 artifact location."""
    artifacts = checkpoint_data.get("artifacts") or {}
    artifact_frame = artifacts.get("resume_from_frame")
    if artifact_frame is not None:
        return int(artifact_frame)

    resume_cursor = checkpoint_data.get("resume_cursor") or {}
    cursor_frame = resume_cursor.get("frame_idx")
    if cursor_frame is not None:
        return int(cursor_frame)

    pending = checkpoint_data.get("pending_detection") or {}
    pending_frame = pending.get("frame_idx")
    if pending_frame is not None:
        return int(pending_frame)

    return int(checkpoint_data.get("frame_count", 0) or 0)


def worker_state_from(checkpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest pipeline-stage checkpoint's worker_state block, if any."""
    by_stage = latest_checkpoint_data_by_stage(checkpoints)
    for stage in reversed(STAGE_ORDER):
        data = by_stage.get(stage.value)
        if data and data.get("worker_state"):
            return dict(data["worker_state"])
    return None


def select_correction_checkpoint(
    checkpoints: list[dict[str, Any]],
) -> tuple[PipelineStage | None, dict[str, Any]]:
    """Prefer mid-track correction context, then initial detection context.

    Reads `artifacts.partial_tracking_s3_key` first (V1 schema) with root-level
    fallback for back-compat.
    """
    by_stage = checkpoint_by_stage(checkpoints)
    track_cp = by_stage.get(PipelineStage.TRACK.value, {})
    track_artifacts = track_cp.get("artifacts") or {}
    if (
        track_cp.get("pending_detection")
        or track_artifacts.get("partial_tracking_s3_key")
        or track_cp.get("partial_tracking_s3_key")
    ):
        return PipelineStage.TRACK, track_cp

    detect_cp = by_stage.get(PipelineStage.DETECT.value, {})
    if detect_cp.get("pending_detection"):
        return PipelineStage.DETECT, detect_cp

    return None, {}
