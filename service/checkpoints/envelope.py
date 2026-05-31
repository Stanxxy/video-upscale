"""V1 checkpoint envelope helpers."""

from __future__ import annotations

from typing import Any, NamedTuple


class WorkerStateSnapshot(NamedTuple):
    """In-memory worker progress at the time a checkpoint is written."""

    progress_percent: float
    current_frame: int
    total_frames: int
    stage_progress_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "progress_percent": float(self.progress_percent),
            "current_frame": int(self.current_frame),
            "total_frames": int(self.total_frames),
            "stage_progress_fraction": float(self.stage_progress_fraction),
        }


def make_envelope(
    *,
    worker_state: WorkerStateSnapshot,
    pending_detection: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    **extras: Any,
) -> dict[str, Any]:
    cp: dict[str, Any] = {
        "schema_version": 1,
        "pending_detection": pending_detection,
        "artifacts": dict(artifacts or {}),
        "worker_state": worker_state.to_dict(),
    }
    cp.update(extras)
    return cp
