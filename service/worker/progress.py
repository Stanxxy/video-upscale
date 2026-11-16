"""Worker progress snapshot helper shared by the highlight job."""
from service.checkpoints import WorkerStateSnapshot


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
