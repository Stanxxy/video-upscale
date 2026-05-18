"""
Bridge between the tracking_pipeline and the service worker.

Provides:
  - run_detect()              — YOLO26 detection on a single frame
  - run_tracking_job()        — wraps run_tracking() with progress + detection callbacks
  - normalize_tracking_output() — normalises tracking_pipeline output for service/worker.py
"""
import logging
import os
from typing import Callable, Optional

import cv2

from service.models import JobCancelledError

logger = logging.getLogger(__name__)

from tracking import detect_persons as _detect_persons, run_tracking as _run_tracking


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def run_detect(
    video_path: str,
    frame_idx: int = 0,
    threshold: float = 0.5,
    yolo_model: str = "yolo26m",
) -> list[dict]:
    """
    Run YOLO26 on a single frame.

    Returns a list of candidates:
        [{"candidate_id": int, "box": [x1,y1,x2,y2], "confidence": float}, ...]
    Sorted by descending confidence.
    """
    try:
        persons, _ = _detect_persons(
            video_path,
            frame_idx=frame_idx,
            threshold=threshold,
            yolo_model=yolo_model,
        )
    except Exception as e:
        logger.warning("run_detect failed: %s", e)
        return []

    return [
        {
            "candidate_id": i,
            "box": p["box"],
            "confidence": p["confidence"],
        }
        for i, p in enumerate(
            sorted(persons, key=lambda p: p["confidence"], reverse=True)
        )
    ]


def capture_frame_jpeg(video_path: str, frame_idx: int) -> Optional[bytes]:
    """Read a single frame and encode it as JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Tracking job
# ---------------------------------------------------------------------------

def run_tracking_job(
    video_path: str,
    box_a: list,
    box_b: list,
    output_dir: str,
    *,
    sam2_model_id: str = "facebook/sam2.1-hiera-base-plus",
    yolo_model: str = "yolo26m",
    detection_threshold: float = 0.5,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step_size: int = 60,
    max_history: int = 8,
    max_missing_frames: int = 15,
    force_cpu: bool = False,
    frame_stride: int = 1,
    prop_stride: int = 1,
    enable_pose: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    detection_cb: Optional[Callable[..., Optional[tuple]]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Run the SAM2 tracking loop with on-demand YOLO for track loss.

    Args:
        video_path: Path to the downloaded video file.
        box_a: [x1, y1, x2, y2] for athlete A (confirmed by human).
        box_b: [x1, y1, x2, y2] for athlete B (confirmed by human).
        output_dir: Directory to write tracking.json and debug artefacts.
        max_missing_frames: Frames of mask collapse before triggering user intervention.
        progress_cb: Called with (frames_done, total_frames, global_idx) on the
            first processed frame and every 30 frames thereafter.
        detection_cb: Called with (reason, frame_jpeg_bytes, **kwargs) when mid-tracking
            human input is needed. kwargs includes yolo_detections=[{box, confidence}, ...].
            Return (box_a, box_b), None, or raise HumanVerificationSuspend after persisting
            suspend state so tracking exits immediately (replacement job can run).
        should_stop: Optional callable() -> bool. If True, tracking exits cleanly
            and JobCancelledError is raised (e.g. when DELETE /job/{id} is called).

    Returns:
        Absolute path to tracking.json.

    Raises:
        JobCancelledError if should_stop returned True (job cancelled).
        RuntimeError if run_tracking() returns None for other reasons.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_path = _run_tracking(
        video_path=video_path,
        output_dir=output_dir,
        box_a=box_a,
        box_b=box_b,
        start_frame=start_frame,
        end_frame=end_frame,
        sam2_model_id=sam2_model_id,
        step_size=step_size,
        max_history=max_history,
        detection_threshold=detection_threshold,
        yolo_model=yolo_model,
        max_missing_frames=max_missing_frames,
        force_cpu=force_cpu,
        save_json=True,
        detection_callback=detection_cb,
        progress_callback=progress_cb,
        should_stop=should_stop,
        frame_stride=frame_stride,
        prop_stride=prop_stride,
        enable_pose=enable_pose,
    )

    if json_path is None:
        # should_stop → cancel; HumanVerificationSuspend in pipeline → suspend;
        # detection_cb returned None without suspend → suspend (legacy).
        # Return None so worker raises JobSuspendedError except on cancel.
        if should_stop and should_stop():
            raise JobCancelledError("Tracking stopped by user (job cancelled)")
        return None

    return os.path.abspath(json_path)


# ---------------------------------------------------------------------------
# Schema adapter
# ---------------------------------------------------------------------------

def normalize_tracking_output(raw: dict) -> dict:
    """
    Convert tracking_pipeline JSON schema → service/worker.py schema.

    Changes:
      - frame_idx  → frame
      - drops: local_idx, state, iou, source (per-athlete)

    The upscale pipeline expects:
        {"video": ..., "fps": ..., "frames": [{"frame": N, "timestamp": T,
          "athletes": [{"track_id": K, "box": [...], "keypoints": [...]}]}]}
    """
    adapted_frames = []
    for entry in raw.get("frames", []):
        athletes = []
        for a in entry.get("athletes", []):
            athletes.append({
                "track_id": a["track_id"],
                "box": a["box"],
                "keypoints": a.get("keypoints"),
            })
        adapted_frames.append({
            "frame": entry["frame_idx"],
            "timestamp": entry.get("timestamp", 0.0),
            "athletes": athletes,
        })

    return {
        "video": raw.get("video", ""),
        "fps": raw.get("fps", 30.0),
        "frames": adapted_frames,
    }
