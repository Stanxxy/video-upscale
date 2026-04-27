"""
BJJ Athlete Tracking Pipeline: YOLO26 detect -> User verify -> SAM2 track.

Orchestrates:
1. YOLO26 person detection on initial frame
2. Human verification / manual box selection
3. SAM2 tracking with user intervention on track loss

Usage (from repo root):
    python -m tracking_pipeline.pipeline --video ../input_video.mp4 --start_time 0:04 --end_time 0:30
    python -m tracking_pipeline --video ../input_video.mp4 --auto
    python -m tracking_pipeline.pipeline --video ../input_video.mp4 --cpu --step_size 30
"""
import argparse
import json
import os
import time

import cv2

from .detect import detect_persons
from .select_boxes import select_boxes_from_detections, manual_draw_boxes, read_frame
from .hybrid_tracking import run_tracking


# --- SAM2 model ID mapping ---
SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def parse_time(time_str):
    """Parse MM:SS or HH:MM:SS time string to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid time format '{time_str}'. Use MM:SS or HH:MM:SS")


def _format_time(seconds):
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _get_video_info(video_path):
    """Get video FPS and total frame count."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def run_pipeline(
    video_path,
    detection_frame=0,
    detection_threshold=0.5,
    yolo_model="yolo26m",
    sam2_model="base_plus",
    step_size=60,
    max_history=8,
    max_missing_frames=15,
    force_cpu=False,
    output_dir="output",
    auto=False,
    start_time=None,
    end_time=None,
):
    """
    Full pipeline: detect -> verify -> track.

    Args:
        video_path: Path to input video.
        detection_frame: Frame offset for initial detection (relative to start).
        detection_threshold: YOLO26 confidence threshold.
        yolo_model: YOLO26 model variant ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x").
        sam2_model: SAM2 model key ("tiny", "small", "base_plus", "large").
        step_size: SAM2 propagation chunk size (frames).
        max_history: SAM2 memory pruning window.
        max_missing_frames: Frames of mask collapse before triggering YOLO + user intervention.
        force_cpu: Force CPU device.
        output_dir: Directory for outputs.
        auto: Skip human verification, auto-select top 2 detections.
        start_time: Start time in seconds (from MM:SS input).
        end_time: End time in seconds (from MM:SS input).
    """
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # --- Compute frame range ---
    fps, total_frames = _get_video_info(video_path)
    start_frame = 0
    end_frame = None

    if start_time is not None:
        start_frame = int(start_time * fps)
        start_frame = min(start_frame, total_frames - 1)
    if end_time is not None:
        end_frame = int(end_time * fps)
        end_frame = min(end_frame, total_frames)

    duration = total_frames / fps
    print(f"[pipeline] Video: {_format_time(duration)} "
          f"({total_frames} frames @ {fps:.1f} fps)")
    if start_time is not None or end_time is not None:
        print(f"[pipeline] Time range: "
              f"{_format_time(start_time or 0)} - "
              f"{_format_time(end_time or duration)} "
              f"(frames {start_frame} - {end_frame or total_frames})")

    abs_detection_frame = start_frame + detection_frame

    # ===========================================================
    # STEP 1: YOLO26 Person Detection
    # ===========================================================
    print()
    print("=" * 60)
    print("STEP 1: YOLO26 Person Detection")
    print("=" * 60)

    persons, (h, w) = detect_persons(
        video_path, abs_detection_frame, detection_threshold,
        yolo_model, force_cpu,
    )

    det_path = os.path.join(output_dir, "detections.json")
    with open(det_path, "w") as f:
        json.dump({
            "video": video_path,
            "frame_idx": abs_detection_frame,
            "video_size": [w, h],
            "persons": persons,
        }, f, indent=2)

    if persons:
        print(f"[pipeline] {len(persons)} person(s) detected. "
              f"Saved to {det_path}")
    else:
        print("[pipeline] No persons detected by YOLO26. "
              "Falling back to manual selection.")

    # ===========================================================
    # STEP 2: Verify / Select Athletes
    # ===========================================================
    print()
    print("=" * 60)
    print("STEP 2: Verify / Select Athletes")
    print("=" * 60)

    frame = read_frame(video_path, abs_detection_frame)

    if auto and persons:
        # Auto mode: pick top 2 by confidence
        sorted_persons = sorted(
            persons, key=lambda p: p["confidence"], reverse=True,
        )
        verified = []
        for i, p in enumerate(sorted_persons[:2]):
            verified.append({
                "box": p["box"],
                "track_id": i + 1,
            })
        print(f"[pipeline] Auto-selected {len(verified)} athlete(s)")
    elif auto and not persons:
        print("[pipeline] Auto mode but no detections — switching to manual.")
        verified = manual_draw_boxes(frame)
        if verified is None:
            print("[pipeline] Selection cancelled.")
            return None
    else:
        # Interactive verification (handles empty list -> manual draw)
        verified = select_boxes_from_detections(frame, persons)
        if verified is None:
            print("[pipeline] Selection cancelled.")
            return None

    for v in verified:
        label = "A" if v["track_id"] == 1 else "B"
        print(f"[pipeline] Athlete {label}: box={v['box']}")

    # Save verified boxes
    verified_path = os.path.join(output_dir, "verified_boxes.json")
    with open(verified_path, "w") as f:
        json.dump(verified, f, indent=2)

    # Extract boxes for tracking
    box_a = verified[0]["box"] if len(verified) >= 1 else None
    box_b = verified[1]["box"] if len(verified) >= 2 else None

    if box_a is None or box_b is None:
        print("[pipeline] Need exactly 2 athlete boxes. Got "
              f"{len(verified)}. Aborting.")
        return None

    # ===========================================================
    # STEP 3: SAM2 Tracking (YOLO26 on-demand for track loss)
    # ===========================================================
    print()
    print("=" * 60)
    print("STEP 3: Hybrid Tracking")
    print("=" * 60)

    sam2_model_id = SAM2_MODELS.get(sam2_model, sam2_model)

    json_path = run_tracking(
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
    )

    # ===========================================================
    # DONE
    # ===========================================================
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"Outputs: {output_dir}/")
    print(f"  detections.json     - YOLO26 initial detection results")
    print(f"  verified_boxes.json - User-verified athlete boxes")
    print(f"  tracking.json       - Per-frame tracking data")
    print(f"  tracked_output.mp4  - Annotated video with masks + boxes")
    print(f"  debug_frames/       - Debug frame samples")
    print(f"{'=' * 60}")

    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="BJJ Athlete Tracking: YOLO26 + SAM2 + State Machine",
    )
    parser.add_argument("--video", required=True,
                        help="Path to input video")
    parser.add_argument("--detection_frame", type=int, default=0,
                        help="Frame offset for initial detection (relative to start)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="YOLO26 detection confidence threshold")
    parser.add_argument("--yolo_model", default="yolo26m",
                        choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
                        help="YOLO26 model variant (default: yolo26m)")
    parser.add_argument("--sam2_model", default="base_plus",
                        choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--start_time", type=parse_time, default=None,
                        help="Start time as MM:SS (e.g. 1:30)")
    parser.add_argument("--end_time", type=parse_time, default=None,
                        help="End time as MM:SS (e.g. 3:00)")
    parser.add_argument("--step_size", type=int, default=60,
                        help="SAM2 propagation chunk size (frames)")
    parser.add_argument("--max_history", type=int, default=8,
                        help="SAM2 memory pruning window")
    parser.add_argument("--max_missing_frames", type=int, default=15,
                        help="Frames of mask collapse before triggering YOLO + user intervention")
    parser.add_argument("--auto", action="store_true",
                        help="Skip verification, auto-select top 2 detections")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU device")
    parser.add_argument("--output_dir", default="output",
                        help="Output directory")
    args = parser.parse_args()

    run_pipeline(
        args.video,
        detection_frame=args.detection_frame,
        detection_threshold=args.threshold,
        yolo_model=args.yolo_model,
        sam2_model=args.sam2_model,
        step_size=args.step_size,
        max_history=args.max_history,
        max_missing_frames=args.max_missing_frames,
        force_cpu=args.cpu,
        output_dir=args.output_dir,
        auto=args.auto,
        start_time=args.start_time,
        end_time=args.end_time,
    )


if __name__ == "__main__":
    main()
