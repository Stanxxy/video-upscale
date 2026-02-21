"""
Full pipeline: RF-DETR detect -> SAM 2.1 track -> visualize.

Usage:
    python pipeline.py --video ../input_video.mp4 --max_frames 100
    python pipeline.py --video ../input_video.mp4 --max_frames 300 --no_visualize
    python pipeline.py --video ../input_video.mp4 --cpu
"""
import argparse
import json
import os
import time

from detect import detect_persons
from track import track_objects
from visualize import render_annotated_video, render_tracking_summary


def run_pipeline(video_path, detection_frame=0, detection_threshold=0.5,
                 rfdetr_size="base", sam2_model="tiny", max_frames=None,
                 sampling_rate=1, force_cpu=False, visualize=True,
                 output_dir="output"):
    """Full detect-then-track pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # Step 1: Detect
    print("=" * 60)
    print("STEP 1: RF-DETR Person Detection")
    print("=" * 60)
    persons, (h, w) = detect_persons(
        video_path, detection_frame, detection_threshold, rfdetr_size,
        force_cpu,
    )

    if not persons:
        print("[pipeline] No persons detected. Try lowering --threshold.")
        return None

    det_path = os.path.join(output_dir, "detections.json")
    with open(det_path, "w") as f:
        json.dump({
            "video": video_path,
            "frame_idx": detection_frame,
            "video_size": [w, h],
            "persons": persons,
        }, f, indent=2)
    print(f"[pipeline] Detections saved to {det_path}")

    # Step 2: Track
    print()
    print("=" * 60)
    print("STEP 2: SAM 2.1 Video Object Tracking")
    print("=" * 60)
    tracking_json, raw_results = track_objects(
        video_path, persons, sam2_model, max_frames, sampling_rate, force_cpu,
    )

    track_path = os.path.join(output_dir, "tracking.json")
    with open(track_path, "w") as f:
        json.dump(tracking_json, f, indent=2)
    print(f"[pipeline] Tracking saved to {track_path}")

    # Step 3: Visualize
    if visualize:
        print()
        print("=" * 60)
        print("STEP 3: Visualization")
        print("=" * 60)

        video_out = os.path.join(output_dir, "tracked_output.mp4")
        render_annotated_video(video_path, raw_results, video_out,
                               max_frames, sampling_rate)

        summary_out = os.path.join(output_dir, "tracking_summary.jpg")
        render_tracking_summary(video_path, raw_results, summary_out)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Outputs: {output_dir}/")
    print(f"  detections.json  - RF-DETR results")
    print(f"  tracking.json    - Per-frame tracking data")
    if visualize:
        print(f"  tracked_output.mp4    - Annotated video")
        print(f"  tracking_summary.jpg  - Sample frame grid")
    print(f"{'=' * 60}")

    return tracking_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RF-DETR + SAM 2.1 BJJ Athlete Tracking Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--detection_frame", type=int, default=0,
                        help="Frame to run detection on")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--rfdetr_size", default="base",
                        choices=["base", "large"])
    parser.add_argument("--sam2_model", default="base_plus",
                        choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames to process")
    parser.add_argument("--sampling_rate", type=int, default=1,
                        help="Process every Nth frame")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Skip visualization step")
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    run_pipeline(
        args.video,
        detection_frame=args.detection_frame,
        detection_threshold=args.threshold,
        rfdetr_size=args.rfdetr_size,
        sam2_model=args.sam2_model,
        max_frames=args.max_frames,
        sampling_rate=args.sampling_rate,
        force_cpu=args.cpu,
        visualize=not args.no_visualize,
        output_dir=args.output_dir,
    )
