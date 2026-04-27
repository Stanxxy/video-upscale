"""
YOLO26 person detection for initial frame(s).

Usage:
    python detect.py --video ../input_video.mp4 --frame 0 --threshold 0.5
    python detect.py --video ../input_video.mp4 --frame 0 --cpu
"""
import argparse
import json
import os
import time

import cv2
from ultralytics import YOLO

from .device import get_device

PERSON_CLASS_ID = 0  # COCO class 0 = person


def detect_persons(video_path, frame_idx=0, threshold=0.5, yolo_model="yolo26m",
                   force_cpu=False):
    """
    Run YOLO26 on a single frame, filter to person detections only.

    Returns:
        persons: list of {"box": [x1,y1,x2,y2], "confidence": float, "track_id": int}
        frame_shape: (height, width)
    """
    device = get_device(force_cpu)
    print(f"[detect] Device: {device}")

    # Load video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    model_name = yolo_model if yolo_model.endswith(".pt") else f"{yolo_model}.pt"
    print(f"[detect] Loading {model_name}...")
    t0 = time.time()

    try:
        model = YOLO(model_name)
        results = model(
            frame,
            conf=threshold,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            device=str(device),
        )[0]
        print(f"[detect] Inference took {time.time() - t0:.2f}s")

    except Exception as e:
        print(f"[detect] YOLO26 failed: {e}")
        if not force_cpu:
            print("[detect] Retrying on CPU...")
            return detect_persons(video_path, frame_idx, threshold, yolo_model,
                                  force_cpu=True)
        raise

    # Parse results — classes=[0] already filters to persons
    persons = []
    boxes = results.boxes
    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().tolist()
        conf = float(boxes.conf[i].cpu())
        persons.append({
            "box": [round(c, 1) for c in box],
            "confidence": round(conf, 4),
            "track_id": len(persons) + 1,
        })

    print(f"[detect] Found {len(persons)} person(s) (threshold={threshold})")
    for p in persons:
        print(f"  ID {p['track_id']}: box={p['box']} conf={p['confidence']}")

    return persons, frame.shape[:2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 Person Detection")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to detect on")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--yolo_model", default="yolo26m",
                        choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--output", default="output/detections.json")
    args = parser.parse_args()

    persons, (h, w) = detect_persons(args.video, args.frame, args.threshold,
                                     args.yolo_model, args.cpu)

    result = {
        "video": args.video,
        "frame_idx": args.frame,
        "video_size": [w, h],
        "persons": persons,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[detect] Saved to {args.output}")
