"""
Human verification/correction of detected bounding boxes.

Shows RF-DETR detections overlaid on a video frame. The user can:
- Accept specific detections by index
- Manually redraw boxes if detections are wrong (e.g. intertwined athletes)

Usage:
    python select_boxes.py --video ../input_video.mp4 --frame 0
"""
import argparse
import cv2
import numpy as np


COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]


def draw_detections(frame, detections):
    """Draw numbered detection boxes on a frame."""
    viz = frame.copy()
    for i, det in enumerate(detections):
        box = det["box"]
        x1, y1, x2, y2 = [int(c) for c in box]
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)

        label = f"[{i}] conf={det.get('confidence', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(viz, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(viz, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return viz


def select_boxes_from_detections(frame, detections):
    """
    Show detections and let user accept or redraw.

    Args:
        frame: BGR frame (numpy array)
        detections: List of {"box": [x1,y1,x2,y2], "confidence": float, ...}

    Returns:
        List of 2 dicts: [{"box": [...], "track_id": 1}, {"box": [...], "track_id": 2}]
        or None if cancelled.
    """
    if not detections:
        print("[select] No detections to verify. Drawing manually.")
        return manual_draw_boxes(frame)

    viz = draw_detections(frame, detections)

    # Add instructions
    instructions = [
        "DETECTION VERIFICATION",
        f"Found {len(detections)} person(s). Options:",
        "  Press digit keys to select Athlete A, then Athlete B",
        "  Press 'm' to manually draw both boxes",
        "  Press 'q' or ESC to cancel",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(viz, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Verify Detections", viz)
    cv2.setWindowProperty("Verify Detections", cv2.WND_PROP_TOPMOST, 1)

    selected = []
    print(f"\n[select] {len(detections)} detection(s) shown.")
    print("[select] Press digit key to select Athlete A, then Athlete B.")
    print("[select] Press 'm' to manually draw. Press 'q'/ESC to cancel.")

    while len(selected) < 2:
        key = cv2.waitKey(0) & 0xFF

        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return None

        if key == ord('m'):
            cv2.destroyAllWindows()
            return manual_draw_boxes(frame)

        # Digit keys 0-9
        if ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx < len(detections):
                if idx not in [s["_idx"] for s in selected]:
                    track_id = len(selected) + 1
                    selected.append({
                        "box": detections[idx]["box"],
                        "track_id": track_id,
                        "_idx": idx,
                    })
                    label = "A" if track_id == 1 else "B"
                    print(f"[select] Athlete {label} = detection [{idx}]")

                    # Redraw with selection highlighted
                    viz2 = draw_detections(frame, detections)
                    for s in selected:
                        si = s["_idx"]
                        box = detections[si]["box"]
                        x1, y1, x2, y2 = [int(c) for c in box]
                        label_text = f"Athlete {'A' if s['track_id'] == 1 else 'B'}"
                        cv2.rectangle(viz2, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        cv2.putText(viz2, label_text, (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2)
                    cv2.imshow("Verify Detections", viz2)
                else:
                    print(f"[select] Detection [{idx}] already selected.")
            else:
                print(f"[select] No detection [{idx}]. Max index: {len(detections) - 1}")

    cv2.destroyAllWindows()

    # Clean up internal keys
    result = []
    for s in selected:
        result.append({
            "box": [round(c, 1) for c in s["box"]],
            "track_id": s["track_id"],
        })
    return result


def manual_draw_boxes(frame):
    """Let user draw two boxes manually using cv2.selectROI."""
    print("\n[select] MANUAL MODE: Draw bounding box for Athlete A")
    print("[select] Click and drag to draw. Press ENTER to confirm, ESC to cancel.")

    box_a = cv2.selectROI("Draw Athlete A", frame, fromCenter=False,
                          showCrosshair=True)
    cv2.destroyAllWindows()

    if box_a == (0, 0, 0, 0):
        print("[select] Cancelled.")
        return None

    print("[select] Draw bounding box for Athlete B")
    # Show frame with Athlete A drawn
    viz = frame.copy()
    x, y, w, h = box_a
    cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(viz, "Athlete A", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    box_b = cv2.selectROI("Draw Athlete B", viz, fromCenter=False,
                          showCrosshair=True)
    cv2.destroyAllWindows()

    if box_b == (0, 0, 0, 0):
        print("[select] Cancelled.")
        return None

    def roi_to_xyxy(roi):
        x, y, w, h = roi
        return [int(x), int(y), int(x + w), int(y + h)]

    return [
        {"box": roi_to_xyxy(box_a), "track_id": 1},
        {"box": roi_to_xyxy(box_b), "track_id": 2},
    ]


def read_frame(video_path, frame_idx=0):
    """Read a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Box Selection")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    args = parser.parse_args()

    frame = read_frame(args.video, args.frame)
    result = manual_draw_boxes(frame)

    if result:
        for r in result:
            label = "A" if r["track_id"] == 1 else "B"
            print(f"Athlete {label}: {r['box']}")
    else:
        print("Selection cancelled.")
