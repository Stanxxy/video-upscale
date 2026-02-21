"""
Visualization utilities for tracking results.
Renders annotated frames with bounding boxes, masks, and track IDs.
"""
import cv2
import numpy as np
from tqdm import tqdm

# Distinct colors for tracked objects (BGR for OpenCV)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]


def overlay_mask(frame, mask, color, alpha=0.3):
    """Overlay a binary mask on a frame with transparency."""
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    colored = np.zeros_like(frame)
    colored[mask > 0] = color
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)


def draw_box_and_label(frame, box, track_id, color, confidence=None):
    """Draw bounding box with track ID label."""
    x1, y1, x2, y2 = [int(c) for c in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{track_id}"
    if confidence is not None:
        label += f" {confidence:.2f}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def render_annotated_video(video_path, tracking_results, output_path,
                           max_frames=None, sampling_rate=1):
    """Render annotated video with bounding boxes and semi-transparent masks."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    local_idx = 0
    written = 0
    pbar = tqdm(desc="Rendering video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sampling_rate == 0:
            if local_idx in tracking_results:
                result = tracking_results[local_idx]
                masks = result.get("masks")

                for ai, athlete in enumerate(result["athletes"]):
                    tid = athlete["track_id"]
                    color = COLORS[(tid - 1) % len(COLORS)]

                    # Draw mask overlay
                    if masks is not None:
                        obj_idx = ai
                        mask = masks[obj_idx, 0]
                        if hasattr(mask, "cpu"):
                            mask = mask.cpu()
                        frame = overlay_mask(frame, mask, color)

                    draw_box_and_label(frame, athlete["box"], tid, color)

            local_idx += 1

        out.write(frame)
        written += 1
        frame_idx += 1
        pbar.update(1)

        if max_frames and local_idx >= max_frames:
            break

    pbar.close()
    cap.release()
    out.release()
    print(f"[visualize] Saved annotated video ({written} frames) to {output_path}")


def render_tracking_summary(video_path, tracking_results, output_path,
                            num_samples=6):
    """Create a grid image showing tracking at evenly spaced frames."""
    sorted_indices = sorted(tracking_results.keys())
    if not sorted_indices:
        print("[visualize] No tracking results to summarize")
        return

    # Pick evenly spaced sample frames
    step = max(1, len(sorted_indices) // num_samples)
    sample_indices = sorted_indices[::step][:num_samples]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames at once (we need random access by local_idx)
    all_frames = {}
    frame_idx = 0
    local_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if local_idx in sample_indices:
            all_frames[local_idx] = frame
        local_idx += 1
        frame_idx += 1
    cap.release()

    # Annotate sample frames
    annotated = []
    for idx in sample_indices:
        if idx not in all_frames:
            continue
        frame = all_frames[idx].copy()
        result = tracking_results[idx]
        masks = result.get("masks")

        for ai, athlete in enumerate(result["athletes"]):
            tid = athlete["track_id"]
            color = COLORS[(tid - 1) % len(COLORS)]
            if masks is not None:
                mask = masks[ai, 0]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu()
                frame = overlay_mask(frame, mask, color, alpha=0.4)
            draw_box_and_label(frame, athlete["box"], tid, color)

        # Add frame number label
        cv2.putText(frame, f"Frame {idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        annotated.append(frame)

    if not annotated:
        print("[visualize] No frames to render in summary")
        return

    # Arrange in a grid (2 rows)
    cols = min(3, len(annotated))
    rows = (len(annotated) + cols - 1) // cols

    # Resize each frame for the grid
    cell_w, cell_h = 640, 360
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, img in enumerate(annotated):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (cell_w, cell_h))
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = resized

    cv2.imwrite(output_path, grid)
    print(f"[visualize] Saved tracking summary ({len(annotated)} frames) "
          f"to {output_path}")
