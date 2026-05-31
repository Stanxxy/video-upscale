"""Tracking JSON and visualization output helpers."""
import json
import os

import cv2
import numpy as np

from tracking_pipeline.hybrid.yolo26_detector import NumpyEncoder

def _build_athlete_dicts(frame_boxes, frame_kpts, frame_scores,
                         frame_sources, display_map):
    """Build the athlete dict list for frame_callback."""
    athletes = []
    for track_id, box in frame_boxes.items():
        disp_id = display_map.get(track_id, track_id)
        athlete = {
            "track_id": disp_id,
            "box": [round(c, 1) for c in box],
            "source": frame_sources.get(track_id, "unknown"),
        }
        kpts = frame_kpts.get(track_id)
        if kpts is not None:
            if hasattr(kpts, "tolist"):
                athlete["keypoints"] = kpts.tolist()
            else:
                athlete["keypoints"] = kpts
        kpt_sc = frame_scores.get(track_id)
        if kpt_sc is not None:
            if hasattr(kpt_sc, "tolist"):
                athlete["keypoint_scores"] = kpt_sc.tolist()
            else:
                athlete["keypoint_scores"] = kpt_sc
        athletes.append(athlete)
    return athletes

# COCO 17-joint skeleton connections for pose visualization
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # upper body
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]


def _write_viz_frame(out, frame_bgr, frame_boxes, frame_masks,
                     frame_kpts, frame_scores, frame_sources, display_map,
                     state, iou, global_idx, debug_dir, local_idx):
    """Draw masks, boxes, skeleton, labels, and status on frame, then write to video."""
    viz = frame_bgr.copy()
    colors = {1: (0, 255, 0), 2: (255, 0, 0)}  # Green=A, Blue=B

    for track_id, box in frame_boxes.items():
        disp_id = display_map.get(track_id, track_id)
        color = colors.get(disp_id, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box)

        # Semi-transparent mask overlay
        if track_id in frame_masks:
            overlay = np.zeros_like(viz)
            overlay[frame_masks[track_id]] = color
            viz = cv2.addWeighted(viz, 1.0, overlay, 0.4, 0)

        # Pose skeleton (confidence-aware: suppress low-confidence limbs)
        kpts = frame_kpts.get(track_id)
        kpt_sc = frame_scores.get(track_id)
        if kpts is not None:
            pts = kpts if isinstance(kpts, list) else kpts.tolist()
            scs = None
            if kpt_sc is not None:
                scs = kpt_sc if isinstance(kpt_sc, list) else kpt_sc.tolist()
            # Draw limb connections (only if both endpoints confident)
            for i, j in _SKELETON_EDGES:
                if i < len(pts) and j < len(pts):
                    if scs and (scs[i] <= 0 or scs[j] <= 0):
                        continue  # skip limbs with suppressed endpoints
                    p1 = (int(pts[i][0]), int(pts[i][1]))
                    p2 = (int(pts[j][0]), int(pts[j][1]))
                    if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                        cv2.line(viz, p1, p2, color, 2, cv2.LINE_AA)
            # Draw joint dots (skip suppressed joints)
            for idx_j, pt in enumerate(pts):
                px, py = int(pt[0]), int(pt[1])
                if px > 0 and py > 0:
                    if scs and idx_j < len(scs) and scs[idx_j] <= 0:
                        continue
                    cv2.circle(viz, (px, py), 3, color, -1, cv2.LINE_AA)

        # Bounding box + label
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
        source = frame_sources.get(track_id, "?")
        label = f"ID {disp_id} ({source})"
        cv2.putText(viz, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Status bar
    status = f"{state.value} | IoU:{iou:.2f} | F:{global_idx}"
    cv2.putText(viz, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out.write(viz)

    # Save debug frame every 120 frames
    if local_idx % 120 == 0:
        cv2.imwrite(
            os.path.join(debug_dir, f"debug_{global_idx:05d}.jpg"), viz,
        )


def _append_frame_to_json(json_file, is_first, global_idx, local_idx, fps,
                       state, iou, frame_boxes, frame_kpts, frame_scores,
                       frame_sources, display_map):
    """Write one frame's data directly to the JSON file (streaming).

    Returns updated is_first flag.
    """
    frame_entry = {
        "frame_idx": global_idx,
        "local_idx": local_idx,
        "timestamp": round(global_idx / fps, 4),
        "state": state.value,
        "iou": round(iou, 4),
        "athletes": _build_athlete_dicts(
            frame_boxes, frame_kpts, frame_scores, frame_sources, display_map
        ),
    }

    if not is_first:
        json_file.write(",\n")
    json_file.write(json.dumps(frame_entry, cls=NumpyEncoder))
    json_file.flush()
    return False
