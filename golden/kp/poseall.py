"""Stage 2a: RTMW-x whole-body keypoints (133) for every sizeable person the detector found, before identity.

Identity then reads kit colours at body parts located by the skeleton (shorts, gloves, shin guards, head)
instead of bands of a whole-body mask, so it works with any detector that gives boxes.

Usage: poseall.py SEG_DIR DET ENGINE    (reads masks_<DET>.mot.txt; writes pose_<DET>.npz: frame, tid, box, kp, score)
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

from trt_engine import Engine

SIZE = (288, 384)    # model input w, h
MEAN, STD = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])
TOP, MIN_H = 6, 0.14   # persons per frame (largest first); minimum box height as a share of the frame


def prep(img, box_xywh):
    x, y, w, h = box_xywh
    center, scale = bbox_xyxy2cs(np.array([x, y, x + w, y + h]), padding=1.25)
    crop, scale = top_down_affine(SIZE, scale, center, img)
    return ((crop - MEAN) / STD).transpose(2, 0, 1).astype(np.float32), center, scale


def pose(eng, img, boxes):
    """Keypoints [N, 133, 2] and SimCC scores [N, 133] for xywh boxes on one image."""
    items = [prep(img, b) for b in boxes]
    sx, sy = eng(np.stack([it[0] for it in items]))
    locs, scores = get_simcc_maximum(sx, sy)
    pts = [locs[i] / 2.0 / np.array(SIZE) * items[i][2] + items[i][1] - items[i][2] / 2 for i in range(len(items))]
    return np.array(pts, np.float32), scores.astype(np.float32)


def read_mot(path):
    boxes = defaultdict(dict)
    for line in open(path):
        f, tid, x, y, w, h = line.split(",")[:6]
        boxes[int(f) - 1][int(tid)] = [float(x), float(y), float(w), float(h)]
    return boxes


def main(seg_dir, det, engine_path):
    seg = Path(seg_dir)
    mot = read_mot(seg / f"masks_{det}.mot.txt")
    eng = Engine(engine_path)
    cap = cv2.VideoCapture(str(seg.parent.parent / "segs" / f"{seg.name}.mp4"))
    frames, tids, boxes, kps, scs = [], [], [], [], []
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        cands = [(tid, b) for tid, b in mot.get(f, {}).items() if b[3] >= MIN_H * img.shape[0] and b[2] >= 12]
        cands = sorted(cands, key=lambda c: -c[1][2] * c[1][3])[:TOP]
        if cands:
            pts, scores = pose(eng, img, [b for _, b in cands])
            for i, (tid, b) in enumerate(cands):
                frames.append(f); tids.append(tid); boxes.append(b); kps.append(pts[i]); scs.append(scores[i])
        f += 1
    np.savez_compressed(seg / f"pose_{det}.npz", frame=np.array(frames, np.int32), tid=np.array(tids, np.int32),
                        box=np.array(boxes, np.float32), kp=np.array(kps, np.float16), score=np.array(scs, np.float16),
                        nframes=f)
    print(f"{seg.name} {det}: {f} frames, {len(frames)} person poses")


if __name__ == "__main__":
    main(*sys.argv[1:4])
