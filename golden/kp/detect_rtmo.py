"""Stage 1 (pose-native alternative): RTMO-l finds every person and their 17 body keypoints in one pass per frame.

A person is its skeleton: the box is the extent of the confident keypoints (padded), and two detections whose
skeletons nearly coincide are one person (RTMO's own boxes are loose in FP16 and overlap in clinches).

Writes the same MOT text the SAM 3 / DART detectors write (frame,id,x,y,w,h,conf,-1,-1,-1; frame is 1-based), so
the later stages can swap detectors. Ids: each person is matched (Hungarian, box overlap) to the tracks seen in
the last 15 frames, so a track survives short misses instead of restarting. No masks.

With --zoom every frame gets a second pass on the region around the fighter-sized people, enlarged to the full
model input: small or tangled fighters (overhead camera, clinch) come out as two skeletons more often.

Usage: detect_rtmo.py VIDEO ENGINE OUT_PREFIX [--zoom]    (writes OUT_PREFIX.mot.txt and OUT_PREFIX.json)
"""
import json
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from trt_engine import Engine

SIZE, SCORE, KP_OK, SAME = 640, 0.3, 0.3, 0.12   # SAME: mean keypoint distance / body size
KEEP, MATCH_IOU = 15, 0.25                         # frames a lost track stays matchable; minimum overlap to match


def iou_matrix(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-6)


def people(scores, kps, vis, shape):
    """Skeleton boxes, best score first, duplicates (near-identical skeletons) dropped."""
    boxes, confs, kept = [], [], []
    for i in np.argsort(-scores):
        ok = vis[i] > KP_OK
        if scores[i] < SCORE or ok.sum() < 6:
            continue
        pts = kps[i][ok]
        x1, y1, x2, y2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        size = np.sqrt(max((x2 - x1) * (y2 - y1), 1.0))
        if any(np.linalg.norm(kps[i] - kps[j], axis=1).mean() < SAME * size for j in kept):
            continue
        kept.append(i)
        px, py = 0.12 * (x2 - x1) + 6, 0.10 * (y2 - y1) + 6
        boxes.append([max(x1 - px, 0), max(y1 - 1.6 * py, 0), min(x2 + px, shape[1]), min(y2 + py, shape[0])])
        confs.append(scores[i])
    return np.array(boxes).reshape(-1, 4), np.array(confs)


def infer(eng, img, x0=0, y0=0):
    """RTMO on an image (a crop at offset x0, y0): scores, keypoints in frame coordinates, visibilities."""
    ratio = min(SIZE / img.shape[0], SIZE / img.shape[1])
    pad = np.full((SIZE, SIZE, 3), 114, np.uint8)
    small = cv2.resize(img, (max(int(img.shape[1] * ratio), 1), max(int(img.shape[0] * ratio), 1)))
    pad[:small.shape[0], :small.shape[1]] = small
    dets, kps = eng(pad.transpose(2, 0, 1)[None].astype(np.float32))
    return dets[0, :, 4], kps[0, :, :, :2] / ratio + [x0, y0], kps[0, :, :, 2]


def zoom_region(boxes, shape):
    """Region around the fighter-sized people (largest four), padded; None when it is most of the frame anyway."""
    H, W = shape[:2]
    big = [b for b in boxes if b[3] - b[1] >= 0.14 * H]
    big = sorted(big, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:4]
    if not big:
        return None
    b = np.array(big)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    px, py = 0.12 * (x2 - x1) + 10, 0.12 * (y2 - y1) + 10
    x1, y1, x2, y2 = int(max(x1 - px, 0)), int(max(y1 - py, 0)), int(min(x2 + px, W)), int(min(y2 + py, H))
    if (x2 - x1) * (y2 - y1) > 0.6 * W * H or x2 - x1 < 40 or y2 - y1 < 40:
        return None
    return x1, y1, x2, y2


def main(video, engine_path, out, zoom=""):
    eng = Engine(engine_path)
    cap = cv2.VideoCapture(video)
    tracks, next_id, f, rows, counts = {}, 1, 0, [], []   # tracks: id -> (box, last frame)
    t0 = time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        scores, kps, vis = infer(eng, img)
        boxes, confs = people(scores, kps, vis, img.shape)
        region = zoom_region(boxes, img.shape) if zoom else None
        if region:
            x1, y1, x2, y2 = region
            s2, k2, v2 = infer(eng, img[y1:y2, x1:x2], x1, y1)
            # the enlarged pass sees more detail: its skeletons win ties with the full-frame ones
            boxes, confs = people(np.concatenate([s2 + 0.05, scores]), np.concatenate([k2, kps]),
                                  np.concatenate([v2, vis]), img.shape)
        ids = [0] * len(boxes)
        live = [t for t, (_, seen) in tracks.items() if f - seen <= KEEP]
        if len(boxes) and live:
            m = iou_matrix(boxes, np.array([tracks[t][0] for t in live]))
            age = np.array([f - tracks[t][1] for t in live])
            for i, j in zip(*linear_sum_assignment(1 - m + 0.01 * age[None])):
                if m[i, j] >= MATCH_IOU:
                    ids[i] = live[j]
        for i in range(len(boxes)):
            if ids[i] == 0:
                ids[i], next_id = next_id, next_id + 1
            x1, y1, x2, y2 = boxes[i]
            rows.append(f"{f + 1},{ids[i]},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{confs[i]:.3f},-1,-1,-1")
            tracks[ids[i]] = (boxes[i], f)
        counts.append(len(boxes))
        f += 1
    open(out + ".mot.txt", "w").write("\n".join(rows) + "\n")
    stats = dict(frames=f, wall_ms_per_frame=round((time.time() - t0) * 1000 / max(f, 1), 1),
                 tracks_mean=round(float(np.mean(counts)), 2))
    json.dump(stats, open(out + ".json", "w"))
    print(json.dumps(stats))


if __name__ == "__main__":
    main(*sys.argv[1:5])
