"""Pick the two fighters out of DARTF person tracks, without per-video tuning (runs on gx10 next to the masks).

Per frame and track, inside the person's own mask: the torso band (22-42% of the mask height) and the shorts
band (50-66%). Fighters are the large people with a skin-coloured torso (the referee and corner staff are
clothed). Their shorts colours are split into two clusters (k-means, k=2, on LAB); the darker cluster is A
(red box), the lighter B (blue box). A track takes the majority label of its frames within +-10 frames (identity
re-checked continuously), and a frame keeps at most one box per fighter. A video whose shorts clusters are too
close is reported as ``separable: false`` and should not be labelled with boxes.

Usage: track_assign.py MASK_DIR VIDEO OUT_JSON
"""
import glob
import json
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

SCALE = 0.5
MIN_SEPARATION = 25.0   # LAB distance between the two shorts clusters


def band(lab, mask, top, bottom):
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    y0, y1 = ys.min(), ys.max()
    sel = (ys >= y0 + (y1 - y0) * top) & (ys <= y0 + (y1 - y0) * bottom)
    if sel.sum() < 60:
        return None
    return np.median(lab[ys[sel], xs[sel]].astype(float), axis=0)


def skin(lab_px):
    """OpenCV LAB (0-255): skin is warm (a, b above neutral 128) and neither black nor white."""
    L, a, b = lab_px
    return 35 <= L <= 225 and a >= 134 and b >= 134


def main(mask_dir, video, out):
    cap = cv2.VideoCapture(video)
    files = sorted(glob.glob(f"{mask_dir}/*.npz"))
    feats = {}
    for fi, path in enumerate(files):
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        H, W = img.shape[:2]
        z = np.load(path)
        for k in z.files:
            if not k.startswith("id"):
                continue
            mask = cv2.resize(z[k].astype(np.float32), (W, H)) > 0
            ys, xs = np.nonzero(mask)
            if len(ys) < 0.004 * H * W:   # crowd and cornermen are small
                continue
            torso, shorts = band(lab, mask, 0.22, 0.42), band(lab, mask, 0.50, 0.66)
            if torso is None or shorts is None or not skin(torso):
                continue
            box = [float(xs.min() / SCALE), float(ys.min() / SCALE),
                   float((xs.max() - xs.min()) / SCALE), float((ys.max() - ys.min()) / SCALE)]
            feats[(fi, int(k[2:]))] = (shorts, box, len(ys))
    frames = len(files)
    if len(feats) < 50:
        json.dump({"frames": frames, "separable": False, "reason": "too few fighter detections", "boxes": {}}, open(out, "w"))
        return
    X = np.array([f[0] for f in feats.values()], dtype=np.float32)
    _, labels, centres = cv2.kmeans(X, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5,
                                    cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    dark = int(np.argmin(centres[:, 0]))
    separation = float(np.linalg.norm(centres[0] - centres[1]))
    counts = np.bincount(labels, minlength=2)
    vote = {key: ("a" if lab_ == dark else "b") for key, lab_ in zip(feats, labels)}
    by_track = defaultdict(dict)
    for (f, tid), v in vote.items():
        by_track[tid][f] = v
    result = {}
    for (f, tid), (_, box, area) in feats.items():
        near = Counter(v for g, v in by_track[tid].items() if abs(g - f) <= 10)
        lab_, n = near.most_common(1)[0]
        if n < 3 or n < 0.7 * sum(near.values()):
            continue
        score = n + area / 1e6
        cur = result.setdefault(f, {})
        if lab_ not in cur or score > cur[lab_][0]:
            cur[lab_] = (score, box)
    result = {f: {k: v[1] for k, v in d.items()} for f, d in result.items()}
    both = sum(1 for d in result.values() if len(d) == 2)
    one = sum(1 for d in result.values() if len(d) == 1)
    json.dump({
        "frames": frames, "separable": bool(separation >= MIN_SEPARATION and min(counts) >= 0.15 * counts.sum()),
        "separation": round(separation, 1), "cluster_sizes": counts.tolist(),
        "centres_lab": np.round(centres, 1).tolist(), "a_cluster": dark,
        "both_pct": round(both / max(frames, 1), 3), "one_pct": round(one / max(frames, 1), 3),
        "boxes": {str(f + 1): v for f, v in result.items()},
    }, open(out, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
