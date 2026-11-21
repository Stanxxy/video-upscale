"""Stage 2: which DARTF person track is fighter A, which is B, every frame (referee and crowd excluded).

Per detection, inside its own mask: torso band skin rate (fighters are bare-chested, the referee is clothed) and
the shorts-band colour (LAB). Tracks whose median skin rate is low are not fighters. Shorts colours split into two
clusters; the cluster closer to fighter A's named shorts colour (or the darker one) is A. A track takes the
majority kit label of its own detections within +-12 frames, then each frame assigns at most one detection per
fighter with the Hungarian method on kit disagreement plus box discontinuity.

Flags per frame: ``kit_conflict`` (the chosen detection's own kit vote disagrees with its slot), ``switch``
(the track id behind A or B changed), ``merged`` (one fighter mask about as big as both: a clinch blob).

Usage: identity.py SEG_DIR NAME_A NAME_B     (SEG_DIR has masks_light/, masks_light.mot.txt; writes identity.json,
masks_ab.npz)
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

W, H = 640, 360
NAMED = {"black": (20, 128, 128), "white": (235, 128, 128), "grey": (130, 128, 128), "gray": (130, 128, 128),
         "light grey": (175, 128, 128), "dark brown": (55, 140, 145), "brown": (75, 140, 150),
         "red": (110, 175, 155), "blue": (70, 140, 95), "green": (110, 105, 140), "orange": (150, 165, 180),
         "yellow": (220, 115, 200), "neon yellow": (225, 100, 210)}


def named_shorts(name):
    """LAB guess for the shorts colour in a kit name ('Red shin guards, grey shorts' -> grey)."""
    name = name.lower()
    for part in name.split(","):
        if "shorts" in part or "tights" in part:
            words = part.replace("shorts", "").replace("tights", "").strip()
            for key in sorted(NAMED, key=len, reverse=True):
                if key in words:
                    return np.array(NAMED[key], float)
    return None


def band(lab, mask, top, bottom):
    ys, xs = np.nonzero(mask)
    if len(ys) < 150:
        return None, 0.0
    y0, y1 = ys.min(), ys.max()
    sel = (ys >= y0 + (y1 - y0) * top) & (ys <= y0 + (y1 - y0) * bottom)
    if sel.sum() < 40:
        return None, 0.0
    px = lab[ys[sel], xs[sel]].astype(float)
    skin = ((px[:, 0] >= 35) & (px[:, 0] <= 225) & (px[:, 1] >= 134) & (px[:, 2] >= 134)).mean()
    return np.median(px, axis=0), float(skin)


def read_mot(path):
    boxes = defaultdict(dict)
    for line in open(path):
        f, tid, x, y, w, h = line.split(",")[:6]
        boxes[int(f) - 1][int(tid)] = [float(x), float(y), float(w), float(h)]
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-6)


def main(seg_dir, name_a, name_b):
    seg = Path(seg_dir)
    files = sorted(glob.glob(str(seg / "masks_light" / "*.npz")))
    mot = read_mot(seg / "masks_light.mot.txt")
    cap = cv2.VideoCapture(str(seg.parent.parent / "segs" / f"{seg.name}.mp4"))
    dets = {}          # (f, tid) -> dict
    masks = {}         # (f, tid) -> packed 288x288 mask
    for fi, path in enumerate(files):
        ok, img = cap.read()
        if not ok:
            break
        lab = cv2.cvtColor(cv2.resize(img, (W, H)), cv2.COLOR_BGR2LAB)
        z = np.load(path)
        for k in z.files:
            if not k.startswith("id"):
                continue
            tid = int(k[2:])
            m288 = z[k].astype(np.float32) > 0
            mask = cv2.resize(m288.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            area = mask.mean()
            if area < 0.004 or tid not in mot.get(fi, {}):
                continue
            torso, skin = band(lab, mask, 0.15, 0.42)
            shorts, _ = band(lab, mask, 0.48, 0.66)
            if shorts is None:
                continue
            dets[(fi, tid)] = dict(area=float(area), skin=skin, shorts=shorts, box=mot[fi][tid])
            masks[(fi, tid)] = m288
    nframes = len(files)
    by_track = defaultdict(list)
    for (f, tid), d in dets.items():
        by_track[tid].append(d)
    fighter_tracks = {tid for tid, ds in by_track.items()
                      if np.median([d["skin"] for d in ds]) >= 0.30 and len(ds) >= 5}
    pool = [(k, d) for k, d in dets.items() if k[1] in fighter_tracks]
    X = np.array([d["shorts"] for _, d in pool], np.float32)
    _, labels, centres = cv2.kmeans(X, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5,
                                    cv2.KMEANS_PP_CENTERS)
    ga, gb = named_shorts(name_a), named_shorts(name_b)
    if ga is not None and gb is not None:
        straight = np.linalg.norm(centres[0] - ga) + np.linalg.norm(centres[1] - gb)
        crossed = np.linalg.norm(centres[1] - ga) + np.linalg.norm(centres[0] - gb)
        a_cluster = 0 if straight <= crossed else 1
        rule = "named shorts colour"
    else:
        a_cluster = int(np.argmin(centres[:, 0]))
        rule = "darker shorts"
    separation = float(np.linalg.norm(centres[0] - centres[1]))
    proto = {"a": centres[a_cluster], "b": centres[1 - a_cluster]}
    # per detection kit probability of being A (softmax over distance to the two prototypes)
    for k, d in pool:
        da, db = np.linalg.norm(d["shorts"] - proto["a"]), np.linalg.norm(d["shorts"] - proto["b"])
        d["pa"] = float(1 / (1 + np.exp((da - db) / max(separation / 4, 1))))
    track_frames = defaultdict(dict)
    for (f, tid), d in pool:
        track_frames[tid][f] = d["pa"]
    single_area = np.median([d["area"] for _, d in pool])
    out, prev = {}, {"a": None, "b": None}
    prev_id = {"a": None, "b": None}
    stats = Counter()
    for f in range(nframes):
        cands = [(tid, dets[(f, tid)]) for tid in fighter_tracks if (f, tid) in dets]
        cands = sorted(cands, key=lambda c: -c[1]["area"])[:3]
        frame = {"flags": []}
        if cands:
            cost = np.zeros((len(cands), 2))
            for i, (tid, d) in enumerate(cands):
                near = [p for g, p in track_frames[tid].items() if abs(g - f) <= 12]
                pa = float(np.mean(near))
                d["pa_track"] = pa
                for j, slot in enumerate(("a", "b")):
                    kit = 1 - pa if slot == "a" else pa
                    cont = 1 - iou(d["box"], prev[slot]) if prev[slot] is not None else 0.5
                    cost[i, j] = 2.0 * kit + 0.7 * cont
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] > 1.9:   # neither kit nor continuity supports this detection as that fighter
                    continue
                tid, d = cands[i]
                slot = ("a", "b")[j]
                frame[slot] = dict(id=tid, box=[round(v, 1) for v in d["box"]], pa=round(d["pa"], 3),
                                   pa_track=round(d["pa_track"], 3))
                if (slot == "a" and d["pa"] < 0.35) or (slot == "b" and d["pa"] > 0.65):
                    frame["flags"].append(f"kit_conflict_{slot}")
                if prev_id[slot] is not None and prev_id[slot] != tid:
                    frame["flags"].append(f"switch_{slot}")
                prev[slot], prev_id[slot] = d["box"], tid
            got = [s for s in ("a", "b") if s in frame]
            if len(got) == 1 and dets[(f, frame[got[0]]["id"])]["area"] > 1.6 * single_area:
                frame["flags"].append("merged")
        n = sum(s in frame for s in ("a", "b"))
        stats[f"fighters_{n}"] += 1
        for fl in frame["flags"]:
            stats[fl.rsplit("_", 1)[0] if fl != "merged" else fl] += 1
        out[f] = frame
    packed = np.zeros((nframes, 2, 288 * 288 // 8), np.uint8)
    for f, fr in out.items():
        for j, slot in enumerate(("a", "b")):
            if slot in fr:
                packed[f, j] = np.packbits(masks[(f, fr[slot]["id"])].ravel())
    np.savez_compressed(seg / "masks_ab.npz", packed=packed)
    summary = dict(frames=nframes, fighter_tracks=sorted(fighter_tracks), all_tracks=len(by_track),
                   separation=round(separation, 1), a_rule=rule, names=[name_a, name_b],
                   both_pct=round(stats["fighters_2"] / nframes, 3), one_pct=round(stats["fighters_1"] / nframes, 3),
                   none_pct=round(stats["fighters_0"] / nframes, 3), kit_conflicts=stats["kit_conflict"],
                   switches=stats["switch"], merged=stats["merged"], camera_gate=stats["fighters_2"] / nframes >= 0.7)
    json.dump(dict(summary=summary, frames=out), open(seg / "identity.json", "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:4])
