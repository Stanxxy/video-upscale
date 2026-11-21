"""Stage 3 variants: redo each fighter's pose with the opponent painted grey, so joints cannot land on the other body.

  samask  the opponent's SAM 3 instance mask (needs the mask dumps of the SAM 3 detector), minus the fighter's own
  skmask  the opponent's body drawn from his first-pass skeleton (thick limbs, torso, head), minus the fighter's
          own drawn body: needs no masks, so it works with the light detectors

Usage: posefix.py SEG_DIR DET ENGINE MODE    (reads identity_<DET>.json, kp_<DET>.npz; writes kp_<DET>_<MODE>.npz)
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from poseall import pose
from trt_engine import Engine

LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12)]


def body_mask(kp, sc, shape):
    """Rough body region from a skeleton: limbs as thick lines, the torso filled, a disc for the head."""
    m = np.zeros(shape[:2], np.uint8)
    ok = sc >= 2.0
    if not ok[[5, 6, 11, 12]].any():
        return m > 0
    tl = np.linalg.norm((kp[5] + kp[6]) / 2 - (kp[11] + kp[12]) / 2) if ok[[5, 6, 11, 12]].all() else 80.0
    thick = int(max(tl * 0.28, 6))
    for a, b in LIMBS:
        if ok[a] and ok[b]:
            cv2.line(m, tuple(map(int, kp[a])), tuple(map(int, kp[b])), 1, thick)
    if ok[[5, 6, 11, 12]].all():
        cv2.fillPoly(m, [np.round(kp[[5, 6, 12, 11]]).astype(np.int32)], 1)
    if ok[0]:
        cv2.circle(m, tuple(map(int, kp[0])), int(max(tl * 0.3, 8)), 1, -1)
    return m > 0


def main(seg_dir, det, engine_path, mode):
    seg = Path(seg_dir)
    ident = json.load(open(seg / f"identity_{det}.json"))["frames"]
    z = np.load(seg / f"kp_{det}.npz")
    kp0, sc0 = z["kp"], z["score"]
    files = sorted(glob.glob(str(seg / f"masks_{det}" / "*.npz"))) if mode == "samask" else None
    eng = Engine(engine_path)
    cap = cv2.VideoCapture(str(seg.parent.parent / "segs" / f"{seg.name}.mp4"))
    kp, sc = kp0.copy(), sc0.copy()
    redone = 0
    for f in range(len(kp0)):
        ok, img = cap.read()
        if not ok:
            break
        fr = ident[str(f)]
        if "a" not in fr or "b" not in fr:
            continue   # nobody to grey out
        H, W = img.shape[:2]
        if mode == "samask":
            zz = np.load(files[f])
            region = [cv2.resize((zz[f"id{fr[s]['id']}"].astype(np.float32) > 0).astype(np.uint8), (W, H),
                                 interpolation=cv2.INTER_NEAREST) > 0 for s in "ab"]
        else:
            region = [body_mask(kp0[f, j], sc0[f, j], img.shape) for j in range(2)]
        srcs = []
        for j in range(2):
            im = img.copy()
            im[region[1 - j] & ~region[j]] = 128
            srcs.append(im)
        for j, s in enumerate("ab"):
            pts, scores = pose(eng, srcs[j], [fr[s]["box"]])
            kp[f, j], sc[f, j] = pts[0], scores[0]
        redone += 1
    np.savez_compressed(seg / f"kp_{det}_{mode}.npz", kp=kp, score=sc)
    moved = np.linalg.norm(kp[:, :, [9, 10, 15, 16]] - kp0[:, :, [9, 10, 15, 16]], axis=-1)
    print(f"{seg.name} {det} {mode}: {redone} frames redone, wrist/ankle moved > 20 px on "
          f"{float(np.nanmean(moved > 20)):.3f} of joint-frames")


if __name__ == "__main__":
    main(*sys.argv[1:5])
