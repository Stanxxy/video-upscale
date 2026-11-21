"""By-eye pose check sheets: frames where the two fighter boxes overlap, the same crop drawn once per pose variant
in a shuffled order (so the judge does not know which panel is which).

Usage: posecheck.py DATA OUT DET VARIANTS SEG...    VARIANTS like "base,samask,skmask" (base = kp_<DET>.npz)
Writes OUT/<seg>_<frame>.jpg (panels left to right numbered 1..n) and OUT/key.json (panel order per sheet).
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 5), (0, 6)]
COL = {0: (40, 40, 230), 1: (230, 120, 30)}     # BGR: A red, B blue
PER_SEG, PANEL = 12, 560


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    return iw * ih / max(a[2] * a[3] + b[2] * b[3] - iw * ih, 1e-6)


def main(data, out, det, variants, segs):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    variants = variants.split(",")
    rng, key = random.Random(7), {}
    for seg in segs:
        d = data / "out" / seg
        ident = json.load(open(d / f"identity_{det}.json"))["frames"]
        kps = {v: np.load(d / (f"kp_{det}.npz" if v == "base" else f"kp_{det}_{v}.npz")) for v in variants}
        over = [int(f) for f, fr in ident.items() if "a" in fr and "b" in fr and iou(fr["a"]["box"], fr["b"]["box"]) > 0.12]
        picks = [over[i] for i in np.linspace(0, len(over) - 1, min(PER_SEG, len(over))).astype(int)] if over else []
        cap = cv2.VideoCapture(str(data / "segs" / f"{seg}.mp4"))
        for f in sorted(set(picks)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            fr = ident[str(f)]
            xs = [fr[s]["box"][0] for s in "ab"] + [fr[s]["box"][0] + fr[s]["box"][2] for s in "ab"]
            ys = [fr[s]["box"][1] for s in "ab"] + [fr[s]["box"][1] + fr[s]["box"][3] for s in "ab"]
            x0, x1, y0, y1 = int(max(min(xs) - 25, 0)), int(min(max(xs) + 25, img.shape[1])), int(max(min(ys) - 25, 0)), int(min(max(ys) + 25, img.shape[0]))
            order = variants[:]
            rng.shuffle(order)
            panels = []
            for n, v in enumerate(order):
                im = img.copy()
                kp, sc = kps[v]["kp"][f], kps[v]["score"][f]
                for j in range(2):
                    for a, b in EDGES:
                        if sc[j, a] >= 2.0 and sc[j, b] >= 2.0:
                            cv2.line(im, tuple(map(int, kp[j, a])), tuple(map(int, kp[j, b])), COL[j], 2)
                    for k in (7, 8, 13, 14):
                        if sc[j, k] >= 2.0:
                            cv2.circle(im, tuple(map(int, kp[j, k])), 4, COL[j], -1)
                    for k in (9, 10, 15, 16):
                        if sc[j, k] >= 2.0:
                            cv2.circle(im, tuple(map(int, kp[j, k])), 7, COL[j], -1)
                            cv2.circle(im, tuple(map(int, kp[j, k])), 7, (255, 255, 255), 1)
                im = im[y0:y1, x0:x1]
                im = cv2.resize(im, (PANEL, int(im.shape[0] * PANEL / im.shape[1])))
                cv2.rectangle(im, (0, 0), (34, 34), (255, 255, 255), -1)
                cv2.putText(im, str(n + 1), (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 3)
                panels.append(im)
            cv2.imwrite(str(out / f"{seg}_{f:04d}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
            key[f"{seg}_{f:04d}"] = order
    json.dump(key, open(out / "key.json", "w"), indent=1)
    print(len(key), "sheets")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
