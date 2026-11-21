"""Window lists for the classifier runs.

oracle: windows around the founder's labelled attacks and negatives in the gold minute (CW R2 3:50-4:52), so the
  classifier is measured without detection loss. Items closer than 0.9 s share a window; windows are padded
  0.7 s and split at 3.2 s.
sensing: windows from the sensing candidates (any segment): candidates within 1.0 s form one exchange window (at most 2 s of peaks),
  padded 0.8 s before and 0.4 s after.

sliding: 1.8 s windows every 1.4 s over whole segments, for labelling passes that must not depend on a strike
  detector. A window is dropped only when it is plainly idle: nobody tracked, or both fighters tracked and no
  wrist, ankle or knee moves faster than 2 torso lengths per second relative to its owner's hips.

Usage: windows.py oracle OUT_JSON | windows.py sensing CANDS_DIR OUT_JSON SEG...
       windows.py sliding OUT_DIR DET OUT_JSON SEG... [--length FRAMES --stride FRAMES]   (OUT_DIR/<seg>/kp_<DET>.npz)
"""
import json
import sys
from pathlib import Path

GOLD_START_MS, FPS = 230000, 25


def f_of(t_ms):
    return round((t_ms - GOLD_START_MS) * FPS / 1000)


def oracle(out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from golden.gold import EXPECTED, NEGATIVE
    items = sorted([(t, "pos") for t, *_ in EXPECTED] + [(t, "neg") for t, *_ in NEGATIVE])
    groups = []
    for t, k in items:
        if groups and t - groups[-1][-1][0] <= 900 and t - groups[-1][0][0] <= 2000:
            groups[-1].append((t, k))
        else:
            groups.append([(t, k)])
    wins = [dict(id=f"gold-o{i:02d}", seg="gold", f0=f_of(g[0][0] - 700), f1=f_of(g[-1][0] + 700),
                 t0_ms=g[0][0] - 700, items=[t for t, _ in g]) for i, g in enumerate(groups)]
    json.dump(wins, open(out, "w"), indent=1)
    print(len(wins), "oracle windows")


def sensing(cands_dir, out, segs):
    wins = []
    for seg in segs:
        cands = sorted(json.load(open(Path(cands_dir) / seg / "candidates.json")), key=lambda c: c["peak_f"])
        groups = []
        for c in cands:
            if groups and c["peak_f"] - groups[-1][-1]["peak_f"] <= FPS and c["peak_f"] - groups[-1][0]["peak_f"] <= 2 * FPS:
                groups[-1].append(c)
            else:
                groups.append([c])
        for i, g in enumerate(groups):
            f0, f1 = g[0]["peak_f"] - 20, g[-1]["peak_f"] + 10
            top = max(g, key=lambda c: c["speed_tl_s"])
            li = {"punch": 0, "kick": 4, "knee": 6}[top["group"]] + (top["side"] == "R")
            wins.append(dict(id=f"{seg}-s{i:03d}", seg=seg, f0=max(0, f0), f1=f1, t0_ms=None,
                             cands=[c["id"] for c in g], peak_f=top["peak_f"], focus=dict(fighter=top["fighter"], li=li)))
    json.dump(wins, open(out, "w"), indent=1)
    print(len(wins), "sensing windows")


def sliding(out_dir, det, out, segs, length=45, stride=35, idle=2.0):
    import numpy as np
    wins, dropped = [], 0
    for seg in segs:
        z = np.load(Path(out_dir) / seg / f"kp_{det}.npz")
        kp, sc = z["kp"], z["score"]
        kp = np.where((sc >= 2.0)[..., None], kp, np.nan)
        pelvis = (kp[:, :, 11] + kp[:, :, 12]) / 2
        tl = np.nanmedian(np.linalg.norm((kp[:, :, 5] + kp[:, :, 6]) / 2 - pelvis, axis=-1), axis=0)   # per fighter
        rel = kp[:, :, [9, 10, 13, 14, 15, 16]] - pelvis[:, :, None]
        speed = np.linalg.norm(rel[2:] - rel[:-2], axis=-1) / 2 * FPS / tl[None, :, None]            # [n-2, 2, 6]
        tracked = ~np.isnan(pelvis[:, :, 0])
        n = len(kp)
        for i, f0 in enumerate(range(0, n - 10, stride)):
            f1 = min(f0 + length - 1, n - 1)
            seen = tracked[f0:f1 + 1].mean(axis=0)
            sp = speed[f0:max(f1 - 1, f0 + 1)]
            top = float(np.nanmax(sp)) if np.isfinite(sp).any() else None
            if seen.max() < 0.3 or (seen.min() >= 0.8 and top is not None and top < idle):
                dropped += 1
                continue
            wins.append(dict(id=f"{seg}-w{i:03d}", seg=seg, f0=f0, f1=f1, top_speed=None if top is None else round(top, 1),
                             tracked=[round(float(v), 2) for v in seen]))
    json.dump(wins, open(out, "w"), indent=1)
    print(len(wins), "sliding windows kept,", dropped, "idle or untracked dropped")


if __name__ == "__main__":
    if sys.argv[1] == "oracle":
        oracle(sys.argv[2])
    elif sys.argv[1] == "sliding":
        args, opts = sys.argv[5:], {}
        while "--length" in args or "--stride" in args:
            i = next(i for i, x in enumerate(args) if x in ("--length", "--stride"))
            opts[args[i][2:]] = int(args[i + 1])
            args = args[:i] + args[i + 2:]
        sliding(sys.argv[2], sys.argv[3], sys.argv[4], args, **opts)
    else:
        sensing(sys.argv[2], sys.argv[3], sys.argv[4:])
