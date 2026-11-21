"""Window lists for the classifier runs.

oracle: windows around the founder's labelled attacks and negatives in the gold minute (CW R2 3:50-4:52), so the
  classifier is measured without detection loss. Items closer than 0.9 s share a window; windows are padded
  0.7 s and split at 3.2 s.
sensing: windows from the sensing candidates (any segment): candidates within 1.0 s form one exchange window (at most 2 s of peaks),
  padded 0.8 s before and 0.4 s after.

Usage: windows.py oracle OUT_JSON | windows.py sensing CANDS_DIR OUT_JSON SEG...
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


if __name__ == "__main__":
    if sys.argv[1] == "oracle":
        oracle(sys.argv[2])
    else:
        sensing(sys.argv[2], sys.argv[3], sys.argv[4:])
