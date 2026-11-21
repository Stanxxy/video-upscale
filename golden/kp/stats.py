"""Per-window stage numbers for the report: identity summary, keypoint and limb-track statistics, stage time and
peak memory. Usage: stats.py SEG_DIR...   (writes SEG_DIR/stats.json)"""
import json
import re
import sys
from pathlib import Path

import numpy as np

NAMES = ["L wrist", "R wrist", "L elbow", "R elbow", "L ankle", "R ankle", "L knee", "R knee"]
LIMBS = [9, 10, 7, 8, 15, 16, 13, 14]

for seg in map(Path, sys.argv[1:]):
    ident = json.load(open(seg / "identity.json"))["summary"]
    k, l = np.load(seg / "kp_rtmw.npz"), np.load(seg / "limbs.npz")
    have = ~np.isnan(k["kp"][:, :, 0, 0])
    sc, cross = k["score"][have][:, LIMBS], k["cross"][have][:, LIMBS]
    src, dis = l["source"][have], l["disagree"][have]
    stage = {}
    for name in ("identity", "keypoints", "limbs", "candidates"):
        p = seg / f"{name}.mem"
        hit = re.search(r"^(\d+) KB", p.read_text(), re.M) if p.exists() else None
        mem = hit.group(1) if hit else None
        stage[name] = dict(peak_gb=round(int(mem) / 1e6, 2) if mem else None)
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", (seg / "masks_light.log").read_text())
    stage["masks"] = dict(peak_gb=round(int(m.group(1)) / 1e6, 2) if m else None,
                          ms_per_frame=round(json.load(open(seg / "masks_light.json"))["wall_ms_per_frame"], 1))
    out = dict(identity=ident, stage=stage,
               keypoints=dict(fighter_frames=int(have.sum()), low_score_pct=round(float((sc < 2.5).mean()), 3),
                              cross_mask_pct=round(float(cross.mean()), 4),
                              per_joint_low={n: round(float((sc[:, i] < 2.5).mean()), 3) for i, n in enumerate(NAMES)}),
               limbs=dict(pose_pct=round(float((src == 1).mean()), 3), tracker_fill_pct=round(float((src == 2).mean()), 3),
                          none_pct=round(float((src == 0).mean()), 3), disagree_pct=round(float(dis.mean()), 3),
                          per_joint_disagree={n: round(float(dis[:, i].mean()), 3) for i, n in enumerate(NAMES)}),
               candidates=len(json.load(open(seg / "candidates.json"))))
    json.dump(out, open(seg / "stats.json", "w"), indent=1)
    print(seg.name, json.dumps(out["limbs"]), out["candidates"])
