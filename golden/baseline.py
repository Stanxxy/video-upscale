"""G4 classifier baseline: who attacks and which technique classes, from tracked pose (no video model).

Per sequence, both fighters' COCO-17 keypoints (ViTPose on the tracker's boxes) over the span +-0.5 s give
hand-made features in the spirit of BoxMind's strike features: each fighter's limb ends (wrists, elbows,
knees, ankles) approaching the opponent's head and torso, limb extension, kick height, level change, the
distance between the fighters (clinch) and box shape (ground). Gradient-boosted trees, evaluated by
leave-videos-out cross-validation; mirrored copies (A<->B swapped) make the actor model symmetric.

    python -m golden.baseline train --tasks export.jsonl --pose-dir DIR [--report out.json]
    python -m golden.baseline gold   --pose cw_pose.json --boxes boxes2.json --fps 25 --offset-ms 228000 --model model.pkl

Labels: the founder's correction where one exists (verdict accept/corrected), otherwise the model's pre-label
(reported separately: pre-labels are noisy and the score then measures agreement with the pre-labeller).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

KIND = {"strike_punch": "punch", "strike_kick": "kick", "strike_knee": "knee", "strike_elbow": "elbow",
        "ground_strike": "ground_strike", "takedown_attempt": "takedown", "clinch_entry": "clinch"}
KINDS = ("punch", "kick", "knee", "elbow", "ground_strike", "takedown", "clinch")
NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
ENDS = {"wrist": (LWR, RWR), "elbow": (LEL, REL), "knee": (LKN, RKN), "ankle": (LAN, RAN)}
PAD_MS, MIN_SCORE = 500, 0.3


def _pt(kp, i):
    return np.array(kp["xy"][i]) if kp["s"][i] >= MIN_SCORE else None


def _mean(pts):
    pts = [p for p in pts if p is not None]
    return np.mean(pts, axis=0) if pts else None


def fighter_series(frames, me, other, scale):
    """Per-frame measurements for fighter ``me`` against ``other`` (dicts of frame -> keypoints)."""
    rows = []
    for f in frames:
        a, b = me.get(f), other.get(f)
        row = {}
        if a is not None:
            hip, sh = _mean([_pt(a, LHIP), _pt(a, RHIP)]), _mean([_pt(a, LSH), _pt(a, RSH)])
            if hip is not None:
                ank = [p for p in (_pt(a, LAN), _pt(a, RAN)) if p is not None]
                if ank:
                    row["kick_height"] = max((hip[1] - p[1]) / scale for p in ank)
                row["hip_y"] = hip[1] / scale
            if sh is not None:
                wr = [p for p in (_pt(a, LWR), _pt(a, RWR)) if p is not None]
                if wr:
                    row["extension"] = max(np.linalg.norm(p - sh) / scale for p in wr)
        if a is not None and b is not None:
            torso = _mean([_pt(b, LSH), _pt(b, RSH), _pt(b, LHIP), _pt(b, RHIP)])
            head = _pt(b, NOSE)
            for name, idx in ENDS.items():
                pts = [p for p in (_pt(a, i) for i in idx) if p is not None]
                for tname, target in (("torso", torso), ("head", head)):
                    if pts and target is not None:
                        row[f"{name}_{tname}"] = min(np.linalg.norm(p - target) for p in pts) / scale
            mine = _mean([_pt(a, LSH), _pt(a, RSH), _pt(a, LHIP), _pt(a, RHIP)])
            if mine is not None and torso is not None:
                row["gap"] = np.linalg.norm(mine - torso) / scale
        rows.append(row)
    return rows


def summarise(rows, fps):
    out = {}
    n = max(len(rows), 1)
    out["seen"] = sum(1 for r in rows if r) / n
    keys = sorted({k for r in rows for k in r})
    for k in ("kick_height", "extension", "hip_y", "gap") + tuple(f"{e}_{t}" for e in ENDS for t in ("torso", "head")):
        vals = [r.get(k) for r in rows]
        xs = np.array([v for v in vals if v is not None], dtype=float)
        if not len(xs):
            out[f"{k}_min"] = out[f"{k}_max"] = out[f"{k}_mean"] = np.nan
            continue
        out[f"{k}_min"], out[f"{k}_max"], out[f"{k}_mean"] = xs.min(), xs.max(), xs.mean()
        if k not in ("kick_height", "extension", "hip_y", "gap"):
            # approach speed toward the opponent (scale units per second) and time in contact range
            speeds = [(vals[i - 1] - vals[i]) * fps for i in range(1, len(vals)) if vals[i] is not None and vals[i - 1] is not None]
            out[f"{k}_approach"] = max(speeds) if speeds else np.nan
            out[f"{k}_contact"] = float(np.mean(xs < 0.35))
        if k == "hip_y":
            out["hip_drop"] = xs.max() - xs.min()
    del keys
    return out


def features(pose, boxes, start_ms, end_ms, fps):
    """Feature dict for one sequence; ``pose``/``boxes``: frame(int) -> {a, b}."""
    frames = list(range(int((start_ms - PAD_MS) / 1000 * fps) + 1, int((end_ms + PAD_MS) / 1000 * fps) + 2))
    per = {lab: {f: pose[f][lab] for f in frames if f in pose and lab in pose[f]} for lab in ("a", "b")}
    heights = [boxes[f][lab][3] for f in frames if f in boxes for lab in boxes[f]]
    scale = float(np.median(heights)) if heights else 300.0
    feats = {}
    for me, other in (("a", "b"), ("b", "a")):
        s = summarise(fighter_series(frames, per[me], per[other], scale), fps)
        feats.update({f"{me}_{k}": v for k, v in s.items()})
    widths = [boxes[f][lab][2] / max(boxes[f][lab][3], 1) for f in frames if f in boxes for lab in boxes[f]]
    feats["aspect_max"] = max(widths) if widths else np.nan
    feats["duration"] = (end_ms - start_ms) / 1000
    return feats


def mirrored(feats):
    out = {}
    for k, v in feats.items():
        out[("b_" + k[2:]) if k.startswith("a_") else ("a_" + k[2:]) if k.startswith("b_") else k] = v
    return out


def vector(feats, names):
    return np.array([feats.get(n, np.nan) for n in names], dtype=float)


def label_of(task):
    c = task.get("correction") or {}
    if c.get("verdict") in ("accept", "corrected"):
        return c.get("actor"), {KIND[s["action"]] for s in c["steps"] if s["action"] in KIND}, "founder"
    if c.get("verdict") in ("not_sequence", "skip"):
        return None, None, "dropped"
    p = task["proposal"]
    return p.get("actor"), {KIND[s["action"]] for s in p["steps"] if s["action"] in KIND}, "prelabel"


def load_dataset(tasks_path, pose_dir: Path, fps=10):
    rows = []
    cache = {}
    for line in Path(tasks_path).read_text().splitlines():
        t = json.loads(line)
        actor, kinds, source = label_of(t)
        if source == "dropped":
            continue
        vid = t["source"]["youtube_id"]
        if vid not in cache:
            pp, bp = pose_dir / f"{vid}.pose.json", pose_dir / f"{vid}.boxes.json"
            if not pp.exists() or not bp.exists():
                cache[vid] = None
            else:
                from golden.render_boxes import filled_boxes
                cache[vid] = ({int(k): v for k, v in json.loads(pp.read_text()).items()},
                              filled_boxes(json.loads(bp.read_text())))
        if cache[vid] is None:
            continue
        pose, boxes = cache[vid]
        f = features(pose, boxes, t["span"]["start_ms"], t["span"]["end_ms"], fps)
        if f["a_seen"] < 0.3 and f["b_seen"] < 0.3:
            continue
        rows.append(dict(video=vid, feats=f, actor=actor, kinds=kinds, source=source))
    return rows


def train(tasks_path, pose_dir, report_path=None, model_path=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    rows = load_dataset(tasks_path, Path(pose_dir))
    names = sorted({k for r in rows for k in r["feats"]})
    videos = sorted({r["video"] for r in rows})
    report = dict(sequences=len(rows), videos=len(videos), label_sources=dict(Counter(r["source"] for r in rows)))
    folds = GroupKFold(n_splits=min(5, len(videos)))
    groups = np.array([r["video"] for r in rows])
    # actor: mirrored copies double the data and make the model symmetric in A/B
    act = [r for r in rows if r["actor"] in ("a", "b")]
    Xa = np.array([vector(r["feats"], names) for r in act])
    ya = np.array([r["actor"] == "a" for r in act], dtype=int)
    ga = np.array([r["video"] for r in act])
    pred = np.zeros(len(act))
    for tr, te in folds.split(Xa, ya, ga):
        Xtr = np.vstack([Xa[tr], np.array([vector(mirrored(act[i]["feats"]), names) for i in tr])])
        ytr = np.concatenate([ya[tr], 1 - ya[tr]])
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
        m.fit(Xtr, ytr)
        # average the prediction and its mirror so P(A)+P(B)=1 exactly
        p = m.predict_proba(Xa[te])[:, 1]
        pm = m.predict_proba(np.array([vector(mirrored(act[i]["feats"]), names) for i in te]))[:, 1]
        pred[te] = (p + 1 - pm) / 2
    report["actor"] = dict(n=len(act), accuracy=round(float(np.mean((pred > 0.5) == ya)), 3),
                           majority=round(float(max(ya.mean(), 1 - ya.mean())), 3),
                           by_source={s: round(float(np.mean([(pred[i] > 0.5) == ya[i] for i in range(len(act)) if act[i]["source"] == s])), 3)
                                      for s in {r["source"] for r in act}})
    X = np.array([vector(r["feats"], names) for r in rows])
    report["kinds"] = {}
    for kind in KINDS:
        y = np.array([kind in r["kinds"] for r in rows], dtype=int)
        if y.sum() < 15 or (len(y) - y.sum()) < 15:
            report["kinds"][kind] = dict(positives=int(y.sum()), skipped="too few")
            continue
        p = np.zeros(len(rows))
        for tr, te in folds.split(X, y, groups):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        report["kinds"][kind] = dict(positives=int(y.sum()), auc=round(float(roc_auc_score(y, p)), 3),
                                     f1=round(float(f1_score(y, p > 0.5)), 3), base_rate=round(float(y.mean()), 3))
    if model_path:
        final = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
        final.fit(np.vstack([Xa, np.array([vector(mirrored(r["feats"]), names) for r in act])]), np.concatenate([ya, 1 - ya]))
        pickle.dump(dict(names=names, actor=final), open(model_path, "wb"))
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report


def gold(pose_path, boxes_path, fps, offset_ms, model_path):
    """Who threw each founder-labelled technique (CW R2 3:50-4:47), from pose alone."""
    from golden import gold as g
    from golden.render_boxes import filled_boxes

    model = pickle.load(open(model_path, "rb"))
    pose = {int(k): v for k, v in json.load(open(pose_path)).items()}
    boxes = filled_boxes(json.load(open(boxes_path)))
    right = wrong = none = 0
    for t, actor, name, _ in g.EXPECTED:
        rel = t - offset_ms
        f = features(pose, boxes, rel - 700, rel + 300, fps)
        if f["a_seen"] < 0.3 or f["b_seen"] < 0.3:
            none += 1
            continue
        x = vector(f, model["names"])[None]
        xm = vector(mirrored(f), model["names"])[None]
        p = (model["actor"].predict_proba(x)[0, 1] + 1 - model["actor"].predict_proba(xm)[0, 1]) / 2
        if (p > 0.5) == (actor == "a"):
            right += 1
        else:
            wrong += 1
    result = dict(right=right, wrong=wrong, no_pose=none, of=len(g.EXPECTED))
    print(json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tasks", required=True)
    t.add_argument("--pose-dir", required=True)
    t.add_argument("--report")
    t.add_argument("--model")
    g = sub.add_parser("gold")
    g.add_argument("--pose", required=True)
    g.add_argument("--boxes", required=True)
    g.add_argument("--fps", type=int, default=25)
    g.add_argument("--offset-ms", type=int, default=228000)
    g.add_argument("--model", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.tasks, a.pose_dir, a.report, a.model)
    else:
        gold(a.pose, a.boxes, a.fps, a.offset_ms, a.model)


if __name__ == "__main__":
    main()
