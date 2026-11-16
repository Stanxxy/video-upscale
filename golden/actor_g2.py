"""Who attacks, from tracked pose, on the bake-off rounds (broadcast cameras, where tracker identity holds).

Train: every G2 sequence outside the founder-labelled minute, labelled by the founder's correction where one
exists, else by the pre-label. Test: the founder's 30 labelled techniques (CW R2 3:50-4:47), each scored on the
pose in [t-700 ms, t+300 ms], compared with the pre-label model's own attribution on the same techniques.

    python -m golden.actor_g2 --tasks g2/tasks.jsonl --track-dir g2/track [--export export.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from golden import gold
from golden.baseline import features, label_of, mirrored, vector
from golden.render_boxes import filled_boxes

# round video: (source, offset of the 10 fps copy, tracker cluster "a" is the census's fighter-a?)
ROUNDS = {"XFBseoY4ER0": ("cw", 230000, True),     # A dark brown shorts = darker cluster
          "1ACTKcuQZyQ": ("fa", 520000, False)}    # A orange shorts; the darker cluster is B (black shorts)
GOLD_SPAN = (225000, 292000)
FPS = 10


def load_round(track_dir: Path, name: str, same: bool):
    pose = {int(k): v for k, v in json.loads((track_dir / f"{name}.pose.json").read_text()).items()}
    boxes = filled_boxes(json.loads((track_dir / f"{name}.boxes.json").read_text()))
    if not same:   # relabel tracker clusters so "a" is the census's fighter-a
        swap = {"a": "b", "b": "a"}
        pose = {f: {swap[k]: v for k, v in d.items()} for f, d in pose.items()}
        boxes = {f: {swap[k]: v for k, v in d.items()} for f, d in boxes.items()}
    return pose, boxes


def main(tasks_path, track_dir, export=None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    track_dir = Path(track_dir)
    data = {src: load_round(track_dir, name, same) for src, (name, _, same) in ROUNDS.items()}
    corrected = {}
    if export:
        for line in Path(export).read_text().splitlines():
            t = json.loads(line)
            corrected[t["task_id"]] = t
    train = []
    for line in Path(tasks_path).read_text().splitlines():
        t = corrected.get(json.loads(line)["task_id"], json.loads(line))
        src = t["source"]["youtube_id"]
        actor, _, source = label_of(t)
        if actor not in ("a", "b") or src not in ROUNDS:
            continue
        s, e = t["span"]["start_ms"], t["span"]["end_ms"]
        if src == "XFBseoY4ER0" and s < GOLD_SPAN[1] and e > GOLD_SPAN[0]:
            continue
        off = ROUNDS[src][1]
        pose, boxes = data[src]
        f = features(pose, boxes, s - off, e - off, FPS)
        if f["a_seen"] < 0.3 or f["b_seen"] < 0.3:
            continue
        train.append((f, actor, source))
    names = sorted({k for f, _, _ in train for k in f})
    X = np.array([vector(f, names) for f, _, _ in train] + [vector(mirrored(f), names) for f, _, _ in train])
    y = np.array([a == "a" for _, a, _ in train] + [a != "a" for _, a, _ in train], dtype=int)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0).fit(X, y)
    pose, boxes = data["XFBseoY4ER0"]
    right = wrong = none = 0
    for t, actor, _, _ in gold.EXPECTED:
        rel = t - ROUNDS["XFBseoY4ER0"][1]
        f = features(pose, boxes, rel - 700, rel + 300, FPS)
        if f["a_seen"] < 0.3 or f["b_seen"] < 0.3:
            none += 1
            continue
        p = (model.predict_proba(vector(f, names)[None])[0, 1] + 1 - model.predict_proba(vector(mirrored(f), names)[None])[0, 1]) / 2
        right += (p > 0.5) == (actor == "a")
        wrong += (p > 0.5) != (actor == "a")
    result = dict(train_sequences=len(train), train_label_sources={s: sum(1 for *_, x in train if x == s) for s in {x for *_, x in train}},
                  gold=dict(right=int(right), wrong=int(wrong), no_pose=none, of=len(gold.EXPECTED)))
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--track-dir", required=True)
    ap.add_argument("--export")
    a = ap.parse_args()
    main(a.tasks, a.track_dir, a.export)
