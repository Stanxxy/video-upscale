"""Precision / recall / F1 on the gold minute with one-to-one matching, so a system cannot win by listing a lot.

A predicted attack matches a founder label when it is the same fighter and kind within 0.7 s; each label and
each prediction is used once (closest first). Reported: who+kind precision, recall, F1; exact technique name among
the matches; and the same with fighter only.
Usage: f1gold.py WINDOWS_JSON RUN_JSONL...    (rows: window, rep, answer.attacks; times relative to the window)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.gold import EXPECTED, kind  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import GOLD_START_MS, pred_kind  # noqa: E402


def one_to_one(preds, same):
    pairs = sorted((abs(p["t"] - t), i, j) for i, p in enumerate(preds) for j, (t, a, n, o) in enumerate(EXPECTED)
                   if abs(p["t"] - t) <= 700 and same(p, a, n))
    used_p, used_g, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); out.append((i, j))
    return out


def evaluate(wins, rows):
    by_win = {w["id"]: w for w in wins}
    preds = []
    for r in rows:
        t0 = GOLD_START_MS + by_win[r["window"]]["f0"] * 40
        for a in r["answer"]["attacks"]:
            preds.append(dict(t=t0 + a["time_s"] * 1000, fighter=a["fighter"].lower(), kind=pred_kind(a),
                              technique=a["technique"], action=a["action"]))
    kept = []                       # overlapping windows report the same attack twice
    for p in sorted(preds, key=lambda p: p["t"]):
        if not any(k["fighter"] == p["fighter"] and k["kind"] == p["kind"] and abs(k["t"] - p["t"]) <= 320 for k in kept):
            kept.append(p)
    preds = kept
    m = one_to_one(preds, lambda p, a, n: p["fighter"] == a and p["kind"] == kind(n))
    who = one_to_one(preds, lambda p, a, n: p["fighter"] == a)
    name = sum(EXPECTED[j][2] in (preds[i]["technique"], preds[i]["action"]) for i, j in m)
    n, g = max(len(preds), 1), len(EXPECTED)
    f1 = lambda h: 2 * h / (n + g)
    return dict(predicted=len(preds), kind_hits=len(m), kind_precision=len(m) / n, kind_recall=len(m) / g, kind_f1=f1(len(m)),
                name_hits=name, name_f1=f1(name), who_hits=len(who), who_f1=f1(len(who)))


if __name__ == "__main__":
    wins = json.load(open(sys.argv[1]))
    for path in sys.argv[2:]:
        rows = [json.loads(line) for line in open(path)]
        rows = [r for r in rows if r["window"] in {w["id"] for w in wins if w["seg"] == "gold"}]
        reps = sorted({r.get("rep", 0) for r in rows})
        per = [evaluate(wins, [r for r in rows if r.get("rep", 0) == k]) for k in reps]
        mean = {k: round(sum(p[k] for p in per) / len(per), 2) for k in per[0]}
        print(json.dumps(dict(run=Path(path).stem, reps=len(reps), **mean)))
