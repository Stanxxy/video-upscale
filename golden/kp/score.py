"""Score classifier runs on the gold minute against the founder's labels.

Per labelled attack (30): right_fighter = an attack by the labelled fighter within 0.7 s (any kind; comparable to the
bake-off's 21/30 for Kimi); right_kind = same fighter and same kind (punch/kick/knee/takedown/clinch); name = the
founder's name among that fighter's attacks within 0.7 s (13/30 for Kimi); wrong_fighter_only = only the opponent
has an attack of that kind there. Per negative (6 blocks / defence): counted when the fighter is given a strike
within 0.4 s. Also a technique confusion table for punches and kicks the founder named.

Usage: score.py ORACLE_WINDOWS RUN_JSONL...   (prints one JSON line per run file and rep, then the mean)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.gold import EXPECTED, NEGATIVE, STRIKES, kind  # noqa: E402

GOLD_START_MS = 230000
SPECIFIC = {"jab", "cross", "hook", "overhand", "uppercut", "side_kick", "roundhouse", "teep", "calf_kick",
            "clinch_knee", "double_leg", "single_leg"}


def pred_kind(p):
    t = p["technique"]
    if t in ("jab", "cross", "hook", "uppercut", "overhand") or p["action"] == "strike_punch":
        return "punch"
    if p["action"] == "strike_kick" or t in ("roundhouse", "calf_kick", "teep", "side_kick", "front_kick"):
        return "kick"
    if p["action"] == "strike_knee" or t in ("clinch_knee", "straight_knee"):
        return "knee"
    if p["action"] == "takedown_attempt":
        return "takedown"
    if p["action"] == "clinch_entry":
        return "clinch"
    return p["action"]


def score(wins, rows):
    by_win = {w["id"]: w for w in wins}
    preds = []
    for r in rows:
        w = by_win[r["window"]]
        t0 = GOLD_START_MS + w["f0"] * 40
        for a in r["answer"]["attacks"]:
            preds.append(dict(a, t=t0 + a["time_s"] * 1000, fighter=a["fighter"].lower(), kind=pred_kind(a)))
    near = lambda t, d: [p for p in preds if abs(p["t"] - t) <= d]
    s = Counter()
    conf = defaultdict(Counter)
    for t, a, n, o in EXPECTED:
        ps = near(t, 700)
        mine = [p for p in ps if p["fighter"] == a]
        same = [p for p in mine if p["kind"] == kind(n)]
        s["right_fighter"] += bool(mine)
        s["right_kind"] += bool(same)
        s["name"] += any(n in (p["technique"], p["action"]) for p in mine)
        s["wrong_fighter_only"] += (not same) and any(p["fighter"] != a and p["kind"] == kind(n) for p in ps)
        if n in SPECIFIC:
            s["specific"] += 1
            if same:
                best = min(same, key=lambda p: abs(p["t"] - t))
                conf[n][best["technique"]] += 1
                s["specific_right"] += best["technique"] == n
            else:
                conf[n]["(missed)"] += 1
    for t, a, what in NEGATIVE:
        s["defence_counted"] += any(p["fighter"] == a and (p["action"] in STRIKES if what == "strike" else p["action"] == what)
                                    for p in near(t, 400))
    s["predicted"] = len(preds)
    return dict(s), {k: dict(v) for k, v in conf.items()}


def main(wins_path, *runs):
    wins = json.load(open(wins_path))
    for path in runs:
        rows = [json.loads(l) for l in open(path)]
        reps = sorted({r["rep"] for r in rows})
        per = [score(wins, [r for r in rows if r["rep"] == k]) for k in reps]
        keys = sorted({k for s, _ in per for k in s})
        mean = {k: round(sum(s.get(k, 0) for s, _ in per) / len(per), 1) for k in keys}
        cost = round(sum(r["cost"] for r in rows), 2)
        print(json.dumps(dict(run=Path(path).stem, reps=len(reps), mean=mean, cost=cost, confusion=per[0][1])))


if __name__ == "__main__":
    main(*sys.argv[1:])
