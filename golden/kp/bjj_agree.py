"""Agreement between two VLM Studio BJJ runs of the same footage (no ground truth: a cheaper setting is compared
with production, and a production rerun gives the noise floor).

Positions: per second, the same position name and the same role for each player (who has guard, who is on top).
Techniques: one-to-one matching of events with the same actor and the same action class whose starts are within
2 s; reported as matched / events in each run, and outcome agreement among the matches.
Cost: list-price dollars from the token counts in each run's journal (Hydrogen $1.50 / $7.50 per million).

Usage: bjj_agree.py RUNS_DIR RUN_A RUN_B      (run ids or unique prefixes)
"""
import glob
import json
import sqlite3
import sys
from pathlib import Path

PRICE = {"gemini-4-hydrogen": (1.50, 7.50)}


def find(root, rid):
    hits = [p for p in glob.glob(str(Path(root) / f"{rid}*")) if Path(p).is_dir()]
    if len(hits) != 1:
        raise SystemExit(f"{rid}: {len(hits)} matching runs")
    return Path(hits[0])


def cost(run):
    total, calls = 0.0, 0
    for (body,) in sqlite3.connect(run / "journal.sqlite").execute("select body from heads"):
        u = json.loads(body).get("usage") or {}
        if u.get("kind") != "model":
            continue
        p_in, p_out = PRICE[u["model"]]
        total += (u.get("input_tokens") or 0) * p_in / 1e6 + ((u.get("output_tokens") or 0) + (u.get("thinking_tokens") or 0)) * p_out / 1e6
        calls += 1
    return total, calls


def seconds(run):
    doc = json.load(open(run / "positions.json"))
    out = {}
    for p in doc["positions"] or []:
        roles = dict(zip(p["player_ids"], p.get("roles") or []))
        for s in range(p["bounds"]["start_ms"] // 1000, p["bounds"]["end_ms"] // 1000):
            out[s] = (p.get("position"), roles.get("fighter-a"), roles.get("fighter-b"))
    return out


def events(run):
    doc = json.load(open(run / "techniques.json"))
    return [dict(t=e["bounds"]["start_ms"], actor=e.get("actor_player_id"), cls=tuple(sorted(e.get("action_classes") or [])),
                 outcome=e.get("outcome")) for e in doc["techniques"] or []]


def compare(root, a, b):
    ra, rb = find(root, a), find(root, b)
    pa, pb = seconds(ra), seconds(rb)
    both = set(pa) & set(pb)
    pos = sum(pa[s][0] == pb[s][0] for s in both)
    pos_roles = sum(pa[s] == pb[s] for s in both)
    ea, eb = events(ra), events(rb)
    pairs = sorted((abs(x["t"] - y["t"]), i, j) for i, x in enumerate(ea) for j, y in enumerate(eb)
                   if x["actor"] == y["actor"] and x["cls"] == y["cls"] and abs(x["t"] - y["t"]) <= 2000)
    ua, ub, m = set(), set(), []
    for _, i, j in pairs:
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j); m.append((i, j))
    ca, cb = cost(ra), cost(rb)
    return dict(seconds=len(both), position_same=round(pos / max(len(both), 1), 3),
                position_and_roles_same=round(pos_roles / max(len(both), 1), 3),
                events_a=len(ea), events_b=len(eb), matched=len(m),
                technique_f1=round(2 * len(m) / max(len(ea) + len(eb), 1), 3),
                outcome_same=round(sum(ea[i]["outcome"] == eb[j]["outcome"] for i, j in m) / max(len(m), 1), 3),
                cost_a=round(ca[0], 3), cost_b=round(cb[0], 3), calls_a=ca[1], calls_b=cb[1])


if __name__ == "__main__":
    print(json.dumps(compare(*sys.argv[1:4])))
