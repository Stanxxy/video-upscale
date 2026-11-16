"""G2 pre-label model bake-off: every registry model answers the same census windows.

    python -m golden.bakeoff run --models kimi-k3.1-pro,qwen4-max --rounds cw-r2,cw-r3,fa-r2 [--windows 0]
    python -m golden.bakeoff report

Answers are cached per model, round and window (a rerun only calls what is missing or failed),
so the bake-off can be resumed and re-scored for free. Run with OPENROUTER_API_KEY in the env.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from golden import census, gold, media, runner

HERE = Path(__file__).resolve().parent
OUT = media.ROOT / "x-mmagold" / "bakeoff"
RUNS = media.ROOT / "vlm-core" / "runs"


def registry():
    return yaml.safe_load((HERE / "models.yaml").read_text())


def rounds():
    return yaml.safe_load((HERE / "rounds.yaml").read_text())


def scope_of(r) -> census.Scope:
    return census.Scope(r["start_ms"], r["end_ms"], tuple(r["fighters"]))


def call_window(model_id: str, entry: dict, round_id: str, r: dict, index: int) -> dict:
    scope = scope_of(r)
    crops = census.read_crops(RUNS / r["crops_run"], scope.fighters)
    return runner.call_window(entry, scope, index, media.source_path(r["source"]), crops,
                              OUT / model_id / round_id / f"w{index:02d}.json")


def run(models, round_ids, windows=None, parallel=6):
    reg, rs = registry(), rounds()
    jobs = []
    for m in models:
        for rid in round_ids:
            n = len(scope_of(rs[rid]).windows())
            for i in (windows if windows is not None else range(n)):
                if i < n:
                    jobs.append((m, reg[m], rid, rs[rid], i))
    started = time.time()
    with ThreadPoolExecutor(parallel) as pool:
        for rec in pool.map(lambda j: call_window(*j), jobs):
            cost = runner.record_cost(rec)
            print(f"{'ok ' if rec['ok'] else 'ERR'} w{rec['index']:02d} {rec['window']} ${cost:.4f}"
                  + ("" if rec["ok"] else f" {rec['attempts'][-1]['error'][:160]}"), flush=True)
    print(f"{len(jobs)} windows in {time.time() - started:.0f}s")


def collect(model_id, round_id, r):
    """(merged flat sequences or None when a window is missing, window stats)."""
    scope = scope_of(r)
    n = len(scope.windows())
    recs = []
    for i in range(n):
        p = OUT / model_id / round_id / f"w{i:02d}.json"
        recs.append(json.loads(p.read_text()) if p.exists() else None)
    cost = latency = 0.0
    ok = calls = 0
    for rec in recs:
        if rec is None:
            continue
        for a in rec.get("attempts", []):
            calls += 1
            cost += (a.get("reply") or {}).get("cost") or 0
        if rec.get("ok"):
            ok += 1
            calls += 1
            cost += rec["reply"].get("cost") or 0
            latency += rec["reply"].get("latency_s") or 0
    stats = dict(windows=n, ok=ok, calls=calls, cost=round(cost, 4), mean_latency_s=round(latency / ok, 1) if ok else None,
                 minutes=(r["end_ms"] - r["start_ms"]) / 60000)
    if ok < n:
        return None, stats
    from bjj_analysis_core.mma_outputs import SequenceCensusV1
    batch = census.merge(scope, [SequenceCensusV1.model_validate(rec["census"]) for rec in recs])
    stats.update(merged_copies=batch.merged_copies)
    return census.flat(batch), stats


def report():
    reg, rs = registry(), rounds()
    rows = []
    for m in reg:
        if not (OUT / m).exists():
            continue
        row = dict(model=m, rounds={}, cost=0.0, minutes=0.0, windows=0, ok=0, calls=0)
        for rid, r in rs.items():
            if not (OUT / m / rid).exists():
                continue
            seqs, stats = collect(m, rid, r)
            row["cost"] += stats["cost"]
            row["windows"] += stats["windows"]
            row["ok"] += stats["ok"]
            row["calls"] += stats["calls"]
            if seqs is not None:
                row["minutes"] += stats["minutes"]
                entry = dict(stats, sequences=len(seqs), steps=sum(len(s["steps"]) for s in seqs),
                             unknown_actor=sum(1 for s in seqs if s["actor"] is None))
                if rid == "cw-r2":
                    entry["gold"] = gold.score_minute(seqs)
                row["rounds"][rid] = entry
                (OUT / m / rid / "merged.json").write_text(json.dumps(seqs, indent=1))
            else:
                row["rounds"][rid] = dict(stats, incomplete=True)
        row["cost"] = round(row["cost"], 3)
        row["cost_per_min"] = round(row["cost"] / row["minutes"], 3) if row["minutes"] else None
        rows.append(row)
    (OUT / "report.json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        g = row["rounds"].get("cw-r2", {}).get("gold")
        gs = (f"right {g['right_fighter']}/30 other {g['other_fighter_only']} names {g['name_present']}/30 "
              f"defence {g['defence_counted']}/6 dup {g['duplicates']} | ex found {g['ex_found']}/10 starter {g['ex_starter']}/10 "
              f"types {g['ex_types']}/10 count {g['ex_count']}/10") if g else "no gold score"
        print(f"{row['model']:20s} ok {row['ok']}/{row['windows']} calls {row['calls']} ${row['cost']:.2f} "
              f"(${row['cost_per_min'] or 0:.3f}/min) | {gs}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", required=True)
    r.add_argument("--rounds", default="cw-r2,cw-r3,fa-r2")
    r.add_argument("--windows", default=None, help="comma-separated window indexes (default: all)")
    r.add_argument("--parallel", type=int, default=6)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.models.split(","), a.rounds.split(","), [int(x) for x in a.windows.split(",")] if a.windows else None, a.parallel)
    else:
        report()


if __name__ == "__main__":
    main()
