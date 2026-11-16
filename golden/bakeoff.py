"""G2 pre-label model bake-off: every registry model answers the same census windows.

    python -m golden.bakeoff run --models kimi-k3.1-pro,qwen4-max --rounds cw-r2,cw-r3,fa-r2 [--windows 0]
    python -m golden.bakeoff report
    python -m golden.bakeoff g2-tasks --model kimi-k3.1-pro --out g2.jsonl     # founder corrects these first
    python -m golden.bakeoff rescore --export export.jsonl                     # every model vs the corrections

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


def g2_tasks(model_id: str, out: Path):
    """Correction tasks for the bake-off rounds from one model's merged answers (batch g2, queued first)."""
    from golden import tasks
    rs, made = rounds(), []
    for index, (rid, r) in enumerate(rs.items()):
        seqs, stats = collect(model_id, rid, r)
        if seqs is None:
            raise SystemExit(f"{model_id} {rid} is incomplete")
        src = dict(youtube_id=r["source"], title=f"Bake-off round {rid}", channel="", license="evaluation",
                   url=f"https://www.youtube.com/watch?v={r['source']}", fighters=list(r["fighters"]))
        made += tasks.build([s for s in seqs if s["steps"]], batch="g2", source_index=index, source=src,
                            video=media.source_path(r["source"]), video_duration_ms=r["end_ms"] + 2000, model=model_id,
                            prompt_version="census-exchange-v1", clip_dir=media.ROOT / "x-mmagold" / "g2" / "clips")
    tasks.write_jsonl(made, out)
    print(f"{len(made)} tasks -> {out}")


def rescore(export: Path):
    """Score every model against the founder's corrections of batch g2 (export of the task table).

    A corrected sequence (accept/corrected) is a gold sequence: its fighter, span and technique kinds. A model is
    credited when one of its sequences overlaps the span (found) and belongs to the same fighter (right fighter);
    'not a sequence' spans count the model's sequences that overlap them (false alarms). Bias: the tasks started
    from one model's proposals, so sequences every model missed are only in the founder's free-text notes.
    """
    gold_by_source = {}
    for line in Path(export).read_text().splitlines():
        t = json.loads(line)
        c = t.get("correction") or {}
        if t["batch"] != "g2" or c.get("verdict") not in ("accept", "corrected", "not_sequence"):
            continue
        gold_by_source.setdefault(t["source"]["youtube_id"], []).append(
            dict(span=t["span"], verdict=c["verdict"], actor=c.get("actor"),
                 kinds={gold.kind(s.get("technique") or s["action"]) for s in c.get("steps") or []}))
    rows = []
    for m in registry():
        totals = dict(model=m, gold=0, found=0, right_fighter=0, kinds_exact=0, not_seq=0, false_alarm=0, rounds=0)
        for rid, r in rounds().items():
            seqs, _ = collect(m, rid, r) if (OUT / m / rid).exists() else (None, None)
            golds = [g for g in gold_by_source.get(r["source"], []) if r["start_ms"] <= g["span"]["start_ms"] < r["end_ms"]]
            if seqs is None or not golds:
                continue
            totals["rounds"] += 1
            for g in golds:
                lo, hi = g["span"]["start_ms"] - PAD, g["span"]["end_ms"] + PAD
                over = [q for q in seqs if q["start"] <= hi and q["end"] >= lo]
                if g["verdict"] == "not_sequence":
                    totals["not_seq"] += 1
                    totals["false_alarm"] += bool(over)
                    continue
                totals["gold"] += 1
                totals["found"] += bool(over)
                mine = [q for q in over if q["actor"] == g["actor"]]
                totals["right_fighter"] += bool(mine)
                totals["kinds_exact"] += {gold.kind(st["technique"] or st["action"]) for q in mine for st in q["steps"]} == g["kinds"]
        if totals["rounds"]:
            rows.append(totals)
    for t in rows:
        print(f"{t['model']:20s} rounds {t['rounds']} | found {t['found']}/{t['gold']} right fighter {t['right_fighter']}/{t['gold']} "
              f"kinds exact {t['kinds_exact']}/{t['gold']} | false alarms {t['false_alarm']}/{t['not_seq']}")
    (OUT / "rescore.json").write_text(json.dumps(rows, indent=1))
    return rows


PAD = 600


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", required=True)
    r.add_argument("--rounds", default="cw-r2,cw-r3,fa-r2")
    r.add_argument("--windows", default=None, help="comma-separated window indexes (default: all)")
    r.add_argument("--parallel", type=int, default=6)
    sub.add_parser("report")
    t = sub.add_parser("g2-tasks")
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    rs = sub.add_parser("rescore")
    rs.add_argument("--export", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.models.split(","), a.rounds.split(","), [int(x) for x in a.windows.split(",")] if a.windows else None, a.parallel)
    elif a.cmd == "g2-tasks":
        g2_tasks(a.model, Path(a.out))
    elif a.cmd == "rescore":
        rescore(Path(a.export))
    else:
        report()


if __name__ == "__main__":
    main()
