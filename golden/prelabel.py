"""G4: pre-label whole videos with the bake-off winner, then write correction tasks.

    python -m golden.prelabel run --model kimi-k3.1-pro --videos ID[,ID...] [--parallel 8]
    python -m golden.prelabel tasks --model kimi-k3.1-pro --batch g4 --out tasks.jsonl

Inputs per video (temp_data/x-mmagold/g4): ``videos/<id>.mp4`` (source) and ``names/<id>.json`` (name_fighters:
the two fighters named by their kit). Fighter identity is by those names, not tracker boxes: the tracker's
appearance split failed on overhead fisheye footage (red mat read as skin, fighters lying down, merged tracks),
which is most of the G4 sources. Every window is sent; the model returns nothing for non-live footage.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from golden import census, runner, tasks
from golden.media import ROOT

G4 = ROOT / "x-mmagold" / "g4"
HERE = Path(__file__).resolve().parent


def duration_ms(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(float(out) * 1000)


def video_path(video_id: str) -> Path:
    return G4 / "videos" / f"{video_id}.mp4"


def names(video_id: str) -> tuple[str, str]:
    n = json.loads((G4 / "names" / f"{video_id}.json").read_text())
    return n["fighter_a"], n["fighter_b"]


def scope_for(video_id: str) -> census.Scope:
    return census.Scope(0, duration_ms(video_path(video_id)), names(video_id))


def out_dir(model_id: str, video_id: str) -> Path:
    return G4 / "prelabel" / model_id / video_id


def run(model_id: str, video_ids: list[str], parallel: int):
    entry = yaml.safe_load((HERE / "models.yaml").read_text())[model_id]
    jobs = []
    for vid in video_ids:
        scope = scope_for(vid)
        n = len(scope.windows())
        print(f"{vid}: {n} windows, fighters {scope.fighters}", flush=True)
        jobs += [(vid, scope, i) for i in range(n)]
    total = 0.0

    def one(job):
        vid, scope, i = job
        return vid, runner.call_window(entry, scope, i, video_path(vid), [], out_dir(model_id, vid) / f"w{i:03d}.json")

    with ThreadPoolExecutor(parallel) as pool:
        for vid, rec in pool.map(one, jobs):
            total += runner.record_cost(rec)
            if not rec["ok"]:
                print(f"ERR {vid} w{rec['index']} {rec['attempts'][-1]['error'][:160]}", flush=True)
    print(f"{len(jobs)} windows, ${total:.2f}")


def merged(model_id: str, video_id: str):
    from bjj_analysis_core.mma_outputs import SequenceCensusV1

    scope = scope_for(video_id)
    censuses = []
    for i in range(len(scope.windows())):
        rec = json.loads((out_dir(model_id, video_id) / f"w{i:03d}.json").read_text())
        if not rec["ok"]:
            raise SystemExit(f"{video_id} window {i} has no valid answer; rerun `run` first")
        censuses.append(SequenceCensusV1.model_validate(rec["census"]))
    return census.flat(census.merge(scope, censuses))


def build_tasks(model_id: str, batch: str, out: Path):
    candidates = {c["youtube_id"]: c for c in json.loads((ROOT / "x-mmagold" / "g3" / "candidates.json").read_text())}
    order = sorted((c for c in candidates.values() if c.get("g4_rank")), key=lambda c: c["g4_rank"])
    all_tasks = []
    for index, c in enumerate(order):
        vid = c["youtube_id"]
        if not out_dir(model_id, vid).exists():
            continue
        try:
            seqs = [s for s in merged(model_id, vid) if s["steps"]]
        except SystemExit as incomplete:
            print(f"{vid}: skipped ({incomplete})")
            continue
        source = dict(youtube_id=vid, title=c["title"], channel=c["channel_title"], license=c["license"], url=c["url"],
                      fighters=list(names(vid)))
        made = tasks.build(seqs, batch=batch, source_index=index, source=source, video=video_path(vid),
                           video_duration_ms=scope_for(vid).end_ms, model=model_id, prompt_version="census-exchange-v1",
                           clip_dir=G4 / "clips")
        print(f"{vid}: {len(made)} tasks")
        all_tasks += made
    tasks.write_jsonl(all_tasks, out)
    print(f"{len(all_tasks)} tasks -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--videos", required=True)
    r.add_argument("--parallel", type=int, default=8)
    t = sub.add_parser("tasks")
    t.add_argument("--model", required=True)
    t.add_argument("--batch", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.model, a.videos.split(","), a.parallel)
    else:
        build_tasks(a.model, a.batch, Path(a.out))


if __name__ == "__main__":
    main()
