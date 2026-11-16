"""G4: pre-label whole boxed videos with the bake-off winner, then write correction tasks.

    python -m golden.prelabel run --model kimi-k3.1-pro --videos ID[,ID...] [--parallel 8]
    python -m golden.prelabel tasks --model kimi-k3.1-pro --batch g4a --out tasks.jsonl

Inputs per video (temp_data/x-mmagold/g4): ``videos/<id>.mp4`` (source), ``track/<id>.boxes.json`` (gx10
track_assign at 10 fps), ``boxed/<id>.mp4`` and ``boxed/<id>-0.png``/``-1.png`` (render_boxes). Windows where
the tracker sees too little of the fighters (walkouts, interviews, replays, breaks) are not sent: they get an
empty census, which is what the model is asked to return for non-live footage anyway.
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
from golden.render_boxes import presence

G4 = ROOT / "x-mmagold" / "g4"
HERE = Path(__file__).resolve().parent
NAMES = ("Red box A", "Blue box B")
TRACK_FPS = 10
MIN_ANY, MIN_BOTH = 0.5, 0.2   # share of a window's tracking frames with >=1 / 2 fighters boxed


def duration_ms(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(float(out) * 1000)


def scope_for(video_id: str) -> census.Scope:
    return census.Scope(0, duration_ms(G4 / "boxed" / f"{video_id}.mp4"), NAMES)


def live_windows(video_id: str, scope: census.Scope) -> list[bool]:
    data = json.loads((G4 / "track" / f"{video_id}.boxes.json").read_text())
    seen = presence(data, TRACK_FPS)
    out = []
    for w in scope.windows():
        frames = range(int(w.start_ms / 1000 * TRACK_FPS) + 1, int(w.end_ms / 1000 * TRACK_FPS) + 1)
        counts = [seen.get(f, 0) for f in frames]
        n = max(len(counts), 1)
        out.append(sum(c >= 1 for c in counts) / n >= MIN_ANY and sum(c == 2 for c in counts) / n >= MIN_BOTH)
    return out


def crops(video_id: str):
    return [(pid, name, (G4 / "boxed" / f"{video_id}-{i}.png").read_bytes())
            for i, (pid, name) in enumerate(zip(census.PLAYER_IDS, NAMES))]


def out_dir(model_id: str, video_id: str) -> Path:
    return G4 / "prelabel" / model_id / video_id


def run(model_id: str, video_ids: list[str], parallel: int):
    entry = yaml.safe_load((HERE / "models.yaml").read_text())[model_id]
    jobs = []
    for vid in video_ids:
        scope = scope_for(vid)
        live = live_windows(vid, scope)
        (out_dir(model_id, vid)).mkdir(parents=True, exist_ok=True)
        (out_dir(model_id, vid) / "live.json").write_text(json.dumps(live))
        print(f"{vid}: {sum(live)}/{len(live)} windows live", flush=True)
        jobs += [(vid, scope, i) for i, ok in enumerate(live) if ok]
    total = 0.0

    def one(job):
        vid, scope, i = job
        return vid, runner.call_window(entry, scope, i, G4 / "boxed" / f"{vid}.mp4", crops(vid),
                                       out_dir(model_id, vid) / f"w{i:03d}.json")

    with ThreadPoolExecutor(parallel) as pool:
        for vid, rec in pool.map(one, jobs):
            total += runner.record_cost(rec)
            if not rec["ok"]:
                print(f"ERR {vid} w{rec['index']} {rec['attempts'][-1]['error'][:160]}", flush=True)
    print(f"{len(jobs)} windows, ${total:.2f}")


def merged(model_id: str, video_id: str):
    from bjj_analysis_core.mma_outputs import SequenceCensusV1

    scope = scope_for(video_id)
    live = json.loads((out_dir(model_id, video_id) / "live.json").read_text())
    censuses = []
    for i, ok in enumerate(live):
        if not ok:
            censuses.append(SequenceCensusV1(schema_version=1, window_index=i, sequences=()))
            continue
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
        if not (out_dir(model_id, vid) / "live.json").exists():
            continue
        seqs = [s for s in merged(model_id, vid) if s["steps"]]
        source = dict(youtube_id=vid, title=c["title"], channel=c["channel_title"], license=c["license"], url=c["url"])
        made = tasks.build(seqs, batch=batch, source_index=index, source=source, video=G4 / "boxed" / f"{vid}.mp4",
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
