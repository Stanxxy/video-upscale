"""Merged pre-label sequences -> correction tasks (JSONL + clips) for ``helping_scripts/golden_labels.py import``.

One task per sequence. Its clip is the sequence span with ``PAD_MS`` either side, cut from the boxed video when
there is one (red A / blue B: the identity the founder corrects against), 480p, no audio.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from golden import media

PAD_MS = 1000
OUTCOME = {"successful": "landed", "unsuccessful": "missed"}
TECHNIQUE_NAMES = None


def _names():
    global TECHNIQUE_NAMES
    if TECHNIQUE_NAMES is None:
        from golden.census import bundle
        TECHNIQUE_NAMES = bundle()["technique_names"]
    return TECHNIQUE_NAMES


def proposal_steps(steps):
    names = _names()
    out = []
    for s in steps:
        technique = s.get("technique")
        if technique is not None and technique not in names.get(s["action"], ()):
            technique = None
        out.append(dict(action=s["action"], technique=technique, outcome=OUTCOME.get(s["outcome"], "unclear"),
                        repeat=int(s.get("repeat") or 1)))
    return out


def task_id(batch: str, youtube_id: str, start_ms: int, actor) -> str:
    h = hashlib.sha1(youtube_id.encode()).hexdigest()[:8]
    return f"{batch}-{h}-{start_ms}{actor or 'x'}"


def build(sequences, *, batch: str, source_index: int, source: dict, video: Path, video_duration_ms: int,
          model: str, prompt_version: str, clip_dir: Path) -> list[dict]:
    """``sequences``: flat merged sequences (census.flat). ``source``: youtube_id, title, channel, license, url."""
    tasks = []
    for s in sequences:
        start, end = max(0, s["start"] - PAD_MS), min(video_duration_ms, s["end"] + PAD_MS)
        tid = task_id(batch, source["youtube_id"], s["start"], s["actor"])
        clip = media.cut(video, start, end, height=480, fps=25, crf=28, out=clip_dir / f"{tid}.mp4")
        tasks.append(dict(
            task_id=tid, batch=batch, order=f"{batch}:{source_index:03d}:{s['start']:08d}:{s['actor'] or 'x'}",
            source=source, span=dict(start_ms=s["start"], end_ms=s["end"]),
            clip=dict(start_ms=start, end_ms=end, file=str(clip)),
            proposal=dict(actor=s["actor"], summary=s["summary"], steps=proposal_steps(s["steps"])),
            model=model, prompt_version=prompt_version,
        ))
    return tasks


def write_jsonl(tasks, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
