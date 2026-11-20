"""VLM Studio sequence review: one run's technique sequences, judged one by one.

The review unit is a whole sequence (one fighter's chain of techniques, not timed one
by one). Flags say what is wrong with it, a count says how many techniques it really
had, and a note says what really happened. Labels are stored per video, keyed by the
fighter and the sequence start, so they stay valid as ground truth for later runs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ACTORS = ("fighter-a", "fighter-b")
Flag = Literal[
    "correct", "wrong_fighter", "missed_techniques", "extra_techniques", "wrong_names",
    "wrong_landed", "not_real", "duplicate", "timing_off",
]


class SequenceLabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^(fighter-a|fighter-b)@\d{1,9}$")
    source: Literal["run", "added"]
    flags: list[Flag] = Field(default_factory=list)
    count: Optional[int] = Field(default=None, ge=0, le=60)  # techniques it really had
    note: str = Field(default="", max_length=800)


def _name(step):
    name = (step.get("technique") or step["action"]).replace("_", " ")
    if step.get("repeat", 1) > 1:
        name += f" ×{step['repeat']}"
    return name


def sequences(events):
    """The run's sequences in time order (events without steps are one-step sequences)."""
    out = []
    for t in events:
        actor = t.get("actor_player_id")
        steps = t.get("steps") or [dict(action=t["action_classes"][0], technique=None, outcome=t["outcome"],
                                        strike_target=t.get("strike_target"), repeat=1, landed=None)]
        b = t["bounds"]
        out.append(dict(
            key=f"{actor or t.get('actor_sentinel')}@{b['start_ms']}",
            start_ms=b["start_ms"], end_ms=b["end_ms"], actor=actor, sentinel=t.get("actor_sentinel"),
            summary=t.get("notes") or "", outcome=t["outcome"],
            chain=[dict(name=_name(s), action=s["action"], outcome=s["outcome"], target=s.get("strike_target"),
                        landed=s.get("landed"), repeat=s.get("repeat", 1), evidence=s.get("evidence") or "")
                   for s in steps],
        ))
    return sorted(out, key=lambda s: (s["start_ms"], s["key"]))


def _path(root: Path, youtube_id: str) -> Path:
    return root / "sequence-review" / f"labels-{youtube_id}.json"


def load_labels(root: Path, youtube_id: str) -> dict:
    path = _path(root, youtube_id)
    return json.loads(path.read_text()) if path.exists() else {}


def save_label(root: Path, youtube_id: str, label: SequenceLabelIn) -> dict:
    """Set or clear one label (no flags, no count and no note clears it); atomic write."""
    labels = load_labels(root, youtube_id)
    note = label.note.strip()
    if not label.flags and label.count is None and not note:
        labels.pop(label.key, None)
    else:
        actor, at = label.key.split("@")
        labels[label.key] = dict(
            source=label.source, flags=sorted(set(label.flags)), count=label.count, note=note,
            actor=actor, t_ms=int(at), labelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    labels = dict(sorted(labels.items(), key=lambda kv: (kv[1]["t_ms"], kv[0])))
    path = _path(root, youtube_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, indent=1))
    os.replace(tmp, path)
    return labels
