"""VLM Studio strike review: two runs' strikes side by side, and hand labels as ground truth.

Two runs of the same video (for example 3.8 Flash and Hydrogen) are compared over the
part of the video both analysed. A strike of one run and a strike of the other are the
same strike when the same fighter throws them within MATCH_MS of each other (nearest
pairs first). The reviewer labels every item; labels are stored per video, keyed by the
fighter and the strike time, so they stay valid as ground truth for any later run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

STRIKES = frozenset({"strike_punch", "strike_kick", "strike_knee", "strike_elbow", "ground_strike"})
MATCH_MS = 400
ACTORS = ("fighter-a", "fighter-b")
Verdict = Literal["landed", "missed", "contact_unclear", "not_a_strike", "wrong_fighter", "unsure"]
REAL = frozenset({"landed", "missed", "contact_unclear", "wrong_fighter"})


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^(fighter-a|fighter-b)@\d{1,9}$")
    verdict: Optional[Verdict]  # None clears the label
    source: Literal["both", "a", "b", "added"]
    note: str = Field(default="", max_length=500)


def _strike(t, index):
    return dict(
        index=index,
        t_ms=(t["bounds"]["start_ms"] + t["bounds"]["end_ms"]) // 2,
        bounds=t["bounds"],
        actor=t.get("actor_player_id"),
        classes=list(t["action_classes"]),
        target=t.get("strike_target"),
        outcome=t.get("outcome"),
        significant=t.get("significant"),
        knockdown=bool(t.get("knockdown")),
        notes=t.get("notes") or "",
        evidence=t.get("outcome_evidence") or t.get("evidence") or "",
    )


def strikes(techniques, start_ms, end_ms):
    """The strikes of one run inside [start_ms, end_ms), in time order."""
    found = [
        _strike(t, i)
        for i, t in enumerate(techniques)
        if set(t.get("action_classes") or ()) & STRIKES and t.get("actor_player_id") in ACTORS
    ]
    return sorted((s for s in found if start_ms <= s["t_ms"] < end_ms), key=lambda s: (s["t_ms"], s["index"]))


def match(a, b, tolerance_ms=MATCH_MS):
    """Review items: matched pairs plus each run's unmatched strikes, in time order."""
    candidates = sorted(
        (abs(x["t_ms"] - y["t_ms"]), i, j)
        for i, x in enumerate(a)
        for j, y in enumerate(b)
        if x["actor"] == y["actor"] and abs(x["t_ms"] - y["t_ms"]) <= tolerance_ms
    )
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j), pairs.append((i, j))
    items = [dict(source="both", a=a[i], b=b[j]) for i, j in pairs]
    items += [dict(source="a", a=x, b=None) for i, x in enumerate(a) if i not in used_a]
    items += [dict(source="b", a=None, b=y) for j, y in enumerate(b) if j not in used_b]
    for item in items:
        ref = item["a"] or item["b"]  # run A's time names a matched strike
        item["t_ms"], item["actor"] = ref["t_ms"], ref["actor"]
        item["key"] = f"{ref['actor']}@{ref['t_ms']}"
    return sorted(items, key=lambda item: (item["t_ms"], item["actor"]))


def _labels_path(root: Path, youtube_id: str) -> Path:
    return root / "strike-review" / f"labels-{youtube_id}.json"


def load_labels(root: Path, youtube_id: str) -> dict:
    path = _labels_path(root, youtube_id)
    return json.loads(path.read_text()) if path.exists() else {}


def save_label(root: Path, youtube_id: str, label: LabelIn) -> dict:
    """Set or clear one label; the file is replaced atomically."""
    labels = load_labels(root, youtube_id)
    if label.verdict is None:
        labels.pop(label.key, None)
    else:
        actor, at = label.key.split("@")
        labels[label.key] = dict(
            verdict=label.verdict, source=label.source, note=label.note, actor=actor, t_ms=int(at),
            labelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    labels = dict(sorted(labels.items(), key=lambda kv: (kv[1]["t_ms"], kv[0])))
    path = _labels_path(root, youtube_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, indent=1))
    os.replace(tmp, path)
    return labels


def load_pairs(root: Path) -> list:
    """Prepared comparisons ({label, a, b} with run ids), if any were written for this Studio."""
    path = root / "strike-review" / "pairs.json"
    return json.loads(path.read_text()) if path.exists() else []
