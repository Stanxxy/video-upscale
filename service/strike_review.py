"""VLM Studio technique review: two runs' techniques side by side, hand labels as ground truth.

Two runs of the same video (for example the per-strike workflow and the sequence
workflow) are compared over the part of the video both analysed. The review unit is one
technique: a per-strike or grappling event, or one step of a sequence (shown with its
whole chain). Units of the two runs are the same technique when the same fighter does
them within MATCH_MS (both strikes) or MATCH_GRAPPLING_MS (otherwise) of each other,
nearest pairs first, whatever class each run gave it: a takedown one run calls a punch
is one item. Labels are stored per video, keyed by the fighter and the unit's time
(a strike's contact, any other technique's start), so they stay valid for later runs.
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
MATCH_GRAPPLING_MS = 1000
ACTORS = ("fighter-a", "fighter-b")
# landed/missed read "worked"/"failed" for a technique that is not a strike;
# not_a_strike is "not a technique" (the stored value predates sequences).
Verdict = Literal[
    "landed", "missed", "contact_unclear", "not_a_strike", "wrong_fighter",
    "wrong_class", "wrong_technique", "unsure",
]
REAL = frozenset({"landed", "missed", "contact_unclear", "wrong_fighter", "wrong_class", "wrong_technique"})


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^(fighter-a|fighter-b)@\d{1,9}$")
    verdict: Optional[Verdict]  # None clears the label
    source: Literal["both", "a", "b", "added"]
    note: str = Field(default="", max_length=500)


def _name(step):
    name = (step.get("technique") or step["action"]).replace("_", " ")
    return name + (f" ×{step['repeat']}" if step.get("repeat", 1) > 1 else "")


def _unit(t, index, step=None, position=None):
    """One technique: an event without steps, or one step of a sequence event."""
    source = step or t
    classes = [step["action"]] if step else list(t["action_classes"])
    strike = bool(set(classes) & STRIKES)
    bounds = source["bounds"]
    if step is None:
        at = (bounds["start_ms"] + bounds["end_ms"]) // 2 if strike else bounds["start_ms"]
    else:
        at = bounds["end_ms"] if strike else bounds["start_ms"]  # a strike step ends at contact
    steps = t.get("steps")
    return dict(
        index=index,
        step=position,
        t_ms=at,
        bounds=bounds,
        actor=t.get("actor_player_id"),
        classes=classes,
        strike=strike,
        technique=(step or {}).get("technique"),
        repeat=(step or {}).get("repeat", 1),
        landed=(step or {}).get("landed"),
        target=source.get("strike_target"),
        outcome=source.get("outcome"),
        significant=source.get("significant"),
        knockdown=bool(source.get("knockdown")),
        notes=t.get("notes") or "",
        evidence=(step or {}).get("evidence") or t.get("outcome_evidence") or t.get("evidence") or "",
        chain=[_name(s) for s in steps] if steps else None,
    )


def techniques(events, start_ms, end_ms):
    """The techniques of one run inside [start_ms, end_ms), in time order."""
    found = []
    for i, t in enumerate(events):
        if t.get("actor_player_id") not in ACTORS:
            continue
        if t.get("steps"):
            found += [_unit(t, i, s, k) for k, s in enumerate(t["steps"])]
        else:
            found.append(_unit(t, i))
    return sorted((s for s in found if start_ms <= s["t_ms"] < end_ms), key=lambda s: (s["t_ms"], s["index"], s["step"] or 0))


def match(a, b):
    """Review items: matched pairs plus each run's unmatched techniques, in time order."""
    candidates = sorted(
        (abs(x["t_ms"] - y["t_ms"]), i, j)
        for i, x in enumerate(a)
        for j, y in enumerate(b)
        if x["actor"] == y["actor"]
        and abs(x["t_ms"] - y["t_ms"]) <= (MATCH_MS if x["strike"] and y["strike"] else MATCH_GRAPPLING_MS)
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
