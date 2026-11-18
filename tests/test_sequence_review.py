"""VLM Studio sequence review: one run's sequences and their labels."""
import json

import pytest
from pydantic import ValidationError

from service import sequence_review as sr


def event(start, end, actor="fighter-a", steps=None, **extra):
    return dict(bounds=dict(start_ms=start, end_ms=end), actor_player_id=actor, action_classes=["strike_punch"],
                outcome="successful", notes="jab, cross", steps=steps, strike_target="head", **extra)


def test_sequences_list_each_chain_in_time_order_and_events_without_steps_as_one_step():
    steps = [dict(action="strike_punch", technique="jab", outcome="unsuccessful", strike_target="head", repeat=1, landed=None),
             dict(action="ground_strike", technique="punch", outcome="successful", strike_target="head", repeat=4, landed=2)]
    found = sr.sequences([event(5000, 6000), event(1000, 2500, steps=steps)])
    assert [s["key"] for s in found] == ["fighter-a@1000", "fighter-a@5000"]
    assert [c["name"] for c in found[0]["chain"]] == ["jab", "punch ×4"] and found[0]["chain"][1]["landed"] == 2
    assert [c["name"] for c in found[1]["chain"]] == ["strike punch"]


def test_labels_keep_flags_count_and_note_and_clear_only_when_empty(tmp_path):
    sr.save_label(tmp_path, "XFBseoY4ovl", sr.SequenceLabelIn(key="fighter-b@9000", source="run", flags=["correct"]))
    labels = sr.save_label(tmp_path, "XFBseoY4ovl", sr.SequenceLabelIn(
        key="fighter-a@1000", source="run", flags=["missed_techniques", "wrong_names", "wrong_names"], count=3,
        note=" jab, hook, low kick "))
    assert list(labels) == ["fighter-a@1000", "fighter-b@9000"]
    assert labels["fighter-a@1000"] | {"labelled_at": None} == dict(
        source="run", flags=["missed_techniques", "wrong_names"], count=3, note="jab, hook, low kick",
        actor="fighter-a", t_ms=1000, labelled_at=None)
    stored = json.loads((tmp_path / "sequence-review" / "labels-XFBseoY4ovl.json").read_text())
    assert stored["fighter-b@9000"]["flags"] == ["correct"]
    assert "fighter-b@9000" not in sr.save_label(tmp_path, "XFBseoY4ovl", sr.SequenceLabelIn(key="fighter-b@9000", source="run"))
    with pytest.raises(ValidationError):
        sr.SequenceLabelIn(key="fighter-a@1000", source="run", flags=["blurry"])
