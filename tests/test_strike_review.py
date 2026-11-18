"""VLM Studio technique review: matching two runs' techniques and storing hand labels."""
import json

import pytest
from pydantic import ValidationError

from service import strike_review as sr


def tech(t_ms, actor="fighter-a", classes=("strike_punch",), outcome="successful"):
    return dict(bounds=dict(start_ms=t_ms - 250, end_ms=t_ms + 250), actor_player_id=actor, action_classes=list(classes),
                strike_target="head", outcome=outcome, significant=True, knockdown=False, notes="jab", outcome_evidence="snaps head")


def step(start, end, action="strike_punch", technique="jab", **extra):
    return dict(bounds=dict(start_ms=start, end_ms=end), action=action, technique=technique, outcome="successful",
                strike_target="head" if action.startswith("strike") or action == "ground_strike" else None,
                significant=True, knockdown=False, evidence="seen", **{"repeat": 1, "landed": None, **extra})


def sequence(*steps, actor="fighter-a"):
    return dict(bounds=dict(start_ms=steps[0]["bounds"]["start_ms"], end_ms=steps[-1]["bounds"]["end_ms"]),
                actor_player_id=actor, action_classes=list(dict.fromkeys(s["action"] for s in steps)),
                outcome="successful", notes="jab, cross into a double leg", steps=list(steps))


def test_every_technique_inside_the_shared_scope_is_compared():
    found = sr.techniques([tech(1000), tech(2000, classes=("takedown_attempt",)), tech(9000), tech(500, actor=None)], 0, 5000)
    # A strike is placed at its contact (the event's middle), anything else at its start.
    assert [(s["t_ms"], s["strike"]) for s in found] == [(1000, True), (1750, False)]
    assert found[0]["classes"] == ["strike_punch"] and found[0]["evidence"] == "snaps head"


def test_each_sequence_step_is_one_unit_carrying_its_chain():
    chain = sequence(step(1000, 1300), step(1400, 1700, technique="cross"),
                     step(1900, 3000, "takedown_attempt", "double_leg"),
                     step(3200, 4000, "ground_strike", "punch", repeat=4, landed=2))
    units = sr.techniques([chain], 0, 10_000)
    assert [(u["t_ms"], u["technique"], u["step"]) for u in units] == [
        (1300, "jab", 0), (1700, "cross", 1), (1900, "double_leg", 2), (4000, "punch", 3),
    ]
    assert units[2]["chain"] == ["jab", "cross", "double leg", "punch ×4"] and units[3]["landed"] == 2


def test_a_takedown_one_run_calls_a_punch_is_one_item():
    a = sr.techniques([sequence(step(1900, 3000, "takedown_attempt", "double_leg"))], 0, 10_000)
    b = sr.techniques([tech(2600)], 0, 10_000)
    (item,) = sr.match(a, b)
    assert item["source"] == "both" and item["a"]["classes"] == ["takedown_attempt"] and item["b"]["classes"] == ["strike_punch"]


def test_matching_pairs_nearest_same_fighter_strikes_within_tolerance():
    a = sr.techniques([tech(1000), tech(1300), tech(5000, actor="fighter-b"), tech(8000)], 0, 10_000)
    b = sr.techniques([tech(1250), tech(5100, actor="fighter-a"), tech(8500)], 0, 10_000)
    items = sr.match(a, b)
    by = {(i["source"], i["t_ms"]) for i in items}
    # 1250 is nearer to 1300 than to 1000; 5100 is the other fighter; 8500 is 500 ms away.
    assert by == {("both", 1300), ("a", 1000), ("a", 5000), ("a", 8000), ("b", 5100), ("b", 8500)}
    both = next(i for i in items if i["source"] == "both")
    assert both["key"] == "fighter-a@1300" and both["b"]["t_ms"] == 1250
    assert [i["t_ms"] for i in items] == sorted(i["t_ms"] for i in items)


def test_labels_are_stored_per_video_cleared_and_validated(tmp_path):
    sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(key="fighter-b@5100", verdict="landed", source="b"))
    labels = sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(key="fighter-a@1300", verdict="not_a_strike", source="both"))
    assert list(labels) == ["fighter-a@1300", "fighter-b@5100"]  # file kept in time order
    stored = json.loads((tmp_path / "strike-review" / "labels-XFBseoY4ER0.json").read_text())
    assert stored["fighter-b@5100"]["verdict"] == "landed" and stored["fighter-b@5100"]["t_ms"] == 5100
    assert sr.load_labels(tmp_path, "EsqIo-wSBg4") == {}
    cleared = sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(key="fighter-b@5100", verdict=None, source="b"))
    assert list(cleared) == ["fighter-a@1300"]
    for bad in (dict(key="referee@1", verdict="landed", source="a"), dict(key="fighter-a@1", verdict="maybe", source="a"),
                dict(key="fighter-a@1", verdict="landed", source="a", extra=1)):
        with pytest.raises(ValidationError):
            sr.LabelIn(**bad)


def test_timing_and_several_techniques_are_flags_beside_the_verdict(tmp_path):
    labels = sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(
        key="fighter-a@1300", verdict="landed", source="both", note="cut starts late",
        flags=["timing_off", "holds_several", "timing_off"], count=2))
    assert labels["fighter-a@1300"]["flags"] == ["holds_several", "timing_off"]
    assert labels["fighter-a@1300"]["count"] == 2 and labels["fighter-a@1300"]["verdict"] == "landed"
    only_timing = sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(
        key="fighter-a@1300", verdict="landed", source="both", flags=["timing_off"], count=3))
    assert only_timing["fighter-a@1300"]["count"] is None  # a count belongs to holds_several
    with pytest.raises(ValidationError):
        sr.LabelIn(key="fighter-a@1300", verdict="landed", source="both", flags=["blurry"])
    correct = sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(key="fighter-a@1300", verdict="missed", source="both", flags=["correct"]))
    assert correct["fighter-a@1300"]["flags"] == ["correct"]


def test_a_note_or_flag_is_kept_without_a_verdict_and_only_an_empty_label_is_removed(tmp_path):
    kept = sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(key="fighter-a@900", verdict=None, source="a", note=" starts late "))
    assert kept["fighter-a@900"]["verdict"] is None and kept["fighter-a@900"]["note"] == "starts late"
    flagged = sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(key="fighter-a@900", verdict=None, source="a", flags=["timing_off"]))
    assert flagged["fighter-a@900"]["flags"] == ["timing_off"] and flagged["fighter-a@900"]["note"] == ""
    assert sr.save_label(tmp_path, "XFBseoY4ER0", sr.LabelIn(key="fighter-a@900", verdict=None, source="a")) == {}
