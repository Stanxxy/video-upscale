"""VLM Studio strike review: matching two runs' strikes and storing hand labels."""
import json

import pytest
from pydantic import ValidationError

from service import strike_review as sr


def tech(t_ms, actor="fighter-a", classes=("strike_punch",), outcome="successful"):
    return dict(bounds=dict(start_ms=t_ms - 250, end_ms=t_ms + 250), actor_player_id=actor, action_classes=list(classes),
                strike_target="head", outcome=outcome, significant=True, knockdown=False, notes="jab", outcome_evidence="snaps head")


def test_only_strikes_inside_the_shared_scope_are_compared():
    found = sr.strikes([tech(1000), tech(2000, classes=("takedown_attempt",)), tech(9000), tech(500, actor=None)], 0, 5000)
    assert [s["t_ms"] for s in found] == [1000]
    assert found[0]["classes"] == ["strike_punch"] and found[0]["evidence"] == "snaps head"


def test_matching_pairs_nearest_same_fighter_strikes_within_tolerance():
    a = sr.strikes([tech(1000), tech(1300), tech(5000, actor="fighter-b"), tech(8000)], 0, 10_000)
    b = sr.strikes([tech(1250), tech(5100, actor="fighter-a"), tech(8500)], 0, 10_000)
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
