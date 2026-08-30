"""Score / go-no-go decision tests."""
from __future__ import annotations

from eval.adcc.sam3_scout.score_report import decide_go_no_go, score_arm_events, gold_event_for_clip


def _clip(**kwargs):
    base = {
        "clip_id": "c1",
        "match_id": "m1",
        "t_s": 100.0,
        "action_class": "sweep",
        "outcome": "successful",
        "occlusion_heavy": False,
        "stratum": "sweep_back_take",
    }
    base.update(kwargs)
    return base


def test_score_arm_detects_class_and_outcome():
    gold = gold_event_for_clip(_clip())
    preds = [{"start_s": 95.0, "end_s": 105.0, "action_class": "sweep", "outcome": "successful"}]
    s = score_arm_events(gold, preds, tau_t=2.0)
    assert s["detected"] and s["action_class_ok"] and s["outcome_ok"]


def test_no_go_when_b_and_c_do_not_beat_a_on_hard():
    def arms(a_det, b_det, c_det):
        def one(d):
            return {"detected": d, "action_class_ok": d, "outcome_ok": d}
        return {"A": one(a_det), "B": one(b_det), "C": one(c_det)}

    clips = [
        {"occlusion_heavy": True, "stratum": "submission", "arms": arms(True, True, True)},
        {"occlusion_heavy": True, "stratum": "submission", "arms": arms(True, False, False)},
    ]
    # Both hard clips: B,C never strictly beat A
    d = decide_go_no_go(clips)
    assert d["verdict"] in ("no-go", "investigate")


def test_justified_when_hard_lift():
    def one(d, a=False, o=False):
        return {"detected": d, "action_class_ok": a, "outcome_ok": o}

    clips = [
        {
            "occlusion_heavy": True,
            "stratum": "submission",
            "arms": {
                "A": one(True, False, False),
                "B": one(True, True, True),
                "C": one(True, False, False),
            },
        }
    ]
    d = decide_go_no_go(clips)
    assert d["verdict"] == "justified"
