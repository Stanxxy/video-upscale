"""``service/worker/majority_vote.py`` — per-match majority-vote actor
reconciliation (2026-07-26-engine13-rescope-single-call-cutover.md AC8-11;
Brooks's architecture ruling §2, re-ruled §2a)."""
from __future__ import annotations

from service.worker import majority_vote


def _clip(player_id=None, player_name=None, identity_uncertain=None, actor_sentinel=None, **extra):
    base = {
        "start_s": 0.0, "end_s": 10.0, "position": "mount",
        "action_class": "submission_arm_lock", "outcome": "successful",
        "player_id": player_id, "player_name": player_name,
        "identity_uncertain": identity_uncertain, "actor_sentinel": actor_sentinel,
        "notes": "n",
    }
    base.update(extra)
    return base


# --------------------------------------------------------------------------- #
# Clear majority
# --------------------------------------------------------------------------- #
def test_clear_majority_reconciles_every_clip_to_the_winner():
    clips = [
        _clip(player_id="p1", player_name="Alice"),
        _clip(player_id="p1", player_name="Alice"),
        _clip(player_id="p2", player_name="Bob"),
    ]
    reconciled, flip_count = majority_vote.reconcile_match_actors(clips)
    assert [c["player_id"] for c in reconciled] == ["p1", "p1", "p1"]
    assert all(c["player_name"] == "Alice" for c in reconciled)
    assert all(c["identity_uncertain"] is False for c in reconciled)
    assert all(c["actor_sentinel"] is None for c in reconciled)
    assert flip_count == 1  # only the p2 clip disagreed with the winner


def test_clear_majority_even_overrides_a_sentinel_clip():
    """Founder intent per Brooks §2a: "make per-player stats reflect the
    reconciled player" — even a clip the model itself couldn't resolve
    gets the benefit of the match's determined identity."""
    clips = [
        _clip(player_id="p1", player_name="Alice"),
        _clip(player_id="p1", player_name="Alice"),
        _clip(player_id=None, player_name=None, identity_uncertain=True, actor_sentinel="contested"),
    ]
    reconciled, flip_count = majority_vote.reconcile_match_actors(clips)
    assert reconciled[2]["player_id"] == "p1"
    assert reconciled[2]["player_name"] == "Alice"
    assert reconciled[2]["identity_uncertain"] is False
    assert flip_count == 1  # the sentinel clip disagreed with (had no) the winner


def test_single_highlight_match_trivially_reconciles_to_its_own_vote():
    clips = [_clip(player_id="p1", player_name="Alice")]
    reconciled, flip_count = majority_vote.reconcile_match_actors(clips)
    assert reconciled[0]["player_id"] == "p1"
    assert reconciled[0]["identity_uncertain"] is False
    assert flip_count == 0


# --------------------------------------------------------------------------- #
# Ties / no majority — NEVER a forced pick (Brooks §2a).
# --------------------------------------------------------------------------- #
def test_tie_never_forces_a_winner_becomes_uncertain_unclear():
    clips = [
        _clip(player_id="p1", player_name="Alice"),
        _clip(player_id="p2", player_name="Bob"),
    ]
    reconciled, flip_count = majority_vote.reconcile_match_actors(clips)
    assert all(c["player_id"] is None for c in reconciled)
    assert all(c["player_name"] is None for c in reconciled)
    assert all(c["identity_uncertain"] is True for c in reconciled)
    assert all(c["actor_sentinel"] == "unclear" for c in reconciled)
    assert flip_count == len(clips)  # maximally disagreeing — no winner to agree with


def test_three_way_tie_never_forces_a_winner():
    clips = [
        _clip(player_id="p1"), _clip(player_id="p2"), _clip(player_id="p3"),
    ]
    reconciled, flip_count = majority_vote.reconcile_match_actors(clips)
    assert all(c["identity_uncertain"] is True for c in reconciled)
    assert flip_count == 3


def test_all_sentinel_match_no_votes_at_all_becomes_uncertain():
    clips = [
        _clip(player_id=None, actor_sentinel="contested", identity_uncertain=True),
        _clip(player_id=None, actor_sentinel="unclear", identity_uncertain=True),
    ]
    reconciled, flip_count = majority_vote.reconcile_match_actors(clips)
    assert all(c["identity_uncertain"] is True for c in reconciled)
    assert all(c["actor_sentinel"] == "unclear" for c in reconciled)
    assert flip_count == 2


def test_empty_match_returns_empty_no_crash():
    reconciled, flip_count = majority_vote.reconcile_match_actors([])
    assert reconciled == []
    assert flip_count == 0


# --------------------------------------------------------------------------- #
# Non-identity fields pass through untouched; no in-place mutation.
# --------------------------------------------------------------------------- #
def test_non_identity_fields_pass_through_unchanged():
    clip = _clip(player_id="p1", player_name="Alice", start_s=15.0, end_s=32.0,
                  position="side_control", action_class="submission_choke", outcome="successful", notes="j | e")
    reconciled, _ = majority_vote.reconcile_match_actors([clip, _clip(player_id="p1", player_name="Alice")])
    assert reconciled[0]["start_s"] == 15.0
    assert reconciled[0]["end_s"] == 32.0
    assert reconciled[0]["position"] == "side_control"
    assert reconciled[0]["action_class"] == "submission_choke"
    assert reconciled[0]["outcome"] == "successful"
    assert reconciled[0]["notes"] == "j | e"


def test_original_clip_dicts_never_mutated_in_place():
    original = _clip(player_id="p2", player_name="Bob")
    clips = [_clip(player_id="p1", player_name="Alice"), _clip(player_id="p1", player_name="Alice"), original]
    majority_vote.reconcile_match_actors(clips)
    assert original["player_id"] == "p2"  # untouched — reconcile_match_actors returned NEW dicts
    assert original["player_name"] == "Bob"
