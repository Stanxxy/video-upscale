"""Per-match majority-vote actor reconciliation
(2026-07-26-engine13-rescope-single-call-cutover.md AC8-11; Brooks's
architecture ruling §2, re-ruled §2a).

Deterministic, engine-side (Python, non-Gemini) — runs ONCE, after a
job's full per-chunk analyze loop, over every collected clip in the match
(one job == one match; this codebase has no cross-match partitioning
concept anywhere else either). The actor axis (reference images, per-
highlight independent call) is unchanged; this module is a *consumer* of
its per-highlight output, never a replacement for it.

**What "majority vote" means here, precisely (AC10's own definition,
carried over unchanged by the §2a re-ruling):** one authoritative
``player_id`` winner is computed per match — the plurality (max-count)
``player_id`` among every clip that resolved to a REAL (non-sentinel,
non-``None``) player identity. If there is a unique plurality winner, its
identity is written onto EVERY clip in the match (not just the ones that
already agreed) — Brooks's §2a explicit correction: "the founder's ask was
never 'tell us it's unstable' — it was 'make per-player stats reflect the
reconciled player.'" ``cross_highlight_flip_rate`` (fraction of the
match's clips whose OWN raw ``player_id`` disagreed with the winner) stays
the audit/telemetry signal, additive on ``attribution_metrics``.

**Ties / no clear majority — NEVER a forced pick (Brooks §2a, quoted
directly):** "Where the vote has no clear majority (an even split, or too
few highlights in a match to vote meaningfully), the reconciled result
must be identity_uncertain=True / an actor_sentinel value — never an
arbitrarily-picked winner." Every clip in a tied/no-vote match becomes
``player_id=None``, ``player_name=None``, ``identity_uncertain=True``,
``actor_sentinel="unclear"`` — and the WHOLE match counts as maximally
disagreeing (``flip_count == len(clips)``) for the flip-rate signal, since
there is no defined winner for any individual clip to be judged "agreeing"
or "disagreeing" against.

**Never mutates the caller's clip dicts in place** — returns NEW dicts
(``dict(clip)`` copies with the four identity fields overwritten), so a
caller holding a reference to the original list is never surprised by a
side effect it didn't ask for.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

# The single project-standard "no real identity resolved" sentinel used for
# a reconciliation tie/no-majority outcome — see
# shared_lib.models.simplified_taxonomy.AXIS2_ACTOR_SENTINELS (["contested",
# "unclear"]); "unclear" is the semantically-correct member here (the
# AMBIGUITY is about which athlete this is, not about whether the action
# itself was contested).
NO_MAJORITY_SENTINEL: str = "unclear"


def reconcile_match_actors(clips: list[dict]) -> tuple[list[dict], int]:
    """Returns ``(reconciled_clips, flip_count)`` — ``reconciled_clips`` is
    a NEW list (same length/order as ``clips``), each entry a shallow copy
    with ``player_id``/``player_name``/``identity_uncertain``/
    ``actor_sentinel`` overwritten per the match's majority-vote outcome.
    ``flip_count`` is the number of clips whose reconciled identity differs
    from what that clip's OWN raw actor call produced (used for
    ``cross_highlight_flip_rate = flip_count / len(clips)``, callers'
    responsibility — this function doesn't compute the rate itself so it
    stays honest about ``len(clips) == 0`` being the caller's edge case to
    handle, not silently defaulted here).

    Every OTHER key on each clip dict (``position``/``action_class``/
    ``outcome``/``start_s``/``end_s``/``notes``/etc.) passes through
    UNCHANGED — this function only ever touches the four identity fields.
    """
    votes = Counter(c["player_id"] for c in clips if c.get("player_id"))

    if votes:
        top_count = max(votes.values())
        winners = [pid for pid, count in votes.items() if count == top_count]
    else:
        winners = []

    if len(winners) == 1:
        return _apply_winner(clips, winners[0])
    return _apply_no_majority(clips)


def _apply_winner(clips: list[dict], winner_id: str) -> tuple[list[dict], int]:
    winner_name: Optional[str] = None
    for c in clips:
        if c.get("player_id") == winner_id and c.get("player_name"):
            winner_name = c["player_name"]
            break

    reconciled: list[dict] = []
    flip_count = 0
    for clip in clips:
        if clip.get("player_id") != winner_id:
            flip_count += 1
        new_clip = dict(clip)
        new_clip["player_id"] = winner_id
        new_clip["player_name"] = winner_name
        new_clip["identity_uncertain"] = False
        new_clip["actor_sentinel"] = None
        reconciled.append(new_clip)
    return reconciled, flip_count


def _apply_no_majority(clips: list[dict]) -> tuple[list[dict], int]:
    reconciled: list[dict] = []
    for clip in clips:
        new_clip = dict(clip)
        new_clip["player_id"] = None
        new_clip["player_name"] = None
        new_clip["identity_uncertain"] = True
        new_clip["actor_sentinel"] = NO_MAJORITY_SENTINEL
        reconciled.append(new_clip)
    # No defined winner exists for any clip to agree/disagree with — the
    # whole match counts as maximally disagreeing (see module docstring).
    return reconciled, len(clips)


__all__ = ["NO_MAJORITY_SENTINEL", "reconcile_match_actors"]
