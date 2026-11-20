"""Axis vocabulary and candidate construction for the vision engine.

Two jobs:

1. **Axis vocabulary, sourced directly from shared_lib.** The names re-exported
   below are shared_lib's own objects (leaf-module import, never
   ``from shared_lib.models import ...``); nothing here hand-copies a value
   list.
2. **Sanitization + axis-only candidate construction.** Gemini's structured
   output is ENUM-constrained, so the ``sanitize_*`` functions are the
   last-mile defensive layer: the fail path is an EXPLICIT sentinel plus a
   logged warning, never a silent drop. ``build_axis_only_candidate`` builds
   the ``schema_version=3`` candidate the highlight orchestrator publishes
   (position, actor, action class, outcome; the specific move lives in
   ``notes``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from shared_lib.models.simplified_taxonomy import (  # noqa: F401 (re-exported)
    AXIS1_POSITION,
    AXIS2_ACTOR_SENTINELS,
    AXIS3_ACTION,
    AXIS4_OUTCOME,
    HARD_PAIR_ACTIONS,
    TAXONOMY_VERSION,
    is_valid_actor_sentinel,
    is_valid_axis1,
    is_valid_axis3,
    is_valid_axis4,
)

logger = logging.getLogger(__name__)


def sanitize_axis1_position(values: Optional[List[str]]) -> List[str]:
    """Filter to valid AXIS1_POSITION members, logging anything dropped.

    Empty list (no values, or all values dropped) is a legal "unlabeled"
    result — axis1 has no forced fallback member (unlike axis3's
    `transition`/axis4's `unclear`) because inventing a position the model
    didn't actually report would be a fabrication, not a safe default.
    """
    values = values or []
    valid = [v for v in values if v in AXIS1_POSITION]
    dropped = [v for v in values if v not in AXIS1_POSITION]
    if dropped:
        logger.warning("taxonomy_mapper: dropped invalid axis1_position value(s) %s", dropped)
    return valid


def sanitize_axis3_action(values: Optional[List[str]]) -> List[str]:
    """Filter to valid AXIS3_ACTION members, logging anything dropped.

    If the model reported SOME value(s) but none survive filtering, fall
    back to the explicit `["transition"]` catch-all ("something happened,
    wasn't a recognized class") rather than an empty list.
    A genuinely empty input (model reported nothing) stays `[]`.
    """
    values = values or []
    valid = [v for v in values if v in AXIS3_ACTION]
    dropped = [v for v in values if v not in AXIS3_ACTION]
    if dropped:
        logger.warning("taxonomy_mapper: dropped invalid axis3_action value(s) %s", dropped)
    if values and not valid:
        logger.warning(
            "taxonomy_mapper: axis3_action had no valid values after sanitization; "
            "falling back to ['transition']"
        )
        return ["transition"]
    return valid


def sanitize_axis4_outcome(value: Optional[str]) -> str:
    """Valid AXIS4_OUTCOME member, else explicit `unclear` (never None)."""
    if value in AXIS4_OUTCOME:
        return value
    if value is not None:
        logger.warning("taxonomy_mapper: invalid axis4_outcome %r; falling back to 'unclear'", value)
    return "unclear"




def build_axis_only_candidate(clip: Dict[str, Any], video_id: str, *, candidate_cls: Optional[Type] = None):
    """Build a ``schema_version=3`` ``VideoEventCandidate`` from one
    ``executors.highlight_analyze_node`` synthesized clip dict.

    ``clip`` is expected to already carry the actor axis's resolved fields
    (``player_id``/``player_name``/``identity_uncertain``/``actor_sentinel``)
    — folded in at the executor level, NOT re-derived here, so there is
    exactly one place this mapping logic lives.

    ``candidate_cls`` (testability seam, default ``None`` -> the real
    ``shared_lib.models.sns_event_models.VideoEventCandidate``): tests inject
    a duck-typed stand-in to verify which clip field derives which kwarg
    independently of the real model's own field constraints.
    """
    from shared_lib.models.sns_event_models import VideoEventCandidate as _RealVideoEventCandidate

    cls = candidate_cls or _RealVideoEventCandidate

    player_id = clip.get("player_id")
    player_name = clip.get("player_name")
    actor_sentinel = clip.get("actor_sentinel")
    identity_uncertain = clip.get("identity_uncertain")
    # role stays a human-readable descriptor, never an identity source.
    role = player_name or player_id or actor_sentinel or "Unknown"

    position = clip.get("position")
    action_class = clip.get("action_class")
    outcome = clip.get("outcome")

    axis1_position = sanitize_axis1_position([position] if position else [])
    axis3_action = sanitize_axis3_action([action_class] if action_class else [])
    axis4_outcome = sanitize_axis4_outcome(outcome)

    notes = clip.get("notes") or ""

    return cls(
        role=role,
        player_name=player_name,
        player_id=player_id,
        identity_uncertain=identity_uncertain,
        notes=notes,
        schema_version=3,
        axis1_position=axis1_position,
        axis3_action=axis3_action,
        axis4_outcome=axis4_outcome,
        actor_sentinel=actor_sentinel,
    )
