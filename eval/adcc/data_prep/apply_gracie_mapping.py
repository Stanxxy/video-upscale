"""Applies ``eval/adcc/gracie_mapping/v1.yaml`` to a single ADCC gold event.

This module contains ZERO hardcoded taxonomy rulings — every raw-value list
(sequenceType table, submission-subtype lookup, POSITION ENTRY split
thresholds, position buckets, compat sets) lives in the checked-in YAML
artifact and is consumed here as pure data (plan §4: "Structure it so
apply_gracie_mapping.py consumes it as pure data"). An unmapped
``sequenceType`` is a hard error, never a silent drop or guess.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("eval.adcc.data_prep.apply_gracie_mapping")

_MAPPING_PATH = Path(__file__).resolve().parent.parent / "gracie_mapping" / "v1.yaml"

__all__ = [
    "load_mapping",
    "UnknownSequenceTypeError",
    "MappingError",
    "apply_mapping",
    "resolve_submission_subtype",
    "resolve_position_entry",
    "bucket_position",
    "resolve_event_position",
    "compat_classes_for",
]


class MappingError(ValueError):
    """Base class for all data-quality errors raised while applying the
    Gracie mapping to a raw gold event."""


class UnknownSequenceTypeError(MappingError):
    """Raised when a raw event's ``sequenceType`` is not a key in
    ``sequence_type_map`` — hard error per plan §4 ("Unknown sequenceType =
    hard error at prep time"), never a silent drop."""


@functools.lru_cache(maxsize=1)
def load_mapping(path: Optional[str] = None) -> dict:
    """Load + cache ``v1.yaml`` (pure data, no code). ``path`` override exists
    for tests exercising a synthetic mapping; production callers always use
    the default (the real checked-in artifact)."""
    mapping_path = Path(path) if path else _MAPPING_PATH
    with open(mapping_path, "r") as f:
        return yaml.safe_load(f)


def _is_attempt(movements: str) -> bool:
    return movements.strip().upper().startswith("ATTEMPT (")


def resolve_submission_subtype(ending_position: str, movements: str, mapping: dict) -> tuple[str, str]:
    """Returns ``(action_class, track)`` for a SUBMISSION row per §4.2.

    Checks the exact ``endingPosition`` lookup first (near-100% reliable per
    Gracie's spot-check), then a ``movements`` keyword fallback, then gives up
    with ``("submission_unclear", "soft")`` — never fabricates a confident
    subtype. Match FEASIBILITY for this row is unaffected either way (§5.1:
    SUBMISSION is family-wide compat regardless of subtype) — ``track``
    controls report/confusion-matrix granularity only.
    """
    subtypes = mapping["submission_subtype"]
    ep = ending_position.strip().upper()
    for family, raw_values in subtypes.items():
        if family == "movements_keyword_fallback":
            continue
        if ep in raw_values:
            return f"submission_{family}", "hard"

    mv = movements.strip().upper()
    fallback = subtypes["movements_keyword_fallback"]
    for family, keywords in fallback.items():
        if any(kw in mv for kw in keywords):
            return f"submission_{family}", "hard"

    return "submission_unclear", "soft"


_GENERALIZATION_RULES: list[tuple[tuple[str, ...], str]] = [
    (("ASHI", "50-50", "X-GUARD", "SPIDERWEB", "HAMSTRING", "FLANK", "WEDGE", "CLAMP", "K-GUARD"), "open_guard"),
    (("SIDE CONTROL", "NORTH SOUTH", "SMASH", "STACK"), "side_control"),
    (("MOUNT",), "mount"),
    (("BACK",), "back_control"),
    (("TURTLE", "CRAB"), "turtle"),
    (("HEADLOCK", "DOGFIGHT", "SCRAMBLE"), "scramble"),
    (("STANDING", "SINGLE LEG", "DISTANCE"), "standing"),
    (("CLOSED GUARD",), "closed_guard"),
    (("HALF GUARD",), "half_guard"),
    (("GUARD",), "open_guard"),
]


def _generalize_unseen_position(raw_upper: str) -> str:
    """Documented, LOGGED fallback for a raw position value not present in
    ``position_buckets`` — plan §4.4's "Generalization principle for unseen
    values" (forward guidance for corpus growth, never fired on the current
    258-match corpus since that table is exhaustive for it — see
    ``tests/test_apply_gracie_mapping.py::test_generalize_unseen_position``
    for the one place this path is actually exercised, via a synthetic
    value). Substring match, first rule wins; raises if nothing matches."""
    logger.warning(
        "apply_gracie_mapping: generalizing unseen raw position %r — not present in "
        "gracie_mapping/v1.yaml position_buckets (extend the table, this is a heuristic fallback)",
        raw_upper,
    )
    for keywords, bucket in _GENERALIZATION_RULES:
        if any(kw in raw_upper for kw in keywords):
            return bucket
    raise MappingError(f"cannot generalize unseen raw position {raw_upper!r} into any of the 9 position buckets")


def bucket_position(raw_position: str, mapping: dict) -> str:
    """Bucket a raw position string into one of the 9 ``POSITION_VALUES``
    (§4.4). Exact-lookup first, generalization heuristic second."""
    raw_upper = raw_position.strip().upper()
    for bucket, raw_values in mapping["position_buckets"].items():
        if raw_upper in raw_values:
            return bucket
    return _generalize_unseen_position(raw_upper)


def resolve_event_position(event: dict, prev_event: Optional[dict], mapping: dict) -> str:
    """The position bucket recorded for a normalized gold event: always the
    event's OWN ``startingPosition`` (verified: all 336 SUBMISSION rows in
    the real corpus carry a legitimate, non-submission-name startingPosition
    — never empty, never itself a submission name). Falls back to the
    immediately prior event's ``endingPosition`` only if this event's own
    ``startingPosition`` is itself unbucketable as a real position (defensive;
    never observed in the current corpus, guards against future data drift
    per §4.4's "inherit position from the prior event" language)."""
    try:
        return bucket_position(event["startingPosition"], mapping)
    except MappingError:
        if prev_event is not None:
            return bucket_position(prev_event["endingPosition"], mapping)
        raise


def resolve_position_entry(event: dict, prev_event: Optional[dict], mapping: dict) -> Optional[dict]:
    """Applies the §4.3 POSITION ENTRY split. Returns a partial dict with
    keys ``action_class``/``outcome``/``scoreable``/``track``, or ``None`` if
    this row should be DROPPED entirely (administrative no-op or a
    dedup-suppressed repeat back_take) — never fabricated into an event.
    """
    rules = mapping["position_entry_split"]
    sp = event["startingPosition"].strip().upper()
    ep = event["endingPosition"].strip().upper()
    mv = event["movements"].strip().upper()

    back_eps = set(rules["back_take_ending_positions"])
    admin_mv = set(v.strip().upper() for v in rules["administrative_movements"])
    leg_eps = set(rules["leg_entanglement_ending_positions"])
    pin_eps = set(rules["top_pin_target_ending_positions"])
    guard_family = set(rules["guard_family_starting_positions"])
    sweep_kw = rules["sweep_movement_keywords"]

    # Rule 1: back_take, with immediate-repeat dedup against the prior gold
    # event (ANY sequenceType, chronological order within this match_id).
    if ep in back_eps:
        prev_ep = (prev_event.get("endingPosition", "").strip().upper() if prev_event else None)
        if prev_ep in back_eps:
            return None  # dedup-suppressed: "still riding the back" repeat
        return {"action_class": "back_take", "outcome": "successful", "scoreable": True, "track": "hard"}

    # Rule 2: no state change + administrative movements -> drop.
    if sp == ep and mv in admin_mv:
        return None

    # Rule 3: leg-entanglement entry, no sub/back-take -> soft transition.
    if ep in leg_eps:
        return {"action_class": "transition", "outcome": "unclear", "scoreable": False, "track": "soft"}

    # Rule 4: top-pin-target ending -> guard_pass or sweep, but ONLY for a
    # genuine top-initiated pass (A1 fix, plan §4.3/Part A1 — BLOCKING
    # over-generalization bug in the prior version of this rule, which fired
    # guard_pass for ANY pin-ending row regardless of start). A real
    # guard_pass requires the top player to be COMPLETING a pass, i.e.
    # starting from a guard-family position. Same-position (start==end) is
    # a same-position dedup, analogous to the back_take immediate-repeat
    # dedup in rule 1 above: structurally never a pass. Any other non-
    # guard-family start (SIDE CONTROL, MOUNT, NORTH SOUTH, TURTLE, BACK,
    # FRONT HEADLOCK, STACK, SMASH PASS, STANDING, DOGFIGHT, CRAB RIDE) is
    # pin-consolidation / base-recovery / turtle-breakdown, not a pass ->
    # routes to the same soft, non-scoreable transition bucket as rule 5.
    if ep in pin_eps:
        if sp == ep or sp not in guard_family:
            return {"action_class": "transition", "outcome": "unclear", "scoreable": False, "track": "soft"}
        if any(kw in mv for kw in sweep_kw):
            return {"action_class": "sweep", "outcome": "successful", "scoreable": True, "track": "hard"}
        return {"action_class": "guard_pass", "outcome": "successful", "scoreable": True, "track": "hard"}

    # Rule 5: catch-all -> soft transition.
    return {"action_class": "transition", "outcome": "unclear", "scoreable": False, "track": "soft"}


def apply_mapping(event: dict, prev_event: Optional[dict], mapping: dict) -> Optional[dict]:
    """Apply the full §4 mapping to ONE raw gold event (already known to
    belong to a single match, ``prev_event`` being the chronologically prior
    raw event in THAT SAME match, or ``None`` for the first event).

    Returns a normalized-event partial dict (missing only ``t_s``, which the
    caller attaches after ``parse_timestamp``) with keys: ``action_class``,
    ``position_bucket``, ``outcome``, ``scoreable``, ``track``,
    ``raw_sequenceType``, ``raw_positions`` (``{starting, ending}``),
    ``raw_movements``. Returns ``None`` if the row should be DROPPED
    entirely (§4.5). Raises ``UnknownSequenceTypeError`` for any
    ``sequenceType`` not in ``sequence_type_map`` — never a silent skip.
    """
    seq_type = event["sequenceType"]
    seq_map = mapping["sequence_type_map"]
    if seq_type not in seq_map:
        raise UnknownSequenceTypeError(
            f"unmapped sequenceType {seq_type!r} — extend gracie_mapping/v1.yaml's "
            f"sequence_type_map before scoring this corpus (known keys: {sorted(seq_map)})"
        )

    rule = seq_map[seq_type]
    position_bucket = resolve_event_position(event, prev_event, mapping)

    if rule["action_class"] == "SPLIT:submission_subtype":
        action_class, track = resolve_submission_subtype(event["endingPosition"], event["movements"], mapping)
        outcome = "unsuccessful" if _is_attempt(event["movements"]) else "successful"
        scoreable = True
    elif rule["action_class"] == "SPLIT:position_entry":
        resolved = resolve_position_entry(event, prev_event, mapping)
        if resolved is None:
            return None
        action_class = resolved["action_class"]
        outcome = resolved["outcome"]
        scoreable = resolved["scoreable"]
        track = resolved["track"]
    else:
        action_class = rule["action_class"]
        outcome = rule["outcome"]
        scoreable = rule["scoreable"]
        track = rule["track"]

    return {
        "action_class": action_class,
        "position_bucket": position_bucket,
        "outcome": outcome,
        "scoreable": scoreable,
        "track": track,
        "raw_sequenceType": seq_type,
        "raw_positions": {"starting": event["startingPosition"], "ending": event["endingPosition"]},
        "raw_movements": event["movements"],
    }


def compat_classes_for(normalized_event: dict, mapping: dict) -> frozenset[str]:
    """§5.1 match-feasibility compat set for a normalized gold event. Family-
    wide for any row whose ``raw_sequenceType`` is a key in
    ``match_compat.family_by_raw_sequence_type`` (currently only
    ``"SUBMISSION"``, per the plan's one explicit many-to-many case);
    singleton exact-class match otherwise."""
    family_map = mapping["match_compat"]["family_by_raw_sequence_type"]
    raw_seq_type = normalized_event["raw_sequenceType"]
    if raw_seq_type in family_map:
        return frozenset(family_map[raw_seq_type])
    return frozenset({normalized_event["action_class"]})
