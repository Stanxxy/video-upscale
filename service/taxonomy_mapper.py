"""Simplified 4-axis BJJ taxonomy support for the vision engine (T2, s9).

Re-authored 2026-07-12 for the simplified-4-axis-taxonomy adoption
(``working_log/plans/2026-07-12-simplified-4axis-taxonomy-adoption.md``, D5).
Replaces the old free-string-technique + post-hoc frozenset sanitization
(``map_to_frontend_taxonomy`` and its category-inference heuristics
``_infer_standup_action`` / ``_infer_guard_play_action`` /
``_infer_positional_action`` / ``_infer_defense_action``, all DELETED here —
grep-verified unused outside their own now-removed test file, a dead branch
left over from the pre-4-axis pipeline's ``STANDUP_GAME``/``GUARD_PLAY``/
``GUARD_PASSING``/``POSITIONAL_DOMINANCE``/``SUBMISSION_OFFENSE``/
``DEFENSE_ESCAPES`` category system, which ``bjj_analysis_taxonomy.md``
already told callers never to use).

**S12 Phase 1b addendum (2026-07-20) — this module now serves TWO callers
with TWO different output shapes.** ``clip_to_event``/``dual_emit_legacy_fields``
below remain the DUAL-EMIT path (``schema_version=2``) for the dormant
tracking pipeline's ``sns.py::clip_to_event`` — untouched, still load-bearing
for that (dormant-but-functional) code path. ``build_axis_only_candidate``
(new, bottom of this module) is the AXIS-ONLY path (``schema_version=3``)
for the v2 ``highlight-scan-critique-analyze`` production orchestrator's
``sns.py::clip_to_axis_only_event`` — it NEVER calls ``dual_emit_legacy_fields``
or any of the ``*_TO_LEGACY_*`` tables; ``action``/``technique``/``result``/
``confidence`` are left ``None`` (requires shared_lib's relaxed
``VideoEventCandidate`` schema — see the S12 Phase 1b production wiring
design doc §5.1/§8.1 for the hard cross-repo sequencing constraint). A future
maintainer should not assume the whole module is one code path.

This module now has three jobs:

1. **Axis vocabulary, sourced directly from shared_lib** (item 1; revised
   2026-07-12 by CEO ruling on Evaluator pass 1 MEDIUM-1). ``shared_lib`` is
   already a runtime dependency of this engine (``requirements-service.txt``)
   and ``shared_lib.models.simplified_taxonomy`` is the canonical Python
   module for this vocabulary (T1, ``shared_lib`` v1.1.0) — this module
   imports its public names (leaf-module import, per the shared_lib import-
   boundary convention: never ``from shared_lib.models import ...``) directly
   rather than hand-maintaining a second copy. An earlier revision of this
   module vendored a byte-identical local ``taxonomy_manifest.json`` with its
   own parity test; the CEO ruling correctly identified that as the exact
   five-hand-copy failure mode this whole initiative exists to close
   (INS-088) — "adjust independently without a shared_lib bump" is the
   disease, not a feature, and plan AC#1 explicitly demands zero hand-
   maintained value lists. There is nothing left here to drift, so there is
   nothing left here to parity-test against shared_lib; a slim parity test
   against ``bjj-dataset-harness/src/harness/taxonomy.py`` remains
   (``tests/test_taxonomy_harness_parity.py``) because harness regeneration
   from the manifest is still an open D2 tail (per D2's own text: "harness
   ``taxonomy.py`` stays authoritative for probe axes ... regenerated from
   the same manifest" — not yet wired as of T0/T1), so THAT axis pair can
   still independently drift.

2. **Sanitization + dual-emit mapping** (items 3-5). Gemini's structured
   output is already ENUM-constrained by ``analyzer._build_response_schema``
   (D5) so most values are valid by construction; the sanitize_* functions
   here are the last-mile defensive layer (SDK/library bugs, hand-crafted
   test fixtures, future non-Gemini callers) — fail path is an EXPLICIT
   sentinel (never null/blank, D4's rule applied to the write path) plus a
   logged warning, never a silent drop. ``resolve_technique_shortlist`` maps
   Gemini's free-text ``technique_guess`` (D1, LeCun's drift signal) into the
   curated per-class shortlist or ``other``. The ``*_TO_LEGACY_*`` tables +
   ``dual_emit_legacy_fields`` are the T2 sharpest-seam mitigation (plan §7
   risk #2): every new-taxonomy write must ALSO populate legacy
   ``action``/``technique``/``result`` with values recognized by
   ``event_filter_utils.py``'s search validators, or search 422s instantly.

3. **Axis-only candidate construction** (S12 Phase 1b, ``build_axis_only_candidate``).
   The v2 pipeline's terminal shape — no dual-emit, no legacy fields — see
   the addendum above.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Type

# Leaf-module import (never `from shared_lib.models import ...`), per the
# shared_lib import-boundary convention (AGENTS.md). These names are
# RE-EXPORTED as this module's own public API — every consumer in this repo
# (analyzer.py, sns.py, this module's own dual-emit tables below) reads the
# axis vocabulary through `taxonomy_mapper.AXIS1_POSITION` etc., but the
# single source of truth is shared_lib; nothing here hand-copies a value
# list (CEO ruling, Evaluator pass 1 MEDIUM-1 — see module docstring item 1).
from shared_lib.models.simplified_taxonomy import (  # noqa: F401 (re-exported)
    AXIS1_POSITION,
    AXIS2_ACTOR_SENTINELS,
    AXIS3_ACTION,
    AXIS4_OUTCOME,
    HARD_PAIR_ACTIONS,
    TAXONOMY_VERSION,
    TECHNIQUE_SHORTLISTS,
    is_valid_actor_sentinel,
    is_valid_axis1,
    is_valid_axis3,
    is_valid_axis4,
    is_valid_technique_for_class,
    is_valid_technique_shortlist_value,
)

logger = logging.getLogger(__name__)

# Derived (not hand-maintained) from the imported TECHNIQUE_SHORTLISTS —
# used by resolve_technique_shortlist below.
_ALL_SHORTLIST_VALUES: frozenset = frozenset(
    v for values in TECHNIQUE_SHORTLISTS.values() for v in values
)


# ---------------------------------------------------------------------------
# Legacy enum value sets (frontend/backend ground truth) — kept from the
# pre-4-axis version of this module. Canonical source:
# bjj-vision-backend/shared_lib/src/shared_lib/models/bjj_taxonomy.py
# These are the ONLY values the dual-emit tables below are allowed to
# produce (asserted at import time, below) — this is what keeps the T2
# search-422 seam (plan §7 risk #2) closed.
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset({
    "guard_top", "guard_bottom", "half_guard_top", "half_guard_bottom",
    "side_control", "mount", "back_control", "knee_on_belly",
    "turtle_top", "turtle_bottom", "north_south",
    "sweep", "pass", "reversal", "scramble",
    "submission_attempt", "submission_defense",
    "takedown", "takedown_defense", "guard_pull", "guard_retention",
    "escape", "standing", "clinch", "reset", "other",
})

VALID_TECHNIQUES = frozenset({
    # Chokes
    "rear_naked_choke", "guillotine", "darce", "anaconda", "arm_triangle",
    "ezekiel", "loop_choke", "bow_and_arrow", "clock_choke",
    "baseball_bat_choke", "north_south_choke", "paper_cutter",
    "cross_collar_choke", "triangle_choke",
    # Arm locks
    "armbar", "kimura", "americana", "omoplata", "wrist_lock",
    "bicep_slicer", "tarikoplata", "baratoplata",
    # Leg locks
    "heel_hook", "inside_heel_hook", "outside_heel_hook",
    "straight_ankle_lock", "toe_hold", "knee_bar", "calf_slicer",
    "estima_lock",
    # Sweeps
    "scissor_sweep", "flower_sweep", "hip_bump_sweep", "pendulum_sweep",
    "butterfly_sweep", "x_guard_sweep", "single_leg_x_sweep",
    "hook_sweep", "lasso_sweep", "spider_sweep", "de_la_riva_sweep",
    "berimbolo", "sickle_sweep",
    # Passes
    "toreando_pass", "knee_cut_pass", "leg_drag", "over_under_pass",
    "double_under_pass", "stack_pass", "pressure_pass", "long_step_pass",
    "smash_pass", "body_lock_pass", "x_pass",
    # Takedowns
    "single_leg", "double_leg", "high_crotch", "ankle_pick", "snap_down",
    "arm_drag", "body_lock_takedown", "foot_sweep_takedown", "hip_throw",
    "trip", "suplex", "fireman_carry",
    # Escapes
    "bridge_escape", "hip_escape", "elbow_escape", "trap_and_roll",
    "granby_roll", "inversion_escape", "standing_escape",
    # Guard types
    "closed_guard", "open_guard", "butterfly_guard", "spider_guard",
    "lasso_guard", "de_la_riva_guard", "reverse_de_la_riva", "x_guard",
    "single_leg_x", "half_guard", "deep_half_guard", "z_guard",
    "rubber_guard", "worm_guard", "lapel_guard", "seated_guard",
    # Transitions / misc
    "back_take", "mount_transition", "side_control_transition",
    "leg_entanglement", "sprawl", "underhook", "overhook", "frame",
    "posture_break", "grip_break", "other",
})

VALID_RESULTS = frozenset({
    "success", "tap_out", "points", "advantage",
    "neutral", "position_maintained", "transition",
    "failed", "blocked", "countered", "escaped", "defended",
    "knockout", "decision", "draw", "dq",
})

# ---------------------------------------------------------------------------
# Sanitization (item 5) — Gemini output is already enum-constrained by
# analyzer._build_response_schema, so these are a last-mile defensive layer.
# Fail path is an EXPLICIT sentinel + a logged warning, never a silent drop
# or None (D4's "never null/blank" rule applied at the write path).
# ---------------------------------------------------------------------------

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
    wasn't a recognized class") rather than an empty list — mirrors
    shared_lib's `taxonomy_migration.py` read-path convention (RESET/OTHER ->
    `[transition]`, never silently empty when an action clearly occurred).
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


# ---------------------------------------------------------------------------
# technique_guess -> curated shortlist (item 3, D1)
# ---------------------------------------------------------------------------

def _normalize(text: Optional[str]) -> str:
    """Normalize freeform technique text to snake_case for matching.

    Preserves underscores already present in the input (so an already-
    snake_case shortlist value like ``rear_naked_choke`` round-trips through
    this function unchanged) — the ``[^a-z0-9\\s_]`` strip below explicitly
    keeps ``_`` alongside alnum/whitespace, unlike a naive "strip all
    non-alnum" pass which would silently collapse it to ``rearnakedchoke``.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"['’]", "", text)            # remove apostrophes (D'Arce -> DArce)
    text = re.sub(r"[-]", " ", text)             # hyphens to spaces
    text = re.sub(r"[^a-z0-9\s_]", "", text)     # strip other non-alnum (keep underscores)
    text = re.sub(r"\s+", "_", text.strip())     # spaces to underscores
    return text


# Common abbreviations / alternate phrasings -> curated D1 shortlist value.
# Every target value here MUST be a member of some TECHNIQUE_SHORTLISTS list
# (asserted at import time below).
TECHNIQUE_GUESS_ALIASES: Dict[str, str] = {
    "rnc": "rear_naked_choke",
    "rear_naked": "rear_naked_choke",
    "rear_naked_choke_attempt": "rear_naked_choke",
    "guillotine_choke": "guillotine",
    "triangle_choke": "triangle",
    "darce": "darce_anaconda",
    "darce_choke": "darce_anaconda",
    "d_arce": "darce_anaconda",
    "d_arce_choke": "darce_anaconda",
    "anaconda": "darce_anaconda",
    "anaconda_choke": "darce_anaconda",
    "cross_collar_choke": "collar_choke",
    "cross_choke": "collar_choke",
    "arm_triangle_choke": "arm_triangle",
    "kata_gatame": "arm_triangle",
    "head_and_arm_choke": "arm_triangle",
    "kimura_lock": "kimura",
    "straight_ankle_lock": "ankle_lock",
    "straight_footlock": "ankle_lock",
    "footlock": "ankle_lock",
    "kneebar": "knee_bar",
    "toehold": "toe_hold",
    "heelhook": "heel_hook",
    "inside_heel_hook": "heel_hook",
    "outside_heel_hook": "heel_hook",
    "single_leg_takedown": "single_leg",
    "double_leg_takedown": "double_leg",
    "ankle_pick_takedown": "ankle_pick",
    "hip_throw": "throw",
    "osoto_gari": "throw",
    "seoi_nage": "throw",
    "uchi_mata": "throw",
    "trip_takedown": "trip",
    "snap_down_takedown": "snap_down",
    "jump_guard_pull": "jump_pull",
    "flying_guard_pull": "jump_pull",
    "sit_down_guard_pull": "sit_pull",
    "sitting_guard_pull": "sit_pull",
    "knee_cut": "knee_cut_pass",
    "knee_slice": "knee_cut_pass",
    "knee_slice_pass": "knee_cut_pass",
    "toreando": "toreando_pass",
    "toreano_pass": "toreando_pass",
    "bullfighter_pass": "toreando_pass",
    "over_under": "over_under_pass",
    "double_under": "double_under_pass",
    "back_take": "turtle_back_take",
    "shrimp": "hip_escape",
    "shrimp_escape": "hip_escape",
    "granby": "granby_roll",
}


def resolve_technique_shortlist(
    technique_guess: Optional[str], axis3_action: List[str],
) -> Optional[str]:
    """Normalize a free-text ``technique_guess`` into a D1 curated shortlist
    value, scoped to ``axis3_action``'s class(es).

    Returns ``None`` only when ``axis3_action`` is empty (nothing to
    shortlist against — mirrors ``taxonomy_migration.py``'s read-path
    convention). Otherwise always returns a value that is a member of at
    least one populated class's shortlist (a resolved match/alias, or the
    class's own ``other`` bucket) — this is what keeps
    ``VideoEventCandidate``'s ``_validate_technique_shortlist`` model
    validator satisfied unconditionally.
    """
    if not axis3_action:
        return None
    classes_with_shortlist = [c for c in axis3_action if c in TECHNIQUE_SHORTLISTS]
    if not classes_with_shortlist:
        return None

    normalized = _normalize(technique_guess)
    if normalized:
        # Direct match against any in-scope class's shortlist.
        for cls in classes_with_shortlist:
            if normalized in TECHNIQUE_SHORTLISTS[cls]:
                return normalized
        # Alias match, still scoped to the in-scope classes.
        aliased = TECHNIQUE_GUESS_ALIASES.get(normalized)
        if aliased:
            for cls in classes_with_shortlist:
                if aliased in TECHNIQUE_SHORTLISTS[cls]:
                    return aliased

    # No usable guess, or nothing matched in scope -> the "other" bucket for
    # whichever in-scope class has one (all D1 shortlists include "other").
    logger.debug(
        "taxonomy_mapper: technique_guess %r did not resolve to a shortlist value "
        "for axis3_action=%s; falling back to 'other'", technique_guess, axis3_action,
    )
    return "other"


# ---------------------------------------------------------------------------
# New -> legacy dual-emit mapping (item 4, plan §7 risk #2 — the sharpest
# seam). Every new-taxonomy write must ALSO populate legacy
# action/technique/result with a value `event_filter_utils.py`'s search
# validators recognize, or search 422s the instant this engine ships.
# ---------------------------------------------------------------------------

# axis3_action is multi-label; legacy `action` is scalar. When more than one
# class co-occurs, this precedence picks ONE — submissions first (highest
# stats significance, feeds legacy SUBMISSION_SUCCESS_RATE), then the
# remaining classes ordered by how much information is lost if a
# higher-precedence class isn't present; `transition` (weakest signal) last.
_AXIS3_LEGACY_PRECEDENCE: List[str] = [
    "submission_choke",
    "submission_arm_lock",
    "submission_leg_lock",
    "takedown_attempt",
    "guard_pass",
    "sweep",
    "back_take",
    "escape",
    "guard_pull",
    "transition",
]

# axis3 action class -> legacy ActionType value. Verbatim twins used where
# they exist (guard_pull, sweep, escape); `submission_*` -> the legacy
# SUBMISSION_ATTEMPT bucket (all three submission sub-types share it, since
# legacy ActionType has no arm/leg/choke split at the action level — that
# split lived only in the ~90-value TechniqueType, which the shortlist/
# technique mapping below reconstructs where possible). `guard_pass` has no
# verbatim ActionType twin (`pass`, not `guard_pass`) — `pass` used instead.
#
# `back_take` has no ActionType analog at all. The obvious-looking candidate,
# `ActionType.BACK_CONTROL`, is DELIBERATELY REJECTED (Evaluator pass 1
# review): BACK_CONTROL is one of the enumerated `positional_actions()`
# members and feeds the legacy POSITIONAL_CONTROL time-tracking calculation
# (`bjj_taxonomy.py::ActionType.positional_actions()` /
# `statistics_calculator.py`'s time-in-position accounting) — that
# calculation assumes each `BACK_CONTROL`-tagged event marks a *span of
# control being held*, not a discrete transition event. Mapping every
# `back_take` (a point-in-time "took the back" action, not a duration) into
# that bucket would silently pollute POSITIONAL_CONTROL with instantaneous
# events masquerading as held-position spans. `ActionType.REVERSAL`
# ("reverses position from bottom to top") is the closest available legacy
# action that is NOT a positional-time-tracking member — it correctly
# characterizes back_take as a discrete transition, matching how the legacy
# schema itself distinguishes "Transition Actions" (SWEEP/PASS/REVERSAL/
# SCRAMBLE) from "Positional Actions" (GUARD_TOP/.../BACK_CONTROL/...) in
# bjj_taxonomy.py's own section comments. The TechniqueType-level twin
# (`back_take`, see the shortlist table below) recovers the specific-
# technique fidelity `reversal` alone would lose.
#
# `transition` (the axis3 catch-all) maps to legacy `scramble` — both denote
# "movement happened, no clean class" in their respective taxonomies.
ACTION_CLASS_TO_LEGACY_ACTION: Dict[str, str] = {
    "submission_choke": "submission_attempt",
    "submission_arm_lock": "submission_attempt",
    "submission_leg_lock": "submission_attempt",
    "takedown_attempt": "takedown",
    "guard_pull": "guard_pull",
    "sweep": "sweep",
    "guard_pass": "pass",
    "back_take": "reversal",
    "escape": "escape",
    "transition": "scramble",
}

# axis4_outcome (scalar, 3 values) -> legacy ResultType value.
OUTCOME_TO_LEGACY_RESULT: Dict[str, str] = {
    "successful": "success",
    "unsuccessful": "failed",
    "unclear": "neutral",
}

# D1 shortlist value -> legacy TechniqueType value, VERBATIM STRING TWINS
# ONLY (31 of the 39 non-"other" shortlist values, verified directly against
# bjj_taxonomy.py's TechniqueType member list — the plan's own estimate of
# "20 values" undercounted; documented in the engineering report, not
# silently corrected). A near-miss rename (e.g. shortlist `triangle` vs
# legacy `triangle_choke`, `ankle_lock` vs `straight_ankle_lock`,
# `darce_anaconda` vs separate `darce`/`anaconda`, `collar_choke` vs
# `cross_collar_choke`, `throw` vs `hip_throw`, `sit_pull`/`jump_pull` (no
# legacy analog at all), `turtle_back_take` vs `back_take`) is deliberately
# NOT fuzzy-matched — only an exact string twin counts, per the task's
# "verbatim legacy twin" instruction. Everything else (including every
# class's own "other") falls back to legacy technique "other".
TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE: Dict[str, str] = {
    "rear_naked_choke": "rear_naked_choke",
    "guillotine": "guillotine",
    "arm_triangle": "arm_triangle",
    "armbar": "armbar",
    "kimura": "kimura",
    "americana": "americana",
    "omoplata": "omoplata",
    "heel_hook": "heel_hook",
    "knee_bar": "knee_bar",
    "toe_hold": "toe_hold",
    "single_leg": "single_leg",
    "double_leg": "double_leg",
    "ankle_pick": "ankle_pick",
    "trip": "trip",
    "snap_down": "snap_down",
    "scissor_sweep": "scissor_sweep",
    "flower_sweep": "flower_sweep",
    "butterfly_sweep": "butterfly_sweep",
    "hip_bump_sweep": "hip_bump_sweep",
    "x_guard_sweep": "x_guard_sweep",
    "toreando_pass": "toreando_pass",
    "knee_cut_pass": "knee_cut_pass",
    "leg_drag": "leg_drag",
    "over_under_pass": "over_under_pass",
    "double_under_pass": "double_under_pass",
    "body_lock_pass": "body_lock_pass",
    "berimbolo": "berimbolo",
    "hip_escape": "hip_escape",
    "bridge_escape": "bridge_escape",
    "standing_escape": "standing_escape",
    "granby_roll": "granby_roll",
}

# --- Fail-loud import-time guarantees: every value this module can EVER
# emit into a legacy field is a member of the old enum's value set. This is
# what actually closes plan §7 risk #2 — not just documentation.
assert set(ACTION_CLASS_TO_LEGACY_ACTION.values()) <= VALID_ACTIONS, (
    "ACTION_CLASS_TO_LEGACY_ACTION emits a value outside VALID_ACTIONS"
)
assert set(OUTCOME_TO_LEGACY_RESULT.values()) <= VALID_RESULTS, (
    "OUTCOME_TO_LEGACY_RESULT emits a value outside VALID_RESULTS"
)
assert set(TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE.values()) <= VALID_TECHNIQUES, (
    "TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE emits a value outside VALID_TECHNIQUES"
)
assert set(ACTION_CLASS_TO_LEGACY_ACTION) == set(AXIS3_ACTION), (
    "ACTION_CLASS_TO_LEGACY_ACTION must cover every axis3 class"
)
assert set(OUTCOME_TO_LEGACY_RESULT) == set(AXIS4_OUTCOME), (
    "OUTCOME_TO_LEGACY_RESULT must cover every axis4 outcome"
)
assert set(TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE) <= _ALL_SHORTLIST_VALUES, (
    "TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE keys must all be real shortlist values"
)


def primary_axis3_class(axis3_action: List[str]) -> Optional[str]:
    """Pick ONE axis3 class from a (possibly multi-label) list, by
    ``_AXIS3_LEGACY_PRECEDENCE``, for scalar legacy-field derivation."""
    for cls in _AXIS3_LEGACY_PRECEDENCE:
        if cls in axis3_action:
            return cls
    return None


def axis3_to_legacy_action(axis3_action: Optional[List[str]]) -> str:
    """Multi-label axis3_action -> scalar legacy ActionType value.

    Empty/unresolvable input -> legacy `other` (verbatim ActionType member),
    never a fabricated guess.
    """
    cls = primary_axis3_class(axis3_action or [])
    if cls is None:
        return "other"
    return ACTION_CLASS_TO_LEGACY_ACTION[cls]


def axis4_to_legacy_result(axis4_outcome: Optional[str]) -> str:
    """Scalar axis4_outcome -> legacy ResultType value. Unrecognized/None
    input -> legacy `neutral` (mirrors axis4's own `unclear` semantics)."""
    return OUTCOME_TO_LEGACY_RESULT.get(axis4_outcome, "neutral")


def technique_shortlist_to_legacy_technique(shortlist_value: Optional[str]) -> str:
    """D1 shortlist value -> legacy TechniqueType value (verbatim twins
    only, see TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE docstring). Anything
    without a verbatim twin (including every class's own "other") -> legacy
    `other`, never a fabricated near-match."""
    if not shortlist_value:
        return "other"
    return TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE.get(shortlist_value, "other")


def dual_emit_legacy_fields(
    axis3_action: Optional[List[str]],
    axis4_outcome: Optional[str],
    technique_shortlist: Optional[str],
) -> Dict[str, str]:
    """Derive the legacy ``action``/``technique``/``result`` triple from the
    new-taxonomy axes — the T2 dual-emit contract (plan §7 risk #2). Always
    returns values drawn from VALID_ACTIONS/VALID_TECHNIQUES/VALID_RESULTS
    (asserted at import time above), so a ``VideoEventCandidate`` built from
    this output can never trip the backend's legacy-enum search validators.
    """
    return {
        "action": axis3_to_legacy_action(axis3_action),
        "technique": technique_shortlist_to_legacy_technique(technique_shortlist),
        "result": axis4_to_legacy_result(axis4_outcome),
    }


# ---------------------------------------------------------------------------
# S12 Phase 1b — axis-only candidate construction (item 13, design §5.2/§5.5).
# The v2 highlight-scan-critique-analyze pipeline's terminal shape:
# schema_version=3, legacy action/technique/result/confidence left None
# (never fabricated — requires shared_lib's relaxed VideoEventCandidate,
# §8.1's hard cross-repo sequencing constraint). NEVER touches
# dual_emit_legacy_fields/ACTION_CLASS_TO_LEGACY_ACTION/etc. above.
# ---------------------------------------------------------------------------
def build_axis_only_candidate(clip: Dict[str, Any], video_id: str, *, candidate_cls: Optional[Type] = None):
    """Build a ``schema_version=3`` (axis-only) ``VideoEventCandidate`` from
    one ``executors.highlight_analyze_node`` synthesized clip dict.

    ``clip`` is expected to already carry the actor axis's resolved fields
    (``player_id``/``player_name``/``identity_uncertain``/``actor_sentinel``)
    — folded in at the executor level (design §4.4/item 5), NOT re-derived
    here, so there is exactly one place this mapping logic lives (single
    source of truth; a second ``actor_result`` argument duplicating the same
    data would only invite the two to drift).

    ``candidate_cls`` (testability seam, default ``None`` -> the REAL
    ``shared_lib.models.sns_event_models.VideoEventCandidate``): shared_lib
    1.3.0 (installed) relaxed ``action``/``technique``/``result``/
    ``confidence`` to ``Optional`` for a ``schema_version=3`` candidate, so
    the real class now constructs successfully with this function's ``None``
    legacy values — see ``tests/test_taxonomy_mapper_axis_only.py``'s
    ``test_real_shared_lib_construction_succeeds`` for the end-to-end proof.
    Tests still ALSO inject a duck-typed stand-in class to verify the
    MAPPING logic (which clip field derives which kwarg) independently of
    the real model's own field constraints.

    **2026-07-26 single-call cutover:** this function's own mapping logic is
    UNCHANGED and call-count-agnostic — it reads ``clip.get("position")``/
    ``"action_class"``/``"outcome"``/... off a plain dict and does not care
    whether one Gemini call produced them (the current shape) or the deleted
    two-call+validator design did. See
    ``tests/test_taxonomy_mapper_axis_only.py``'s
    ``test_maps_the_single_call_executor_output_shape`` for a pinned
    regression against the REAL shape ``executors.highlight_analyze_node``
    emits post-cutover.
    """
    from shared_lib.models.sns_event_models import VideoEventCandidate as _RealVideoEventCandidate

    cls = candidate_cls or _RealVideoEventCandidate

    player_id = clip.get("player_id")
    player_name = clip.get("player_name")
    actor_sentinel = clip.get("actor_sentinel")
    identity_uncertain = clip.get("identity_uncertain")
    # role stays a human-readable descriptor (never a new identity source,
    # design §5.2) — same "Unknown" capital-U fallback convention as
    # clip_to_event's role_descriptor above, for the rare case none of
    # player_name/player_id/actor_sentinel resolved.
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
        track_id=None,  # always None for v2 — no tracking exists (design §4.4)
        identity_uncertain=identity_uncertain,
        action=None,
        technique=None,
        result=None,
        confidence=None,
        notes=notes,
        schema_version=3,
        axis1_position=axis1_position,
        axis3_action=axis3_action,
        axis4_outcome=axis4_outcome,
        actor_sentinel=actor_sentinel,
        # v2's technique call does not produce sub-technique granularity yet
        # — an honest scope gap (design §5.2), not a fabricated guess.
        technique_shortlist=None,
        technique_guess=None,
    )
