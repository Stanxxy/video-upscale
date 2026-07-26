"""Taxonomy + actor axis helpers for ``highlight_analyze_node`` (PASS 2/3 of
``highlight-scan-(critique-)analyze``). Pure functions only (schema +
prompt) — no Gemini client; ``service/pipelines/executors.py``'s
``highlight_analyze_node`` is the only caller that actually talks to Gemini,
via the shared ``simplified_tags.SimplifiedTagsTimeAnalyzer.analyze_chunk``
(reusing its offset-quantization/retry/instrumentation plumbing, INS-107
discipline).

**Single-call cutover (2026-07-26-engine13-rescope-single-call-cutover.md,
AC1/OQ1) — the position/technique/validator triple this module used to hold
is DELETED, not deprecated-in-place.** LeCun's 5 cutover gates
(``VERDICTS_V2.md``, qualified-model record) ran the two-call+validator
design (Gate 1) against a single flat taxonomy call, TWICE, at two model
tiers: the two-call design lost by a WIDER margin at the stronger, qualified
model (hit_T-hit_S: -1 -> -5, every single disagreement favoring the
single-call baseline) — "more thinking consolidated the model's errors into
a confident, systematic bias rather than reducing them." Gate 2 additionally
found the validator's HIGH ``media_resolution`` default ACTIVELY REGRESSED a
correct answer at the qualified model. Brooks's architecture ruling (§1)
directs physical deletion in the same PR that lands the replacement — a
config flag reviving this design "in case we need it" is exactly the
fallback logic the project rule bans. Resurrection, if ever wanted, is a
``git log`` exercise (this module's own history), not a dormant code path.

``build_single_call_schema``/``build_single_call_prompt`` REPLACE the
position+technique pair: ONE call judges position, action_class, AND
outcome together, still as a single flat verdict — "the highlight's own
(already critique-corrected) window IS the unit of judgment," Gracie's
framing for the original per-axis calls, carries over unchanged for the
combined call. No adversarial second-guessing pass exists anymore (that was
the validator's job, and Gate 2 showed HIGH-resolution second-guessing
active regressed a correct answer) — ``executors.highlight_analyze_node``
builds ONE synthesized clip per highlight directly from this call's verdict,
spanning the highlight's own authoritative bounds.

**Kept, unchanged in shape (Gate 1/2 did not touch this axis):** the actor
axis (``ACTOR_SYSTEM_PROMPT``/``build_actor_schema``/``build_actor_prompt``)
— a fourth, independent reference-image-backed identity call, a *consumer*
of the taxonomy call's clip, not a replacement for it.
"""
from __future__ import annotations

from string import Template
from typing import Optional

from google.genai import types

from service.pipelines.simplified_tags import ACTION_CLASS_VALUES, OUTCOME_VALUES, POSITION_VALUES

# Leaf-module import (never `from shared_lib.models import ...`), per the
# shared_lib import-boundary convention — same discipline as
# service/taxonomy_mapper.py. AXIS2_ACTOR_SENTINELS is the "no hand-copied
# vocabulary" source of truth for the neutral/indeterminate actor values
# (S12 Phase 1b production wiring design §4.3).
from shared_lib.models.simplified_taxonomy import AXIS2_ACTOR_SENTINELS

# --------------------------------------------------------------------------- #
# Single-call taxonomy axis — ONE call judging position + action_class +
# outcome together, replacing the position+technique+validator triple that
# Gates 1/2 (VERDICTS_V2.md) rejected at BOTH model tiers (qualified-model
# margin widened from -1 to -5; HIGH validator resolution actively
# regressed a correct answer). No adversarial second-guessing pass — a
# single flat verdict per highlight, no timestamps, no list. See
# 2026-07-26-engine13-rescope-single-call-cutover.md AC1.
# --------------------------------------------------------------------------- #
SINGLE_CALL_SYSTEM_PROMPT: str = """
You are tagging ONE BJJ highlight clip with three axes at once: POSITION, ACTION CLASS, and OUTCOME. The clip's own window IS the unit of judgment — decide all three from what's actually visible in THIS clip.

POSITION — use EXACTLY one of these values, never invent a new one, never combine two:

- standing — includes clinch
- closed_guard — legs locked, ankles crossed
- open_guard — any open-guard variant (butterfly, spider, De La Riva, X-guard, etc. all collapse here)
- half_guard — one leg trapped
- side_control — includes knee-on-belly
- mount — full mount
- back_control — hooks or body triangle in
- turtle — either player turtled
- scramble — no stable position established; this is an HONEST "not sure / still resolving" bucket, not a failure on your part

It is legitimate — and expected — for a clip to be a pure standing exchange with nothing else established: use "standing" for that, plainly, rather than forcing a guard/mount/side-control label that isn't there.

ACTION CLASS — use EXACTLY one of these values:

- takedown_attempt — any standing-to-ground technique
- guard_pull — bottom player pulls guard voluntarily
- sweep — bottom player reverses to top
- guard_pass — top player passes to side control/mount/back
- back_take — transition into back control
- submission_choke — any choke, neck is the target
- submission_arm_lock — any arm/wrist joint attack
- submission_leg_lock — any leg/ankle/knee joint attack
- escape — inferior player returns to neutral/better position
- transition — position changed with no clear technique credit; this is the HONEST "something happened, wasn't a named move" bucket. If nothing resembling a named technique is being attempted at all — pure positional control, stalling, grip fighting with no entry — use "transition" rather than forcing one of the named techniques onto it.

OUTCOME — use EXACTLY one of these values:

- successful — tap, sweep/pass/takedown/escape completed, points scored, for the athlete attempting the action
- unsuccessful — the attempt didn't land
- unclear — a scramble/contested moment resolved without a clean read

Outcome is the hardest of these three calls, especially for submissions where the finishing grip or the tapping hand is hidden from camera. HONEST UNCERTAINTY IS REQUIRED HERE: if you cannot actually see whether it landed — because the grip is occluded, the camera cuts away, or bodies are stacked — say "unclear." A confident guess you can't actually back with a visible cue is worse than an honest "unclear"; it erodes trust when a coach checks it against the tape.

You MUST justify your call in one to two sentences citing the SPECIFIC visual cue(s) you used for position AND action_class (e.g. "ankles crossed and locked behind the top player's back; top player's arm trapped between both of bottom player's legs, hips elevated"). If you chose "unclear" for outcome, briefly say WHAT was hidden (e.g. "tapping hand not visible, camera angle blocked by referee"). A label with no cited cue is not acceptable.
""".strip()

_SINGLE_CALL_INITIAL_TEMPLATE = Template("""
Here is the native video clip for this highlight (its own corrected time window — you do not need to report timestamps, only position/action_class/outcome).

Context from the earlier body-movement scan (for orientation only — do not let it bias your read if the video shows something different): "$description"

Decide the ONE position value, the ONE action_class value, and the ONE outcome value for this clip, per the system instructions. If nothing resembling a named technique is happening, use "transition" for action_class — that is a legitimate, expected answer, not a failure.

Return ONLY valid JSON matching this shape — no prose:
{
    "position": "<one Position enum value>",
    "action_class": "<one Action Class enum value>",
    "outcome": "<one Outcome enum value>",
    "justification": "string — one to two sentences citing the specific visual cue(s) used; if outcome is 'unclear', also say what was hidden"
}
""".strip())


def build_single_call_schema() -> types.Schema:
    """ONE flat verdict: ``{position, action_class, outcome, justification}``
    — NOT a list, NOT timestamped, NO actor field (identity is a separate,
    independent call — see the actor axis below). All three taxonomy fields
    are hard STRING enums; ``justification`` is required free text (Gracie's
    original per-axis discipline, carried over: "A label with no cited cue
    is not acceptable")."""
    return types.Schema(
        type=types.Type.OBJECT,
        required=["position", "action_class", "outcome", "justification"],
        properties={
            "position": types.Schema(
                type=types.Type.STRING, enum=list(POSITION_VALUES),
                description="Body configuration established/most contested in this clip — exactly one enum value.",
            ),
            "action_class": types.Schema(
                type=types.Type.STRING, enum=list(ACTION_CLASS_VALUES),
                description="Action/technique class attempted in this clip — exactly one enum value.",
            ),
            "outcome": types.Schema(
                type=types.Type.STRING, enum=list(OUTCOME_VALUES),
                description="Outcome of the attempt — exactly one enum value.",
            ),
            "justification": types.Schema(
                type=types.Type.STRING,
                description=(
                    "One to two sentences citing the specific visual cue(s) used for position and "
                    "action_class; if outcome='unclear', what was hidden."
                ),
            ),
        },
    )


def build_single_call_prompt(description: Optional[str]) -> str:
    """``$description`` substituted (``string.Template.safe_substitute``)."""
    return _SINGLE_CALL_INITIAL_TEMPLATE.safe_substitute(
        description=description or "unknown — no description was provided by the scan pass",
    )


# --------------------------------------------------------------------------- #
# Actor axis — independent call #3 (S12 Phase 1b production wiring design
# §4.1/§4.3): WHICH named athlete (or a contested/unclear sentinel) is the
# primary actor in this clip. A FOURTH, independent axis call — flat +1 per
# highlight, never re-invoked by the validator loop (same call-count
# treatment as position/technique). Reference images are supplied as
# inline ``Part``s by the caller (``executors.highlight_analyze_node``, via
# ``SimplifiedTagsTimeAnalyzer.analyze_chunk``'s ``extra_parts`` kwarg) —
# this module only builds the schema/prompt TEXT, never touches image bytes.
# --------------------------------------------------------------------------- #
ACTOR_SYSTEM_PROMPT: str = """
You are judging ONE axis only: WHO is the primary actor (the athlete initiating or dominating the exchange) in this BJJ clip. Do not think about position, technique, or outcome at all right now — those are separate, independent judgments made by someone else.

You have been given one or more REFERENCE IMAGES, each labeled with an athlete's name and id. Use ONLY visual comparison against those reference images (gi/rashguard color, build, visible features) to decide which labeled athlete is the primary actor in the clip. If reference images are not decisive — occlusion, both athletes visually similar, a genuinely contested/neutral moment, or no reference images were provided at all — use the honest "contested" or "unclear" sentinel rather than guessing. NEVER invent an athlete identity that was not given to you in a reference image; NEVER output a bare positional description (e.g. "top player") as if it were an identity — that is a different, already-separate judgment.

You MUST also report whether you are confident in this identity call (``identity_uncertain``) — this is independent of which value you chose for ``actor``: a model can name a specific athlete AND still flag low confidence (e.g. a good-but-not-certain visual match), or resolve to a sentinel with high confidence (e.g. a clearly contested scramble where neither athlete is dominant).

You MUST justify your call in one sentence citing the SPECIFIC visual cue you used (e.g. "blue gi matches reference image 2's athlete" or "both athletes tangled, neither reference image clearly matches the dominant position").
""".strip()

_ACTOR_INITIAL_TEMPLATE = Template("""
Here is the native video clip for this highlight (its own corrected time window — you do not need to report timestamps, only actor identity).

$reference_preamble

Context from the earlier body-movement scan (for orientation only — do not let it bias your identity call): "$description"

Decide the ONE actor value for the primary actor in this clip, per the system instructions.

Return ONLY valid JSON matching this shape — no prose:
{
    "actor": "<a labeled player_id, or a contested/unclear sentinel value>",
    "identity_uncertain": true | false,
    "justification": "string — one sentence citing the specific visual cue used"
}
""".strip())


def build_actor_schema(player_choices: list[str]) -> types.Schema:
    """ONE flat verdict: ``{actor, identity_uncertain, justification}`` — NOT
    a list, NOT timestamped (see module docstring). The ``actor`` enum is
    built PER JOB (not a static module constant, unlike ``POSITION_VALUES``/
    ``ACTION_CLASS_VALUES``) — valid player-identity values are dynamic per
    match: ``player_choices`` (the job's real ``player_id`` values) plus the
    always-legal ``AXIS2_ACTOR_SENTINELS``. ``identity_uncertain`` is the
    model's own honesty channel, independent of whether ``actor`` resolved
    to a sentinel or a real ``player_id``."""
    return types.Schema(
        type=types.Type.OBJECT,
        required=["actor", "identity_uncertain", "justification"],
        properties={
            "actor": types.Schema(
                type=types.Type.STRING,
                enum=[*player_choices, *AXIS2_ACTOR_SENTINELS],
                description=(
                    "The primary actor's player_id (from the labeled reference images), "
                    "or a contested/unclear sentinel — exactly one enum value."
                ),
            ),
            "identity_uncertain": types.Schema(
                type=types.Type.BOOLEAN,
                description="Honest confidence flag for this identity call, independent of the chosen actor value.",
            ),
            "justification": types.Schema(
                type=types.Type.STRING,
                description="One sentence citing the specific visual cue used to decide this label.",
            ),
        },
    )


def build_actor_prompt(description: Optional[str], player_references: list[dict]) -> str:
    """``$description``/``$reference_preamble`` substituted
    (``string.Template.safe_substitute``). ``player_references`` is the
    job's ``RunContext.player_references`` list (``{"player_id",
    "player_name", ...}`` dicts, per ``highlight_ingest.py``) — used ONLY to
    build the labeling preamble text ("Reference image 1 = athlete ... (id
    ...)"); the actual inline image ``Part``s are attached separately by the
    caller. An empty list (e.g. the QA playground, which never populates
    ``player_references``) produces an explicit "no reference images"
    preamble rather than a silently empty one."""
    if player_references:
        lines = [
            f"Reference image {i + 1} = athlete \"{ref.get('player_name') or ref.get('player_id')}\" "
            f"(id \"{ref.get('player_id')}\")"
            for i, ref in enumerate(player_references)
        ]
        preamble = (
            "Reference images (attached below, in this order):\n" + "\n".join(lines)
        )
    else:
        preamble = (
            "No reference images are available for this job — you cannot resolve a named "
            "identity; use the contested/unclear sentinel."
        )
    return _ACTOR_INITIAL_TEMPLATE.safe_substitute(
        description=description or "unknown — no description was provided by the scan pass",
        reference_preamble=preamble,
    )


__all__ = [
    "SINGLE_CALL_SYSTEM_PROMPT",
    "ACTOR_SYSTEM_PROMPT",
    "build_single_call_schema",
    "build_single_call_prompt",
    "build_actor_schema",
    "build_actor_prompt",
]
