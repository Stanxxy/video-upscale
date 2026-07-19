"""Position/technique/validator axis helpers (v2 build plan,
``2026-07-18-highlight-scan-critique-analyze-v2-plan.md``, "Node 3 —
highlight_analyze"). Pure functions only (schema + prompt) — no Gemini
client; ``service/pipelines/executors.py``'s ``highlight_analyze_node`` is
the only caller that actually talks to Gemini, via the shared
``simplified_tags.SimplifiedTagsTimeAnalyzer.analyze_chunk`` (reusing its
offset-quantization/retry/instrumentation plumbing, INS-107 discipline).

**Design (Gracie's final draft, dropped in verbatim from
``v2_node_prompts.md``, 2026-07-18) — ONE verdict per highlight, not a list
of timestamped sub-events.** Earlier engineering scaffolding for this module
modeled position/technique as LISTS of chunk-relative-timestamped tags (one
call producing N sub-spans), mirroring ``simplified_tags``'s
``simplified-tags-time-v1`` shape. Gracie's real prompts are narrower and
simpler: the highlight's OWN (already critique-corrected) window IS the unit
of judgment — "Decide the ONE position value that is established or being
most contested in this clip" / "Decide the ONE action_class value and the
ONE outcome value for this clip." Neither call reports timestamps, an actor,
a confidence score, or a list — each returns a single flat object. The
validator then reconciles those two single verdicts (not a merged list) via
an adversarial review + a strict ditch test, returning a single
``final_position``/``final_action_class``/``final_outcome`` triple (or a
ditch verdict). ``executors.highlight_analyze_node`` builds ONE synthesized
clip per non-ditched highlight (spanning the highlight's own authoritative
bounds), not a merge-by-time-overlap of independent tag lists.
"""
from __future__ import annotations

from string import Template
from typing import Optional

from google.genai import types

from service.pipelines.simplified_tags import ACTION_CLASS_VALUES, OUTCOME_VALUES, POSITION_VALUES

# --------------------------------------------------------------------------- #
# Position axis — independent call #1: WHAT POSITION, and ONLY position.
# --------------------------------------------------------------------------- #
POSITION_SYSTEM_PROMPT: str = """
You are judging ONE axis only: WHAT POSITION is established or being contested in this BJJ clip. Do not think about technique, submission, or outcome at all right now — that is a separate, independent judgment made by someone else. Body configuration is large-scale and rarely occluded, so this is the most reliable signal you will be asked for.

Use EXACTLY one of these values — never invent a new one, never combine two:

- standing — includes clinch
- closed_guard — legs locked, ankles crossed
- open_guard — any open-guard variant (butterfly, spider, De La Riva, X-guard, etc. all collapse here)
- half_guard — one leg trapped
- side_control — includes knee-on-belly
- mount — full mount
- back_control — hooks or body triangle in
- turtle — either player turtled
- scramble — no stable position established; this is an HONEST "not sure / still resolving" bucket, not a failure on your part

It is legitimate — and expected — for a clip to be a pure standing exchange with nothing else established: use "standing" for that, plainly, rather than forcing a guard/mount/side-control label that isn't there. Use "scramble" only when position is genuinely too chaotic/contested to call even standing/guard cleanly.

You MUST justify your call in one sentence citing the SPECIFIC visual cue you used (e.g. "ankles crossed and locked behind the top player's back" or "both players upright, no grips below the waist"). A label with no cited cue is not acceptable.
""".strip()

_POSITION_INITIAL_TEMPLATE = Template("""
Here is the native video clip for this highlight (its own corrected time window — you do not need to report timestamps, only the position).

Context from the earlier body-movement scan (for orientation only — do not let it bias which enum you pick if the video shows something different): "$description"

Decide the ONE position value that is established or being most contested in this clip, per the system instructions.

Return ONLY valid JSON matching this shape — no prose:
{
    "position": "<one Position enum value>",
    "justification": "string — one sentence citing the specific visual cue"
}
""".strip())


def build_position_schema() -> types.Schema:
    """ONE flat verdict: ``{position, justification}`` — NOT a list, NOT
    timestamped (see module docstring). ``position`` is a hard STRING enum;
    ``justification`` is required free text (Gracie: "A label with no cited
    cue is not acceptable")."""
    return types.Schema(
        type=types.Type.OBJECT,
        required=["position", "justification"],
        properties={
            "position": types.Schema(
                type=types.Type.STRING, enum=list(POSITION_VALUES),
                description="Body configuration established/most contested in this clip — exactly one enum value.",
            ),
            "justification": types.Schema(
                type=types.Type.STRING,
                description="One sentence citing the specific visual cue used to decide this label.",
            ),
        },
    )


def build_position_prompt(description: Optional[str]) -> str:
    """``$description`` substituted (``string.Template.safe_substitute``)."""
    return _POSITION_INITIAL_TEMPLATE.safe_substitute(
        description=description or "unknown — no description was provided by the scan pass",
    )


# --------------------------------------------------------------------------- #
# Technique axis — independent call #2: action_class + outcome, ONLY.
# --------------------------------------------------------------------------- #
TECHNIQUE_SYSTEM_PROMPT: str = """
You are judging TWO axes only: WHAT TECHNIQUE/ACTION is being attempted, and its OUTCOME. This is INDEPENDENT of position — someone else is judging position separately, and both of you may be right at once (a guard pass attempt can be happening while the position shown is still half_guard; a submission attempt can be live while the position is mount). Do not try to make your answer consistent with a position label — you don't have one.

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

- successful — tap, sweep/pass/takedown/escape completed, points scored, for the athlete you identified as attempting the action
- unsuccessful — the attempt didn't land
- unclear — a scramble/contested moment resolved without a clean read

Outcome is the hardest of these two calls, especially for submissions where the finishing grip or the tapping hand is hidden from camera. HONEST UNCERTAINTY IS REQUIRED HERE: if you cannot actually see whether it landed — because the grip is occluded, the camera cuts away, or bodies are stacked — say "unclear." A confident guess you can't actually back with a visible cue is worse than an honest "unclear"; it erodes trust when a coach checks it against the tape.

You MUST justify your action_class call in one sentence citing the specific visual cue (e.g. "top player's arm trapped between both of bottom player's legs, hips elevated" for an arm lock attempt). If you chose "unclear" for outcome, briefly say WHAT was hidden (e.g. "tapping hand not visible, camera angle blocked by referee").
""".strip()

_TECHNIQUE_INITIAL_TEMPLATE = Template("""
Here is the native video clip for this highlight (its own corrected time window — you do not need to report timestamps, only technique/outcome).

Context from the earlier body-movement scan (for orientation only — do not let it bias which enum you pick if the video shows something different): "$description"

Decide the ONE action_class value and the ONE outcome value for this clip, per the system instructions. If nothing resembling a named technique is happening, use "transition" — that is a legitimate, expected answer, not a failure.

Return ONLY valid JSON matching this shape — no prose:
{
    "action_class": "<one Action Class enum value>",
    "outcome": "<one Outcome enum value>",
    "justification": "string — one sentence citing the specific visual cue for action_class; if outcome is 'unclear', also say what was hidden"
}
""".strip())


def build_technique_schema() -> types.Schema:
    """ONE flat verdict: ``{action_class, outcome, justification}`` — NOT a
    list, NOT timestamped, NO actor field (see module docstring)."""
    return types.Schema(
        type=types.Type.OBJECT,
        required=["action_class", "outcome", "justification"],
        properties={
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
                description="One sentence citing the visual cue for action_class; if outcome='unclear', what was hidden.",
            ),
        },
    )


def build_technique_prompt(description: Optional[str]) -> str:
    """``$description`` substituted (``string.Template.safe_substitute``)."""
    return _TECHNIQUE_INITIAL_TEMPLATE.safe_substitute(
        description=description or "unknown — no description was provided by the scan pass",
    )


# --------------------------------------------------------------------------- #
# Validator — adversarial reconciliation + strict ditch authority.
# --------------------------------------------------------------------------- #
_VALIDATOR_SYSTEM_BODY: str = """
You are the adversarial check on two independent judgments already made about this BJJ clip: a POSITION call and a TECHNIQUE/OUTCOME call. Your job is NOT to rubber-stamp them. Do three things, in this exact order, and never skip the first one:

1. ARGUE THE OPPOSITE FIRST. Before you decide anything, build the strongest case you can that EACH label is WRONG. For the position label: what position would you have called instead, and why? For the action_class/outcome labels: what would you have called instead? You MUST cite specific visual evidence for your opposing case — a frame-level detail, a body configuration, a grip you can see — never just assert "I disagree." Only after making this case do you decide whether the original labels actually survive it.

2. CHECK FOR A GENUINE CONTRADICTION between the position label and the action_class label — NOT mere rarity, a real structural impossibility given what each enum means (see the incompatible-combination list you were given separately). A guard_pass attempt paired with a side_control or mount ending position is normal and expected — that is success, not a contradiction. Only flag it if the combination could not physically occur under the taxonomy's own definitions.

3. APPLY THE DITCH TEST — the STRICT bar. Recommend status="ditched" ONLY if there was NO intentional grip change, weight change, or hip change by EITHER athlete anywhere in this clip — the case where a coach watching would say "nothing happened here." Do NOT ditch:
   - a failed sweep, failed pass, failed takedown, or failed submission attempt — these are real unsuccessful attempts, not non-events, and the human reviewer needs to see them;
   - a stall or grip-fighting sequence where either athlete is actively fighting for grips, even if no technique lands;
   - guard retention / guard recovery, even if the "technique" resolves to "transition."
   If you are unsure whether something happened, do NOT ditch — ditching is for the rare, unambiguous non-event only.

Your final verdict is exactly one of:
- "agree" — both labels survive your adversarial case; report them as the final labels (unchanged).
- "disagree" — one specific label is wrong; say which one (position/action_class/outcome) and what it should be instead, with the visual evidence that changed your mind.
- "ditch" — this clip passes the strict ditch test; give the reason.

Every verdict, including "agree," must cite the specific visual evidence you used — never a bare confirmation with no cue.
""".strip()

# Reference list the SYSTEM prompt above refers to as "given to you
# separately" (Gracie's markdown keeps this as its own section) —
# concatenated onto the system instruction actually sent to Gemini so the
# model has it in-context, not just referenced.
IMPOSSIBLE_COMBOS_TEXT: str = """
IMPOSSIBLE position x action_class combinations (validator contradiction check):

1. guard_pass x closed_guard (as ending position) — a completed pass cannot end in closed guard; that's `transition`.
2. submission_leg_lock x mount OR back_control — no leg access from either.
3. guard_pull x mount/side_control/back_control — guard-pull is a standing entry, can't originate from an established top position.
4. back_take x closed_guard (as ending position) — back_take is defined as transition INTO back control.
""".strip()

VALIDATOR_SYSTEM_PROMPT: str = f"{_VALIDATOR_SYSTEM_BODY}\n\n{IMPOSSIBLE_COMBOS_TEXT}"

_VALIDATOR_INITIAL_TEMPLATE = Template("""
Here is the native video clip for this highlight.

Scan's body-movement description (context only): "$description"

POSITION call: "$position_label" — justification given: "$position_justification"

TECHNIQUE call: action_class="$technique_label", outcome="$outcome_label" — justification given: "$technique_justification"

Follow the system instructions in order: argue the opposite first (cite evidence), check for a genuine position×action_class contradiction, then apply the strict ditch test. Reach exactly one verdict.

Return ONLY valid JSON matching this shape — no prose:
{
    "adversarial_case": "string — the strongest case you built AGAINST the given labels, with cited evidence, before you decided",
    "verdict": "agree" | "disagree" | "ditch",
    "final_position": "<Position enum value or null>",
    "final_action_class": "<Action Class enum value or null>",
    "final_outcome": "<Outcome enum value or null>",
    "wrong_label": "position" | "action_class" | "outcome" | null,
    "reason": "string — required for disagree/ditch, null-able for agree",
    "evidence": "string — the specific frame-level visual cue(s) that decided this verdict"
}
""".strip())


def build_validator_schema() -> types.Schema:
    """ONE flat verdict, highlight-level (NOT per-clip — build plan "Ditched
    contract"). ``final_position``/``final_action_class``/``final_outcome``/
    ``wrong_label``/``reason`` are NULLABLE (Gracie: "null-able for agree")
    — not everything is required on every verdict; ``adversarial_case``/
    ``verdict``/``evidence`` are always required (Gracie: "Every verdict,
    including agree, must cite the specific visual evidence")."""
    return types.Schema(
        type=types.Type.OBJECT,
        required=["adversarial_case", "verdict", "evidence"],
        properties={
            "adversarial_case": types.Schema(
                type=types.Type.STRING,
                description="The strongest case built AGAINST the given labels, with cited evidence, before deciding.",
            ),
            "verdict": types.Schema(
                type=types.Type.STRING, enum=["agree", "disagree", "ditch"],
                description="Exactly one of agree/disagree/ditch.",
            ),
            "final_position": types.Schema(
                type=types.Type.STRING, enum=list(POSITION_VALUES), nullable=True,
                description="Final position label (unchanged on agree, corrected on disagree, null-able).",
            ),
            "final_action_class": types.Schema(
                type=types.Type.STRING, enum=list(ACTION_CLASS_VALUES), nullable=True,
                description="Final action_class label (unchanged on agree, corrected on disagree, null-able).",
            ),
            "final_outcome": types.Schema(
                type=types.Type.STRING, enum=list(OUTCOME_VALUES), nullable=True,
                description="Final outcome label (unchanged on agree, corrected on disagree, null-able).",
            ),
            "wrong_label": types.Schema(
                type=types.Type.STRING, enum=["position", "action_class", "outcome"], nullable=True,
                description="Which label was wrong, when verdict='disagree'. Null otherwise.",
            ),
            "reason": types.Schema(
                type=types.Type.STRING, nullable=True,
                description="Required in spirit for disagree/ditch; null-able for agree.",
            ),
            "evidence": types.Schema(
                type=types.Type.STRING,
                description="The specific frame-level visual cue(s) that decided this verdict.",
            ),
        },
    )


def build_validator_prompt(
    *, description: Optional[str], position_label: str, position_justification: str,
    technique_label: str, outcome_label: str, technique_justification: str,
) -> str:
    """``$description``/``$position_label``/``$position_justification``/
    ``$technique_label``/``$outcome_label``/``$technique_justification``
    substituted (``string.Template.safe_substitute``) — the EXACT six token
    names Gracie's text uses. All six MUST be supplied by the caller."""
    return _VALIDATOR_INITIAL_TEMPLATE.safe_substitute(
        description=description or "unknown — no description was provided by the scan pass",
        position_label=position_label, position_justification=position_justification,
        technique_label=technique_label, outcome_label=outcome_label,
        technique_justification=technique_justification,
    )


__all__ = [
    "POSITION_SYSTEM_PROMPT",
    "TECHNIQUE_SYSTEM_PROMPT",
    "VALIDATOR_SYSTEM_PROMPT",
    "IMPOSSIBLE_COMBOS_TEXT",
    "build_position_schema",
    "build_position_prompt",
    "build_technique_schema",
    "build_technique_prompt",
    "build_validator_schema",
    "build_validator_prompt",
]
