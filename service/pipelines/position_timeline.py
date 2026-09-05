"""Continuous position-state contracts for the QA VLM Studio timeline pipeline.

Gemini proposes an initial state and transition *boundaries*. Python owns the
interval construction and duration arithmetic so independently generated model
ranges can never create hidden gaps or overlaps.
"""

from __future__ import annotations

from collections import defaultdict
import io
import re
from string import Template
from typing import Iterable, Mapping, Sequence

from google.genai import types

from service.pipelines.simplified_tags import ACTION_CLASS_VALUES, OUTCOME_VALUES
from service.pipelines.highlight_critique import DEFAULT_CRITIQUE_INITIAL_PROMPT


ACTIVE_POSITIONS: tuple[str, ...] = (
    "standing",
    "closed_guard",
    "open_guard",
    "half_guard",
    "side_control",
    "mount",
    "back_control",
    "turtle",
    "scramble",
    "other",
)
TIMELINE_POSITIONS: tuple[str, ...] = ACTIVE_POSITIONS + ("inactive",)
PLAYER_ROLES: tuple[str, ...] = ("top", "bottom", "neutral", "not_applicable")
TECHNIQUE_ACTOR_SENTINELS: tuple[str, ...] = ("contested", "unclear")

# A bare event name or an alias of an existing core position cannot explain
# the geometry of an ``other`` interval. Reject these exact-note failures so a
# taxonomy miss cannot silently contaminate duration statistics. Descriptive
# notes that include real geometry remain valid.
_NON_GEOMETRIC_OTHER_NOTES: frozenset[str] = frozenset(
    {
        "standing",
        "closed guard",
        "full guard",
        "open guard",
        "half guard",
        "side control",
        "side mount",
        "mount",
        "full mount",
        "back control",
        "back mount",
        "rear mount",
        "turtle",
        "scramble",
        "armbar",
        "arm lock",
        "kimura",
        "americana",
        "omoplata",
        "triangle",
        "triangle choke",
        "choke",
        "rear naked choke",
        "guillotine",
        "heel hook",
        "kneebar",
        "ankle lock",
        "toe hold",
        "sweep",
        "pass",
        "guard pass",
        "escape",
        "takedown",
        "back take",
        "submission",
        "transition",
    }
)


def _normalized_position_note(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

POSITION_SCAN_SYSTEM_PROMPT = """
You reconstruct the continuous Brazilian Jiu-Jitsu positional state of two
target athletes in arbitrary source footage. The source may be a formal
competition, an in-house match, open mat or live sparring, a technique drill,
or uncontrolled phone footage from a random gym. Track exactly the two target
athletes identified by the labelled reference images. Ignore every other pair,
coach, referee, spectator, and athlete who enters the frame. Never identify the
targets from rashguard or gi color, corner color, scoreboard text, commentary,
or reference-image order alone.

Return the state at the requested scope start and every later instant where
the position, either athlete's geometric top/bottom/neutral role, or the
active/inactive status changes. Return transition boundaries, never
independently invented ranges. Exactly one state covers every instant in the
scope. Do not judge which athlete is dominant, controlling, winning, or more
effective.

Use only these active positions and decide from current body geometry, never
from the name of the technique that led there:

- standing: both athletes are upright or in a standing clinch.
- closed_guard: the bottom athlete's legs encircle the top athlete with ankles
  crossed or otherwise visibly closed behind them. "Full guard" maps here.
- open_guard: the bottom athlete uses unclosed legs/feet to frame, hook, or
  control the top athlete and no more specific outside-taxonomy leg
  entanglement is established.
- half_guard: the bottom athlete's legs visibly trap one of the top athlete's
  legs.
- side_control: the top athlete has passed the legs and pins from the side in a
  chest-to-chest or comparable side-pin relationship. "Side mount" maps here.
- mount: one athlete straddles the other's torso or hips while facing them.
- back_control: one athlete is attached behind the other's shoulders/hips in a
  sustained chest-to-back relationship with a seatbelt, hook, body triangle,
  rear body lock, or equivalent rear control. "Back mount" and "rear mount"
  map here. This is a geometric category, not a scoring judgment, so do not
  require two hooks or an official points signal.
- turtle: one athlete is compact on knees plus hands/elbows with the torso
  facing the mat while the opponent is beside, above, or behind them, but has
  not established the rear attachment required for back_control. Turtle
  remains turtle while an opponent circles behind or attacks the back;
  back_control takes precedence only once that rear control is visibly
  established, and the state returns to turtle when that attachment is lost.
- scramble: the athletes are actively changing relationship and no listed or
  outside-taxonomy position is yet established. End scramble as soon as a
  stable body relationship is visible.
- other: a real, visibly established geometric subposition outside the list.

Use neutral roles when neither athlete has an established geometric top/bottom
relationship, including ordinary standing and unresolved scrambles. For other,
describe concrete body geometry in position_note, for example 50/50,
north-south, knee-on-belly, ashi garami, saddle, or single-leg X. A technique
or action name is never a position or sufficient position_note: armbar,
triangle, choke, sweep, pass, escape, takedown, back take, and submission are
events. If a submission creates an outside-taxonomy configuration, describe
the visible torso/hip/leg relationship instead of naming the submission. Never
use other for back mount, rear mount, side mount, full guard, or turtle; map
those aliases to their core category. Do not collapse a clearly visible named leg entanglement
into generic open_guard. Use open_guard only when no more
specific outside-taxonomy configuration is visibly established.

Preserve the last established active state through brief occlusion, camera
motion, or a person crossing the frame when the same exchange visibly
continues and there is no reset cue. Use inactive only for a real interruption:
a referee stoppage or reset, a mutual training reset or full disengagement, a
round break, coach instruction or drill pause, replay, hard edit, camera cut
away from the target pair, pre/post-round footage, or a span where the target
pair is not engaged and no BJJ position can truthfully be assigned. Training
footage does not require a referee cue; a visible partner reset, round timer,
coach pause, or drill restart is sufficient. For inactive, set both roles to
not_applicable and give a concrete inactive_reason.

Ground every boundary in visible body relationship or a visible interruption
cue. When evidence is ambiguous, prefer the last visibly established state
until a new state or interruption is established; never invent a transition
from an expected BJJ sequence.
""".strip()

POSITION_SCAN_INITIAL_PROMPT = """
Analyze the native BJJ source footage from absolute source-video second
$start_sec through $end_sec. The first state must describe exactly $start_sec,
and every transition timestamp must be an ABSOLUTE source-video second on that
same clock. Do not assume the scope begins at a round boundary: infer the state
already established at $start_sec from the visible relationship and immediate
context.

Athlete A is the player_id on the first labelled reference and Athlete B is
the player_id on the second labelled reference. Track only those two targets,
even if other athletes or simultaneous rolling pairs are visible. Apply the
competition/training interruption rules from the system instruction.

Make a chronological pass through the entire requested scope before
finalizing. Do not stop scanning after a salient submission or keep its label
through the later exchange. After every attempt, defense, sweep/reversal,
guard engagement/disengagement, pass, back exposure, rear-control attachment
or loss, and settled scramble, reassess the position and roles from visible
body geometry. Long proposed intervals require the same continued scrutiny as
short ones. Emit every established state change, including repeated returns
between guard, side_control, turtle, and back_control. Return only the required
structured JSON.
""".strip()

_POSITION_SCAN_INITIAL_TEMPLATE = Template(POSITION_SCAN_INITIAL_PROMPT)

TECHNIQUE_SCAN_SYSTEM_PROMPT = """
Find sparse, intentional Brazilian Jiu-Jitsu technique events in a native
match video. Cast wide for recall: takedown attempts, guard pulls, sweeps,
guard passes, back takes, submissions, escapes, and intentional transitions
all count whether successful or unsuccessful. Static position holding and
ordinary grip adjustment without a technique attempt do not count. Describe
visible body movement without assigning a position or technique label; a
later grounded analyzer supplies the label. Exclude broadcast replays so one
real technique is never reported twice.

BOUNDARY CONTRACT — every highlight must be outcome-complete, not
motion-complete. Treat start_s and end_s as independent decisions:

- Start at the earliest intentional setup movement causally connected to the
  attempt, not at generic grip fighting before the attempt and not at the
  later visually dramatic moment.
- End only after the attempt's visible result is established. For a successful
  action, include the finish and the resulting control, reversal, escape, or
  stoppage settling. For an unsuccessful action, include the completed
  defense, escape from the threat, forced reset, or the attacker's clear
  abandonment/switch away from the attempt. If the result remains unclear,
  follow the same action through the last relevant visible evidence and say
  why it is unclear.
- A viewer watching only start_s through end_s must see enough evidence to
  judge the attempt as successful, unsuccessful, or unclear. Never end merely
  because the attacking motion pauses, first meets resistance, or stops being
  visually dramatic. Do not add a fixed tail or unrelated later exchange.

Return one intentional technique event per highlight. If a sweep or reversal
is followed by a submission, guard pull, pass, escape, or another intentional
action, return separate highlights even when their bounds touch or overlap.
Never combine a compound sequence into one highlight merely because the
actions flow together. The opponent's immediate defense remains inside the
original attempt's highlight because it establishes that attempt's outcome;
if that defense also becomes a distinct intentional counter-technique, emit a
separate overlapping highlight for the counter without truncating the
original attempt before its result is visible.
""".strip()

TECHNIQUE_SCAN_INITIAL_PROMPT = """
Scan the requested BJJ match scope from absolute second $start_sec through
$end_sec. Return an ordered list of every intentional technique event worth
close analysis. For each event, report absolute start_s/end_s, a concrete
body-movement description without a taxonomy label, and reasoning explaining
the visible intentional grip, weight, hip, or limb change. The description
must cover the attempt's setup/execution AND the visible result cue; do not
describe only the attacker's last motion.

Before finalizing each end_s, inspect the later frames and apply the
outcome-complete boundary contract. A viewer watching only start_s through
end_s must be able to judge the event as successful, unsuccessful, or unclear.
If the requested scope ends while the same attempt is unresolved, use
$end_sec as end_s and state in reasoning that the outcome remains unclear at
the scope boundary; never invent a result outside the available video.

Return one intentional technique event per highlight and separate highlights
for distinct sequential or overlapping actions. In particular, a sweep or
reversal followed by a submission attempt is two highlights, not one compound
highlight. Return only the required structured JSON.
""".strip()

TECHNIQUE_CRITIQUE_POSITION_INITIAL_PROMPT = (
    DEFAULT_CRITIQUE_INITIAL_PROMPT
    + """

ADVISORY POSITION-TIMELINE CONTEXT (independently produced and fallible):
$position_context_json

Use this only as a search aid for the described movement and its likely
before/after state. It may be wrong or slightly time-shifted. Never confirm,
reject, or resize the event from the position labels alone: visible body-movement evidence in this clip is authoritative.
"""
).strip()


def build_position_scan_prompt(start_sec: float, end_sec: float, override: str | None = None) -> str:
    template = Template(override.strip()) if override and override.strip() else _POSITION_SCAN_INITIAL_TEMPLATE
    return template.safe_substitute(start_sec=start_sec, end_sec=end_sec)


POSITION_CRITIQUE_SYSTEM_PROMPT = """
You verify one proposed boundary in a continuous BJJ position timeline. Use
the labelled reference images to preserve athlete identity. Decide whether a
real change of position, athlete role, or active/inactive state happens in the
provided local window. If confirmed, report the earliest absolute match
second at which the new state is visibly established. Do not judge dominance
or control. Use other only with a precise subposition note; use inactive only
with an explicit interruption reason.

Apply the scan taxonomy from visible geometry. In particular, resolve
back_control versus turtle explicitly: back_control requires a sustained
chest-to-back/rear attachment such as a seatbelt, hook, body triangle, or rear
body lock; turtle is a mat-facing knees-plus-hands/elbows posture without that
established rear attachment. Map back mount/rear mount to back_control and side
mount to side_control. A technique or action name such as armbar, sweep, pass,
escape, or back take is not a position and is not a sufficient other note. If
the proposed other note merely names an action, replace state_after with the
geometric core position or a precise outside-taxonomy body configuration that
is actually visible.

Search anywhere in the provided local window, not only at the proposed frame.
If the real transition occurs earlier or later than the rough timestamp, set
confirmed=true and correct corrected_at_s to that visible boundary. Reject only
when the proposed state change never becomes established anywhere in the
window.
""".strip()

POSITION_CRITIQUE_INITIAL_PROMPT = """
This native-video window spans absolute seconds $window_start_s to
$window_end_s. PROPOSED TRANSITION: the rough scan placed a boundary at
$proposed_at_s. The state immediately before it was $state_before_json. The
rough proposed state after it was $state_after_json.

Confirm or reject that state change. If confirmed, correct corrected_at_s on
the absolute match clock and return the complete state_after. If rejected,
set confirmed=false; still return the best visible state_after required by
the schema, but it will not be used. Return only structured JSON.

The rough timestamp is only a search anchor. A matching change later in this
window is a confirmed transition with a later corrected_at_s, not a rejection.
""".strip()

_POSITION_CRITIQUE_TEMPLATE = Template(POSITION_CRITIQUE_INITIAL_PROMPT)


def build_position_critique_prompt(
    *,
    window_start_s: float,
    window_end_s: float,
    proposed_at_s: float,
    state_before: Mapping,
    state_after: Mapping,
    override: str | None = None,
) -> str:
    import json

    template = (
        Template(override.strip())
        if override and override.strip()
        else _POSITION_CRITIQUE_TEMPLATE
    )
    return template.safe_substitute(
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        proposed_at_s=proposed_at_s,
        state_before_json=json.dumps(dict(state_before), sort_keys=True),
        state_after_json=json.dumps(dict(state_after), sort_keys=True),
    )


TECHNIQUE_SYSTEM_PROMPT = """
Judge one sparse BJJ technique event independently of position. Use the two
labelled reference images to ground the actor to an allowed player_id. Return
one action class and one outcome. A failed attempt remains an event. Use
contested when both athletes intentionally create the action and unclear only
when the actor cannot be resolved from the references. Give an optional
specific-technique guess only when visible biomechanics support it. Never
emit a position, controller, dominance, or winner judgment.

The scan-described body movement is the target event. Surrounding video may
contain setup, counters, or the next technique for context. Do not substitute
a later or overlapping technique merely because it is more visually salient.
Classify the described movement itself; a separate highlight handles every
neighboring event.
""".strip()

_TECHNIQUE_TEMPLATE = Template(
    """
Analyze the native-video technique highlight described as: "$description".
The highlight's authoritative event bounds are absolute seconds $start_s to
$end_s; the video part may include surrounding context. Athlete ids allowed:
$player_ids. Return the one grounded technique event as structured JSON. Do
not return timestamps or a position; Python attaches the approved bounds and
the independently built position interval later.

The scan-described movement is the target event. Judge that movement even if
a different technique begins later inside the video context. Do not replace
the target with a neighboring action.

ADVISORY POSITION-TIMELINE CONTEXT (independently produced and fallible):
$position_context_json
Use it as supporting context only. If it conflicts with visible body-movement evidence, trust the video and explain the visible technique—not the timeline.
""".strip()
)


def build_technique_prompt(
    description: str | None,
    *,
    start_s: float,
    end_s: float,
    player_ids: tuple[str, str],
    position_context: Mapping | None = None,
) -> str:
    import json

    return _TECHNIQUE_TEMPLATE.safe_substitute(
        description=description or "no scan description supplied",
        start_s=start_s,
        end_s=end_s,
        player_ids=", ".join(player_ids),
        position_context_json=json.dumps(dict(position_context or {}), sort_keys=True),
    )


TECHNIQUE_VALIDATOR_SYSTEM_PROMPT = """
Perform one adversarial review of a proposed grounded BJJ technique event.
First make the strongest evidence-based case that its actor, action class, or
outcome is wrong. First state whether any intentional technique is visible,
then return agree, correct, or ditch. Ditch only when no intentional technique
attempt occurred; failed attempts are never ditched. A different intentional
technique is a correction, never a ditch. A mismatch with the scanner's
description is not grounds to ditch when any intentional BJJ action is visible.
Review the proposed fields against the scan-described target movement. Do not
substitute a neighboring event from the surrounding context; that event has
its own highlight and validator call.
Do not add or judge a position, dominance, controller, or match winner.
""".strip()

_TECHNIQUE_VALIDATOR_TEMPLATE = Template(
    """
ADVERSARIAL TECHNIQUE REVIEW
Body-movement scan: "$description"
Proposed technique JSON: $technique_json

ADVISORY POSITION-TIMELINE CONTEXT (independently produced and fallible):
$position_context_json
Use it only to challenge or corroborate the proposal; visible body-movement evidence is authoritative when the timeline conflicts with the clip.

Return the final complete technique fields and the review reason. On agree,
echo the surviving values. On correct, replace only values the visible video
contradicts. On ditch, the fields remain schema-valid but Python discards the
event. Set intentional_technique_visible=true for every real attempt, including
failed attempts and actions different from the scanner description. Return
only structured JSON.
""".strip()
)


def build_technique_validator_prompt(
    description: str | None,
    technique: Mapping,
    *,
    position_context: Mapping | None = None,
) -> str:
    import json

    return _TECHNIQUE_VALIDATOR_TEMPLATE.safe_substitute(
        description=description or "no scan description supplied",
        technique_json=json.dumps(dict(technique), sort_keys=True),
        position_context_json=json.dumps(dict(position_context or {}), sort_keys=True),
    )


def build_reference_parts(references: Sequence) -> list[types.Part]:
    """Create labelled inline-image parts once for a grounded native call."""
    parts: list[types.Part] = []
    for reference in references:
        player_id = _read_reference_field(reference, "player_id")
        player_name = _read_reference_field(reference, "player_name")
        if isinstance(reference, Mapping):
            image_bytes = reference.get("image_bytes")
            mime_type = str(reference.get("mime_type") or "image/jpeg")
            image = reference.get("image")
        else:
            image_bytes = getattr(reference, "image_bytes", None)
            mime_type = str(getattr(reference, "mime_type", None) or "image/jpeg")
            image = getattr(reference, "image", None)
        if not isinstance(image_bytes, (bytes, bytearray)):
            if image is None:
                raise TimelineValidationError(
                    f"reference {player_id!r} has no loaded image bytes"
                )
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=92)
            image_bytes = buffer.getvalue()
            mime_type = "image/jpeg"
        parts.append(
            types.Part(text=f"REFERENCE: {player_name} (player_id={player_id})")
        )
        parts.append(types.Part.from_bytes(data=bytes(image_bytes), mime_type=mime_type))
    return parts


def _position_state_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        required=[
            "position",
            "player_a_role",
            "player_b_role",
            "position_note",
            "inactive_reason",
            "confidence",
            "evidence",
        ],
        properties={
            "position": types.Schema(type=types.Type.STRING, enum=list(TIMELINE_POSITIONS)),
            "player_a_role": types.Schema(type=types.Type.STRING, enum=list(PLAYER_ROLES)),
            "player_b_role": types.Schema(type=types.Type.STRING, enum=list(PLAYER_ROLES)),
            "position_note": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Required concrete body-geometry description when position='other'; "
                    "a technique/action name alone is invalid. Empty otherwise."
                ),
            ),
            "inactive_reason": types.Schema(
                type=types.Type.STRING,
                description="Required interruption reason when position='inactive'; empty otherwise.",
            ),
            "confidence": types.Schema(type=types.Type.NUMBER, format="float"),
            "evidence": types.Schema(
                type=types.Type.STRING,
                description="Visible body-relationship or interruption cue supporting this state.",
            ),
        },
    )


def build_position_scan_schema() -> types.Schema:
    """Initial state + ordered transition boundaries; never independent ranges."""
    transition_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["at_s", "state_after", "evidence"],
        properties={
            "at_s": types.Schema(
                type=types.Type.NUMBER,
                format="float",
                description="Absolute match second where state_after begins.",
            ),
            "state_after": _position_state_schema(),
            "evidence": types.Schema(
                type=types.Type.STRING,
                description="Visible cue that marks this transition boundary.",
            ),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["initial_state", "transitions"],
        properties={
            "initial_state": _position_state_schema(),
            "transitions": types.Schema(type=types.Type.ARRAY, items=transition_schema),
        },
    )


def build_position_critique_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        required=["confirmed", "corrected_at_s", "state_after", "note"],
        properties={
            "confirmed": types.Schema(
                type=types.Type.BOOLEAN,
                description="Whether a real state transition occurs in this critique window.",
            ),
            "corrected_at_s": types.Schema(
                type=types.Type.NUMBER,
                format="float",
                description="Corrected absolute match second for the transition.",
            ),
            "state_after": _position_state_schema(),
            "note": types.Schema(
                type=types.Type.STRING,
                description="Visible evidence for confirmation/rejection and timestamp correction.",
            ),
        },
    )


def build_technique_schema(player_ids: tuple[str, str]) -> types.Schema:
    actor_values = [*player_ids, *TECHNIQUE_ACTOR_SENTINELS]
    return types.Schema(
        type=types.Type.OBJECT,
        required=[
            "actor_player_id",
            "identity_uncertain",
            "action_class",
            "outcome",
            "specific_technique_guess",
            "confidence",
            "evidence",
        ],
        properties={
            "actor_player_id": types.Schema(type=types.Type.STRING, enum=actor_values),
            "identity_uncertain": types.Schema(type=types.Type.BOOLEAN),
            "action_class": types.Schema(
                type=types.Type.STRING, enum=list(ACTION_CLASS_VALUES)
            ),
            "outcome": types.Schema(type=types.Type.STRING, enum=list(OUTCOME_VALUES)),
            "specific_technique_guess": types.Schema(
                type=types.Type.STRING,
                description="Specific technique name when visibly supportable; empty if unclear.",
            ),
            "confidence": types.Schema(type=types.Type.NUMBER, format="float"),
            "evidence": types.Schema(
                type=types.Type.STRING,
                description="Visible biomechanical cue supporting action, actor, and outcome.",
            ),
        },
    )


def build_technique_validator_schema(player_ids: tuple[str, str]) -> types.Schema:
    base = build_technique_schema(player_ids)
    properties = dict(base.properties)
    properties.update(
        {
            "verdict": types.Schema(
                type=types.Type.STRING, enum=["agree", "correct", "ditch"]
            ),
            "intentional_technique_visible": types.Schema(
                type=types.Type.BOOLEAN,
                description=(
                    "True when any intentional BJJ technique attempt is visible, "
                    "even if its class differs from the proposal. False only for a real false positive."
                ),
            ),
            "reason": types.Schema(type=types.Type.STRING),
        }
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=[
            "verdict",
            "intentional_technique_visible",
            *list(base.required),
            "reason",
        ],
        properties=properties,
    )


class TimelineValidationError(ValueError):
    """Raised when a model-produced timeline cannot satisfy its hard contract."""


def validate_technique_validator_decision(
    payload: Mapping, *, context: str
) -> tuple[str, bool]:
    """Reject contradictory keep/ditch decisions before timeline output."""
    verdict = payload.get("verdict")
    if verdict not in ("agree", "correct", "ditch"):
        raise TimelineValidationError(
            f"{context}: validator returned invalid verdict {verdict!r}"
        )
    intentional_visible = payload.get("intentional_technique_visible")
    if not isinstance(intentional_visible, bool):
        raise TimelineValidationError(
            f"{context}: validator omitted intentional-technique visibility"
        )
    if verdict == "ditch" and intentional_visible:
        raise TimelineValidationError(
            f"{context}: validator returned contradictory ditch: "
            "an intentional technique was marked visible"
        )
    if verdict != "ditch" and not intentional_visible:
        raise TimelineValidationError(
            f"{context}: validator kept an event while marking no intentional "
            "technique visible"
        )
    return verdict, intentional_visible


def _read_reference_field(reference, field_name: str) -> str:
    if isinstance(reference, Mapping):
        value = reference.get(field_name)
    else:
        value = getattr(reference, field_name, None)
    return str(value or "").strip()


def validate_player_references(references: Sequence) -> tuple[str, str]:
    """Return the two grounded player ids or fail before any model call."""
    if len(references) != 2:
        raise TimelineValidationError(
            "position-timeline-techniques requires exactly two labelled player references"
        )
    player_ids = tuple(_read_reference_field(ref, "player_id") for ref in references)
    if not all(player_ids):
        raise TimelineValidationError("both player references require a non-empty player_id")
    if player_ids[0] == player_ids[1]:
        raise TimelineValidationError("player reference ids must be distinct")
    for ref in references:
        if not _read_reference_field(ref, "player_name"):
            raise TimelineValidationError("both player references require a non-empty player_name")
        if isinstance(ref, Mapping):
            loaded_image_bytes = ref.get("image_bytes")
            loaded_image = ref.get("image")
        else:
            loaded_image_bytes = getattr(ref, "image_bytes", None)
            loaded_image = getattr(ref, "image", None)
        # The QA route validates source exclusivity, decodes the image, and
        # deliberately hands executors a source-agnostic loaded reference.
        # Accept that internal representation instead of demanding the raw
        # URL/base64 transport field a second time.
        if isinstance(loaded_image_bytes, (bytes, bytearray)) or loaded_image is not None:
            continue
        external_source = _read_reference_field(ref, "image_s3_uri_or_url")
        inline_source = _read_reference_field(ref, "image_base64")
        if not external_source and not inline_source:
            raise TimelineValidationError(
                "both player references require an external image or inline crop"
            )
        if external_source and inline_source:
            raise TimelineValidationError(
                "each player reference must use exactly one image source"
            )
    return player_ids  # type: ignore[return-value]


def _validated_state(state: Mapping, *, context: str) -> dict:
    position = str(state.get("position") or "").strip()
    if position not in TIMELINE_POSITIONS:
        raise TimelineValidationError(
            f"{context}: position must be one of {list(TIMELINE_POSITIONS)}; got {position!r}"
        )

    role_a = str(state.get("player_a_role") or "").strip()
    role_b = str(state.get("player_b_role") or "").strip()
    for label, role in (("player_a_role", role_a), ("player_b_role", role_b)):
        if role not in PLAYER_ROLES:
            raise TimelineValidationError(
                f"{context}: {label} must be one of {list(PLAYER_ROLES)}; got {role!r}"
            )

    position_note = str(state.get("position_note") or "").strip()
    inactive_reason = str(state.get("inactive_reason") or "").strip()
    if position == "other" and not position_note:
        raise TimelineValidationError(
            f"{context}: position='other' requires a specific position_note describing the subposition"
        )
    if (
        position == "other"
        and _normalized_position_note(position_note) in _NON_GEOMETRIC_OTHER_NOTES
    ):
        raise TimelineValidationError(
            f"{context}: position='other' position_note must describe body geometry, "
            f"not only a technique/action or existing position alias; got {position_note!r}"
        )
    if position == "inactive":
        if not inactive_reason:
            raise TimelineValidationError(
                f"{context}: position='inactive' requires an inactive_reason"
            )
        if (role_a, role_b) != ("not_applicable", "not_applicable"):
            raise TimelineValidationError(
                f"{context}: inactive state requires both player roles to be 'not_applicable'"
            )
    elif "not_applicable" in (role_a, role_b):
        raise TimelineValidationError(
            f"{context}: active positions cannot use the 'not_applicable' player role"
        )

    confidence = state.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise TimelineValidationError(f"{context}: confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise TimelineValidationError(f"{context}: confidence must be between 0 and 1")

    return {
        "position": position,
        "player_a_role": role_a,
        "player_b_role": role_b,
        "position_note": position_note,
        "inactive_reason": inactive_reason,
        "confidence": confidence,
        "evidence": str(state.get("evidence") or "").strip(),
    }


def _state_identity(state: Mapping) -> tuple[str, str, str, str, str]:
    return (
        state["position"],
        state["player_a_role"],
        state["player_b_role"],
        state["position_note"],
        state["inactive_reason"],
    )


def _interval(
    start_s: float,
    end_s: float,
    state: Mapping,
    *,
    player_ids: tuple[str, str],
    boundary_provenance: Mapping | None,
) -> dict:
    start_s = round(float(start_s), 3)
    end_s = round(float(end_s), 3)
    return {
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": round(end_s - start_s, 3),
        "position": state["position"],
        "player_roles": {
            player_ids[0]: state["player_a_role"],
            player_ids[1]: state["player_b_role"],
        },
        "position_note": state["position_note"],
        "inactive_reason": state["inactive_reason"],
        "confidence": state["confidence"],
        "evidence": state["evidence"],
        "boundary_provenance": dict(boundary_provenance) if boundary_provenance else None,
    }


def build_position_intervals(
    scope_start_s: float,
    scope_end_s: float,
    initial_state: Mapping,
    transitions: Iterable[Mapping],
    *,
    player_ids: tuple[str, str],
) -> list[dict]:
    """Build exact-coverage intervals from one initial state and boundaries.

    Only transitions explicitly confirmed by the critique stage participate.
    Every participating boundary must carry a corrected absolute timestamp.
    """
    start = round(float(scope_start_s), 3)
    end = round(float(scope_end_s), 3)
    if start >= end:
        raise TimelineValidationError("requested scope must have positive duration")
    if len(player_ids) != 2 or not all(player_ids) or player_ids[0] == player_ids[1]:
        raise TimelineValidationError("player_ids must contain two distinct non-empty ids")

    current_state = _validated_state(initial_state, context="initial_state")
    current_start = start
    current_boundary = None
    intervals: list[dict] = []
    last_boundary = start

    for index, transition in enumerate(transitions):
        if transition.get("confirmed") is not True:
            continue
        raw_boundary = transition.get("corrected_at_s")
        if not isinstance(raw_boundary, (int, float)) or isinstance(raw_boundary, bool):
            raise TimelineValidationError(
                f"transition[{index}]: confirmed transition requires numeric corrected_at_s"
            )
        boundary = round(float(raw_boundary), 3)
        if not start < boundary < end:
            raise TimelineValidationError(
                f"transition[{index}]: corrected boundary must be strictly inside the requested scope"
            )
        if boundary <= last_boundary:
            raise TimelineValidationError(
                "confirmed corrected transition boundaries must be strictly increasing"
            )

        state_after = _validated_state(
            transition.get("state_after") or {},
            context=f"transition[{index}].state_after",
        )
        intervals.append(
            _interval(
                current_start,
                boundary,
                current_state,
                player_ids=player_ids,
                boundary_provenance=current_boundary,
            )
        )
        current_start = boundary
        current_state = state_after
        current_boundary = {
            "original_at_s": transition.get("at_s"),
            "corrected_at_s": boundary,
            "critique_note": str(transition.get("critique_note") or "").strip(),
        }
        last_boundary = boundary

    intervals.append(
        _interval(
            current_start,
            end,
            current_state,
            player_ids=player_ids,
            boundary_provenance=current_boundary,
        )
    )

    merged: list[dict] = []
    for interval in intervals:
        if (
            merged
            and merged[-1]["position"] != "inactive"
            and interval["position"] != "inactive"
            and _state_identity(
                {
                    "position": merged[-1]["position"],
                    "player_a_role": merged[-1]["player_roles"][player_ids[0]],
                    "player_b_role": merged[-1]["player_roles"][player_ids[1]],
                    "position_note": merged[-1]["position_note"],
                    "inactive_reason": merged[-1]["inactive_reason"],
                }
            )
            == _state_identity(
                {
                    "position": interval["position"],
                    "player_a_role": interval["player_roles"][player_ids[0]],
                    "player_b_role": interval["player_roles"][player_ids[1]],
                    "position_note": interval["position_note"],
                    "inactive_reason": interval["inactive_reason"],
                }
            )
        ):
            merged[-1]["end_s"] = interval["end_s"]
            merged[-1]["duration_s"] = round(
                merged[-1]["end_s"] - merged[-1]["start_s"], 3
            )
            continue
        merged.append(interval)

    return merged


def _position_interval_index_at(position_intervals: Sequence[Mapping], at_s: float) -> int:
    for interval_index, interval in enumerate(position_intervals):
        is_last = interval_index == len(position_intervals) - 1
        if interval["start_s"] <= at_s < interval["end_s"] or (
            is_last and at_s == interval["end_s"]
        ):
            return interval_index
    raise TimelineValidationError(f"timestamp {at_s} lies outside the position timeline")


def _position_transition_records(position_intervals: Sequence[Mapping]) -> list[dict]:
    records: list[dict] = []
    for before, after in zip(position_intervals, position_intervals[1:]):
        at_s = float(before["end_s"])
        if at_s != float(after["start_s"]):
            raise TimelineValidationError("position timeline contains a gap or overlap")
        records.append(
            {
                "at_s": at_s,
                "position_before": before["position"],
                "position_after": after["position"],
                "player_roles_before": dict(before["player_roles"]),
                "player_roles_after": dict(after["player_roles"]),
            }
        )
    return records


def build_position_context(
    position_intervals: Sequence[Mapping],
    start_s: float,
    end_s: float,
    *,
    transition_tolerance_s: float = 0.0,
) -> dict:
    """Return compact fallible position context for one technique candidate."""
    if not position_intervals:
        raise TimelineValidationError("position timeline must contain at least one interval")
    start = float(start_s)
    end = float(end_s)
    tolerance = float(transition_tolerance_s)
    if end <= start:
        raise TimelineValidationError("position context range must have positive duration")
    if tolerance < 0:
        raise TimelineValidationError("position context transition tolerance cannot be negative")
    scope_start = float(position_intervals[0]["start_s"])
    scope_end = float(position_intervals[-1]["end_s"])
    if start < scope_start or end > scope_end:
        raise TimelineValidationError("position context range must lie inside the timeline")

    before_index = _position_interval_index_at(position_intervals, start)
    after_index = _position_interval_index_at(position_intervals, end)
    before = position_intervals[before_index]
    after = position_intervals[after_index]
    overlapping_indexes = [
        index
        for index, interval in enumerate(position_intervals)
        if float(interval["start_s"]) < end and float(interval["end_s"]) > start
    ]
    transitions = [
        record
        for record in _position_transition_records(position_intervals)
        if start - tolerance <= record["at_s"] <= end + tolerance
    ]
    return {
        "position_before": before["position"],
        "position_after": after["position"],
        "player_roles_before": dict(before["player_roles"]),
        "player_roles_after": dict(after["player_roles"]),
        "position_note_before": str(before.get("position_note") or ""),
        "position_note_after": str(after.get("position_note") or ""),
        "confidence_before": before.get("confidence"),
        "confidence_after": after.get("confidence"),
        "evidence_before": str(before.get("evidence") or ""),
        "evidence_after": str(after.get("evidence") or ""),
        "overlapping_position_interval_indexes": overlapping_indexes,
        "position_transitions": transitions,
    }


def build_position_aware_candidates(
    scan_highlights: Sequence[Mapping],
    position_intervals: Sequence[Mapping],
    *,
    scope_start_s: float,
    scope_end_s: float,
    transition_pre_s: float,
    transition_post_s: float,
    merge_tolerance_s: float,
) -> list[dict]:
    """Union independent visual scan results with active position transitions.

    A transition can enrich one nearby scan target or create one secondary
    candidate. It never removes or relabels a scan-found movement.
    """
    scope_start = float(scope_start_s)
    scope_end = float(scope_end_s)
    pre = float(transition_pre_s)
    post = float(transition_post_s)
    tolerance = float(merge_tolerance_s)
    if scope_start >= scope_end:
        raise TimelineValidationError("requested scope must have positive duration")
    if pre < 0 or post < 0 or pre + post <= 0:
        raise TimelineValidationError("transition candidate window must have positive duration")
    if tolerance < 0:
        raise TimelineValidationError("merge tolerance cannot be negative")
    if not position_intervals:
        raise TimelineValidationError("position timeline must contain at least one interval")

    active_transitions = [
        record
        for record in _position_transition_records(position_intervals)
        if record["position_before"] != "inactive"
        and record["position_after"] != "inactive"
    ]
    prepared_scans: list[dict] = []
    for scan_index, raw in enumerate(scan_highlights):
        item = dict(raw)
        start = float(item["start_s"])
        end = float(item["end_s"])
        if end <= start or start < scope_start or end > scope_end:
            raise TimelineValidationError(
                f"scan_highlights[{scan_index}] must be a positive range inside the requested scope"
            )
        item["scan_index"] = item.get("index", scan_index)
        item["candidate_sources"] = ["technique_scan"]
        prepared_scans.append(item)

    # One-to-one nearest pairing. A scan event and a position transition may
    # enrich each other, but two distinct transitions must never collapse
    # into one candidate merely because the scan window is broad.
    eligible_pairs: list[tuple[float, int, int]] = []
    for transition_index, transition in enumerate(active_transitions):
        for scan_index, scan in enumerate(prepared_scans):
            if (
                float(scan["start_s"]) - tolerance
                <= transition["at_s"]
                <= float(scan["end_s"]) + tolerance
            ):
                midpoint = (
                    float(scan["start_s"]) + float(scan["end_s"])
                ) / 2.0
                eligible_pairs.append(
                    (abs(midpoint - transition["at_s"]), transition_index, scan_index)
                )
    matched_by_scan: dict[int, list[dict]] = defaultdict(list)
    matched_transition_indexes: set[int] = set()
    matched_scan_indexes: set[int] = set()
    for _, transition_index, scan_index in sorted(eligible_pairs):
        if (
            transition_index in matched_transition_indexes
            or scan_index in matched_scan_indexes
        ):
            continue
        matched_transition_indexes.add(transition_index)
        matched_scan_indexes.add(scan_index)
        matched_by_scan[scan_index].append(active_transitions[transition_index])
    unmatched_transitions = [
        transition
        for index, transition in enumerate(active_transitions)
        if index not in matched_transition_indexes
    ]

    candidates: list[dict] = []
    for scan_index, item in enumerate(prepared_scans):
        if matched_by_scan.get(scan_index):
            item["candidate_sources"].append("position_transition")
        item["position_context"] = build_position_context(
            position_intervals,
            float(item["start_s"]),
            float(item["end_s"]),
            transition_tolerance_s=tolerance,
        )
        candidates.append(item)

    for transition in unmatched_transitions:
        start = round(max(scope_start, transition["at_s"] - pre), 3)
        end = round(min(scope_end, transition["at_s"] + post), 3)
        if end <= start:
            continue
        before = transition["position_before"]
        after = transition["position_after"]
        candidates.append(
            {
                "start_s": start,
                "end_s": end,
                "description": (
                    f"The visible position state changes from {before} to {after} near "
                    f"{transition['at_s']:.3f}s. Treat this only as a candidate; do not infer "
                    "that an intentional technique occurred without matching body-movement evidence."
                ),
                "reasoning": (
                    "Deterministic secondary candidate from a critiqued active-to-active "
                    "position transition; the video must confirm or reject it."
                ),
                "candidate_sources": ["position_transition"],
                "position_context": build_position_context(
                    position_intervals, start, end
                ),
                "transition_at_s": transition["at_s"],
                "adjustment": "position_transition_candidate",
            }
        )

    for candidate in candidates:
        candidate["position_context"]["candidate_sources"] = list(
            candidate.get("candidate_sources") or []
        )
    candidates.sort(key=lambda item: (float(item["start_s"]), float(item["end_s"])))
    for index, candidate in enumerate(candidates):
        candidate["index"] = index
    return candidates


def attach_techniques_to_intervals(
    technique_events: Sequence[Mapping], position_intervals: Sequence[Mapping]
) -> list[dict]:
    """Attach each sparse event to the independently computed midpoint state."""
    if not position_intervals:
        raise TimelineValidationError("position timeline must contain at least one interval")
    scope_start = float(position_intervals[0]["start_s"])
    scope_end = float(position_intervals[-1]["end_s"])
    attached: list[dict] = []
    for event_index, event in enumerate(technique_events):
        try:
            event_start = float(event["start_s"])
            event_end = float(event["end_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise TimelineValidationError(
                f"technique_events[{event_index}] requires numeric start_s/end_s"
            ) from error
        if event_end <= event_start:
            raise TimelineValidationError(
                f"technique_events[{event_index}] must have positive duration"
            )
        if event_start < scope_start or event_end > scope_end:
            raise TimelineValidationError(
                f"technique_events[{event_index}] bounds must lie inside the requested scope"
            )
        midpoint = (event_start + event_end) / 2.0
        matched_index = None
        for interval_index, interval in enumerate(position_intervals):
            is_last = interval_index == len(position_intervals) - 1
            if interval["start_s"] <= midpoint < interval["end_s"] or (
                is_last and midpoint == interval["end_s"]
            ):
                matched_index = interval_index
                break
        if matched_index is None:
            raise TimelineValidationError(
                f"technique_events[{event_index}] midpoint lies outside the position timeline"
            )
        interval = position_intervals[matched_index]
        item = dict(event)
        item["position_interval_index"] = matched_index
        item["position"] = interval["position"]
        item["position_note"] = interval.get("position_note", "")
        context = build_position_context(position_intervals, event_start, event_end)
        item.update(context)
        attached.append(item)
    return attached


def build_objective_statistics(
    intervals: Sequence[Mapping],
    technique_events: Sequence[Mapping],
    *,
    player_ids: tuple[str, str],
) -> dict:
    """Aggregate factual durations and counts without a dominance judgment."""
    by_position: dict[str, dict[str, float]] = defaultdict(lambda: {"duration_s": 0.0})
    by_player: dict[str, dict] = {
        player_id: {
            "by_position": defaultdict(
                lambda: {"top_s": 0.0, "bottom_s": 0.0, "neutral_s": 0.0}
            ),
            "role_totals": {"top_s": 0.0, "bottom_s": 0.0, "neutral_s": 0.0},
        }
        for player_id in player_ids
    }
    active_duration = 0.0
    inactive_duration = 0.0

    for interval in intervals:
        duration = round(float(interval["duration_s"]), 3)
        position = str(interval["position"])
        if position == "inactive":
            inactive_duration += duration
            continue
        active_duration += duration
        by_position[position]["duration_s"] += duration
        roles = interval["player_roles"]
        for player_id in player_ids:
            role = roles[player_id]
            if role not in ("top", "bottom", "neutral"):
                raise TimelineValidationError(
                    f"active interval uses non-statistical role {role!r} for {player_id!r}"
                )
            key = f"{role}_s"
            by_player[player_id]["by_position"][position][key] += duration
            by_player[player_id]["role_totals"][key] += duration

    technique_counts: dict[str, int] = defaultdict(
        int, {player_id: 0 for player_id in player_ids}
    )
    for event in technique_events:
        actor = str(event.get("actor_player_id") or "unclear")
        technique_counts[actor] += 1

    def _round_durations(value):
        if isinstance(value, dict):
            return {key: _round_durations(child) for key, child in value.items()}
        if isinstance(value, float):
            return round(value, 3)
        return value

    return _round_durations(
        {
            "active_duration_s": active_duration,
            "inactive_duration_s": inactive_duration,
            "by_position": dict(by_position),
            "by_player": {
                player_id: {
                    "by_position": dict(data["by_position"]),
                    "role_totals": data["role_totals"],
                }
                for player_id, data in by_player.items()
            },
            "techniques_by_player": dict(technique_counts),
        }
    )


__all__ = [
    "ACTIVE_POSITIONS",
    "PLAYER_ROLES",
    "TECHNIQUE_ACTOR_SENTINELS",
    "TIMELINE_POSITIONS",
    "POSITION_CRITIQUE_INITIAL_PROMPT",
    "POSITION_CRITIQUE_SYSTEM_PROMPT",
    "POSITION_SCAN_INITIAL_PROMPT",
    "POSITION_SCAN_SYSTEM_PROMPT",
    "TECHNIQUE_CRITIQUE_POSITION_INITIAL_PROMPT",
    "TimelineValidationError",
    "attach_techniques_to_intervals",
    "build_position_aware_candidates",
    "build_position_context",
    "build_position_critique_prompt",
    "build_objective_statistics",
    "build_position_critique_schema",
    "build_position_intervals",
    "build_position_scan_prompt",
    "build_position_scan_schema",
    "build_reference_parts",
    "build_technique_schema",
    "build_technique_prompt",
    "build_technique_validator_prompt",
    "build_technique_validator_schema",
    "validate_technique_validator_decision",
    "validate_player_references",
]
