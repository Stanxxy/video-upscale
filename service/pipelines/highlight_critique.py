"""QA-layer ``highlight_critique`` (v2 build plan,
``2026-07-18-highlight-scan-critique-analyze-v2-plan.md``) — the middle
stage of ``highlight-scan-critique-analyze`` and the shared technique
critique stage in ``position-timeline-techniques``:
``video_window -> highlight_scan -> highlight_critique -> highlight_analyze``.

Pure helpers only (schema + prompt content) — no Gemini client here, same
discipline as ``highlight_scan.py``/``chunk_segment.py``.
``service/pipelines/executors.py``'s ``highlight_critique_node`` is the only
caller that actually talks to Gemini (via the shared
``simplified_tags.SimplifiedTagsTimeAnalyzer.analyze_chunk``, reusing its
offset-quantization + retry + instrumentation plumbing — INS-107 single-
formatter discipline, never reimplemented here).

**Purpose (Gracie's VTG-drift finding):** Gemini's rough PASS-1 scan tends to
anchor a highlight's reported ``start_s`` to the most SALIENT moment (the
resolution/tap/landing), which is typically 2-6s LATER than the true quiet
setup cue (a grip fight, a level change, a stance shift) that actually begins
the exchange. This node re-examines each scanned highlight with a window that
extends BACKWARD past the scanned start (``critique_backpad_s``, tunable) and
may extend FORWARD past the scanned end (``critique_forwardpad_s``). The
latter lets an outcome-aware pipeline keep enough evidence to judge whether
the action succeeded, failed, or remained unclear. It reports a corrected
start/end ADDITIVELY (``executors.highlight_critique_node`` never overwrites
the scan's own ``start_s``/``end_s`` — provenance is kept).

**Prompt text is Gracie's final draft** (dropped in verbatim from
``v2_node_prompts.md``, 2026-07-18), not a placeholder. Both prompt layers
are public saved defaults and may be replaced through the QA node config.

**ABSOLUTE-seconds contract (load-bearing, differs from every other
``analyze_chunk`` caller in this package):** every other native-video call in
this codebase (``build_video_prompt``/``chunk_analyze_node``/the position/
technique/validator calls in ``highlight_axes.py``) tells Gemini the clip's
own internal clock starts at 0:00 and asks for CHUNK-RELATIVE seconds, which
the Python caller then converts to absolute match seconds
(``executors._convert_chunk_relative_to_absolute``). The critique prompt
below does the OPPOSITE on purpose: it tells Gemini the window's own
ABSOLUTE match-second bounds (``$window_start_sec``/``$window_end_sec``) and
the scan's original ABSOLUTE bounds (``$scan_start_sec``/``$scan_end_sec``),
and explicitly instructs it to report ``corrected_start_s``/
``corrected_end_s`` on THAT SAME absolute clock, "never relative to this
clip's own 0:00". ``executors.highlight_critique_node`` must NOT run these
values through ``_convert_chunk_relative_to_absolute`` — they are already
absolute match seconds directly off the wire, only sanity-clamped.
"""
from __future__ import annotations

from string import Template
from typing import Mapping, Optional

from google.genai import types

DEFAULT_CRITIQUE_SYSTEM_PROMPT: str = """
You are correcting timestamp drift across BOTH temporal boundaries of one BJJ technique highlight. A prior rough scan may latch onto the most visually salient moment, causing a LATE start, and it may stop when the attacking motion loses salience, causing an EARLY end before the result is visible. Your output defines the clip a user will watch, so it must contain the full attempt and enough visible outcome evidence to judge its result.

You will receive: (1) a video clip that extends BACKWARD beyond the scanner's reported start and may extend FORWARD beyond its reported end, and (2) the scanner's plain-language description of the target body movement. The description identifies the event; neighboring actions in the context do not replace it.

Treat the start and end as independent boundaries. Never preserve the scan's duration, and never treat scan_end_s as a preferred answer. Do THREE things:

1. CONFIRM THE CLIP MATCHES THE DESCRIPTION. Does this clip actually show the described body movement at all? If the description clearly does not match anything visible in this clip, say so plainly — do not force a fit.

2. FIND THE TRUE START. Search BACKWARD from the scanner's reported start for the earliest deliberate setup cue causally connected to the described movement: a level change, grip break/change, underhook or overhook being secured, hip shift that commits weight, new control grip, or stance change. Do not anchor on the tap, landing, pass completion, or other resolution. Do not include generic grip fighting that is not yet part of this attempt.

3. FIND THE TRUE END. Independently search FORWARD through the entire provided window, including frames after the scanner's reported end. The boundary must be outcome-complete, not motion-complete:
   - Successful: include the decisive finish plus the visible tap/stoppage, completed reversal/escape, or resulting control becoming established and no longer immediately disputed.
   - Unsuccessful: include the defender fully neutralizing or escaping the threat and the attacker clearly abandoning, switching away, or being forced to reset. The defender's first reaction or the attacker's first pause is not yet the outcome.
   - Unclear: include every relevant available frame of the same attempt. If the window ends while the attempt or its defense is still developing, set corrected_end_s exactly to window_end_s and begin the note with 'OUTCOME_NOT_VISIBLE_BY_WINDOW_END' rather than pretending the result is visible.

A viewer watching only corrected_start_s through corrected_end_s must be able to judge the event as successful, unsuccessful, or unclear from the video. The immediate defense belongs to the target attempt's outcome even when a distinct counter-technique deserves its own overlapping highlight. Do not append a fixed tail, and do not extend into an unrelated later exchange after the target result is already clear.

If the true setup cue appears to begin AT OR BEFORE the very start of the video you were given (i.e. you cannot see it begin at all — the clip itself starts too late), say so explicitly in your note. That is a real, useful signal, not a failure on your part.

Report both corrected timestamps in ABSOLUTE match seconds (the same clock given to you below), never relative to this clip's own 0:00.
""".strip()

DEFAULT_CRITIQUE_INITIAL_PROMPT: str = """
This clip runs from $window_start_sec s to $window_end_sec s (ABSOLUTE match seconds) — it was extended $critique_backpad_s s BACKWARD from the scanner's originally reported start of $scan_start_sec s and $critique_forwardpad_s s FORWARD from its reported end of $scan_end_sec s. The scanner's original reported span was $scan_start_sec s to $scan_end_sec s.

SCANNER'S DESCRIPTION OF THE BODY MOVEMENT (no technique/position names — just what moved): "$description"

1. Does this clip actually contain the described movement? If not, explain the mismatch plainly.
2. Find the TRUE start — the setup cue — by searching BACKWARD from $scan_start_sec s. Report it in ABSOLUTE match seconds. If the setup cue is not fully visible even at the earliest point of this clip ($window_start_sec s), set corrected_start_s = $window_start_sec and say so in your note (this tells us the backward padding needs to be larger).
3. Independently search FORWARD through the entire clip for the TRUE end. Apply the outcome-complete boundary contract: finish/control/stoppage visibly settles for successful; defense/escape/reset/abandonment completes for unsuccessful; or all relevant available evidence is included for unclear. A viewer watching only the corrected range must be able to judge successful, unsuccessful, or unclear. Do not stop at the last attacking motion, the first defensive reaction, or the scanner's end. If the same attempt is still unresolved at $window_end_sec s, set corrected_end_s = $window_end_sec and begin note with 'OUTCOME_NOT_VISIBLE_BY_WINDOW_END'.

Return ONLY valid JSON matching this shape — no prose:
{
    "movement_confirmed": true,
    "corrected_start_s": 0.0,
    "corrected_end_s": 0.0,
    "note": "string — cite the visible start cue and outcome/end cue; flag mismatch, insufficient padding, or OUTCOME_NOT_VISIBLE_BY_WINDOW_END"
}
""".strip()

_INITIAL_TEMPLATE = Template(DEFAULT_CRITIQUE_INITIAL_PROMPT)


def build_critique_schema() -> types.Schema:
    """Gemini structured-output schema for the critique call:
    ``movement_confirmed`` (bool) + corrected bounds (ABSOLUTE match
    seconds — see module docstring's "ABSOLUTE-seconds contract") + a
    required free-text ``note``."""
    return types.Schema(
        type=types.Type.OBJECT,
        required=["movement_confirmed", "corrected_start_s", "corrected_end_s", "note"],
        properties={
            "movement_confirmed": types.Schema(
                type=types.Type.BOOLEAN,
                description="Whether the provided description genuinely matches this video window.",
            ),
            "corrected_start_s": types.Schema(
                type=types.Type.NUMBER, format="float",
                description="TRUE start of the action, ABSOLUTE match seconds (same clock as the requested scope).",
            ),
            "corrected_end_s": types.Schema(
                type=types.Type.NUMBER, format="float",
                description="Outcome-complete end of the action, ABSOLUTE match seconds (same clock as the requested scope).",
            ),
            "note": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Visible start cue and outcome/end cue, plus any mismatch or insufficient-padding flag. "
                    "Begin with OUTCOME_NOT_VISIBLE_BY_WINDOW_END when the result continues past the window."
                ),
            ),
        },
    )


def build_critique_prompt(
    *, window_start_sec: float, window_end_sec: float, critique_backpad_s: float,
    scan_start_sec: float, scan_end_sec: float, description: Optional[str],
    critique_forwardpad_s: float = 0.0,
    override: Optional[str] = None,
    position_context: Optional[Mapping] = None,
) -> str:
    """Per-highlight "user turn" prompt — Gracie's final draft,
    ``$``-substituted (``string.Template.safe_substitute``, same idiom as
    ``highlight_scan.build_scan_prompt``) with the EXACT token names her text
    uses: ``$window_start_sec``, ``$window_end_sec``, ``$critique_backpad_s``,
    ``$critique_forwardpad_s``, ``$scan_start_sec``, ``$scan_end_sec``,
    ``$description``. All seven MUST be
    supplied by the caller (``executors.highlight_critique_node``) — a
    missing/misnamed token is silently left as a literal ``$token`` by
    ``safe_substitute`` (never a raised error), which is exactly the class of
    bug ``tests/test_pipelines_highlight_v2.py``'s
    ``test_*_prompt_leaves_no_leftover_tokens`` regression guards against.
    """
    import json

    template = Template(override.strip()) if override and override.strip() else _INITIAL_TEMPLATE
    return template.safe_substitute(
        window_start_sec=window_start_sec, window_end_sec=window_end_sec,
        critique_backpad_s=critique_backpad_s,
        critique_forwardpad_s=critique_forwardpad_s,
        scan_start_sec=scan_start_sec, scan_end_sec=scan_end_sec,
        description=description or "unknown — no description was provided by the scan pass",
        position_context_json=json.dumps(dict(position_context or {}), sort_keys=True),
    )


__all__ = [
    "DEFAULT_CRITIQUE_INITIAL_PROMPT",
    "DEFAULT_CRITIQUE_SYSTEM_PROMPT",
    "build_critique_schema",
    "build_critique_prompt",
]
