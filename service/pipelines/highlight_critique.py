"""QA-layer ``highlight_critique`` (v2 build plan,
``2026-07-18-highlight-scan-critique-analyze-v2-plan.md``) — the NEW middle
stage of ``highlight-scan-critique-analyze``:
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
extends BACKWARD past the scanned start (``critique_backpad_s``, tunable) so
Gemini has a chance to find that earlier true beginning, and reports a
corrected start/end ADDITIVELY (``executors.highlight_critique_node`` never
overwrites the scan's own ``start_s``/``end_s`` — provenance is kept).

**Prompt text is Gracie's final draft** (dropped in verbatim from
``v2_node_prompts.md``, 2026-07-18), not a placeholder.

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
from typing import Optional

from google.genai import types

DEFAULT_CRITIQUE_SYSTEM_PROMPT: str = """
You are correcting timestamp drift on a BJJ highlight clip. A prior rough scan reported a span of live grappling action, but that scanner reliably LATCHES ONTO THE MOST VISUALLY SALIENT MOMENT — the tap, the sweep landing, the pin settling — and reports that moment's neighborhood as the "start." The TRUE start of the action is almost always 2-6 seconds EARLIER, at the quiet SETUP CUE that a viewer's eye skips past: a level change, a grip break, an underhook plus a hip-shift, a new control-grip being established, a stance narrowing before a shot. Your job is to find that earlier moment, not to confirm the scanner's guess.

You will receive: (1) a video clip that extends BACKWARD in time beyond the scanner's reported start (so the true setup cue should be visible somewhere inside it), and (2) the scanner's plain-language description of the body movement it saw (no technique names, no position names — just what moved).

Do TWO things, in this order:

1. CONFIRM THE CLIP MATCHES THE DESCRIPTION. Does this clip actually show the described body movement at all? If the description clearly does not match anything visible in this clip, say so plainly — do not force a fit.

2. SEARCH BACKWARD FOR THE TRUE START. Starting from the scanner's reported start and working backward through the clip, find the earliest moment where you can see the deliberate setup for the described movement beginning — NOT the moment it resolves. Concretely:
   - Look for: a level change (dropping levels for a shot), a grip being broken or changed, an underhook or overhook being secured, a hip-shift that commits weight, a control grip (collar, sleeve, pant, wrist) being newly established, a stance narrowing.
   - Do NOT anchor on: the tap, the finish, the landing of a sweep/takedown, the moment a pass completes, or any other resolution — those are LATE, not the start.
   - The true END of the described movement is when the resulting position visibly settles — roughly 2 seconds of stillness — or a clear stoppage (tap, ref call, stand-up). Do not extend past that.

If the true setup cue appears to begin AT OR BEFORE the very start of the video you were given (i.e. you cannot see it begin at all — the clip itself starts too late), say so explicitly in your note. That is a real, useful signal, not a failure on your part.

Report both corrected timestamps in ABSOLUTE match seconds (the same clock given to you below), never relative to this clip's own 0:00.
""".strip()

_INITIAL_TEMPLATE = Template("""
This clip runs from $window_start_sec s to $window_end_sec s (ABSOLUTE match seconds) — it was extended $critique_backpad_s s BACKWARD from the scanner's originally reported start of $scan_start_sec s, so you can search earlier than the scanner did. The scanner's original reported span was $scan_start_sec s to $scan_end_sec s.

SCANNER'S DESCRIPTION OF THE BODY MOVEMENT (no technique/position names — just what moved): "$description"

1. Does this clip actually contain the described movement? If not, explain the mismatch plainly.
2. Find the TRUE start — the setup cue — by searching BACKWARD from $scan_start_sec s. Report it in ABSOLUTE match seconds. If the setup cue is not fully visible even at the earliest point of this clip ($window_start_sec s), set corrected_start_s = $window_start_sec and say so in your note (this tells us the backward padding needs to be larger).
3. Find the TRUE end — roughly 2s of stillness after the position settles, or a clear stoppage. Report it in ABSOLUTE match seconds.

Return ONLY valid JSON matching this shape — no prose:
{
    "movement_confirmed": true,
    "corrected_start_s": 0.0,
    "corrected_end_s": 0.0,
    "note": "string — one or two sentences: what you found, and flag any mismatch or insufficient backward padding"
}
""".strip())


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
                description="Corrected end of the action, ABSOLUTE match seconds (same clock as the requested scope).",
            ),
            "note": types.Schema(
                type=types.Type.STRING,
                description="One or two sentences: what was found, and any mismatch/insufficient-backward-padding flag.",
            ),
        },
    )


def build_critique_prompt(
    *, window_start_sec: float, window_end_sec: float, critique_backpad_s: float,
    scan_start_sec: float, scan_end_sec: float, description: Optional[str],
) -> str:
    """Per-highlight "user turn" prompt — Gracie's final draft,
    ``$``-substituted (``string.Template.safe_substitute``, same idiom as
    ``highlight_scan.build_scan_prompt``) with the EXACT token names her text
    uses: ``$window_start_sec``, ``$window_end_sec``, ``$critique_backpad_s``,
    ``$scan_start_sec``, ``$scan_end_sec``, ``$description``. All six MUST be
    supplied by the caller (``executors.highlight_critique_node``) — a
    missing/misnamed token is silently left as a literal ``$token`` by
    ``safe_substitute`` (never a raised error), which is exactly the class of
    bug ``tests/test_pipelines_highlight_v2.py``'s
    ``test_*_prompt_leaves_no_leftover_tokens`` regression guards against.
    """
    return _INITIAL_TEMPLATE.safe_substitute(
        window_start_sec=window_start_sec, window_end_sec=window_end_sec,
        critique_backpad_s=critique_backpad_s,
        scan_start_sec=scan_start_sec, scan_end_sec=scan_end_sec,
        description=description or "unknown — no description was provided by the scan pass",
    )


__all__ = [
    "DEFAULT_CRITIQUE_SYSTEM_PROMPT",
    "build_critique_schema",
    "build_critique_prompt",
]
