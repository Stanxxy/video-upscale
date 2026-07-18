"""QA-layer ``simplified-tags-v1`` analyze path — a NEW experimental 4-axis
Gemini tagging format (position / actor / action_class / outcome +
``specific_technique_guess``), kept entirely separate from ``analyzer.py``'s
production ``clips-v1`` prompt/schema.

Provenance: ``working_log/BJJ_TAXONOMY_SIMPLIFIED_GEMINI_TAGGING.md`` (proposal).
Bundled default ``system_instruction``: ``bjj_taxonomy_simplified.md`` (repo root,
sibling to production's ``bjj_analysis_taxonomy.md``).

This module builds its OWN Gemini call — reusing the shared ``genai`` client
(``RunContext.gemini_client()``) and ``thinking_config_for`` from
``service/pipelines/executors.py`` / ``service/routes/qa_vlm.py`` — rather than
routing through ``analyzer.BJJTechniqueAnalyzer``/``BJJMultiAgentAnalyzer``,
whose prompt is hardcoded to the clips-v1 taxonomy and is production surface,
never forked/branched for an experimental QA format. QA experiment only; does
NOT touch ``analyzer.py``, ``taxonomy_mapper.py``, ``VideoEvent``, or the worker.
"""
from __future__ import annotations

import json
import re
import logging
import os
from typing import Optional

from google.genai import types

from service.pipelines import gemini_retry

logger = logging.getLogger("service.pipelines.simplified_tags")

# --------------------------------------------------------------------------- #
# Frozen enum value sets (build plan handoff / task contract). Keep these in
# lockstep with SIMPLIFIED_TAXONOMY_PATH's per-axis tables — a mismatch would
# mean the bundled prompt text and the Gemini-enforced enum disagree.
# --------------------------------------------------------------------------- #
POSITION_VALUES: list[str] = [
    "standing", "closed_guard", "open_guard", "half_guard", "side_control",
    "mount", "back_control", "turtle", "scramble",
]
ACTION_CLASS_VALUES: list[str] = [
    "takedown_attempt", "guard_pull", "sweep", "guard_pass", "back_take",
    "submission_choke", "submission_arm_lock", "submission_leg_lock",
    "escape", "transition",
]
OUTCOME_VALUES: list[str] = ["successful", "unsuccessful", "unclear"]

# ``simplified-tags-time-v1`` (chunk-segment-tags PASS 2) ACTOR axis — Gracie
# domain veto: this native-video call sends NO player reference images, so
# Actor MUST collapse to a positional/contested enum, NEVER a name/player_id
# (this INVERTS the frame-based ``build_prompt``'s "never a bare top/bottom
# token" instruction above — that instruction assumed reference-image-backed
# identity resolution, which this native path does not have).
TIME_ACTOR_VALUES: list[str] = ["top", "bottom", "contested"]

SIMPLIFIED_TAXONOMY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bjj_taxonomy_simplified.md"
)


def load_default_taxonomy_text() -> str:
    """Read the bundled simplified-taxonomy asset.

    Mirrors the exact defensive-read idiom already used in ``analyzer.py``
    (``BJJTechniqueAnalyzer._build_contents_and_prompt`` / the Judge's taxonomy
    read in ``BJJMultiAgentAnalyzer.analyze_sequence_async``): a missing file is
    a real, logged condition — not a fabricated taxonomy — surfaced via a short
    marker string so a broken deploy is visibly wrong rather than silently
    running with an empty system instruction.
    """
    try:
        with open(SIMPLIFIED_TAXONOMY_PATH, "r") as f:
            text = f.read()
    except Exception as e:  # noqa: BLE001 — log the real cause, still return a marker string
        logger.warning("simplified_tags: could not read %s: %s", SIMPLIFIED_TAXONOMY_PATH, e)
        return "Simplified taxonomy file not found."
    # Strip leading HTML provenance comment(s) — that metadata is for humans
    # reading the asset, not instruction text for Gemini. Only leading comments
    # are removed; anything after real content is left intact.
    return re.sub(r"\A(?:\s*<!--.*?-->\s*)+", "", text, flags=re.DOTALL).lstrip()


# Computed once at import time (module constant), same pattern as other
# module-level defaults in this package (e.g. qa_vlm.py's model allowlists).
DEFAULT_TAXONOMY_TEXT: str = load_default_taxonomy_text()


def extract_usage_metadata(response) -> Optional[dict]:
    """Plain-dict token accounting from a ``genai`` response's
    ``usage_metadata`` (ADCC eval dashboard, Task 1 — LeCun's real-cost
    preference over placeholder rates). Returns ``None`` if the response
    carries no ``usage_metadata`` at all (e.g. a bare ``SimpleNamespace(text=...)``
    test double); individual fields are ``None`` if the SDK object itself
    didn't populate them. Never fabricates a token count."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    return {
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
    }


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` / ``` ... ``` fences from a Gemini response
    (shared by both ``SimplifiedTagsAnalyzer`` and ``SimplifiedTagsTimeAnalyzer``
    — identical idiom, factored out once rather than duplicated a third time)."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()


def _format_offset_seconds(seconds: float) -> str:
    """Format a seconds value as a protobuf-``Duration``-safe
    ``video_metadata`` offset string (``start_offset``/``end_offset``).

    **Live-bug fix (2026-07-18):** ``Duration`` accepts AT MOST 9 fractional
    digits (nanoseconds). ``highlight_analyze_node``/``chunk_analyze_node``
    compute these offsets via float subtraction/addition (e.g. pre-roll:
    ``33.3 - 5.0 == 28.299999999999997``, 15 fractional digits;
    ``2/3 == 0.6666666666666666``, 16 digits) — a raw ``f"{seconds}s"`` on
    either overflows the 9-digit limit and Gemini rejects the whole request
    with ``400 INVALID_ARGUMENT ... Field 'start_offset' ... failed to parse
    nano seconds``. Quantizing to millisecond precision (3 fractional digits)
    is far finer than a single video frame (~66ms at 15fps, the fastest fps
    this pipeline uses) — zero analysis-accuracy cost — and guarantees the
    emitted string is always a valid Duration regardless of upstream float
    imprecision. This is the ONE place both ``analyze_chunk`` callers
    (``chunk_analyze_node`` and ``highlight_analyze_node``) route their
    ``start_offset``/``end_offset`` through, so fixing it here fixes both.
    """
    quantized = round(float(seconds), 3)
    if quantized == 0:
        quantized = 0.0  # normalize -0.0 -> 0.0, never emit "-0.000s"
    return f"{quantized:.3f}s"


def resolve_system_instruction(override: Optional[str]) -> str:
    """``None``/whitespace-only -> the bundled default taxonomy text (same
    "empty means default" discipline as ``qa_vlm._clean_system_instruction`` +
    ``analyzer.py``'s ``taxonomy_text=None`` convention). A non-empty override
    REPLACES the bundled text entirely."""
    cleaned = override.strip() if override else None
    return cleaned or DEFAULT_TAXONOMY_TEXT


def build_response_schema() -> types.Schema:
    """Gemini structured-output schema for ``simplified-tags-v1``.

    ``position``/``action_class``/``outcome`` are STRING enums (the proven
    string-enum pattern from ``analyzer._build_response_schema``) constrained
    to the frozen value sets above. ``actor`` and ``specific_technique_guess``
    are free-text/ungated per the contract; ``specific_technique_guess`` is
    optional (absent from ``required``).
    """
    tag_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["start_frame", "end_frame", "position", "actor", "action_class", "outcome", "confidence"],
        properties={
            "start_frame": types.Schema(type=types.Type.INTEGER),
            "end_frame": types.Schema(type=types.Type.INTEGER),
            "position": types.Schema(
                type=types.Type.STRING,
                enum=list(POSITION_VALUES),
                description="Axis 1 (body configuration) — exactly one enum value.",
            ),
            "actor": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Axis 2 (free text, NOT an enum) — the athlete in the dominant/initiating "
                    "role (by name/descriptor/player_id), or 'contested'/'unclear' for a "
                    "neutral/indeterminate moment."
                ),
            ),
            "action_class": types.Schema(
                type=types.Type.STRING,
                enum=list(ACTION_CLASS_VALUES),
                description="Axis 3 (action/technique class) — exactly one enum value.",
            ),
            "outcome": types.Schema(
                type=types.Type.STRING,
                enum=list(OUTCOME_VALUES),
                description="Axis 4 (outcome) — exactly one enum value.",
            ),
            "specific_technique_guess": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Optional free-text specific-technique guess (e.g. 'possibly a kimura'). "
                    "Ungated — never defines/gates the enum axes above."
                ),
            ),
            "confidence": types.Schema(type=types.Type.NUMBER, format="float", description="Confidence 0.0-1.0."),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["clips"],
        properties={
            "current_context_summary": types.Schema(type=types.Type.STRING),
            "clips": types.Schema(
                type=types.Type.ARRAY,
                description="One entry per detected grappling tag in this window.",
                items=tag_schema,
            ),
        },
    )


def build_prompt(frame_indices: list[int], previous_context: Optional[str]) -> str:
    """Per-window prompt text. Frame numbers are ABSOLUTE source frame indices
    (not re-indexed into this window's frame list) — same discipline as
    ``analyzer.py``'s ``_build_contents_and_prompt``."""
    return f"""
    Here is a sequence of {len(frame_indices)} frames from a BJJ match.
    The frames correspond to the following ABSOLUTE frame numbers: {frame_indices}.

    PREVIOUS CONTEXT: "{previous_context if previous_context else "Start of the match."}"

    Tag every distinct grappling exchange visible in this window using EXACTLY the
    4-axis simplified taxonomy given in the system instruction. For each tag:
    1. `start_frame`/`end_frame`: use the ABSOLUTE frame numbers listed above, never
       relative indices into this window's frame list.
    2. `position`: EXACTLY one Position enum value (Axis 1).
    3. `actor`: free text — identify WHO is in the dominant/initiating role by
       name/descriptor/player_id (Axis 2), or 'contested'/'unclear' for a neutral
       moment. Never a bare 'top'/'bottom' token.
    4. `action_class`: EXACTLY one Action Class enum value (Axis 3).
    5. `outcome`: EXACTLY one Outcome enum value (Axis 4).
    6. `specific_technique_guess`: OPTIONAL free-text specific-name guess (e.g.
       "possibly a kimura") — never gates or replaces the enum fields above.
    7. `confidence`: 0.0-1.0.
    8. Use PREVIOUS CONTEXT to maintain continuity across windows and update
       `current_context_summary` for the next window.

    Return ONLY valid JSON matching this shape — no prose:
    {{
        "current_context_summary": "string",
        "clips": [
            {{
                "start_frame": int,
                "end_frame": int,
                "position": "<Axis 1 enum value>",
                "actor": "string",
                "action_class": "<Axis 3 enum value>",
                "outcome": "<Axis 4 enum value>",
                "specific_technique_guess": "string (optional)",
                "confidence": 0.0
            }}
        ]
    }}
    """


class SimplifiedTagsAnalyzer:
    """Direct-Gemini analyze path for ``simplified-tags-v1``.

    Deliberately mirrors ``BJJTechniqueAnalyzer.analyze_sequence_async``'s call
    signature (``frames, frame_indices, previous_context, *, response_schema``)
    so ``executors._analyze_frame_windows``'s shared per-window loop can drive
    either analyzer uniformly — the only difference is which class gets
    constructed for a given ``AnalyzeConfig.return_format``. Does not read
    ``taxonomy_path``/``bjj_analysis_taxonomy.md`` at all.
    """

    def __init__(
        self,
        client,
        *,
        model_id: str,
        temperature: float = 0.2,
        thinking_config: Optional[types.ThinkingConfig] = None,
        system_instruction: Optional[str] = None,
        retry_config: Optional[gemini_retry.GeminiRetryConfig] = None,
    ):
        self.client = client
        self.model_id = model_id
        self.temperature = temperature
        self.thinking_config = thinking_config
        self.system_instruction = resolve_system_instruction(system_instruction)
        self.retry_config = retry_config or gemini_retry.GeminiRetryConfig()

    def _build_generate_config(self, response_schema: types.Schema) -> types.GenerateContentConfig:
        kwargs = dict(
            system_instruction=self.system_instruction,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        if self.thinking_config is not None:
            kwargs["thinking_config"] = self.thinking_config
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _parse_response_text(text: str) -> str:
        """Strip markdown fences from Gemini response (same idiom as
        ``BJJTechniqueAnalyzer._parse_response_text``)."""
        return _strip_markdown_fences(text)

    async def analyze_sequence_async(
        self, frames, frame_indices, previous_context=None, *, response_schema=None,
    ):
        if not frames:
            return "No frames."

        prompt = build_prompt(frame_indices, previous_context)
        effective_schema = response_schema if response_schema is not None else build_response_schema()
        contents = [prompt, *frames]

        try:
            logger.info(
                "Gemini simplified-tags-v1: sending %d frames to model %s",
                len(frames), self.model_id,
            )
            response = await gemini_retry.call_with_retry(
                lambda: self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=self._build_generate_config(effective_schema),
                ),
                op_name="simplified-tags-v1",
                retry_config=self.retry_config,
            )
            text = response.text
            logger.info(
                "Gemini simplified-tags-v1: received response (%d chars)",
                len(text) if text else 0,
            )
            return self._parse_response_text(text or "")
        except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error, no fabrication
            logger.error("Gemini simplified-tags-v1 call failed: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})


# =============================================================================== #
# simplified-tags-time-v1 — NATIVE-VIDEO sibling for chunk-segment-tags PASS 2.
#
# Distinct from the frame-keyed format above: ``start_s``/``end_s`` (seconds,
# NOT frame numbers), and the ACTOR axis is INVERTED (schema-enum-constrained
# to top/bottom/contested — Gracie domain veto: this call sends no player
# reference images, so it structurally cannot know names/player_ids).
# =============================================================================== #
def build_video_response_schema() -> types.Schema:
    """Gemini structured-output schema for ``simplified-tags-time-v1``.

    ``start_s``/``end_s`` are returned CHUNK-RELATIVE (seconds from 0:00 of
    whatever native-video segment was actually sent — the caller,
    ``executors.chunk_analyze_node``, adds the real sent ``start_offset`` to
    produce absolute match seconds; this module never sees or fabricates that
    offset). ``actor`` is a hard STRING enum (top/bottom/contested) — the
    schema-level anti-fabrication guard is intentionally stronger here than
    the frame-based format's prompt-only "never a bare top/bottom token"
    instruction, because this path has no reference images to ground an
    identity guess in at all. ``actor_gi_color`` is an OPTIONAL free-text
    color-only descriptor (e.g. "white", "blue") — ungated, never combined
    into a fabricated identity string.
    """
    tag_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["start_s", "end_s", "position", "actor", "action_class", "outcome", "confidence"],
        properties={
            "start_s": types.Schema(
                type=types.Type.NUMBER, format="float",
                description="Chunk-RELATIVE seconds (0.0 = start of THIS clip, not the match).",
            ),
            "end_s": types.Schema(
                type=types.Type.NUMBER, format="float",
                description="Chunk-RELATIVE seconds (0.0 = start of THIS clip, not the match).",
            ),
            "position": types.Schema(
                type=types.Type.STRING,
                enum=list(POSITION_VALUES),
                description="Axis 1 (body configuration) — exactly one enum value.",
            ),
            "actor": types.Schema(
                type=types.Type.STRING,
                enum=list(TIME_ACTOR_VALUES),
                description=(
                    "Axis 2 — EXACTLY one of top/bottom/contested. This call has NO "
                    "player reference images: NEVER a name or player_id, never inferred "
                    "identity — only relative position in the current exchange."
                ),
            ),
            "actor_gi_color": types.Schema(
                type=types.Type.STRING,
                description=(
                    "OPTIONAL free-text gi/rashguard COLOR WORD ONLY (e.g. 'white', "
                    "'blue', 'black') if visibly distinguishable — never a name, never "
                    "combined with a fabricated identity."
                ),
            ),
            "action_class": types.Schema(
                type=types.Type.STRING,
                enum=list(ACTION_CLASS_VALUES),
                description="Axis 3 (action/technique class) — exactly one enum value.",
            ),
            "outcome": types.Schema(
                type=types.Type.STRING,
                enum=list(OUTCOME_VALUES),
                description="Axis 4 (outcome) — exactly one enum value.",
            ),
            "specific_technique_guess": types.Schema(
                type=types.Type.STRING,
                description="Optional free-text specific-technique guess. Ungated.",
            ),
            "confidence": types.Schema(type=types.Type.NUMBER, format="float", description="Confidence 0.0-1.0."),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["clips"],
        properties={
            "current_context_summary": types.Schema(type=types.Type.STRING),
            "clips": types.Schema(
                type=types.Type.ARRAY,
                description="One entry per detected grappling tag in this chunk, CHUNK-RELATIVE seconds.",
                items=tag_schema,
            ),
        },
    )


def build_video_prompt(previous_context: Optional[str]) -> str:
    """Per-chunk prompt for ``simplified-tags-time-v1``.

    The video part sent alongside this prompt is a NATIVE segment (this
    chunk's ``[effective_start, chunk_end]`` window, including PASS 2's
    back-overlap into the prior chunk) — from Gemini's point of view this clip
    starts at 0:00; it never sees the match's absolute clock. LeCun anti-
    suppression requirement: the overlap seconds at the START of this clip
    are a DELIBERATE safety net for PASS 2's dedup — do not omit an event just
    because it might have been reported in the previous chunk.
    """
    return f"""
    Here is a NATIVE video clip from a BJJ match. This clip's OWN internal
    clock starts at 0:00 — you have NO information about where this clip sits
    in the overall match; report all timestamps RELATIVE TO THIS CLIP's own
    0:00, never the match's absolute clock.

    PREVIOUS CONTEXT (narrative state only — who's ahead, current position):
    "{previous_context if previous_context else "Start of the match."}"

    IMPORTANT — this clip OVERLAPS the end of the previous chunk by design (a
    safety net so no event near the seam is lost). Do NOT suppress or skip an
    event just because it might already have been reported for the previous
    chunk — report EVERY distinct exchange you see in THIS clip, even if it
    also appeared near the end of the previous one. De-duplication across
    chunks is handled downstream, not by you.

    Tag every distinct grappling exchange visible in this clip using EXACTLY
    the 4-axis simplified taxonomy given in the system instruction. For each tag:
    1. `start_s`/`end_s`: seconds RELATIVE TO THIS CLIP's own 0:00 (float, not MM:SS).
    2. `position`: EXACTLY one Position enum value (Axis 1).
    3. `actor`: EXACTLY one of "top"/"bottom"/"contested" (Axis 2). This call has
       NO player reference images — you CANNOT know athletes' names or
       player_ids. NEVER output a name or any player_id-like string. If you can
       additionally tell gi color apart, put ONLY the color word (e.g. "white",
       "blue") in the optional `actor_gi_color` field — never fold it into a
       fabricated identity.
    4. `action_class`: EXACTLY one Action Class enum value (Axis 3).
    5. `outcome`: EXACTLY one Outcome enum value (Axis 4).
    6. `specific_technique_guess`: OPTIONAL free-text specific-name guess — never
       gates or replaces the enum fields above.
    7. `confidence`: 0.0-1.0.
    8. Use PREVIOUS CONTEXT to maintain narrative continuity and update
       `current_context_summary` for the next chunk (positions/who's ahead only
       — not a list of events already reported).

    Return ONLY valid JSON matching this shape — no prose:
    {{
        "current_context_summary": "string",
        "clips": [
            {{
                "start_s": 0.0,
                "end_s": 0.0,
                "position": "<Axis 1 enum value>",
                "actor": "<top|bottom|contested>",
                "actor_gi_color": "string (optional)",
                "action_class": "<Axis 3 enum value>",
                "outcome": "<Axis 4 enum value>",
                "specific_technique_guess": "string (optional)",
                "confidence": 0.0
            }}
        ]
    }}
    """


class SimplifiedTagsTimeAnalyzer:
    """NATIVE-VIDEO analyze path for ``simplified-tags-time-v1``
    (``chunk-segment-tags`` PASS 2). Sibling to ``SimplifiedTagsAnalyzer``
    above, but calls Gemini with a ``file_data`` + ``video_metadata`` native
    video segment (URL + start/end offsets) instead of extracted frame
    images — same reuse discipline as ``executors._analyze_video_window``'s
    native branch, not a fork of ``analyzer.py``.

    Returns Gemini's response text AS-IS (chunk-relative ``start_s``/``end_s``
    still un-converted) — the caller (``executors.chunk_analyze_node``) is
    responsible for adding the real sent ``start_offset`` to produce absolute
    match seconds. This class never sees or fabricates that offset.

    **Additive instrumentation (ADCC eval dashboard, Task 1):** after every
    ``analyze_chunk`` call (success OR failure), ``self.last_prompt_text`` /
    ``self.last_raw_response_text`` / ``self.last_usage_metadata`` are set to
    the exact prompt sent, the raw (pre-fence-strip) response text, and a
    plain ``{"prompt_token_count","candidates_token_count","total_token_count"}``
    dict from ``response.usage_metadata`` (``None`` for any field/whole dict
    the SDK response didn't populate — never fabricated). This is a PURE
    ADDITION: the method's return value/signature is unchanged, so every
    existing caller/test that only reads the return string is byte-identical.
    Callers that don't care about instrumentation never read these attributes
    and see no behavior change at all.
    """

    def __init__(
        self,
        client,
        *,
        model_id: str,
        temperature: float = 0.2,
        thinking_config: Optional[types.ThinkingConfig] = None,
        system_instruction: Optional[str] = None,
        retry_config: Optional[gemini_retry.GeminiRetryConfig] = None,
    ):
        self.client = client
        self.model_id = model_id
        self.temperature = temperature
        self.thinking_config = thinking_config
        self.system_instruction = resolve_system_instruction(system_instruction)
        self.retry_config = retry_config or gemini_retry.GeminiRetryConfig()
        # Additive instrumentation slots (Task 1) — see class docstring.
        self.last_prompt_text: Optional[str] = None
        self.last_raw_response_text: Optional[str] = None
        self.last_usage_metadata: Optional[dict] = None

    def _build_generate_config(self, response_schema: types.Schema) -> types.GenerateContentConfig:
        kwargs = dict(
            system_instruction=self.system_instruction,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        if self.thinking_config is not None:
            kwargs["thinking_config"] = self.thinking_config
        return types.GenerateContentConfig(**kwargs)

    async def analyze_chunk(
        self, youtube_url: str, start_sec: float, end_sec: float, previous_context: Optional[str] = None,
        *, response_schema=None, fps: int = 1,
    ) -> str:
        """Analyze ONE native-video chunk ``[start_sec, end_sec)``.

        ``start_sec``/``end_sec`` are the ACTUAL offsets sent to Gemini's
        ``video_metadata`` (i.e. already include PASS 2's back-overlap for
        ``chunk_analyze_node``, or the pre/post-roll expansion for
        ``highlight_analyze_node``) — the response's ``start_s``/``end_s``
        are relative to THIS ``start_sec``, which the caller adds back to get
        absolute match seconds.

        ``fps`` (additive, ``highlight-scan-analyze`` PASS 2 —
        ``HighlightAnalyzeConfig.fps``, real Gemini experiment finding: only
        1 or 10 are supported by that config's allowlist) defaults to 1,
        preserving ``chunk_analyze_node``'s prior hardcoded behavior
        byte-for-byte for every existing caller that doesn't pass it.
        """
        effective_schema = response_schema if response_schema is not None else build_video_response_schema()
        prompt = build_video_prompt(previous_context)
        video_part = types.Part(
            file_data=types.FileData(file_uri=youtube_url),
            video_metadata=types.VideoMetadata(
                start_offset=_format_offset_seconds(start_sec),
                end_offset=_format_offset_seconds(end_sec),
                fps=fps,
            ),
        )

        # Additive instrumentation (Task 1) — recorded regardless of outcome;
        # reset at the top of every call so a prior call's data never leaks
        # into this one's event.
        self.last_prompt_text = prompt
        self.last_raw_response_text = None
        self.last_usage_metadata = None

        try:
            logger.info(
                "Gemini simplified-tags-time-v1: chunk [%.1f-%.1f]s, model %s",
                start_sec, end_sec, self.model_id,
            )
            response = await gemini_retry.call_with_retry(
                lambda: self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=[types.Content(role="user", parts=[types.Part(text=prompt), video_part])],
                    config=self._build_generate_config(effective_schema),
                ),
                op_name="simplified-tags-time-v1",
                retry_config=self.retry_config,
            )
            text = response.text
            self.last_raw_response_text = text
            self.last_usage_metadata = extract_usage_metadata(response)
            logger.info(
                "Gemini simplified-tags-time-v1: received response (%d chars)",
                len(text) if text else 0,
            )
            return _strip_markdown_fences(text or "")
        except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error, no fabrication
            logger.error("Gemini simplified-tags-time-v1 call failed: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})
