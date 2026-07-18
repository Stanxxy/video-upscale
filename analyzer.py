from google import genai
from google.genai import types
import os
import asyncio
import json
import logging
from typing import Optional
from PIL import Image

from service import taxonomy_mapper

logger = logging.getLogger(__name__)

# Default when caller does not pass request_timeout_ms (matches ServiceConfig default).
DEFAULT_GEMINI_REQUEST_TIMEOUT_MS = 600_000


def _effective_timeout_ms(request_timeout_ms: Optional[int]) -> int:
    ms = request_timeout_ms if request_timeout_ms is not None else DEFAULT_GEMINI_REQUEST_TIMEOUT_MS
    return max(5_000, ms)


def _gemini_client(api_key: str, timeout_ms: int) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


def _player_ids_from_refs(player_references) -> list:
    """Extract the ordered list of player_ids from labelled references.

    Returns [] when references carry no player_id (legacy unlabelled refs) so the
    caller can decide not to constrain actor_player_id.
    """
    if not player_references:
        return []
    ids = []
    for ref in player_references:
        pid = ref.get("player_id")
        if pid:
            ids.append(pid)
    return ids


def _build_response_schema(player_ids):
    """Gemini structured-output schema for the sliding-window analyzer.

    Mirrors backend ``models/functions.py:build_response_schema`` (proven pattern:
    ``genai.types.Schema`` + ``response_mime_type='application/json'``).

    D5 (simplified-4-axis-taxonomy adoption, 2026-07-12): promotes the QA
    pipeline's ENUM-constrained axis1/axis3/axis4 schema
    (``service/pipelines/simplified_tags.py``) into production, replacing the
    old free-string ``action``/``technique`` fields (which relied on
    post-hoc frozenset sanitization in ``taxonomy_mapper.py`` — a silent
    ``unknown -> other`` path this closes at the source). ``technique_guess``
    is a deliberately UNCONSTRAINED free-text field (D1) — a drift/regression
    signal only, never gating, mapped into the curated per-class shortlist by
    ``taxonomy_mapper.resolve_technique_shortlist`` downstream in
    ``service/sns.py``.

    ``actor_player_id`` is UNCHANGED from the pre-existing production
    machinery (D5 — do NOT adopt the QA pipeline's free-text/positional
    actor, per INS-073: this call sends reference images and can ground
    identity; the QA chunk-segment path cannot): a STRING enum of the
    video's player_ids. When player_ids is empty the field is an
    unconstrained STRING (legacy/no-reference path).
    """
    actor_schema_kwargs = dict(
        type=types.Type.STRING,
        description=(
            "The player_id of the athlete performing the technique. MUST be one "
            "of the provided players' player_id values."
        ),
    )
    if player_ids:
        actor_schema_kwargs["enum"] = list(player_ids)

    clip_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "start_frame", "end_frame",
            "axis1_position", "axis3_action", "axis4_outcome",
            "actor_player_id",
        ],
        properties={
            "start_frame": types.Schema(type=types.Type.INTEGER),
            "end_frame": types.Schema(type=types.Type.INTEGER),
            "axis1_position": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING, enum=list(taxonomy_mapper.AXIS1_POSITION)),
                description=(
                    "Axis 1 — Position (multi-label). One or more exact enum values "
                    "from the taxonomy describing body configuration."
                ),
            ),
            "axis3_action": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING, enum=list(taxonomy_mapper.AXIS3_ACTION)),
                description=(
                    "Axis 3 — Action/Technique Class (multi-label). One or more exact "
                    "enum values from the taxonomy."
                ),
            ),
            "axis4_outcome": types.Schema(
                type=types.Type.STRING,
                enum=list(taxonomy_mapper.AXIS4_OUTCOME),
                description="Axis 4 — Outcome. Exactly one exact enum value from the taxonomy.",
            ),
            "technique_guess": types.Schema(
                type=types.Type.STRING,
                description=(
                    "OPTIONAL free-text specific-technique guess (e.g. 'possibly a kimura'). "
                    "Ungated — never defines or replaces the enum axes above; a drift/"
                    "regression signal only."
                ),
            ),
            "actor_player_id": types.Schema(**actor_schema_kwargs),
            "identity_uncertain": types.Schema(
                type=types.Type.BOOLEAN,
                description="True if the athletes are too entangled to tell apart confidently.",
            ),
            "reasoning": types.Schema(
                type=types.Type.STRING,
                description="Free-text rationale. May mention gi color / top-bottom — descriptor ONLY, never identity.",
            ),
            "confidence": types.Schema(
                type=types.Type.NUMBER,
                description="Confidence 0.0-1.0.",
                format="float",
            ),
        },
    )

    return types.Schema(
        type=types.Type.OBJECT,
        required=["clips"],
        properties={
            "current_context_summary": types.Schema(type=types.Type.STRING),
            "clips": types.Schema(
                type=types.Type.ARRAY,
                description="One entry per detected grappling clip in this window.",
                items=clip_schema,
            ),
        },
    )


class BJJTechniqueAnalyzer:
    def __init__(
        self,
        api_key,
        taxonomy_path=None,
        request_timeout_ms: Optional[int] = None,
        *,
        model_id: str = "gemini-3.1-flash-lite",
        temperature: float = 0.2,
        thinking_config: Optional[types.ThinkingConfig] = None,
        taxonomy_text: Optional[str] = None,
    ):
        """``model_id``/``temperature``/``thinking_config``/``taxonomy_text`` are
        additive, keyword-only, production-preserving knobs (QA pipeline seam — see
        working_log/plans/2026-07-03-vlm-studio-pipeline-dag-build-plan.md §3.3).
        Defaults are IDENTICAL to the hardcoded values production callers relied on
        before this seam existed, so any caller that omits them gets byte-identical
        behavior. ``thinking_config=None`` (the default) means the GenerateContentConfig
        sent to Gemini OMITS the ``thinking_config`` field entirely, exactly as before.
        ``taxonomy_text=None`` (the default) means the taxonomy/system text is read
        from ``taxonomy_path`` exactly as before; a non-None value REPLACES that file
        read entirely (used by the QA pipeline's per-node ``system_instruction`` override).
        """
        if not api_key:
            raise ValueError("API Key is required for BJJTechniqueAnalyzer")
        to_ms = _effective_timeout_ms(request_timeout_ms)
        self.client = _gemini_client(api_key, to_ms)
        self._request_timeout_ms = to_ms
        self.model_id = model_id
        self.temperature = temperature
        self.thinking_config = thinking_config
        self.taxonomy_text = taxonomy_text
        if taxonomy_path is None:
            taxonomy_path = os.path.join(os.path.dirname(__file__), "bjj_analysis_taxonomy.md")
        self.taxonomy_path = taxonomy_path

    def _build_generate_config(self, system_instruction, response_schema):
        """Build ``GenerateContentConfig`` kwargs additively so the default path
        (no ``thinking_config``) is byte-identical to the pre-seam call site."""
        kwargs = dict(
            system_instruction=system_instruction,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        if self.thinking_config is not None:
            kwargs["thinking_config"] = self.thinking_config
        return types.GenerateContentConfig(**kwargs)

    def _build_contents_and_prompt(self, frames, frame_indices, previous_context, player_references):
        """Build the prompt text, system instruction, and contents list (shared by sync/async)."""
        if self.taxonomy_text is not None:
            taxonomy_text = self.taxonomy_text
        else:
            try:
                with open(self.taxonomy_path, "r") as f:
                    taxonomy_text = f.read()
            except Exception:
                taxonomy_text = "Taxonomy file not found."

        system_instruction = f"""
        {taxonomy_text}

        You are analyzing a chunk of video frames.
        """

        ref_instruction = ""
        if player_references:
            ref_lines = ", ".join(
                f"{ref['player_name']} (player_id {ref.get('player_id', '?')})"
                for ref in player_references
            )
            ref_instruction = f"""
        7. **Athlete Identification (grounded)**: Reference images below are each labelled
           with a player name and player_id. The players in this video are: {ref_lines}.
           For each clip, set `actor_player_id` to the player_id of the athlete performing the
           technique, chosen ONLY from the provided players. If the two athletes are too entangled
           to tell apart, set `identity_uncertain: true` and give your best guess. Mention gi color
           or top/bottom ONLY inside `reasoning`, never as the identity."""

        prompt = f"""
        Here is a sequence of {len(frames)} frames from a BJJ match.
        The frames correspond to the following frame numbers: {frame_indices}.

        PREVIOUS CONTEXT: "{previous_context if previous_context else "Start of the match."}"

        Tag every distinct grappling exchange visible in this window using EXACTLY the
        4-axis simplified taxonomy given in the system instruction.

        INSTRUCTIONS:
        1. **Analyze the Flow**: Use frame numbers to define start/end points.
        2. **Resolve Ambiguity**: Use biomechanics. Every axis below has an honest
           "unclear"/"scramble"/"transition" value — use it instead of guessing confidently
           wrong.
        3. **Context Awareness**: Use PREVIOUS CONTEXT.
        4. **Identify the Actor**: For each tag, set `actor_player_id` to the player_id of the
           athlete performing it, chosen from the provided players (see Athlete Identification below).
        5. **Use Enum Values ONLY**: `axis1_position` (Axis 1 — Position, one or more enum values),
           `axis3_action` (Axis 3 — Action/Technique Class, one or more enum values), and
           `axis4_outcome` (Axis 4 — Outcome, exactly one enum value) MUST use exact values from
           the taxonomy. Do NOT invent new values, do NOT abbreviate, do NOT combine axes.
        6. **Optional technique guess**: `technique_guess` is OPTIONAL free text (e.g. "possibly a
           kimura") — it never gates or replaces the enum axes above.
        7. **Generate Output**: Return ONLY valid JSON matching this format:
        {{
            "current_context_summary": "string",
            "clips": [
                {{
                    "start_frame": int,
                    "end_frame": int,
                    "axis1_position": ["<Axis 1 enum value>", "..."],
                    "axis3_action": ["<Axis 3 enum value>", "..."],
                    "axis4_outcome": "<Axis 4 enum value>",
                    "technique_guess": "string (optional, e.g. 'double leg takedown')",
                    "actor_player_id": "the player_id of the athlete performing the technique (must be one of the provided players)",
                    "identity_uncertain": false,
                    "reasoning": "string — may mention gi color / top-bottom as a descriptor, never as identity",
                    "confidence": 0.0-1.0
                }}
            ]
        }}{ref_instruction}
        """

        contents = [prompt]
        if player_references:
            for ref in player_references:
                contents.append(
                    f"Player {ref['player_name']} (player_id {ref.get('player_id', '?')}):"
                )
                contents.append(ref["image"])
        contents.extend(frames)

        return system_instruction, contents

    @staticmethod
    def _validate_actor_player_ids(parsed_text: str, player_ids) -> None:
        """Strict validation: every clip's actor_player_id must be in the enum.

        Raises ValueError on any out-of-enum value. NO silent gi-color fallback —
        a wrong/missing identity is a hard error so the pipeline never re-introduces
        ungrounded free-text identity. No-op when player_ids is empty (legacy path).
        """
        if not player_ids:
            return
        try:
            parsed = json.loads(parsed_text)
        except (ValueError, TypeError):
            # Parsing handled elsewhere; nothing to validate on unparseable text.
            return
        allowed = set(player_ids)
        for clip in parsed.get("clips", []):
            actor = clip.get("actor_player_id")
            if actor is not None and actor not in allowed:
                raise ValueError(
                    f"Gemini returned out-of-enum actor_player_id={actor!r}; "
                    f"allowed player_ids={sorted(allowed)}"
                )

    @staticmethod
    def _parse_response_text(text: str) -> str:
        """Strip markdown fences from Gemini response."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()

    async def analyze_sequence_async(
        self, frames, frame_indices, previous_context=None, player_references=None,
        *, response_schema=None,
    ):
        """Async version of analyze_sequence — uses client.aio for non-blocking Gemini calls.

        Same return signature as analyze_sequence (returns a JSON string).
        The context chain is preserved by the caller; this method is stateless.

        ``response_schema`` is additive/keyword-only (QA pipeline seam): ``None``
        (the default) reproduces today's ``_build_response_schema(player_ids)`` call
        exactly. A caller may pass an alternate ``types.Schema`` (e.g. the QA
        ``events-v1`` schema) to change the structured-output shape.
        """
        if not frames:
            return "No frames."

        system_instruction, contents = self._build_contents_and_prompt(
            frames, frame_indices, previous_context, player_references,
        )
        player_ids = _player_ids_from_refs(player_references)
        effective_schema = response_schema if response_schema is not None else _build_response_schema(player_ids)

        try:
            logger.info(
                "Gemini single-agent async: sending %d frames to model %s (http_timeout_ms=%d)",
                len(frames), self.model_id, self._request_timeout_ms,
            )
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=self._build_generate_config(system_instruction, effective_schema),
            )
            text = response.text
            logger.info(
                "Gemini single-agent async: received response (%d chars)",
                len(text) if text else 0,
            )
            logger.debug("Gemini async raw response: %.500s", text)
            parsed = self._parse_response_text(text)
        except Exception as e:
            logger.error("Gemini single-agent async call failed: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})

        # Strict identity validation OUTSIDE the API try/except: an out-of-enum
        # actor_player_id is a hard error and must propagate (no gi-color fallback).
        self._validate_actor_player_ids(parsed, player_ids)
        return parsed

    def analyze_sequence(
        self, frames, frame_indices, previous_context=None, player_references=None,
        *, response_schema=None,
    ):
        """``response_schema`` is additive/keyword-only (QA pipeline seam) — see
        ``analyze_sequence_async`` docstring for semantics."""
        if not frames:
            return "No frames."

        system_instruction, contents = self._build_contents_and_prompt(
            frames, frame_indices, previous_context, player_references,
        )
        player_ids = _player_ids_from_refs(player_references)
        effective_schema = response_schema if response_schema is not None else _build_response_schema(player_ids)

        try:
            logger.info(
                "Gemini single-agent: sending %d frames to model %s (http_timeout_ms=%d)",
                len(frames), self.model_id, self._request_timeout_ms,
            )
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=self._build_generate_config(system_instruction, effective_schema),
            )

            text = response.text
            logger.info(
                "Gemini single-agent: received response (%d chars)",
                len(text) if text else 0,
            )
            logger.debug("Gemini raw response: %.500s", text)
            parsed = self._parse_response_text(text)
        except Exception as e:
            logger.error("Gemini single-agent call failed: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})

        # Strict identity validation OUTSIDE the API try/except: an out-of-enum
        # actor_player_id is a hard error and must propagate (no gi-color fallback).
        self._validate_actor_player_ids(parsed, player_ids)
        return parsed


class BJJMultiAgentAnalyzer:
    def __init__(
        self,
        api_key,
        taxonomy_path=None,
        request_timeout_ms: Optional[int] = None,
        *,
        model_id: str = "gemini-3.1-flash-lite",
        temperature: float = 0.2,
        thinking_config: Optional[types.ThinkingConfig] = None,
        taxonomy_text: Optional[str] = None,
    ):
        """Additive, keyword-only, production-preserving knobs — mirrors
        ``BJJTechniqueAnalyzer.__init__`` (see that docstring). ``model_id`` replaces
        ``self.model_id`` used by every internal Gemini call (3 persona agents + Judge).
        ``thinking_config`` is added to every internal call ONLY when not None (default
        omits the field, byte-identical to pre-seam behavior). ``temperature`` is stored
        for API parity with ``BJJTechniqueAnalyzer`` but is intentionally NOT applied to
        the persona-agent calls (each has its own fixed, deliberately-tuned per-persona
        temperature — 0.8/0.6/0.9) or the Judge call (fixed 0.1 for deterministic
        synthesis); overriding those with a single class-level temperature would change
        default behavior, which the seam must not do. ``taxonomy_text``, when not None,
        REPLACES the ``taxonomy_path`` file read used to build the JUDGE's system
        instruction (the 3 persona agents don't read the taxonomy at all — unaffected).
        """
        if not api_key:
            raise ValueError("API Key is required for BJJMultiAgentAnalyzer")
        to_ms = _effective_timeout_ms(request_timeout_ms)
        self.client = _gemini_client(api_key, to_ms)
        self._request_timeout_s = to_ms / 1000.0
        self.model_id = model_id
        self.temperature = temperature
        self.thinking_config = thinking_config
        self.taxonomy_text = taxonomy_text
        if taxonomy_path is None:
            taxonomy_path = os.path.join(os.path.dirname(__file__), "bjj_analysis_taxonomy.md")
        self.taxonomy_path = taxonomy_path

    def _build_generate_config(self, system_instruction, temperature, response_schema=None):
        kwargs = dict(system_instruction=system_instruction, temperature=temperature)
        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema
        if self.thinking_config is not None:
            kwargs["thinking_config"] = self.thinking_config
        return types.GenerateContentConfig(**kwargs)

    async def run_agent(self, role_name, system_prompt, frames, frame_indices, context, temperature=0.7, player_references=None):
        """Runs a single agent with a specific persona."""
        system_instruction_base = """
        You are a BJJ Analyst. Analyze the video frames provided.
        """

        ref_instruction = ""
        if player_references:
            ref_lines = ", ".join(
                f"{ref['player_name']} (player_id {ref.get('player_id', '?')})"
                for ref in player_references
            )
            ref_instruction = f"""
        Reference images below are each labelled with a player name and player_id.
        The players in this video are: {ref_lines}. Refer to each athlete by their
        player name / player_id, never by gi color (gi color is a descriptor only)."""

        prompt = f"""
        You are acting as the {role_name}.
        Analyze these frames ({len(frames)} frames).
        Frame Indices: {frame_indices}
        Previous Context: "{context}"

        Output your analysis. Be specific and detailed according to your role.
        For each technique or action identified, specify which athlete performs it
        by their player name / player_id (see the labelled references). You MAY mention
        gi color or top/bottom as a descriptor, but never as the athlete's identity.{ref_instruction}
        Do NOT output JSON. Output a raw analysis paragraph.
        """

        try:
            contents = [prompt]

            # Add player reference images before the crop frames (labelled by identity)
            if player_references:
                for ref in player_references:
                    contents.append(
                        f"Player {ref['player_name']} (player_id {ref.get('player_id', '?')}):"
                    )
                    contents.append(ref["image"])

            contents.extend(frames)

            logger.info(
                "Gemini multi-agent: sending %d frames to %s agent (timeout %.0fs)",
                len(frames), role_name, self._request_timeout_s,
            )
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=self._build_generate_config(
                        system_instruction_base + "\n\n" + system_prompt, temperature,
                    ),
                ),
                timeout=self._request_timeout_s,
            )
            logger.info("Gemini multi-agent %s: received response (%d chars)",
                        role_name, len(response.text) if response.text else 0)
            return f"--- {role_name} REPORT ---\n{response.text}\n"
        except asyncio.TimeoutError:
            logger.error(
                "Gemini multi-agent %s: timed out after %.0fs",
                role_name, self._request_timeout_s,
            )
            return f"--- {role_name} TIMEOUT ---\nError: request exceeded {self._request_timeout_s:.0f}s\n"
        except Exception as e:
            logger.error("Gemini multi-agent %s failed: %s", role_name, e,
                         exc_info=True)
            return f"--- {role_name} FAILED ---\nError: {str(e)}\n"

    async def analyze_sequence_async(
        self, frames, frame_indices, previous_context=None, player_references=None,
        *, response_schema=None,
    ):
        """``response_schema`` is additive/keyword-only (QA pipeline seam): ``None``
        (the default) reproduces today's ``_build_response_schema(player_ids)`` Judge
        call exactly. Applies ONLY to the final Judge synthesis call; the 3 persona
        agent calls are unstructured free text (unchanged)."""
        if not frames:
            return "No frames."

        tasks = []

        # Agent A: The Biomechanist
        tasks.append(
            self.run_agent(
                "Biomechanist",
                "Focus ONLY on the physics. Describe limb positions, joint angles, leverage points, and entanglements. Identify which specific joints are under pressure. Always specify which athlete (by player name / player_id) is performing each action.",
                frames, frame_indices, previous_context, temperature=0.8,
                player_references=player_references,
            )
        )

        # Agent B: The Referee
        tasks.append(
            self.run_agent(
                "Referee",
                "Focus on IBJJF/ADCC scoring standards. Has a position been held for 3 seconds? Is a submission 'real' or just a setup? Is the pass complete? Always specify which athlete (by player name / player_id) scores or attacks.",
                frames, frame_indices, previous_context, temperature=0.6,
                player_references=player_references,
            )
        )

        # Agent C: The Tactician
        tasks.append(
            self.run_agent(
                "Tactician",
                "Focus on the 'Why'. What is the attacker trying to do? Are they baiting? Describe the strategic flow of the match. Always specify which athlete (by player name / player_id) is driving the action.",
                frames, frame_indices, previous_context, temperature=0.9,
                player_references=player_references,
            )
        )

        logger.info(
            "Gemini multi-agent: running 3 specialists in parallel (%.0fs cap each)",
            self._request_timeout_s,
        )
        agent_reports = await asyncio.gather(*tasks)
        logger.info("Gemini multi-agent: specialists finished; starting Judge")
        full_report_text = "\n".join(agent_reports)

        player_ids = _player_ids_from_refs(player_references)
        if player_ids:
            id_hint = (
                "For each technique, set `actor_player_id` to the player_id of the athlete "
                f"performing it, chosen ONLY from these player_ids: {player_ids}. If the athletes "
                "are too entangled to tell apart, set `identity_uncertain: true` and your best guess. "
                "Mention gi color / top-bottom only in `reasoning`, never as identity."
            )
        else:
            id_hint = (
                "For each technique, set `actor_player_id` to the player_id of the athlete "
                "performing it. Mention gi color / top-bottom only in `reasoning`, never as identity."
            )

        # The Judge
        judge_prompt = f"""
        You are the Head Judge.
        Read the following reports from your specialized agents.

        {full_report_text}

        YOUR TASK:
        1. Synthesize these views into a single ground truth.
        2. Resolve conflicts.
        3. {id_hint}
        4. Use EXACT enum values from the taxonomy for `axis1_position` (Axis 1 — Position,
           one or more values), `axis3_action` (Axis 3 — Action/Technique Class, one or more
           values), and `axis4_outcome` (Axis 4 — Outcome, exactly one value). Do NOT invent
           new values. `technique_guess` is OPTIONAL free text and never gates these axes.
        5. Output the final analysis in the required JSON format:
        {{
            "current_context_summary": "string",
            "clips": [
                {{
                    "start_frame": int,
                    "end_frame": int,
                    "axis1_position": ["<Axis 1 enum value>", "..."],
                    "axis3_action": ["<Axis 3 enum value>", "..."],
                    "axis4_outcome": "<Axis 4 enum value>",
                    "technique_guess": "string (optional, e.g. 'double leg takedown')",
                    "actor_player_id": "the player_id of the athlete performing the technique (must be one of the provided players)",
                    "identity_uncertain": false,
                    "reasoning": "string — may mention gi color / top-bottom as a descriptor, never as identity",
                    "confidence": 0.0-1.0
                }}
            ]
        }}

        TAXONOMY REFERENCE:
        (See system instruction — it contains the full list of valid Axis 1/3/4 enum values)
        """

        try:
            if self.taxonomy_text is not None:
                taxonomy_text = self.taxonomy_text
            else:
                try:
                    with open(self.taxonomy_path, "r") as f:
                        taxonomy_text = f.read()
                except Exception:
                    taxonomy_text = ""

            logger.info(
                "Gemini multi-agent Judge: synthesizing reports (timeout %.0fs)",
                self._request_timeout_s,
            )
            effective_schema = response_schema if response_schema is not None else _build_response_schema(player_ids)
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=judge_prompt,
                    config=self._build_generate_config(
                        taxonomy_text + "\n\nYou are the Synthesizer. Output JSON only.",
                        0.1,
                        response_schema=effective_schema,
                    ),
                ),
                timeout=self._request_timeout_s,
            )

            text = response.text
            logger.info("Gemini multi-agent Judge: received response (%d chars)",
                        len(text) if text else 0)
            logger.debug("Gemini Judge raw response: %.500s", text)

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            parsed = text.strip()

        except asyncio.TimeoutError:
            logger.error(
                "Gemini multi-agent Judge: timed out after %.0fs",
                self._request_timeout_s,
            )
            return json.dumps({"error": f"judge_timeout_after_{self._request_timeout_s:.0f}s"})
        except Exception as e:
            logger.error("Gemini multi-agent Judge failed: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})

        # Strict identity validation OUTSIDE the API try/except — out-of-enum is a hard error.
        BJJTechniqueAnalyzer._validate_actor_player_ids(parsed, player_ids)
        return parsed


# Wrapper for synchronous calls
def analyze_sequence_sync(analyzer, frames, frame_indices, previous_context, player_references=None):
    return asyncio.run(
        analyzer.analyze_sequence_async(frames, frame_indices, previous_context, player_references=player_references)
    )
