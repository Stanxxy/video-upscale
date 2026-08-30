"""Thin OpenRouter (OpenAI-compatible chat-completions) client lane, per
`GATES_RERUN_DESIGN.md` §6.2/§6.4. Mirrors `service/pipelines/claude_frames.py`'s
shape deliberately — frame-sampled axis calls (fps=10, evenly subsampled to
<=20 JPEG frames per window), reusing `claude_frames.subsample_evenly` and
`service.pipelines.frame_source`/`service.routes.qa_vlm._resolve_stream_url`
directly (read-only reuse — never a duplicated frame-extraction
implementation, per the design's own "one-time refactor recommendation"
spirit, even though that refactor itself is not required).

**§6.2 probe RESULT (2026-07-26, real API call, ~$0 cost — a 400/404
response is not billed): Kimi K3 is FRAMES-ONLY.** A direct OpenRouter
request with an OpenAI-style `{"type": "video_url", ...}` content block
against `moonshotai/kimi-k3` returned `404 {"error": {"message": "No
endpoints found that support input video"}}`. Independently confirmed by
OpenRouter's own model-list metadata: `architecture.input_modalities =
["text", "image"]` (no "video"). Both signals agree — this resolves LeCun's
two-branch decision tree with no ambiguity left. Kimi therefore reuses the
EXACT SAME frame-extraction/subsampling path Opus already uses (this
module's `sample_and_encode_frames`), never a native-video call shape.

Confirmed exact OpenRouter model slugs (queried live against
`https://openrouter.ai/api/v1/models`, never guessed):
- Kimi K3: `moonshotai/kimi-k3`
- Opus 4.8: `anthropic/claude-opus-4.8` (`supported_efforts` includes
  `"high"`, matching this project's thinking-level convention exactly)

No fallback logic: a non-2xx OpenRouter response or a malformed/non-JSON
body raises immediately — never a fabricated result.

**[FIX 2026-07-26, Bezos precondition 2 for the frontier full-batch
authorization]** The original version of this module built request content
as ``[text, image, image, ...]`` with a BARE ``prompt_text`` (no framing
sentence) — a real, uncontrolled confound against
``claude_frames.analyze_axis_json``'s ``[image, image, ..., text]``
ordering + explicit "N JPEG frames are temporally ordered ... treat as one
continuous clip, not independent photos" framing sentence (claude_frames.py
lines 184-196). Both Kimi and OpenRouter-routed Opus calls now use the
SAME image-then-text ordering and the SAME framing-sentence template
(``_build_framed_prompt`` below) — the only honest difference is the vision-
library name embedded in the sentence (accurate per model: "Claude Vision"
vs "Kimi K3 vision" — both are, in fact, frames-only, so the underlying
claim ["has no native video and caps ~N images/request"] is true for
either)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from typing import Optional

import httpx

from eval.adcc.s12_gates import frame_source_retry
from service.pipelines import frame_source
from service.pipelines.claude_frames import MAX_CLAUDE_IMAGES, subsample_evenly
from service.routes.qa_vlm import _resolve_stream_url

logger = logging.getLogger("eval.adcc.s12_gates.openrouter_client")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
KIMI_K3_MODEL_ID = "moonshotai/kimi-k3"
OPUS_48_OPENROUTER_MODEL_ID = "anthropic/claude-opus-4.8"
OPUS_48_ANTHROPIC_MODEL_ID = "claude-opus-4-8"  # direct-Anthropic-SDK id (service.pipelines.claude_frames.CLAUDE_MODELS)

# Reasoning-effort mapping — OpenRouter's own unified `reasoning: {"effort": ...}`
# parameter, honored per-provider for models with a reasoning mode (design §6.2).
_REASONING_EFFORT_MAP: dict[str, str] = {"low": "low", "medium": "medium", "high": "high"}


class OpenRouterAPIError(RuntimeError):
    """Raised on any non-2xx OpenRouter response — never swallowed, never
    silently retried beyond what httpx itself does (i.e. not at all here;
    this lane has no retry layer, matching the "no fallback logic" rule —
    a real production integration would need one, this is a probe/smoke
    lane only)."""


def _jpeg_b64_data_url(image) -> str:
    """OpenAI-compatible `data:image/jpeg;base64,...` URL — the
    `image_url.url` shape every OpenAI-compatible provider (incl.
    OpenRouter) accepts, as distinct from Anthropic's native
    `{"type":"image","source":{...}}` block (§6.2: "the two providers'
    multimodal message schemas are NOT interchangeable")."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_framed_prompt(
    *, num_frames: int, window_start: float, window_end: float, fps: int,
    max_images: int, vision_label: str, prompt_text: str,
) -> str:
    """Byte-faithful port of `claude_frames.py`'s trailing text block
    (lines 184-196), parameterized only on the vision-library name so the
    sentence stays factually accurate per model (both Kimi K3 and Claude
    Vision are, confirmed, frames-only — see module docstring)."""
    return (
        f"The following {num_frames} JPEG frames are temporally ordered samples "
        f"from highlight window [{window_start:.3f}, {window_end:.3f}) seconds "
        f"(requested fps={fps}; evenly subsampled to ≤{max_images} because "
        f"{vision_label} has no native video and caps ~{max_images} images/"
        f"request). Treat them as one continuous clip — not independent photos.\n\n"
        f"{prompt_text}"
    )


def sample_and_encode_frames(
    youtube_url: str, window_start: float, window_end: float, fps: int,
    *, max_images: int = MAX_CLAUDE_IMAGES,
) -> list[str]:
    """Resolves a direct stream URL (`_resolve_stream_url`, yt-dlp), samples
    frames at ``fps`` over ``[window_start, window_end)``
    (`frame_source.sample_frames`), evenly subsamples to ``max_images``
    (`claude_frames.subsample_evenly` — the EXACT function Opus's own frame
    path uses, never a re-implementation), and returns OpenAI-compatible
    base64 JPEG data URLs, temporally ordered. Raises on any yt-dlp/ffmpeg
    failure — no placeholder frames."""
    stream_url, headers = _resolve_stream_url(youtube_url)
    native_fps = frame_source_retry.probe_native_fps_with_retry(stream_url, headers)  # [FIX §9.3]
    sampled = frame_source.sample_frames(stream_url, headers, window_start, window_end, native_fps, float(fps))
    if not sampled:
        raise RuntimeError(f"openrouter_client: ffmpeg returned zero frames for [{window_start},{window_end}) at fps={fps}")
    kept = subsample_evenly(sampled, max_images)
    logger.info(
        "openrouter_client: window=[%.3f,%.3f) fps=%s sampled=%d kept=%d (cap=%d)",
        window_start, window_end, fps, len(sampled), len(kept), max_images,
    )
    return [_jpeg_b64_data_url(f.image) for f in kept]


_TRANSIENT_RETRY_MAX_ATTEMPTS = 4  # bounded, real transient-fault handling — NOT a fallback (no masked/degraded result)


async def call_openrouter_chat(
    *, api_key: str, model: str, system_instruction: str, prompt_text: str,
    image_data_urls: list[str], reasoning_effort: Optional[str] = None,
    max_tokens: int = 16384, timeout_s: float = 180.0,
) -> tuple[dict, str, dict]:
    """ONE OpenRouter chat-completions call. Returns ``(parsed_json,
    raw_text, usage_dict)``. ``usage_dict`` includes
    ``reasoning_tokens``/``completion_tokens_details`` when the provider
    returns them (§6.2: "must positively confirm the parameter was honored
    ... a reasoning_tokens/equivalent field in the response's usage
    block") — never fabricated, ``None`` fields when genuinely absent.
    Raises ``OpenRouterAPIError`` on any non-2xx response (after bounded
    retry on 429 only — see below); raises ``RuntimeError`` on a non-JSON/
    malformed/length-truncated body — never a silent fallback.

    **[FIX 2026-07-26, real failure found during the frontier full-batch
    run's first live windows]** Two real, root-caused failure modes, fixed
    here rather than papered over:
    (1) ``max_tokens`` was 4096 — too small once ``reasoning.effort="high"``
    is requested: OpenAI-compatible completions bill/cap REASONING tokens
    against the SAME ``max_tokens`` budget as the final answer (unlike
    Anthropic's separate thinking budget), so a verbose high-effort
    reasoning trace can consume the entire budget and leave
    ``finish_reason="length"`` with ``content=None`` — a real, observed
    failure (Kimi K3, window w1, this batch). Raised to 16384. (2) A real
    transient ``429`` from the upstream provider (observed: Kimi/Moonshot,
    ``retry_after_seconds=1``) is now retried up to
    ``_TRANSIENT_RETRY_MAX_ATTEMPTS`` times honoring the response's own
    ``Retry-After``/``retry_after_seconds`` hint when present — this is
    bounded, provider-signaled transient-fault handling (the same category
    as the Gemini path's own ``GeminiRetryConfig``), not a "fallback" that
    masks or substitutes a degraded result; a persistent 429 after
    exhausting retries still raises."""
    # Image-then-text ordering — matches `claude_frames.py`'s construction
    # (§ Bezos precondition 2, 2026-07-26 — see module docstring "[FIX]" note).
    content: list[dict] = []
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": prompt_text})

    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content},
        ],
    }
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
        payload["include_reasoning"] = True  # ask the provider to surface reasoning-token accounting, if it has any

    resp = None
    for attempt in range(1, _TRANSIENT_RETRY_MAX_ATTEMPTS + 1):
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code != 429 or attempt == _TRANSIENT_RETRY_MAX_ATTEMPTS:
            break
        retry_after_s = 1.0
        try:
            body_for_retry = resp.json()
            retry_after_s = float(
                (body_for_retry.get("error", {}).get("metadata", {}) or {}).get("retry_after_seconds", 1.0)
            )
        except Exception:
            pass
        logger.warning(
            "openrouter_client: 429 rate-limited (attempt %d/%d), retrying after %.1fs — model=%s",
            attempt, _TRANSIENT_RETRY_MAX_ATTEMPTS, retry_after_s, model,
        )
        await asyncio.sleep(retry_after_s)
    if resp.status_code != 200:
        raise OpenRouterAPIError(f"OpenRouter {resp.status_code}: {resp.text[:1000]}")

    body = resp.json()
    finish_reason = (body.get("choices") or [{}])[0].get("finish_reason")
    if finish_reason == "length":
        raise RuntimeError(
            f"OpenRouter response truncated (finish_reason='length') before a final answer was produced — "
            f"max_tokens={max_tokens} was insufficient for this call's reasoning+answer length. model={model}. "
            f"body={body}"
        )
    if "error" in body:
        raise OpenRouterAPIError(f"OpenRouter call failed: {body['error']}")
    try:
        raw_text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"OpenRouter response missing choices[0].message.content: {body}") from e
    if not raw_text:
        raise RuntimeError(f"OpenRouter returned empty content: {body}")

    stripped = raw_text.strip()
    if "```json" in stripped:
        stripped = stripped.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in stripped:
        stripped = stripped.split("```", 1)[1].split("```", 1)[0]
    try:
        data = json.loads(stripped.strip())
    except (ValueError, TypeError) as e:
        # [FIX §9.3, GATES_RERUN_DESIGN.md] Kimi `w8`'s ORIGINAL failure
        # (`"Expecting value: line 157 column 1"`) could not be root-caused
        # against its actual raw text — the 500-char truncation here meant
        # only the bare JSONDecodeError string survived to the ledger/
        # predictions file, not the payload that caused it. A live
        # reproduction of the exact same window/prompt/call succeeded
        # cleanly (single valid JSON object, no anomaly) — consistent with a
        # genuinely intermittent provider-side failure, NOT the same
        # "valid JSON + trailing prose" shape confirmed for Opus `w6` (see
        # `anthropic_direct._extract_final_json_object`'s docstring). Per
        # §9.3's explicit "do NOT assume one fix covers both" mandate, NO
        # parser-recovery change is applied here — this keeps failing loud
        # (`call_failed`, unchanged behavior). Only the truncation limit is
        # widened (500->4000 chars) so any FUTURE occurrence captures enough
        # raw text to actually diagnose, instead of repeating this gap.
        raise RuntimeError(f"OpenRouter non-JSON response: {e}; raw={raw_text[:4000]!r}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"OpenRouter JSON root must be an object, got {type(data).__name__}")

    usage = body.get("usage") or {}
    usage_dict = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost_usd": usage.get("cost"),  # OpenRouter can return real-dollar cost directly in the usage block
    }
    return data, raw_text, usage_dict


async def call_kimi_frames_axis(
    *, api_key: str, youtube_url: str, window_start: float, window_end: float, fps: int,
    system_instruction: str, prompt_text: str, thinking: Optional[str] = None,
) -> tuple[dict, str, dict, float]:
    """Kimi K3 axis call, frames-only branch (confirmed, see module
    docstring). Returns ``(parsed, raw_text, usage_dict, latency_ms)``."""
    t0 = time.perf_counter()
    image_urls = sample_and_encode_frames(youtube_url, window_start, window_end, fps)
    framed_prompt = _build_framed_prompt(
        num_frames=len(image_urls), window_start=window_start, window_end=window_end, fps=fps,
        max_images=MAX_CLAUDE_IMAGES, vision_label="Kimi K3 vision", prompt_text=prompt_text,
    )
    effort = _REASONING_EFFORT_MAP.get(thinking) if thinking else None
    data, raw, usage = await call_openrouter_chat(
        api_key=api_key, model=KIMI_K3_MODEL_ID, system_instruction=system_instruction,
        prompt_text=framed_prompt, image_data_urls=image_urls, reasoning_effort=effort,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return data, raw, usage, latency_ms


async def call_opus_via_openrouter_axis(
    *, api_key: str, youtube_url: str, window_start: float, window_end: float, fps: int,
    system_instruction: str, prompt_text: str, thinking: Optional[str] = None,
) -> tuple[dict, str, dict, float]:
    """Opus 4.8 axis call VIA OPENROUTER (not direct Anthropic) — frames
    branch (Claude Vision has no native video, a hard platform limit, §6.1).
    Returns ``(parsed, raw_text, usage_dict, latency_ms)``."""
    t0 = time.perf_counter()
    image_urls = sample_and_encode_frames(youtube_url, window_start, window_end, fps)
    framed_prompt = _build_framed_prompt(
        num_frames=len(image_urls), window_start=window_start, window_end=window_end, fps=fps,
        max_images=MAX_CLAUDE_IMAGES, vision_label="Claude Vision", prompt_text=prompt_text,
    )
    effort = _REASONING_EFFORT_MAP.get(thinking) if thinking else None
    data, raw, usage = await call_openrouter_chat(
        api_key=api_key, model=OPUS_48_OPENROUTER_MODEL_ID, system_instruction=system_instruction,
        prompt_text=framed_prompt, image_data_urls=image_urls, reasoning_effort=effort,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return data, raw, usage, latency_ms
