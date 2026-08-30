"""Claude (Anthropic) frame-path axis calls for ``highlight_analyze``.

Claude Vision accepts images only (JPEG/PNG/GIF/WebP) — not native YouTube
``file_uri`` video. Anthropic caps ~20 images per Messages request, so a
highlight window sampled at fps=10 can easily produce hundreds of frames; we
sample at the configured fps then **evenly subsample to ≤20**. Never fabricate
frames: yt-dlp / ffmpeg errors surface as-is.

Opus 4.8 thinking: ``budget_tokens`` / ``thinking.type=enabled`` returns 400.
Map UI levels to adaptive thinking + ``output_config.effort`` (or omit thinking
entirely for off). Gemini ``media_resolution`` has no Claude equivalent — ignored.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from typing import TYPE_CHECKING, Optional, Sequence, TypeVar

from service.pipelines import frame_source
from service.routes.qa_vlm import _resolve_stream_url

if TYPE_CHECKING:
    from service.pipelines.executors import RunContext

logger = logging.getLogger("service.pipelines.claude_frames")

# Playground A/B — Highlight Analyze only (not global EVENT_MODELS).
CLAUDE_MODELS: list[str] = ["claude-opus-4-8"]

# Anthropic Messages vision hard cap (~20 images per request). Documented here
# so future fps/window changes do not silently blow past the API limit.
MAX_CLAUDE_IMAGES = 20

T = TypeVar("T")


def is_claude_model(model_id: str) -> bool:
    return bool(model_id) and (
        model_id in CLAUDE_MODELS or model_id.startswith("claude-")
    )


def subsample_evenly(items: Sequence[T], max_n: int = MAX_CLAUDE_IMAGES) -> list[T]:
    """Keep at most ``max_n`` items, spaced evenly across the sequence.

    At fps=10 a rolled highlight window can be hundreds of frames; Claude
    rejects >~20 images/request. We never invent frames — only drop extras
    with even spacing so the temporal span still covers the window.
    """
    n = len(items)
    if n <= max_n:
        return list(items)
    if max_n <= 0:
        raise ValueError(f"subsample_evenly: max_n must be > 0, got {max_n}")
    if max_n == 1:
        return [items[n // 2]]
    # Inclusive endpoints; round can collide on tiny n — dedupe, preserve order.
    indices: list[int] = []
    seen: set[int] = set()
    for i in range(max_n):
        idx = int(round(i * (n - 1) / (max_n - 1)))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return [items[i] for i in indices]


def opus_thinking_kwargs(ui_level: Optional[str]) -> dict:
    """Map UI thinking tabs to Opus 4.8 adaptive thinking + effort.

    Opus 4.8 rejects ``thinking: {type: "enabled", budget_tokens: N}`` (400).
    - off / None → omit thinking (runs without extended thinking)
    - low / medium / high → ``thinking.type=adaptive`` + matching effort
    """
    if ui_level in (None, "off"):
        return {}
    effort = {"low": "low", "medium": "medium", "high": "high"}.get(ui_level, "high")
    return {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
    }


def _strip_json_fences(text: str) -> str:
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def _jpeg_b64(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _ensure_stream(ctx: "RunContext") -> None:
    """Resolve yt-dlp stream URL + native fps once per run (lazy; Gemini native
    path never needs this). Real resolve/probe errors propagate — no guesses."""
    if ctx.stream_url and ctx.native_fps:
        return
    stream_url, headers = _resolve_stream_url(ctx.youtube_url)
    native_fps = frame_source.probe_native_fps(stream_url, headers)
    ctx.stream_url = stream_url
    ctx.stream_headers = headers
    ctx.native_fps = native_fps


def _extract_text(response) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


async def analyze_axis_json(
    ctx: "RunContext",
    *,
    model_id: str,
    window_start: float,
    window_end: float,
    prompt_text: str,
    system_instruction: str,
    fps: int,
    thinking: Optional[str] = None,
) -> tuple[dict, str]:
    """Sample frames for ``[window_start, window_end)``, call Claude.

    Returns ``(parsed_json_object, raw_response_text)`` so playground
    ``axis_calls`` telemetry can show the same raw JSON the UI expands —
    Gemini does this via ``SimplifiedTagsTimeAnalyzer.last_raw_response_text``.

    Raises on yt-dlp/ffmpeg/Anthropic/transport failures or non-JSON responses —
    callers map exceptions to ``(None, error_message)`` the same way as Gemini.
    """
    import anthropic

    if not ctx.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY / BJJ_ANTHROPIC_API_KEY is not configured — "
            "required for Claude highlight_analyze axis calls."
        )
    if window_start >= window_end:
        raise ValueError(
            f"Claude frame path: invalid window [{window_start}, {window_end})"
        )

    _ensure_stream(ctx)
    assert ctx.stream_url is not None and ctx.native_fps is not None

    # sample_frames accepts numeric start/end (ffmpeg -ss/-t); float windows OK.
    sampled = frame_source.sample_frames(
        ctx.stream_url,
        ctx.stream_headers,
        window_start,  # type: ignore[arg-type]
        window_end,  # type: ignore[arg-type]
        ctx.native_fps,
        float(fps),
    )
    if not sampled:
        raise RuntimeError(
            f"Claude frame path: ffmpeg returned zero frames for "
            f"[{window_start}, {window_end}) at fps={fps}"
        )

    kept = subsample_evenly(sampled, MAX_CLAUDE_IMAGES)
    logger.info(
        "claude_frames: window=[%.3f,%.3f) fps=%s sampled=%d kept=%d (cap=%d) model=%s",
        window_start, window_end, fps, len(sampled), len(kept), MAX_CLAUDE_IMAGES, model_id,
    )

    content: list[dict] = []
    for frame in kept:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _jpeg_b64(frame.image),
            },
        })
    content.append({
        "type": "text",
        "text": (
            f"The following {len(kept)} JPEG frames are temporally ordered samples "
            f"from highlight window [{window_start:.3f}, {window_end:.3f}) seconds "
            f"(requested fps={fps}; evenly subsampled to ≤{MAX_CLAUDE_IMAGES} because "
            f"Claude Vision has no native video and caps ~{MAX_CLAUDE_IMAGES} images/"
            f"request). Treat them as one continuous clip — not independent photos.\n\n"
            f"{prompt_text}"
        ),
    })

    client = anthropic.AsyncAnthropic(api_key=ctx.anthropic_api_key)
    create_kwargs: dict = {
        "model": model_id,
        "max_tokens": 4096,
        "system": system_instruction,
        "messages": [{"role": "user", "content": content}],
    }
    create_kwargs.update(opus_thinking_kwargs(thinking))

    response = await client.messages.create(**create_kwargs)
    raw = _extract_text(response)
    if not raw:
        raise RuntimeError("Claude returned empty text content for axis call")
    try:
        data = json.loads(_strip_json_fences(raw))
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Claude non-JSON response: {e}; raw={raw[:500]!r}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"Claude JSON root must be an object, got {type(data).__name__}")
    if "error" in data:
        raise RuntimeError(f"Claude axis call failed: {data['error']}")
    return data, raw
