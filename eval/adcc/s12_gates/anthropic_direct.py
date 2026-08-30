"""Local usage-surfacing wrapper for direct-Anthropic Opus 4.8 axis calls
(Bezos precondition 1, 2026-07-26 frontier-lane full-batch authorization).

Does **not** edit `service/pipelines/claude_frames.py` — reuses its exact
frame-sampling + message-construction PRIVATE helpers (`_ensure_stream`,
`_jpeg_b64`, `_extract_text`, `_strip_json_fences`, `opus_thinking_kwargs`,
`subsample_evenly`, `MAX_CLAUDE_IMAGES`) so the request payload this module
sends is the SAME shape as `claude_frames.analyze_axis_json`'s. The message-
construction block below (image content list + trailing framing-sentence
text block) is necessarily duplicated rather than called through — that
function's return signature is `tuple[dict, str]` (no `response.usage`),
which is the entire reason this wrapper exists; duplicating ~15 lines of
dict-building here is a smaller, more auditable footprint than editing the
production return contract or reaching into a closure for it.

Pricing (verified live against
https://docs.anthropic.com/en/docs/about-claude/pricing on 2026-07-26):
Claude Opus 4.8 — **$5/MTok base input tokens, $25/MTok output tokens**.
Thinking tokens are billed AS output tokens, not an addition to them — the
Anthropic Python SDK's `Usage.output_tokens_details.thinking_tokens` field
is an explicit **breakdown of, not an addition to**, `usage.output_tokens`
(confirmed via the SDK's own type docstring, `anthropic.types.usage.
OutputTokensDetails`, and https://docs.anthropic.com/en/docs/build-with-
claude/extended-thinking's "monitor ... which reports how many of the
BILLED output tokens were internal reasoning"). No prompt caching is used
anywhere in this eval harness (fresh window per call, no repeated system
prompts across a cache TTL) — every input token this harness sends is a
"base input token" by construction, so cache-write/cache-read rates are
irrelevant here and intentionally omitted from the cost formula.
"""
from __future__ import annotations

import json as _json
import logging
import time
from typing import Optional

from eval.adcc.s12_gates import frame_source_retry
from service.pipelines import claude_frames, frame_source
from service.pipelines.executors import RunContext
from service.routes.qa_vlm import _resolve_stream_url

logger = logging.getLogger("eval.adcc.s12_gates.anthropic_direct")


def _ensure_stream_with_retry(ctx: RunContext) -> None:
    """[FIX §9.3] Harness-local mirror of `claude_frames._ensure_stream` —
    duplicated (not called through) SPECIFICALLY so the `probe_native_fps`
    step can use the bounded-retry wrapper (`frame_source_retry`) instead of
    a single-shot call, without editing `claude_frames.py` (production).
    Same short-circuit contract (no-op if already resolved)."""
    if ctx.stream_url and ctx.native_fps:
        return
    stream_url, headers = _resolve_stream_url(ctx.youtube_url)
    native_fps = frame_source_retry.probe_native_fps_with_retry(stream_url, headers)
    ctx.stream_url = stream_url
    ctx.stream_headers = headers
    ctx.native_fps = native_fps

OPUS_48_INPUT_USD_PER_MTOK = 5.0
OPUS_48_OUTPUT_USD_PER_MTOK = 25.0  # includes thinking tokens — see module docstring


def _extract_final_json_object(raw: str) -> dict:
    """[FIX 2026-07-26, GATES_RERUN_DESIGN.md §9.3, root-caused against the
    ACTUAL captured Opus `w6` raw text before writing this — mandatory first
    step, not skipped] `w6`'s raw text was NOT the generic "valid JSON +
    trailing chatter" shape LeCun's instruction described as the assumed
    default case — it was a more specific, more important variant: an
    INCOMPLETE first JSON object (missing the required `outcome` field),
    followed by the model's own self-correction prose ("Wait, the outcome
    field is required."), followed by a SECOND, complete JSON object. A
    naive `json.JSONDecoder().raw_decode()` on the FIRST match (LeCun's
    literal instruction) would silently return the model's own
    self-acknowledged-WRONG draft, not its real answer — worse than the
    original bare `json.loads` failure, which at least failed loud instead
    of returning bad data. Fix: extract ALL complete top-level JSON objects
    via repeated `raw_decode`, and take the LAST one (the model's most
    recent/corrected attempt — correct for both the "answer + trailing
    chatter" case, where first==last, and the observed "draft + correction +
    final" case, where only the last one is right). Every multi-object case
    is logged at WARNING with all discarded object(s) shown in full — never
    silently normalized away."""
    decoder = _json.JSONDecoder()
    text = raw.strip()
    objects: list[dict] = []
    idx = 0
    n = len(text)
    while idx < n:
        # skip whitespace/non-JSON-start chars (prose between objects) to find
        # the next plausible object-start brace, rather than requiring objects
        # to be perfectly adjacent.
        brace_idx = text.find("{", idx)
        if brace_idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, brace_idx)
        except ValueError:
            idx = brace_idx + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx = end_idx
    if not objects:
        raise RuntimeError(f"anthropic_direct: no complete JSON object found in response; raw={raw!r}")
    if len(objects) > 1:
        logger.warning(
            "anthropic_direct: response contained %d complete JSON objects (model kept talking/"
            "self-corrected after an initial answer) — using the LAST one, discarding %d earlier "
            "object(s): %s",
            len(objects), len(objects) - 1, _json.dumps(objects[:-1]),
        )
    return objects[-1]


def opus_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * OPUS_48_INPUT_USD_PER_MTOK
        + (output_tokens / 1_000_000) * OPUS_48_OUTPUT_USD_PER_MTOK
    )


async def analyze_axis_json_with_usage(
    ctx: RunContext,
    *,
    model_id: str,
    window_start: float,
    window_end: float,
    prompt_text: str,
    system_instruction: str,
    fps: int,
    thinking: Optional[str] = None,
) -> tuple[dict, str, dict, float]:
    """Same call as `claude_frames.analyze_axis_json` — byte-identical
    request construction (frame sampling, subsampling, image+framing-text
    content, `opus_thinking_kwargs`) — but returns
    `(data, raw_text, usage_dict, latency_ms)` where `usage_dict` carries
    REAL measured `input_tokens`/`output_tokens`/`thinking_tokens`/`cost_usd`
    from `response.usage`, never estimated or fabricated. Raises on the same
    failure classes as `analyze_axis_json` (yt-dlp/ffmpeg/Anthropic/
    transport/non-JSON) — no fallback, no swallowed errors."""
    import anthropic

    if not ctx.anthropic_api_key:
        raise RuntimeError(
            "anthropic_direct: ANTHROPIC_API_KEY / BJJ_ANTHROPIC_API_KEY is not configured — "
            "required for direct-Anthropic Opus axis calls."
        )
    if window_start >= window_end:
        raise ValueError(f"anthropic_direct: invalid window [{window_start}, {window_end})")

    _ensure_stream_with_retry(ctx)
    assert ctx.stream_url is not None and ctx.native_fps is not None

    sampled = frame_source.sample_frames(
        ctx.stream_url, ctx.stream_headers, window_start, window_end, ctx.native_fps, float(fps),
    )
    if not sampled:
        raise RuntimeError(
            f"anthropic_direct: ffmpeg returned zero frames for [{window_start}, {window_end}) at fps={fps}"
        )

    kept = claude_frames.subsample_evenly(sampled, claude_frames.MAX_CLAUDE_IMAGES)

    # Image-then-text ordering + the SAME temporal-continuity framing sentence as
    # claude_frames.py:184-196 (byte-faithful — copied, not paraphrased).
    content: list[dict] = []
    for frame in kept:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": claude_frames._jpeg_b64(frame.image)},
        })
    content.append({
        "type": "text",
        "text": (
            f"The following {len(kept)} JPEG frames are temporally ordered samples "
            f"from highlight window [{window_start:.3f}, {window_end:.3f}) seconds "
            f"(requested fps={fps}; evenly subsampled to ≤{claude_frames.MAX_CLAUDE_IMAGES} because "
            f"Claude Vision has no native video and caps ~{claude_frames.MAX_CLAUDE_IMAGES} images/"
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
    create_kwargs.update(claude_frames.opus_thinking_kwargs(thinking))

    t0 = time.perf_counter()
    response = await client.messages.create(**create_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # [Real failure mode found in this turn's Kimi/OpenRouter lane — max_tokens
    # truncation before a final answer, see openrouter_client.py's matching FIX
    # note] Check explicitly here too rather than let it surface as a generic
    # JSON-parse error, even though `max_tokens=4096` is kept byte-identical to
    # claude_frames.py's own value (this wrapper's whole purpose is an
    # identical request construction — see module docstring) and Opus's own
    # smoke call did not trip this.
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            f"anthropic_direct: response truncated (stop_reason='max_tokens') before a final answer — "
            f"max_tokens=4096 was insufficient for this call's thinking+answer length. model={model_id}."
        )

    raw = claude_frames._extract_text(response)
    if not raw:
        raise RuntimeError("anthropic_direct: Claude returned empty text content for axis call")
    stripped = claude_frames._strip_json_fences(raw)
    try:
        # [FIX §9.3] first try the strict, fast path (single clean JSON object,
        # the common case) — only fall back to the multi-object recovery scan
        # (see `_extract_final_json_object`'s docstring for why LAST not FIRST)
        # if the strict parse actually fails, so a clean response never pays
        # the scan's extra cost/complexity.
        data = _json.loads(stripped)
    except (ValueError, TypeError) as e:
        try:
            data = _extract_final_json_object(stripped)
        except RuntimeError as e2:
            raise RuntimeError(
                f"anthropic_direct: Claude non-JSON response, no recoverable JSON object found: "
                f"{e2}; original strict-parse error: {e}; raw={raw[:4000]!r}"
            ) from e2
    if not isinstance(data, dict):
        raise RuntimeError(f"anthropic_direct: Claude JSON root must be an object, got {type(data).__name__}")
    if "error" in data:
        raise RuntimeError(f"anthropic_direct: Claude axis call failed: {data['error']}")

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    output_details = getattr(usage, "output_tokens_details", None)
    thinking_tokens = getattr(output_details, "thinking_tokens", None) if output_details is not None else None
    cost_usd = opus_cost_usd(input_tokens, output_tokens)
    usage_dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "cost_usd": cost_usd,
    }
    return data, raw, usage_dict, latency_ms
