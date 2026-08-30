"""Direct-Moonshot native-video Kimi K3 client lane, per
`GATES_RERUN_DESIGN.md` §9. **This is NOT a reuse of `openrouter_client.py`'s
frames-only path** — §9's whole premise is that OpenRouter's aggregation
layer does not expose Moonshot's own native-video capability, which must be
re-probed independently (confirmed true, see §9.1 probe results below).

## §9.1 probe results (2026-07-26, real API calls against `https://api.
moonshot.ai/v1`, verified live, never guessed):

- **Step 0 — model id**: `GET /v1/models` confirms the real, API-facing id
  is exactly `kimi-k3` (matches the assumed slug) — model metadata:
  `"supports_video_in": true`, `"reasoning_efforts": {"support": true,
  "valid_efforts": ["low","high","max"], "default_effort":"max"}`.
- **Step 1 — video capability, THREE mechanisms tried, real evidence for
  each**:
  1. `POST /v1/files` with `purpose="video"` — **succeeds** (200, real file
     id, `status:"ready"`) — this IS a real, officially-supported purpose
     value (confirmed via the API's own 400 error body listing all valid
     purposes: `file-extract, batch, batch_output, lambda, image, video`).
  2. Referencing that uploaded file via a `video_url` content block
     (`{"file_id":...}`, or the file's own `/v1/files/{id}/content` URL) —
     **FAILS**, `400 "unsupported video url: <whatever url string>"` for
     every URL-shaped value tried, including the file's own content-fetch
     URL.
  3. `video_url` pointed at a live public YouTube watch URL (the design's
     Branch B1, "the branch that answers the w17 question") — **FAILS**,
     the SAME `400 "unsupported video url: <url>"` — Moonshot's direct API
     does **NOT** fetch arbitrary external URLs (not even YouTube) any more
     than it accepts an internal file-content URL.
  4. `video_url` with an **inline `data:video/mp4;base64,...` URI** (the
     RAW video bytes embedded directly in the request) — **SUCCEEDS**,
     produces a real, content-grounded answer (verified: the response
     correctly named the actual athletes shown on-screen and referenced the
     visible match score — not a generic/hallucinated non-answer).

  **Conclusion: native video works, via mechanism (4) ONLY.** This is not a
  clean match to either of the design's anticipated "Branch A" (file_id
  reference — tried, failed) or "Branch B1" (external URL fetch — tried,
  failed) — it is closer in spirit to "Branch B2" (embed the actual local
  video content) but via a simpler inline-data-URI mechanism rather than a
  locally-served HTTP URL (no local HTTP server needed).

  **w17 resolution (§9.4's conditional): the SAME structural exclusion
  applies as if Branch A had been the working mechanism.** The only working
  mechanism (4) requires the RAW VIDEO BYTES to exist locally before the
  call can be made — exactly the same yt-dlp dependency Branch A would have
  had. Branch B1 (the mechanism that could have bypassed the local-download
  requirement for `w17` specifically) was tried and explicitly, unambiguously
  fails on "not an accepted URL" grounds, not merely "can't reach this
  particular video." `w17` therefore CANNOT be attempted by this lane either
  — same permanent, external, non-retriable SME-takedown block already
  confirmed twice (Gate 3, frontier lanes).
- **Step 2 — reasoning parameter**: real top-level field name confirmed via
  Moonshot's own pricing/model docs (`https://platform.moonshot.ai/docs/
  pricing/chat-k3.md`, fetched live): `reasoning_effort` (values `low`/
  `high`/`max`, default `max`) — NOT OpenRouter's `reasoning:{"effort":...}`
  nesting. Kimi K3 "always reasons" (`supports_thinking_type:"only"` in the
  models endpoint) — there is no way to disable it. Confirmed HONORED via a
  real pilot call: `usage.completion_tokens_details.reasoning_tokens = 510`
  (nonzero) when `reasoning_effort="high"` was requested with `max_tokens=
  16384` — the SAME max_tokens-vs-reasoning-budget issue already fixed for
  the OpenRouter/Kimi and Anthropic/Opus lanes recurred here identically at
  the default `max_tokens=500` (empty final `content`, full budget consumed
  by `reasoning_content`) — fixed the same way, by raising `max_tokens`.
  **Per the founder's "highest confirmed-honored level" standing rule, this
  lane uses `reasoning_effort="high"`** (not `"max"`) — see the settings
  rationale below the pricing block.

## Real pricing (verified live against `https://platform.moonshot.ai/docs/
pricing/chat-k3.md` on 2026-07-26 — NOT estimated, NOT the OpenRouter
per-call average used only as a pre-probe reference anchor in §9.5):
`kimi-k3` = **$3.00/MTok input (cache-miss), $0.30/MTok input (cache-hit),
$15.00/MTok output** (output price covers reasoning tokens too — the
model's own docs describe `completion_tokens` as the billed total, with
`completion_tokens_details.reasoning_tokens` as a breakdown, the same
"breakdown of, not addition to" pattern already confirmed for Anthropic).
This harness never repeats an identical prompt prefix across calls in a way
that would hit Moonshot's automatic context cache, so the cache-MISS input
rate is used uniformly (same reasoning as `anthropic_direct.py`'s cache
irrelevance note).
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.s12_gates.anthropic_direct import _extract_final_json_object
from eval.adcc.predict import media_resolver

logger = logging.getLogger("eval.adcc.s12_gates.kimi_native_client")

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_K3_MODEL_ID = "kimi-k3"  # verified live, §9.1 Step 0 — never assumed
PROVIDER_ROUTE = "moonshot_direct"

KIMI_K3_INPUT_CACHE_MISS_USD_PER_MTOK = 3.00
KIMI_K3_OUTPUT_USD_PER_MTOK = 15.00

# Chosen per the founder's "highest CONFIRMED-HONORED level" rule — "max" is
# technically higher, but only "high" has been directly evidence-confirmed
# via a real nonzero reasoning_tokens count in this harness's own pilot call
# (see module docstring). Using the confirmed level, not the nominally
# highest unconfirmed one, matches this program's "never claim max thinking
# without confirmed evidence" discipline applied consistently.
REASONING_EFFORT = "high"

MAX_TOKENS = 16384  # [FIX, same class as openrouter_client.py/anthropic_direct.py] avoids the
# reasoning-budget-exhausts-before-final-answer truncation observed at the provider default.


class MoonshotAPIError(RuntimeError):
    """Raised on any non-2xx Moonshot response — never swallowed."""


def kimi_native_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * KIMI_K3_INPUT_CACHE_MISS_USD_PER_MTOK
        + (output_tokens / 1_000_000) * KIMI_K3_OUTPUT_USD_PER_MTOK
    )


def _load_moonshot_key() -> str:
    env_path = gc.S12_GATES_DIR.parent.parent.parent / ".env"
    for line in open(env_path):
        if line.startswith("KIMI_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise RuntimeError(f"kimi_native_client: KIMI_API_KEY not found in {env_path}")


async def ensure_local_clip(window: dict, *, cache_dir: Path = reroute.MEDIA_CACHE_DIR) -> tuple[Path, float]:
    """Returns `(normalized_mp4_path, sec_start)` — `sec_start` is the
    ORIGINAL video's absolute second the downloaded clip's own 0:00
    corresponds to (needed to convert the model's clip-relative `start_s`/
    `end_s` back to absolute match-clock seconds — same discipline as
    `files_api_reroute.ReroutedUpload`). Reuses (never reimplements)
    `files_api_reroute.download_section` + `media_resolver.
    build_mp4_derivative` — the SAME download+normalize pipeline the §G.1
    reroute lane already built and validated. **Cache-aware**: if a
    `{window_id}.normalized.mp4` already exists in `cache_dir` (either from
    earlier §G.1 reroute work, or a prior run of this driver), it is reused
    as-is — zero new yt-dlp spend/time, matching this program's established
    "reuse cached local media" discipline (Gate 4's reference-crop fix)."""
    window_id = window["window_id"]
    norm_path = cache_dir / f"{window_id}.normalized.mp4"
    sec_start_path = cache_dir / f"{window_id}.sec_start.txt"
    if norm_path.exists() and sec_start_path.exists():
        return norm_path, float(sec_start_path.read_text().strip())

    src_path, _section_spec, sec_start = await reroute.download_section(
        window["video_url_watch"], window["window_start_s"], window["window_end_s"], cache_dir, window_id=window_id,
    )
    await media_resolver.build_mp4_derivative(src_path, norm_path)
    sec_start_path.write_text(repr(sec_start))
    return norm_path, sec_start


def video_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:video/mp4;base64,{b64}", len(raw)


async def call_moonshot_chat(
    *, api_key: str, model: str, system_instruction: Optional[str], prompt_text: str,
    video_data_url: str, reasoning_effort: Optional[str] = REASONING_EFFORT,
    max_tokens: int = MAX_TOKENS, timeout_s: float = 240.0,
) -> tuple[dict, str, dict, float]:
    """ONE Moonshot direct chat-completions call with native video. Returns
    `(parsed_json, raw_text, usage_dict, latency_ms)`. `usage_dict` includes
    real `input_tokens`/`output_tokens`/`reasoning_tokens`/`cost_usd`
    (computed from the verified rate card, never estimated). Raises
    `MoonshotAPIError` on non-2xx; raises `RuntimeError` on non-JSON/
    unparseable content — never a silent fallback. Reuses
    `anthropic_direct._extract_final_json_object` for the SAME "model kept
    talking after answering" recovery already built and tested for Opus —
    a strict superset of a bare `json.loads` (tries the fast path first),
    applied prospectively here (not because Kimi K3's failure was CONFIRMED
    to share Opus's shape — it wasn't — but because the recovery function
    can never do worse than a bare parse and is real, tested robustness)."""
    content: list[dict] = [{"type": "video_url", "video_url": {"url": video_data_url}}]
    content.append({"type": "text", "text": prompt_text})
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": content})

    payload: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort  # Moonshot's OWN top-level field, confirmed §9.1 Step 2

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(
            f"{MOONSHOT_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code != 200:
        raise MoonshotAPIError(f"Moonshot {resp.status_code}: {resp.text[:2000]}")

    body = resp.json()
    if "error" in body:
        raise MoonshotAPIError(f"Moonshot call failed: {body['error']}")
    choice = (body.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason")
    raw_text = (choice.get("message") or {}).get("content")
    if finish_reason == "length" or not raw_text:
        raise RuntimeError(
            f"Moonshot response truncated/empty before a final answer (finish_reason={finish_reason!r}) — "
            f"max_tokens={max_tokens} was insufficient, or the model never emitted final `content` "
            f"(reasoning_content len={len((choice.get('message') or {}).get('reasoning_content') or '')}). "
            f"model={model}."
        )

    stripped = raw_text.strip()
    if "```json" in stripped:
        stripped = stripped.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in stripped:
        stripped = stripped.split("```", 1)[1].split("```", 1)[0]
    try:
        data = json.loads(stripped.strip())
    except (ValueError, TypeError):
        data = _extract_final_json_object(stripped)
    if not isinstance(data, dict):
        raise RuntimeError(f"Moonshot JSON root must be an object, got {type(data).__name__}")

    usage = body.get("usage") or {}
    input_tokens = usage.get("prompt_tokens") or 0
    output_tokens = usage.get("completion_tokens") or 0
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    cost_usd = kimi_native_cost_usd(input_tokens, output_tokens)
    usage_dict = {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens, "cost_usd": cost_usd,
    }
    return data, raw_text, usage_dict, latency_ms
