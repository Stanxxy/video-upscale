"""Bounded retry wrapper around `service.pipelines.frame_source.
probe_native_fps` — HARNESS code only (`GATES_RERUN_DESIGN.md` §9.3):
"never edit `service/pipelines/frame_source.py` itself, matching this
program's unbroken discipline of never modifying production code for an
eval-only robustness fix."

Root cause: the frontier-lane full batch (both Kimi/OpenRouter and Opus/
direct-Anthropic) observed the `ffprobe fps probe failed (rc=1)` failure
signature 5 times across 75 total calls (~6.7% of calls, concentrated at
this one specific ffprobe-against-a-freshly-resolved-googlevideo-signed-URL
step) — a real, recurring transient rate, not a one-off. `probe_native_fps`
itself deliberately never guesses a default fps on failure (a wrong assumed
fps corrupts every downstream frame_index) — this wrapper does not change
that discipline; it only retries the SAME real call a bounded number of
times before giving up, exactly the same category of transient-fault
handling as `GeminiRetryConfig` (Gemini path) and the OpenRouter 429 retry
already added this turn — never a fallback that fabricates or estimates a
result.
"""
from __future__ import annotations

import logging
import time

from service.pipelines import frame_source

logger = logging.getLogger("eval.adcc.s12_gates.frame_source_retry")

RETRY_DELAYS_S = (2.0, 4.0, 8.0)  # 3 bounded attempts beyond the first, per §9.3


def probe_native_fps_with_retry(stream_url: str, headers: dict) -> float:
    """Same contract as `frame_source.probe_native_fps` (raises RuntimeError
    on failure, never guesses a default) — retries up to
    `len(RETRY_DELAYS_S)` additional times with fixed backoff on failure,
    logging each retry at WARNING. Re-raises the LAST real exception if all
    attempts are exhausted — never swallowed, never silently defaulted."""
    last_exc: Exception | None = None
    for attempt in range(len(RETRY_DELAYS_S) + 1):
        try:
            return frame_source.probe_native_fps(stream_url, headers)
        except Exception as e:
            last_exc = e
            if attempt < len(RETRY_DELAYS_S):
                delay = RETRY_DELAYS_S[attempt]
                logger.warning(
                    "frame_source_retry: probe_native_fps failed (attempt %d/%d) — %s — retrying in %.0fs",
                    attempt + 1, len(RETRY_DELAYS_S) + 1, e, delay,
                )
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc
