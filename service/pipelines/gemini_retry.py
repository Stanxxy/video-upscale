"""Centralized async retry-with-backoff for QA-layer Gemini calls.

Root cause (confirmed against ``google/genai/_api_client.py``): the SDK ships
its own tenacity-based retry machinery, but it is only active when the caller
passes ``types.HttpOptions(retry_options=...)`` — nobody in this codebase does
(production ``analyzer.py`` included), so ``retry_args(None)`` resolves to
``{'stop': tenacity.stop_after_attempt(1), 'reraise': True}``: ONE HTTP attempt,
zero automatic retries. A transient ``503 UNAVAILABLE``/``429
RESOURCE_EXHAUSTED``/``500 INTERNAL`` therefore surfaces immediately as a hard
failure today. At per-frame QA detect volume (hundreds of sequential Gemini
calls per run) a handful of transient backend blips over a run's lifetime is
statistically expected, not exceptional.

This module is QA-layer only (``service/routes/qa_vlm.py``,
``service/pipelines/executors.py``, ``service/pipelines/simplified_tags.py``).
It does NOT touch ``analyzer.py``, ``taxonomy_mapper.py``, ``VideoEvent``, or
``service/worker/**`` — those are production surface and stay byte-untouched.

Policy:
- Retry ONLY transient failures: HTTP 503/429/500/502/504/408
  (``TRANSIENT_HTTP_CODES``) plus raw connection/timeout errors that never
  reached an HTTP status at all. Everything else (400 invalid-arg, 401/403
  auth, 404 model-not-found, malformed-response ``ValueError``s, etc.)
  re-raises IMMEDIATELY — never retried, never delayed. The existing 404
  model-fallback in ``qa_vlm._run_detect`` is layered ON TOP of / AFTER this
  retry (a 404 is non-transient here, so it reaches the fallback branch on the
  very first attempt, unchanged).
- Exponential backoff with full jitter, config-driven
  (``ServiceConfig.gemini_retry_*``) — never magic numbers baked in here.
- A 429's ``Retry-After`` response header (when present and numeric) is
  honored in PREFERENCE to the computed backoff delay.
- Every retry is logged (attempt number, status, delay, source) so a run log
  shows "retrying in Ns" instead of a silent stall or an opaque hard failure.
- When retries are genuinely exhausted, the LAST real exception is re-raised
  as-is — never swallowed, never replaced with a fabricated result. Callers'
  existing honest-failure paths (an ``error`` NDJSON event, a degraded frame,
  ``_run_detect``'s ``RuntimeError``) are unchanged.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypeVar

from google.genai import errors as genai_errors

logger = logging.getLogger("service.pipelines.gemini_retry")

T = TypeVar("T")

# Matches (a documented superset of) the google-genai SDK's own default
# retriable set (408/429/500/502/503/504 — see _api_client.py
# ``_RETRY_HTTP_STATUS_CODES``). 429 RESOURCE_EXHAUSTED (rate limit) and 503
# UNAVAILABLE (transient backend outage) are the two modes actually observed
# in production QA runs; 500/502/504/408 are included because they are the
# same class of "the request itself was fine, the backend/gateway had a
# blip" failure. NEVER includes 400/401/403/404 — those are real, immediately
# actionable errors (bad request, auth, model-not-found) that retrying would
# only mask/delay.
TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class GeminiRetryConfig:
    """Retry policy knobs — mirrors ``ServiceConfig.gemini_retry_*`` 1:1 so a
    ``RunContext``/route call site can build one straight from live config
    (``from_service_config``) or fall back to these defaults when constructed
    directly (e.g. existing unit tests that build ``RunContext`` without a
    live ``ServiceConfig``)."""

    max_attempts: int = 5
    initial_delay_s: float = 1.0
    max_delay_s: float = 30.0
    exp_base: float = 2.0
    jitter_s: float = 1.0

    @classmethod
    def from_service_config(cls, config) -> "GeminiRetryConfig":
        return cls(
            max_attempts=config.gemini_retry_max_attempts,
            initial_delay_s=config.gemini_retry_initial_delay_s,
            max_delay_s=config.gemini_retry_max_delay_s,
            exp_base=config.gemini_retry_exp_base,
            jitter_s=config.gemini_retry_jitter_s,
        )


def is_transient_error(e: BaseException) -> bool:
    """True for a retryable transport/backend blip; False for a real,
    actionable error that should fail fast (or fall through to
    ``qa_vlm``'s existing 404 model-fallback)."""
    code = getattr(e, "code", None)
    if isinstance(e, genai_errors.APIError) and code in TRANSIENT_HTTP_CODES:
        return True
    # Errors that never got far enough to carry an HTTP status at all.
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover — httpx is a hard genai dependency
        return False
    return isinstance(e, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError))


def _retry_after_seconds(e: BaseException) -> Optional[float]:
    """Extract a server-provided ``Retry-After`` delay (seconds) from an
    ``APIError``'s response headers, if present and numeric.

    Returns ``None`` when absent/unparsable (HTTP-date form is not handled —
    real-world Gemini 429s send a plain integer-seconds value) so the caller
    falls back to the computed exponential-backoff delay. Never raises.
    """
    response = getattr(e, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = None
    for key in ("retry-after", "Retry-After"):
        try:
            raw = headers.get(key)
        except AttributeError:
            return None
        if raw is not None:
            break
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _backoff_delay(attempt: int, retry_config: GeminiRetryConfig) -> float:
    """Exponential backoff with full jitter, capped at ``max_delay_s``.

    ``attempt`` is 1-indexed (the Nth attempt that just failed). Base delay
    grows as ``initial_delay_s * exp_base ** (attempt - 1)``; a uniform random
    jitter in ``[0, jitter_s]`` is added on top, then the total is capped.
    """
    base = retry_config.initial_delay_s * (retry_config.exp_base ** (attempt - 1))
    jitter = random.uniform(0.0, retry_config.jitter_s) if retry_config.jitter_s > 0 else 0.0
    return min(retry_config.max_delay_s, base + jitter)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    op_name: str,
    retry_config: GeminiRetryConfig,
) -> T:
    """Call ``fn()`` (a zero-arg async callable performing ONE Gemini API
    attempt), retrying transient failures with exponential backoff + jitter.

    - Non-transient errors re-raise IMMEDIATELY (no delay, no retry).
    - Transient errors retry up to ``retry_config.max_attempts`` TOTAL
      attempts (including the first). A 429's ``Retry-After`` header, when
      present, is honored in preference to the computed backoff delay.
    - When retries are exhausted, the LAST real exception is re-raised as-is.
      Nothing is fabricated; the caller's existing honest-failure path
      (an ``error`` NDJSON event, ``_run_detect``'s ``RuntimeError``, etc.)
      is unchanged.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001 — inspected below, never swallowed
            transient = is_transient_error(e)
            if not transient or attempt >= retry_config.max_attempts:
                if attempt > 1:
                    logger.error(
                        "%s: giving up after %d/%d attempt(s): %s",
                        op_name, attempt, retry_config.max_attempts, e,
                    )
                raise
            delay = _retry_after_seconds(e)
            source = "Retry-After header"
            if delay is None:
                delay = _backoff_delay(attempt, retry_config)
                source = "computed backoff"
            logger.warning(
                "%s: transient error on attempt %d/%d (%s); retrying in %.2fs (%s)",
                op_name, attempt, retry_config.max_attempts, e, delay, source,
            )
            await asyncio.sleep(delay)
