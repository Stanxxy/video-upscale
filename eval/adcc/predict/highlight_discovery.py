"""Highlight-discovery Gemini call (frozen contract §D) — a NEW, PEER
prediction module. This file is deliberately NOT built on top of
``service/pipelines/executors.py``/``run_pipeline`` — it builds its OWN
``genai.Client`` (mirroring, not importing, ``RunContext.gemini_client``'s
~5-line property below) and its own thinking-config mapping (mirroring, not
importing, ``service/routes/qa_vlm.thinking_config_for``). This is a
deliberate CEO-ruled architecture decision, not an oversight: importing
``executors.py``/``run_pipeline`` here would drag this "any kind of
highlight, coarse pass" harness into chunk-segment-tags' two-pass
(PASS-1 rough-scan + PASS-2 per-chunk) machinery, which this feature has no
use for (1 Gemini call ≈ 1 match, no chunk map, no worthy-chunk loop) — see
``CALL_BUDGET_ESTIMATE.md``/INS-090 for why the eval surface's import graph
is deliberately kept narrow. Every production file this module reads from
(``service/pipelines/gemini_retry.py``, ``service/pipelines/simplified_tags.
extract_usage_metadata``) is a leaf helper with no ``service.routes`` barrel
entanglement — never ``service.pipelines.executors``, never
``service.routes.qa_vlm`` directly.

**Output-envelope discipline (frozen contract §D — no fallback, no
fabrication).** Gemini's structured output (``response_schema`` below) is a
JSON array of ``{start_s, end_s}`` (a point when ``start_s == end_s``),
expressed CLIP-RELATIVE (seconds from the start of the scoped clip, 0-based)
— a harness-owned output-envelope instruction, distinct from the founder's
free-text ``system_prompt``/``followup_prompt`` (those drive WHAT counts as
highlight-worthy; the envelope format is fixed by this module, never left to
the model's own judgement). Any response this module cannot trust as real
highlight data — non-JSON text, a non-array, or an item missing/malformed
``start_s``/``end_s`` — raises ``HighlightDiscoveryError`` and SURFACES as an
error, never regex-scraped or coerced into a best-guess shape.

**Scope + timestamp normalization (frozen contract §D, INS-085).** Every
call is scoped to just the ONE match's window (``scope_policy.
single_match_scope``/``multi_match_scope``, reused verbatim — never the whole
VOD, even for a match inside a shared multi-match VOD, per INS-085's
"absolute-clock trap"), passed as the Gemini ``video_metadata`` start/end
offset exactly like ``executors.chunk_segment_node``/``_analyze_video_window``
already do. Gemini's clip-relative response is normalized back to
VOD-absolute (``window_start_s + clip_relative_seconds``) by THIS module
before it is ever handed to the scorer — the scorer (``highlight_scoring.
score_highlights``) never sees a clip-relative timestamp, only real
VOD-absolute ``t_s``-comparable seconds.

**Timestamp critique/refinement (frozen contract, this increment).** Gemini's
coarse discovery pass exhibits a LATE bias on decisive instants (especially
submissions — it tends to anchor on the tap/celebration rather than the lock
that actually finishes the exchange). ``refine_highlight_timestamp`` below is
a SEPARATE, per-highlight Gemini call (own prompt/schema/error type) that
re-checks a small, PADDED, front-loaded window around each discovered
highlight and asks Gemini to confirm the action and mark the true decisive
instant (the LOCK for a submission, never the tap). Orchestrated entirely
from ``run_highlight_batch_with_global_cap`` — that function drives the
per-highlight loop, the cap/stop checks, and the accept/reject decision; it
never inlines ``generate_content`` itself. No fabrication: an unconfirmed or
suspicious (edge/out-of-bounds) refinement is REJECTED and the ORIGINAL
discovery timestamp is kept, flagged ``refinement_failed=True`` — refinement
can only replace a timestamp with a MORE trustworthy one, never guess.

**Live-bug root cause + fix (2026-07-18).** A real user's ``/highlight/run``
click 400'd with a bare, no-field-name ``google.genai.errors.ClientError:
400 INVALID_ARGUMENT`` — the classic signature of a rejected ``file_uri``
host/path shape, not a named-field config error. Root cause: the ADCC gold
snapshot's ``video_url`` field (``MatchTarget.video_url``, threaded straight
through from ``video_links.json``) is, by INS-084's own design, a YouTube
``https://www.youtube.com/embed/<id>`` URL (the corpus's on-disk file-stem
join key, ``md5(embed_url)`` — never meant to be Gemini-facing). Gemini's
native YouTube ``file_uri`` ingestion only recognizes the CANONICAL
``https://www.youtube.com/watch?v=<id>`` form — confirmed by the ONE calling
context that has ever actually exercised a real, successful Gemini video
call in this codebase, the QA VLM Studio client (``qa_client/vlm_studio.
html``'s ``loadVideo()``), which ALWAYS normalizes to ``watch?v=`` before the
backend ever sees a URL, regardless of what the human pasted. The tag-batch
prediction-capture path (``global_cap_driver.py``/``run_predictions.py``)
threaded the SAME un-normalized ``target.video_url`` into
``RunContext.youtube_url`` and had the identical latent exposure — fixed
there too (2026-07-18, same day), by normalizing UPSTREAM of ``RunContext``
in eval/ so ``service/pipelines/executors.py`` (co-tenant, byte-untouchable)
needs zero changes and always receives an already-canonical URL. The actual
extraction/normalization logic now lives in the shared leaf module
``eval/adcc/predict/youtube_url.py`` (moved out of this file specifically so
both this module AND ``global_cap_driver.py``/``run_predictions.py`` can
import it without a circular import — this module already imports
``global_cap_driver`` for ``BatchRunState``). ``normalize_youtube_watch_url``
below is a thin re-export that keeps THIS module's public contract
(``HighlightDiscoveryError``, exact message) byte-identical to before the
extraction — never a silent guess: an unparseable URL raises, never a
best-effort fallback.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types

from eval.adcc.predict.cost_model import estimate_highlight_call_cost_usd
from eval.adcc.predict.global_cap_driver import BatchRunState, PASS1_PLACEHOLDER_USD, event_cost_usd
from eval.adcc.predict.run_predictions import MatchTarget
from eval.adcc.predict.youtube_url import YoutubeUrlError
from eval.adcc.predict.youtube_url import normalize_youtube_watch_url as _normalize_youtube_watch_url
from eval.adcc.predict.scope_policy import MAX_SLICE_S, ScopeWindow, multi_match_scope, single_match_scope
from service.pipelines import gemini_retry, simplified_tags

logger = logging.getLogger("eval.adcc.predict.highlight_discovery")

__all__ = [
    "HIGHLIGHT_MODELS",
    "DEFAULT_HIGHLIGHT_MODEL",
    "THINKING_LEVELS",
    "LONG_WINDOW_RISK_THRESHOLD_S",
    "OUT_OF_BOUNDS_TOLERANCE_S",
    "FPS_OPTIONS",
    "DEFAULT_FPS",
    "FRAME_QUALITY_OPTIONS",
    "DEFAULT_FRAME_QUALITY",
    "DEFAULT_PAD_BEFORE_S",
    "DEFAULT_PAD_AFTER_S",
    "WIDEN_PAD_BEFORE_S",
    "DEFAULT_REFINE_FPS",
    "REFINE_FRAME_QUALITY",
    "REFINE_PLACEHOLDER_USD",
    "HighlightDiscoveryError",
    "RefinementError",
    "RefinementPreCallError",
    "resolve_highlight_window",
    "is_long_window_risk",
    "validate_fps",
    "resolve_media_resolution",
    "normalize_youtube_watch_url",
    "build_gemini_client",
    "highlight_thinking_config",
    "build_highlight_schema",
    "build_refinement_schema",
    "REJECTION_REASONS",
    "validate_highlight_candidate",
    "discover_match_highlights",
    "refine_highlight_timestamp",
    "decide_refinement_acceptance",
    "run_highlight_batch_with_global_cap",
]

# Bezos fast-follow (non-blocking, evaluator-converged): this module
# deliberately sends a match's WHOLE ``ScopeWindow`` in one Gemini call with
# NO over-cap slicing (frozen contract §D says "1 Gemini call == 1 match" —
# adding chunk-segment-tags-style slicing here is CEO-ruled OUT of scope, not
# a gap to silently fill). That means a long match's window is a real,
# UNMITIGATED recall-degradation risk (a single native-video call covering
# many minutes gives Gemini more to miss, with no error, no signal, nothing
# — just a quietly worse P/R/F1). ``LONG_WINDOW_RISK_THRESHOLD_S`` reuses
# (never redefines) ``chunk_segment.MAX_SCOPE_SEC`` via ``scope_policy.
# MAX_SLICE_S`` -- the same 720s bound production's own two-pass pipeline
# treats as "needs slicing" -- purely as a WARNING surfaced in ``/highlight/
# plan`` and rendered in the client, never a gate and never a new slicing
# mechanism.

# Gemini >=3.1 family only (frozen contract §D). A small, LOCAL constant —
# deliberately NOT imported from ``service/routes/qa_vlm.EVENT_MODELS`` (that
# import would pull the same module whose barrel-entanglement INS-090
# documents; a 3-line duplicated allowlist here is cheaper than that
# coupling, and this module's docstring above is the cross-reference that
# keeps the two lists honest if the model family ever changes).
HIGHLIGHT_MODELS: list[str] = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"]
DEFAULT_HIGHLIGHT_MODEL: str = HIGHLIGHT_MODELS[0]

THINKING_LEVELS: tuple[str, ...] = ("off", "low", "medium", "high")

# Re-exported (never re-defined) from scope_policy.py, which itself re-
# exports chunk_segment.MAX_SCOPE_SEC verbatim -- this module never touches
# that production constant, only reads it, same discipline as scope_policy.py
# already applies (see that module's "MAX_SLICE_S" docstring note).
LONG_WINDOW_RISK_THRESHOLD_S: float = MAX_SLICE_S

# fps + frame_quality controls (CEO-frozen spec, this increment). fps is a
# CLOSED enum (CEO overruled free-numeric input — bounded set, no runaway
# cost from an arbitrary caller-supplied fps). frame_quality's "default"
# leaves ``GenerateContentConfig.media_resolution`` UNSET -- exactly
# TODAY'S code path before this increment existed (verified byte-identical:
# ``types.GenerateContentConfig(media_resolution=None) ==
# types.GenerateContentConfig()`` — see the pydantic field's own ``None``
# default). low/medium/high explicitly set ``types.MediaResolution``.
FPS_OPTIONS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
DEFAULT_FPS: float = 1.0

FRAME_QUALITY_OPTIONS: tuple[str, ...] = ("default", "low", "medium", "high")
DEFAULT_FRAME_QUALITY: str = "default"

_MEDIA_RESOLUTION_BY_QUALITY: dict[str, "types.MediaResolution"] = {
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}


class HighlightDiscoveryError(ValueError):
    """Raised for any Gemini response this harness cannot trust as real
    highlight data, or a caller-supplied ``model``/``thinking_level`` outside
    the allowed set — never silently coerced, never regex-scraped, never
    defaulted around (frozen contract §D: "no regex-scraping, no freeform
    fallback")."""


class RefinementError(HighlightDiscoveryError):
    """Raised for any refinement-call response this harness cannot trust —
    its OWN error type (frozen contract, this increment: "own prompt/schema/
    error type"), kept as a ``HighlightDiscoveryError`` subclass so
    ``run_highlight_batch_with_global_cap``'s existing per-call error
    handling (count as spent, surface, continue) covers refinement calls
    with NO separate except-clause needed — never a silent catch-all."""


class RefinementPreCallError(RefinementError):
    """Raised when refinement's OWN pre-call validation fails (today: the
    padded-window-collapse check) BEFORE a Gemini client is built or
    generate_content is ever invoked (AC1-7). A RefinementError subclass
    (any existing ``except RefinementError`` still catches it) --
    ``run_highlight_batch_with_global_cap`` (Task 3) catches THIS subclass
    FIRST and does NOT increment calls_spent/dollars_spent for it, unlike a
    genuine post-call RefinementError (real Gemini/transport failure, or a
    malformed real response), which DOES count as spent. Given Task 3's
    containment gate rejects an out-of-range candidate before
    refine_highlight_timestamp is ever called, the window-collapse case
    this exception guards is provably unreachable through that gated path
    for any original_start_s <= original_end_s with non-negative padding
    (see this task's design note in the plan) -- kept as defense-in-depth
    for any future/direct caller, not dead code removed on that basis."""


def resolve_highlight_window(target: MatchTarget, video_durations: dict[str, float]) -> ScopeWindow:
    """One match's highlight-discovery scope window — reuses
    ``scope_policy.single_match_scope``/``multi_match_scope`` verbatim (the
    SAME window a real ``chunk-segment-tags`` run would use for this match,
    per ``match_catalog.py``'s own "never let two callers compute two
    different windows for one match" discipline). Raises
    ``HighlightDiscoveryError`` (never guesses a duration) if a single-match
    target's ffprobe duration is missing from ``video_durations``."""
    if target.is_multi_match:
        return multi_match_scope(target.match_id, target.gold_t_s_values)
    if target.video_url_md5 not in video_durations:
        raise HighlightDiscoveryError(
            f"match {target.match_id!r}: single-match video {target.video_url_md5!r} has no ffprobe "
            "duration in --video-durations — cannot resolve its highlight-discovery scope window."
        )
    return single_match_scope(target.match_id, video_durations[target.video_url_md5])


def is_long_window_risk(duration_s: float) -> bool:
    """``True`` iff ``duration_s`` exceeds ``LONG_WINDOW_RISK_THRESHOLD_S``
    (720s, the same bound production's own ``chunk-segment-tags`` treats as
    needing slicing) — a WARNING signal only (Bezos fast-follow). Strictly
    greater-than, never ``>=``: a window exactly AT the production cap is the
    same size a legal single ``chunk-segment-tags`` run would use, so it is
    not yet longer than what the rest of the system already considers safe.
    """
    return duration_s > LONG_WINDOW_RISK_THRESHOLD_S


def build_gemini_client(gemini_api_key: str, request_timeout_ms: int) -> genai.Client:
    """OWN client construction — mirrors ``service/pipelines/executors.py``'s
    ``RunContext.gemini_client`` property VERBATIM (frozen contract §D: "do
    NOT import or fork executors.py"). Any future change to that property
    must be mirrored here by a human diff, not inherited via import — the
    whole point of this PEER module is zero import coupling to the
    chunk-segment-tags production pipeline surface."""
    return genai.Client(
        api_key=gemini_api_key,
        http_options=types.HttpOptions(timeout=request_timeout_ms),
    )


def highlight_thinking_config(model_id: str, ui_level: Optional[str]) -> types.ThinkingConfig:
    """Mirrors ``service/routes/qa_vlm.thinking_config_for``'s mapping (NOT
    imported — see module docstring): ``gemini-3.1-pro-preview`` cannot
    disable thinking outright, so "off" is honored as its lowest available
    level rather than a budget/level combination it would 400 on. Raises
    ``HighlightDiscoveryError`` for a ``ui_level`` outside
    ``THINKING_LEVELS``/``None`` — never silently treated as "off"."""
    if ui_level is not None and ui_level not in THINKING_LEVELS:
        raise HighlightDiscoveryError(f"thinking_level must be one of {THINKING_LEVELS} or None, got {ui_level!r}")
    if ui_level in (None, "off"):
        if model_id == "gemini-3.1-pro-preview":
            return types.ThinkingConfig(thinking_level="low")
        return types.ThinkingConfig(thinking_budget=0)
    return types.ThinkingConfig(thinking_level=ui_level)


def validate_fps(fps: float) -> None:
    """Raises ``HighlightDiscoveryError`` for an ``fps`` outside
    ``FPS_OPTIONS`` — a closed enum (CEO-overruled free-numeric input),
    never silently clamped/rounded to the nearest legal value."""
    if fps not in FPS_OPTIONS:
        raise HighlightDiscoveryError(f"fps must be one of {FPS_OPTIONS}, got {fps!r}")


def resolve_media_resolution(frame_quality: str) -> Optional["types.MediaResolution"]:
    """``"default"`` -> ``None`` (leave ``media_resolution`` UNSET on
    ``GenerateContentConfig`` — exactly today's, pre-this-increment code
    path). ``"low"``/``"medium"``/``"high"`` -> the corresponding
    ``types.MediaResolution`` enum value. Raises ``HighlightDiscoveryError``
    for anything else — never silently coerced to ``"default"``."""
    if frame_quality == DEFAULT_FRAME_QUALITY:
        return None
    if frame_quality not in _MEDIA_RESOLUTION_BY_QUALITY:
        raise HighlightDiscoveryError(
            f"frame_quality must be one of {FRAME_QUALITY_OPTIONS}, got {frame_quality!r}"
        )
    return _MEDIA_RESOLUTION_BY_QUALITY[frame_quality]


def normalize_youtube_watch_url(video_url: str) -> str:
    """Re-exports ``eval.adcc.predict.youtube_url.normalize_youtube_watch_url``
    (the shared leaf-module implementation — moved there so the tag-batch
    prediction-capture path, ``global_cap_driver.py``/``run_predictions.py``,
    can reuse it too without a circular import: this module already imports
    ``global_cap_driver`` for ``BatchRunState``, so ``global_cap_driver``
    importing back from HERE would cycle). Translates the leaf module's
    neutral ``YoutubeUrlError`` into THIS module's own
    ``HighlightDiscoveryError`` — keeps ``hd.normalize_youtube_watch_url``'s
    public contract (function name, exception type, exact message) BYTE-
    IDENTICAL to before the extraction, so every pre-existing caller/test of
    this function is unaffected. See ``youtube_url.py``'s module docstring
    for the full live-bug root-cause writeup."""
    try:
        return _normalize_youtube_watch_url(video_url)
    except YoutubeUrlError as e:
        raise HighlightDiscoveryError(str(e)) from e


def build_highlight_schema() -> types.Schema:
    """The HARNESS-enforced output envelope (frozen contract §D): a JSON list
    of ``{start_s, end_s, reason}`` spans, CLIP-RELATIVE seconds (0-based
    from the start of the scoped clip — this module, not the model, is
    responsible for normalizing to VOD-absolute afterward). A point
    highlight is ``start_s == end_s``.

    ``reason`` (frozen contract §A, this increment) is REQUIRED — Rener's
    format, "[category] — [technique] — [decisive visual cue]", naming a
    single-frame-identifiable landmark. For a submission: name the sub type
    AND anchor on the DECISIVE INSTANT = the LOCK (mechanism secured/
    inescapable), never the tap; instance-specific for chains/recaptures
    (e.g. "the second armbar after the roll-through"). ``reason`` is always
    requested and persisted, even when the refinement pass is OFF — it just
    isn't consumed by anything when refinement is off (it feeds
    ``refine_highlight_timestamp``'s prompt when refinement is on)."""
    return types.Schema(
        type=types.Type.ARRAY,
        description=(
            "Every highlight-worthy moment in this clip. A span with start_s == end_s is an "
            "instantaneous highlight point (e.g. a submission's finish moment); start_s < end_s is a "
            "highlight lasting that duration. Timestamps are SECONDS FROM THE START OF THIS CLIP "
            "(0-based, clip-relative) — never absolute-video time, never MM:SS."
        ),
        items=types.Schema(
            type=types.Type.OBJECT,
            required=["start_s", "end_s", "reason"],
            properties={
                "start_s": types.Schema(
                    type=types.Type.NUMBER, format="float",
                    description="Highlight start, seconds from the start of THIS CLIP (0-based).",
                ),
                "end_s": types.Schema(
                    type=types.Type.NUMBER, format="float",
                    description="Highlight end, seconds from the start of THIS CLIP (0-based); "
                                "equal to start_s for an instantaneous point.",
                ),
                "reason": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Format: '[category] — [technique] — [decisive visual cue]', naming a "
                        "single-frame-identifiable landmark. For a submission: name the sub type AND "
                        "anchor on the decisive instant = the LOCK (mechanism secured/inescapable), "
                        "NEVER the tap. Be instance-specific for chains/recaptures, e.g. 'the second "
                        "armbar after the roll-through'."
                    ),
                ),
            },
        ),
    )


_ENVELOPE_INSTRUCTION = (
    "\n\nOUTPUT FORMAT (harness-enforced, do not deviate): return highlight timestamps as SECONDS "
    "FROM THE START OF THIS CLIP (0-based, clip-relative) — never absolute-video time, never MM:SS. "
    "The harness converts your clip-relative timestamps to the original video's absolute clock itself. "
    "For EVERY highlight, also give a 'reason' in the format '[category] — [technique] — [decisive "
    "visual cue]' naming a single-frame-identifiable landmark. For a submission: name the sub type AND "
    "anchor on the decisive instant = the LOCK (mechanism secured/inescapable), NEVER the tap. Be "
    "instance-specific for chains/recaptures, e.g. 'the second armbar after the roll-through'."
)

# Live investigation (2026-07-18, LONG-video out-of-range timestamps):
# Gemini's returned CLIP-RELATIVE timestamps sometimes drift past the
# clip's own [0, clip_duration] bounds (observed worse on long clips) — a
# real Gemini-behavior finding, not a harness normalization bug (verified:
# window.start_s + clip-relative reproduces the persisted VOD-absolute
# value exactly). This tolerance absorbs ordinary float/rounding noise at
# the exact clip boundary (e.g. a span ending at "the last frame") without
# flagging it — only a MEANINGFULLY past-bounds span counts.
OUT_OF_BOUNDS_TOLERANCE_S: float = 1.0

# Timestamp critique/refinement (frozen contract, this increment). Padding is
# ASYMMETRIC/front-loaded — Gemini's discovery bias is LATE (it tends to
# anchor a submission's finish on the tap, not the lock that actually ends
# it), so the true instant is more likely EARLIER than reported; more pad
# goes backward than forward.
DEFAULT_PAD_BEFORE_S: float = 12.0
DEFAULT_PAD_AFTER_S: float = 5.0
# The ONE conditional widen-retry (frozen contract §B: "retry ONCE with the
# window doubled backward only... then HARD STOP") -- never a second retry,
# never a wider forward pad.
WIDEN_PAD_BEFORE_S: float = 24.0

# Refinement call params (frozen contract §B): a short re-check window makes
# dense frames + high resolution cheap and gives Gemini better grounding to
# pinpoint the exact instant. frame_quality is FIXED at "high" for every
# refinement call (not a separate knob — only refine_fps is founder-exposed).
DEFAULT_REFINE_FPS: float = 4.0
REFINE_FRAME_QUALITY: str = "high"

# Dedicated placeholder $ (frozen contract §C: "do NOT borrow chunk-segment
# vocab/rates") for a refinement call whose real ``usage_metadata`` isn't
# available yet (mirrors global_cap_driver.PASS1_PLACEHOLDER_USD's role for
# the tag-batch driver, but computed via THIS feature's own
# ``estimate_highlight_call_cost_usd`` formula, never chunk-segment's
# per-chunk rate). Derived from the DEFAULT pad envelope + refine fps/quality
# so it stays in lockstep with the real cost formula rather than an
# independently-eyeballed number.
REFINE_PLACEHOLDER_USD: float = estimate_highlight_call_cost_usd(
    DEFAULT_PAD_BEFORE_S + DEFAULT_PAD_AFTER_S, DEFAULT_REFINE_FPS, REFINE_FRAME_QUALITY,
)


def _is_span_out_of_bounds(
    clip_start: float, clip_end: float, clip_duration: float, tol: float = OUT_OF_BOUNDS_TOLERANCE_S,
) -> bool:
    """LOOSE detection: ``True`` iff ``[clip_start, clip_end]`` falls
    MEANINGFULLY outside ``[0, clip_duration]`` (past ``tol`` seconds of
    ordinary float/rounding noise at the exact boundary). Shared by
    ``discover_match_highlights``'s out-of-bounds surfacing AND
    ``decide_refinement_acceptance``'s reject-on-OOB check (frozen contract
    §B: "reuse your OOB check") — the SAME tolerance constant, used in the
    SAME direction (outward margin, "is this clearly past the edge")."""
    return clip_start < -tol or clip_end > clip_duration + tol


def _is_span_near_edge(
    clip_start: float, clip_end: float, clip_duration: float, tol: float = OUT_OF_BOUNDS_TOLERANCE_S,
) -> bool:
    """STRICT interior check (the inverse direction of ``_is_span_out_of_
    bounds``'s outward margin): ``True`` iff either endpoint sits within
    ``tol`` of the clip's own boundary — refinement's "suspicious, at the
    padded window's edge" signal (frozen contract §B: "A refined time at the
    padded window's edge is suspicious → flag, do NOT auto-accept"). Uses
    the SAME ``OUT_OF_BOUNDS_TOLERANCE_S`` constant as the sole tolerance
    magnitude in the codebase, just applied as an INWARD safety margin
    instead of an outward one."""
    return clip_start <= tol or clip_end >= clip_duration - tol


REJECTION_REASONS: tuple[str, ...] = ("non_finite", "invalid_order", "out_of_range")


def validate_highlight_candidate(
    start_s: float, end_s: float, sent_window_start_s: float, sent_window_end_s: float,
) -> Optional[str]:
    """Strict acceptance gate (AC1-1/1-2/1-3) over ONE candidate's VOD-
    absolute [start_s, end_s] -- checked in order, first failure wins:
    (1) non_finite: either endpoint is NaN/inf -- a Gemini response can
    smuggle these past json.loads's own default allow_nan=True AND past a
    plain float() coercion, which raises for neither. (2) invalid_order:
    end_s < start_s. (3) out_of_range: NOT fully contained in
    [sent_window_start_s, sent_window_end_s] -- the INTEGER-SECOND window
    ACTUALLY SENT to Gemini as video_metadata.start_offset/end_offset
    (discover_match_highlights's/refine_highlight_timestamp's own
    ``sent_window_s`` -- NEVER the pre-rounding float ScopeWindow.start_s/
    end_s, which can differ from the sent bound by up to ~1s and would
    either falsely reject a legitimate near-boundary highlight or fail to
    reject a genuinely invalid one). Zero tolerance -- no fuzz band
    (distinct from ``_is_span_out_of_bounds``'s tolerant DIAGNOSTIC
    detector above, which stays unchanged and serves a different purpose:
    a flag surfaced to the founder, never a rejection).

    Returns the first-failing reason string (one of REJECTION_REASONS), or
    None if the candidate is valid. (1)/(2) are structurally already-
    guaranteed for a candidate that reached this function via
    discover_match_highlights/refine_highlight_timestamp's own hard-raise
    parsing checks -- re-checked here anyway so this is ONE trustworthy
    gate for any caller, not an invariant split silently across two call
    sites that could drift apart."""
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        return "non_finite"
    if end_s < start_s:
        return "invalid_order"
    if start_s < sent_window_start_s or end_s > sent_window_end_s:
        return "out_of_range"
    return None


async def discover_match_highlights(
    *,
    match_id: str,
    video_url: str,
    window: ScopeWindow,
    gemini_api_key: str,
    request_timeout_ms: int,
    model: str,
    thinking_level: Optional[str],
    system_prompt: str,
    followup_prompt: str,
    fps: float = DEFAULT_FPS,
    frame_quality: str = DEFAULT_FRAME_QUALITY,
    retry_config: Optional[gemini_retry.GeminiRetryConfig] = None,
) -> dict:
    """ONE Gemini call: coarse "find the highlights" pass over ONE match's
    scoped window (never the whole VOD). Founder supplies BOTH
    ``system_prompt`` (Gemini ``system_instruction``) and ``followup_prompt``
    (the per-turn ask) — this module only appends the fixed, harness-owned
    ``_ENVELOPE_INSTRUCTION`` to the followup so the ``response_schema``
    contract above is honored, never altering the founder's own semantic
    wording.

    ``fps``/``frame_quality`` (this increment's CEO-frozen controls) are
    validated against ``FPS_OPTIONS``/``FRAME_QUALITY_OPTIONS`` and threaded
    straight into ``VideoMetadata.fps``/``GenerateContentConfig.
    media_resolution`` — never silently clamped to the nearest legal value.
    Their DEFAULTS (``fps=1.0``, ``frame_quality="default"``) reproduce
    exactly today's pre-this-increment Gemini call (``fps=1``, no
    ``media_resolution`` set at all). This module does NOT live-verify a
    non-default value's acceptance by the real API — if Gemini rejects a
    value, that surfaces as a real transport/4xx error (frozen contract §D's
    "surface an error" discipline), never a silent fallback to a safer value.

    Returns ``{"match_id", "highlights": [{"start_s", "end_s"} in VOD-
    ABSOLUTE seconds], "raw_highlights": [{"start_s", "end_s"} in the RAW,
    UNMODIFIED clip-relative seconds Gemini actually returned, before
    ``window.start_s`` is added — diagnostic only, never consumed by
    scoring], "out_of_bounds_count", "out_of_bounds_spans" (the raw
    clip-relative spans that fall outside ``[0, clip_duration]``, where
    ``clip_duration`` is the ACTUAL integer-second window sent to Gemini —
    DETECTED, never dropped/clamped here; that policy call is LeCun's, not
    this harness's), "prompt_text", "raw_response_text", "usage_metadata",
    "timing"}``. Raises ``HighlightDiscoveryError`` for any unparseable/
    malformed response or an out-of-family ``model``/``thinking_level``/
    ``fps``/``frame_quality`` — never a silent empty-list fallback (frozen
    contract §D)."""
    if model not in HIGHLIGHT_MODELS:
        raise HighlightDiscoveryError(f"model {model!r} is not in the allowed Gemini >=3.1 family {HIGHLIGHT_MODELS!r}")
    thinking_cfg = highlight_thinking_config(model, thinking_level)
    validate_fps(fps)
    media_resolution = resolve_media_resolution(frame_quality)

    client = build_gemini_client(gemini_api_key, request_timeout_ms)
    # LIVE-BUG FIX: never send the gold snapshot's raw /embed/<id> URL to
    # Gemini's native file_uri ingestion — normalize to the canonical
    # watch?v=<id> form it actually accepts (see module docstring).
    canonical_video_url = normalize_youtube_watch_url(video_url)
    # Live investigation (2026-07-18) re-audit finding: start_offset/
    # end_offset are the ACTUAL integer-second window Gemini is told to
    # ingest — computed ONCE here (not re-derived at the bounds-check below)
    # so the tolerance check compares against the exact same integers sent
    # in the request, never the pre-rounding float ``window.start_s``/
    # ``window.end_s`` (a match-invariant, sub-1-second discrepancy that is
    # NOT the cause of the observed ~15s+ drift, but is worth being exact
    # about for a tolerance-boundary check).
    sent_start_offset_s = int(window.start_s)
    sent_end_offset_s = int(round(window.end_s))
    video_part = types.Part(
        file_data=types.FileData(file_uri=canonical_video_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{sent_start_offset_s}s", end_offset=f"{sent_end_offset_s}s", fps=fps,
        ),
    )
    prompt_text = (followup_prompt or "").rstrip() + _ENVELOPE_INSTRUCTION
    gen_config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=build_highlight_schema(),
        thinking_config=thinking_cfg,
        system_instruction=system_prompt or None,
        media_resolution=media_resolution,
    )

    t0 = time.perf_counter()
    try:
        response = await gemini_retry.call_with_retry(
            lambda: client.aio.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt_text), video_part])],
                config=gen_config,
            ),
            op_name="highlight_discovery:discover_match_highlights",
            retry_config=retry_config or gemini_retry.GeminiRetryConfig(),
        )
    except Exception as e:  # noqa: BLE001 — surface the REAL Gemini/transport error (live-bug fix)
        # LIVE-BUG FIX: a genuine Gemini/transport error (e.g.
        # ``google.genai.errors.ClientError: 400 INVALID_ARGUMENT`` naming
        # the offending request field) previously propagated PAST this
        # module's own ``HighlightDiscoveryError`` contract entirely — it
        # was only ever caught by ``dashboard_app.py``'s broad outer
        # ``except Exception`` in ``/highlight/run``'s ``event_stream()``.
        # That meant: (a) the real, already-billed call was NEVER counted
        # against ``BatchRunState.calls_spent``/``dollars_spent`` in
        # ``run_highlight_batch_with_global_cap`` (violating the SAME "a run
        # of failing calls cannot spend unboundedly without tripping the
        # cap" invariant ``global_cap_driver.is_call_event`` already enforces
        # for the tag-batch driver), and (b) it aborted the WHOLE batch
        # instead of surfacing a per-match error and continuing (breaking
        # this module's own documented contract, see
        # ``run_highlight_batch_with_global_cap``'s docstring). Wrapping it
        # as ``HighlightDiscoveryError`` (mirrors ``executors.py``'s
        # ``chunk_segment_node``/``_analyze_video_window``'s IDENTICAL
        # "catch the real SDK exception, surface str(e) verbatim" discipline
        # -- never a fabricated/generic message) routes it through the
        # EXISTING per-match error handling below instead, which restores
        # both invariants. ``str(e)`` preserves Gemini's own error text
        # (INVALID_ARGUMENT's message body names the offending field) so the
        # NEXT occurrence is diagnosable from the NDJSON ``error`` event and
        # the server log alone, without another live-log excavation.
        raise HighlightDiscoveryError(f"match {match_id!r}: Gemini call failed: {e}") from e
    model_ms = (time.perf_counter() - t0) * 1000

    raw_text = response.text or ""
    try:
        parsed = json.loads(raw_text)
    except (ValueError, TypeError) as e:
        raise HighlightDiscoveryError(f"match {match_id!r}: non-JSON Gemini response: {e}") from e
    if not isinstance(parsed, list):
        raise HighlightDiscoveryError(f"match {match_id!r}: expected a JSON array, got {type(parsed).__name__}")

    # Uses the ACTUAL integer-second window sent to Gemini (see the
    # sent_start_offset_s/sent_end_offset_s re-audit note above), not the
    # pre-rounding float window.duration_s — keeps the bounds-check
    # tolerance boundary exact rather than off by a fraction of a second.
    clip_duration = sent_end_offset_s - sent_start_offset_s
    highlights: list[dict] = []
    raw_highlights: list[dict] = []
    out_of_bounds_spans: list[dict] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict) or "start_s" not in item or "end_s" not in item:
            raise HighlightDiscoveryError(f"match {match_id!r}: item {i} missing start_s/end_s: {item!r}")
        try:
            clip_start = float(item["start_s"])
            clip_end = float(item["end_s"])
        except (TypeError, ValueError) as e:
            raise HighlightDiscoveryError(f"match {match_id!r}: item {i} non-numeric start_s/end_s: {item!r}") from e
        # Non-finite endpoints are NOT hard-raised here: Task 3's batch gate
        # (``validate_highlight_candidate`` → ``highlight_rejected``) owns
        # typed rejection so the raw candidate is preserved as diagnostics
        # (AC1-6). Order is still hard-raised — a reversed span is never a
        # valid discovery item and has no gate-preserving diagnostic path.
        if math.isfinite(clip_start) and math.isfinite(clip_end) and clip_end < clip_start:
            raise HighlightDiscoveryError(f"match {match_id!r}: item {i} end_s < start_s: {item!r}")
        # reason is REQUIRED (frozen contract §A) — always requested and
        # persisted, even when the refinement pass is OFF (it's simply
        # unconsumed then). Never defaulted/fabricated on a missing value.
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise HighlightDiscoveryError(f"match {match_id!r}: item {i} missing/empty required 'reason': {item!r}")

        raw_highlights.append({"start_s": clip_start, "end_s": clip_end, "reason": reason})

        # Live investigation (2026-07-18): Gemini's clip-relative timestamps
        # sometimes drift past the clip's own bounds (worse on long clips).
        # DETECT + SURFACE only — never drop/clamp here (that scoring-policy
        # call belongs to LeCun, not this harness). ``highlights`` below
        # still carries EVERY span, in-bounds or not, unmodified.
        # Skip the diagnostic OOB flag for non-finite spans — comparisons
        # against clip_duration are meaningless; the gate rejects them.
        if (
            math.isfinite(clip_start) and math.isfinite(clip_end)
            and _is_span_out_of_bounds(clip_start, clip_end, clip_duration)
        ):
            out_of_bounds_spans.append({"start_s": clip_start, "end_s": clip_end})

        # Normalize clip-relative -> VOD-absolute (frozen contract §D).
        highlights.append({
            "start_s": window.start_s + clip_start, "end_s": window.start_s + clip_end, "reason": reason,
        })

    return {
        "match_id": match_id,
        "highlights": highlights,
        "raw_highlights": raw_highlights,
        "out_of_bounds_count": len(out_of_bounds_spans),
        "out_of_bounds_spans": out_of_bounds_spans,
        "sent_window_s": (float(sent_start_offset_s), float(sent_end_offset_s)),
        "prompt_text": prompt_text,
        "raw_response_text": raw_text,
        "usage_metadata": simplified_tags.extract_usage_metadata(response),
        "timing": {"model_ms": round(model_ms, 1)},
    }


def build_refinement_schema() -> types.Schema:
    """The refinement call's OWN output envelope (frozen contract §B, own
    schema — never reuses ``build_highlight_schema``): a single JSON OBJECT
    (one highlight refined per call, never an array) — ``{action_confirmed,
    start_s, end_s}``. ``start_s``/``end_s`` are ``nullable`` and MUST be
    null whenever ``action_confirmed`` is false — enforced again defensively
    in ``refine_highlight_timestamp`` (never trusts the schema alone to
    prevent a model from violating its own output contract)."""
    return types.Schema(
        type=types.Type.OBJECT,
        required=["action_confirmed", "start_s", "end_s"],
        properties={
            "action_confirmed": types.Schema(
                type=types.Type.BOOLEAN,
                description="True iff the described highlight action actually occurs somewhere in THIS clip.",
            ),
            "start_s": types.Schema(
                type=types.Type.NUMBER, format="float", nullable=True,
                description=(
                    "The precise decisive-instant start, seconds from the start of THIS CLIP (0-based, "
                    "clip-relative). For a submission: the instant the mechanism is LOCKED/secured and "
                    "inescapable -- NEVER the tap or celebration. MUST be null if action_confirmed is false."
                ),
            ),
            "end_s": types.Schema(
                type=types.Type.NUMBER, format="float", nullable=True,
                description=(
                    "The moment the exchange resolves (tap/stoppage for a submission), seconds from the "
                    "start of THIS CLIP (0-based, clip-relative). MUST be null if action_confirmed is false."
                ),
            ),
        },
    )


def _refinement_prompt_text(reason: str) -> str:
    """Rener's refinement prompt content (frozen contract §B) — given the
    discovery pass's own ``reason`` for this highlight, ask Gemini to (a)
    confirm the action occurs in THIS (padded, re-check) clip, (b) if so,
    mark the precise decisive instant, anchored on the LOCK for a
    submission (never the tap) to correct the discovery pass's late bias."""
    return (
        "This clip is a padded RE-CHECK window around a highlight a coarse first pass already found, "
        f"described as:\n\n{reason}\n\n"
        "First, confirm whether this described action actually occurs somewhere in THIS clip. If it "
        "does NOT occur in this clip, set action_confirmed=false and leave start_s/end_s null — never "
        "guess a time.\n\n"
        "If it DOES occur, mark the PRECISE decisive instant: for a submission, the instant the "
        "mechanism is LOCKED/secured and the opponent is inescapable — this is explicitly NOT the tap "
        "or the celebration afterward, which comes later. Use the lock as start_s; the tap/stoppage as "
        "end_s. For a non-submission highlight, use the decisive instant described above as start_s and "
        "the moment the exchange resolves as end_s.\n\n"
        "Report both timestamps as SECONDS FROM THE START OF THIS CLIP (0-based, clip-relative) — never "
        "absolute-video time, never MM:SS."
    )


async def refine_highlight_timestamp(
    *,
    match_id: str,
    video_url: str,
    match_window: ScopeWindow,
    original_start_s: float,
    original_end_s: float,
    reason: str,
    gemini_api_key: str,
    request_timeout_ms: int,
    model: str,
    thinking_level: Optional[str],
    refine_fps: float = DEFAULT_REFINE_FPS,
    pad_before_s: float = DEFAULT_PAD_BEFORE_S,
    pad_after_s: float = DEFAULT_PAD_AFTER_S,
    retry_config: Optional[gemini_retry.GeminiRetryConfig] = None,
) -> dict:
    """ONE refinement Gemini call — ONE attempt over a PADDED, front-loaded
    re-check window around ``[original_start_s, original_end_s]`` (frozen
    contract §B). This function makes exactly one ``generate_content`` call
    and returns its (already VOD-absolute-normalized) result; it does NOT
    implement the widen-retry or the accept/reject decision — those are
    orchestrated by ``run_highlight_batch_with_global_cap`` (which must
    cap/stop-check BEFORE every individual Gemini call, including a
    widen-retry attempt — encapsulating the retry inside this function would
    hide that second call from the per-call cap check).

    The padded window is ``[original_start_s - pad_before_s, original_end_s
    + pad_after_s]``, CLAMPED to ``match_window`` (never re-checks outside
    the match's own scoped window). ``fps=refine_fps``/``frame_quality=
    "high"`` (fixed, frozen contract §B) — Gemini >=3.1 family only, same
    allowlist as discovery.

    Returns ``{"action_confirmed", "start_s", "end_s" (VOD-absolute, or
    ``None`` iff not confirmed), "padded_window": (start_s, end_s),
    "prompt_text", "raw_response_text", "usage_metadata", "timing"}``.
    Raises ``RefinementError`` for any unparseable/malformed response or an
    out-of-family ``model``/``thinking_level``/``refine_fps`` — never a
    silent fallback (frozen contract: "No fabrication, ever")."""
    if model not in HIGHLIGHT_MODELS:
        raise RefinementError(f"model {model!r} is not in the allowed Gemini >=3.1 family {HIGHLIGHT_MODELS!r}")
    thinking_cfg = highlight_thinking_config(model, thinking_level)
    validate_fps(refine_fps)
    media_resolution = resolve_media_resolution(REFINE_FRAME_QUALITY)

    padded_start = max(match_window.start_s, original_start_s - pad_before_s)
    padded_end = min(match_window.end_s, original_end_s + pad_after_s)
    if padded_end <= padded_start:
        raise RefinementPreCallError(
            f"match {match_id!r}: refinement window collapsed to non-positive duration "
            f"[{padded_start}, {padded_end}] after clamping to the match window — cannot re-check "
            "(pre-call: no Gemini client was built, no generate_content call was made)."
        )

    client = build_gemini_client(gemini_api_key, request_timeout_ms)
    canonical_video_url = normalize_youtube_watch_url(video_url)
    sent_start_offset_s = int(padded_start)
    sent_end_offset_s = int(round(padded_end))
    video_part = types.Part(
        file_data=types.FileData(file_uri=canonical_video_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{sent_start_offset_s}s", end_offset=f"{sent_end_offset_s}s", fps=refine_fps,
        ),
    )
    prompt_text = _refinement_prompt_text(reason)
    gen_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=build_refinement_schema(),
        thinking_config=thinking_cfg,
        media_resolution=media_resolution,
    )

    t0 = time.perf_counter()
    try:
        response = await gemini_retry.call_with_retry(
            lambda: client.aio.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt_text), video_part])],
                config=gen_config,
            ),
            op_name="highlight_discovery:refine_highlight_timestamp",
            retry_config=retry_config or gemini_retry.GeminiRetryConfig(),
        )
    except Exception as e:  # noqa: BLE001 — surface the REAL Gemini/transport error, same discipline as discover_match_highlights
        raise RefinementError(f"match {match_id!r}: refinement Gemini call failed: {e}") from e
    model_ms = (time.perf_counter() - t0) * 1000

    raw_text = response.text or ""
    try:
        parsed = json.loads(raw_text)
    except (ValueError, TypeError) as e:
        raise RefinementError(f"match {match_id!r}: non-JSON refinement response: {e}") from e
    if not isinstance(parsed, dict) or "action_confirmed" not in parsed:
        raise RefinementError(f"match {match_id!r}: expected a JSON object with action_confirmed: {parsed!r}")

    confirmed = bool(parsed["action_confirmed"])
    common = {
        "padded_window": (padded_start, padded_end),
        "sent_window_s": (float(sent_start_offset_s), float(sent_end_offset_s)),
        "prompt_text": prompt_text,
        "raw_response_text": raw_text,
        "usage_metadata": simplified_tags.extract_usage_metadata(response),
        "timing": {"model_ms": round(model_ms, 1)},
    }
    if not confirmed:
        # Schema contract: null start_s/end_s whenever unconfirmed. Never
        # trust the schema alone — a model returning a NON-null time here
        # despite action_confirmed=false is violating its own output
        # contract, which this harness refuses to silently paper over.
        if parsed.get("start_s") is not None or parsed.get("end_s") is not None:
            raise RefinementError(
                f"match {match_id!r}: action_confirmed=false but start_s/end_s are non-null: {parsed!r}"
            )
        return {"action_confirmed": False, "start_s": None, "end_s": None, **common}

    if "start_s" not in parsed or "end_s" not in parsed or parsed["start_s"] is None or parsed["end_s"] is None:
        raise RefinementError(f"match {match_id!r}: action_confirmed=true but start_s/end_s missing/null: {parsed!r}")
    try:
        clip_start = float(parsed["start_s"])
        clip_end = float(parsed["end_s"])
    except (TypeError, ValueError) as e:
        raise RefinementError(f"match {match_id!r}: non-numeric refined start_s/end_s: {parsed!r}") from e
    if not (math.isfinite(clip_start) and math.isfinite(clip_end)):
        raise RefinementError(f"match {match_id!r}: non-finite refined start_s/end_s: {parsed!r}")
    if clip_end < clip_start:
        raise RefinementError(f"match {match_id!r}: refined end_s < start_s: {parsed!r}")

    return {
        "action_confirmed": True,
        "start_s": padded_start + clip_start,
        "end_s": padded_start + clip_end,
        **common,
    }


def decide_refinement_acceptance(
    attempt_result: dict, *, tol: float = OUT_OF_BOUNDS_TOLERANCE_S,
) -> dict:
    """Pure accept/reject decision (frozen contract §B) over ONE
    ``refine_highlight_timestamp`` attempt result. Accepts ONLY if (a)
    ``action_confirmed`` is true, AND (b) the corrected time does not fall
    outside the padded window (reuses ``_is_span_out_of_bounds`` — frozen
    contract: "reuse your OOB check"), AND (c) the corrected time is not
    suspiciously close to the padded window's own edge
    (``_is_span_near_edge`` — "the shift is within the padding envelope" is
    exactly this: a time safely INSIDE the padded window, by construction,
    never implies the true instant needed MORE padding than was given).

    Never mutates ``attempt_result``. Returns
    ``{"accept": bool, "reject_reason": Optional[str]}`` — ``reject_reason``
    is one of ``"not_confirmed"``/``"out_of_bounds"``/``"near_edge"``, or
    ``None`` when accepted."""
    if not attempt_result["action_confirmed"]:
        return {"accept": False, "reject_reason": "not_confirmed"}

    padded_start, padded_end = attempt_result["padded_window"]
    clip_duration = padded_end - padded_start
    # attempt_result's start_s/end_s are already VOD-absolute; re-derive
    # clip-relative (relative to ITS OWN padded_start) to reuse the same
    # clip-relative-space bounds helpers discovery already uses.
    clip_start = attempt_result["start_s"] - padded_start
    clip_end = attempt_result["end_s"] - padded_start

    if _is_span_out_of_bounds(clip_start, clip_end, clip_duration, tol=tol):
        return {"accept": False, "reject_reason": "out_of_bounds"}
    if _is_span_near_edge(clip_start, clip_end, clip_duration, tol=tol):
        return {"accept": False, "reject_reason": "near_edge"}
    return {"accept": True, "reject_reason": None}


def _enriched_highlight(
    h: dict,
    *,
    refined_start_s: Optional[float] = None,
    refined_end_s: Optional[float] = None,
    refined: bool = False,
    action_confirmed: Optional[bool] = None,
    refinement_failed: bool = False,
    rejected: bool = False,
    rejection_reason: Optional[str] = None,
) -> dict:
    """The ONE enriched per-highlight dict constructor (frozen contract §F,
    extended by AC1-6/AC1-7's rejected/rejection_reason fields) -- every
    per-highlight outcome in run_highlight_batch_with_global_cap's loop
    constructs its result through THIS function, never an inline dict
    literal, so a future field addition can never silently miss a call
    site."""
    return {
        "original_start_s": h["start_s"], "original_end_s": h["end_s"],
        "refined_start_s": refined_start_s, "refined_end_s": refined_end_s, "refined": refined,
        "reason": h.get("reason"), "action_confirmed": action_confirmed, "refinement_failed": refinement_failed,
        "rejected": rejected, "rejection_reason": rejection_reason,
    }


def _unattempted_enriched_highlight(h: dict) -> dict:
    """The enriched per-highlight dict (frozen contract §F) for a discovery
    highlight that was NEVER handed to refinement — either because the
    ``refine`` toggle is off, or because the global cap fired before this
    highlight's turn came up in the per-highlight loop. Both cases are
    indistinguishable at the persisted-schema level (there is no separate
    "not reached" state in the frozen §F shape) — ``refined=False``,
    ``refinement_failed=False`` (nothing was attempted, so nothing FAILED),
    ``refined_start_s``/``refined_end_s`` stay ``None`` (never backfilled
    from ``original_*``). Delegates to _enriched_highlight()."""
    return _enriched_highlight(h)


def _rejected_enriched_highlight(h: dict, rejection_reason: str) -> dict:
    """The enriched dict for a highlight that FAILED validate_highlight_
    candidate (AC1-4/AC1-6) -- never refined (refinement is never
    attempted for a rejected candidate), diagnostics-only (rejected=True),
    never scoreable (highlight_scoring.effective_pred_events/
    original_pred_events skip it -- Task 4)."""
    return _enriched_highlight(h, rejected=True, rejection_reason=rejection_reason)


async def run_highlight_batch_with_global_cap(
    targets_and_windows: list[tuple[MatchTarget, ScopeWindow]],
    *,
    gemini_api_key: str,
    request_timeout_ms: int,
    model: str,
    thinking_level: Optional[str],
    system_prompt: str,
    followup_prompt: str,
    state: BatchRunState,
    fps: float = DEFAULT_FPS,
    frame_quality: str = DEFAULT_FRAME_QUALITY,
    refine: bool = True,
    refine_pad_before_s: float = DEFAULT_PAD_BEFORE_S,
    refine_pad_after_s: float = DEFAULT_PAD_AFTER_S,
    refine_fps: float = DEFAULT_REFINE_FPS,
) -> AsyncIterator[dict]:
    """Drives every ``(target, window)`` pair's highlight-discovery Gemini
    call, THEN (when ``refine=True``) a per-highlight refinement pass, under
    the SAME global cap + stop semantics as ``global_cap_driver.
    run_batch_with_global_cap`` (frozen contract §D: "Honor the same global
    cap + stop semantics") — reuses its ``BatchRunState``/``event_cost_usd``/
    ``PASS1_PLACEHOLDER_USD`` verbatim for discovery-call cost bookkeeping,
    and this module's own ``REFINE_PLACEHOLDER_USD``/``estimate_highlight_
    call_cost_usd`` for refinement calls (frozen contract §C: never borrows
    chunk-segment vocab/rates).

    **This function ORCHESTRATES ONLY** — it never inlines
    ``generate_content`` itself; every real Gemini call happens inside
    ``discover_match_highlights``/``refine_highlight_timestamp``, called
    from here.

    **Cap/stop granularity (frozen contract §C, this increment supersedes
    the PRIOR docstring claim here that there was "no mid-call granularity
    to check at" — that was true when 1 Gemini call == 1 match; it is FALSE
    now that refinement can add up to 2 more calls per match, one per
    highlight found).** The cap/stop check now happens THREE places: (1)
    before each match's discovery call (unchanged), (2) before EVERY
    per-highlight refinement call, including the ONE widen-retry attempt —
    each is checked separately, never "once per match" (a match with 5
    highlights found can spend the cap mid-refinement, after highlight 2 of
    5, say). When the cap/stop fires mid-refinement-loop: the PARTIAL result
    for that match is emitted immediately — already-refined highlights keep
    their real outcome, the highlight CURRENTLY being decided and every
    highlight after it in the loop are tagged unrefined
    (``_unattempted_enriched_highlight``) — then the terminal
    ``cap_reached``/``run_stopped`` event fires and the generator returns.
    NEVER re-attempted, NEVER silently dropped — the partial match result
    (with its real matches_id) is still in ``results`` when the caller
    persists them.

    A ``HighlightDiscoveryError``/``RefinementError`` for one call is a real
    spent-but-failed call (mirrors ``global_cap_driver.is_call_event``'s
    treatment of a chunk_segment/chunk_analyze ``error`` event) — counted
    against the cap, surfaced as an ``error`` event, and the batch/loop
    continues (to the next match for a discovery failure; to the next
    highlight for a refinement failure) rather than aborting for one bad
    response.
    """
    yield {"type": "highlight_batch_start", "match_count": len(targets_and_windows), "cap": state.cap}

    results: dict[str, dict] = {}
    for target, window in targets_and_windows:
        if state.stop_requested:
            yield {
                "type": "run_stopped", "reason": state.stop_reason or "stop requested",
                "calls_spent": state.calls_spent, "dollars_spent": round(state.dollars_spent, 4),
                "results": results,
            }
            return
        if state.calls_spent >= state.cap:
            yield {"type": "cap_reached", "calls_spent": state.calls_spent, "cap": state.cap, "results": results}
            return

        yield {"type": "match_start", "match_id": target.match_id, "scope": [window.start_s, window.end_s]}
        try:
            result = await discover_match_highlights(
                match_id=target.match_id, video_url=target.video_url, window=window,
                gemini_api_key=gemini_api_key, request_timeout_ms=request_timeout_ms,
                model=model, thinking_level=thinking_level,
                system_prompt=system_prompt, followup_prompt=followup_prompt,
                fps=fps, frame_quality=frame_quality,
            )
        except HighlightDiscoveryError as e:
            state.calls_spent += 1
            state.dollars_spent += PASS1_PLACEHOLDER_USD
            yield {
                "type": "error", "stage_id": "highlight_discovery", "call_kind": "discovery",
                "match_id": target.match_id, "message": str(e),
                "global_calls_spent": state.calls_spent, "global_dollars_spent": round(state.dollars_spent, 4),
            }
            continue

        state.calls_spent += 1
        state.dollars_spent += event_cost_usd({"type": "chunk_map", "usage_metadata": result["usage_metadata"]})
        yield {
            "type": "highlight_result", "call_kind": "discovery", "match_id": target.match_id,
            "highlights": result["highlights"],
            "prompt_text": result["prompt_text"], "raw_response_text": result["raw_response_text"],
            "usage_metadata": result["usage_metadata"], "timing": result["timing"],
            # Live investigation (2026-07-18): surfaced in the stream so the
            # founder SEES "Gemini returned N highlights past the clip end"
            # live, not just in a persisted file — detection only, never a
            # scoring-policy decision (LeCun's call, not made here).
            "out_of_bounds_count": result["out_of_bounds_count"],
            "out_of_bounds_spans": result["out_of_bounds_spans"],
            "global_calls_spent": state.calls_spent, "global_dollars_spent": round(state.dollars_spent, 4),
        }

        # ------------------------------------------------------------- #
        # Per-highlight refinement orchestration (frozen contract §B/§C).
        # ------------------------------------------------------------- #
        enriched_highlights: list[dict] = []
        cap_or_stop_fired = False
        for h_index, h in enumerate(result["highlights"]):
            reject_reason = validate_highlight_candidate(h["start_s"], h["end_s"], *result["sent_window_s"])
            if reject_reason is not None:
                enriched_highlights.append(_rejected_enriched_highlight(h, reject_reason))
                yield {
                    "type": "highlight_rejected", "match_id": target.match_id, "highlight_index": h_index,
                    "rejection_reason": reject_reason, "start_s": h["start_s"], "end_s": h["end_s"],
                    "reason": h.get("reason"),
                    "message": (
                        f"rejected ({reject_reason}): candidate [{h['start_s']:.2f}, {h['end_s']:.2f}]s "
                        "falls outside the match's sent scope window — zero calls/dollars billed."
                    ),
                }
                continue

            if cap_or_stop_fired or not refine:
                enriched_highlights.append(_unattempted_enriched_highlight(h))
                continue

            if state.stop_requested or state.calls_spent >= state.cap:
                # Mid-refinement cap/stop: THIS highlight and every one
                # after it in the loop are unrefined — never re-attempted,
                # never dropped from the persisted/emitted result.
                enriched_highlights.append(_unattempted_enriched_highlight(h))
                cap_or_stop_fired = True
                continue

            reason = h.get("reason") or ""
            try:
                attempt = await refine_highlight_timestamp(
                    match_id=target.match_id, video_url=target.video_url, match_window=window,
                    original_start_s=h["start_s"], original_end_s=h["end_s"], reason=reason,
                    gemini_api_key=gemini_api_key, request_timeout_ms=request_timeout_ms,
                    model=model, thinking_level=thinking_level, refine_fps=refine_fps,
                    pad_before_s=refine_pad_before_s, pad_after_s=refine_pad_after_s,
                )
            except RefinementPreCallError as e:
                # AC1-7: never reached generate_content -- zero calls/dollars billed.
                yield {
                    "type": "error", "stage_id": "refine_highlight_timestamp", "call_kind": "refine",
                    "match_id": target.match_id, "highlight_index": h_index, "message": str(e), "billed": False,
                    "global_calls_spent": state.calls_spent, "global_dollars_spent": round(state.dollars_spent, 4),
                }
                enriched_highlights.append(_enriched_highlight(h, refinement_failed=True))
                continue
            except RefinementError as e:
                state.calls_spent += 1
                state.dollars_spent += REFINE_PLACEHOLDER_USD
                yield {
                    "type": "error", "stage_id": "refine_highlight_timestamp", "call_kind": "refine",
                    "match_id": target.match_id, "highlight_index": h_index, "message": str(e), "billed": True,
                    "global_calls_spent": state.calls_spent, "global_dollars_spent": round(state.dollars_spent, 4),
                }
                enriched_highlights.append(_enriched_highlight(h, refinement_failed=True))
                continue

            state.calls_spent += 1
            state.dollars_spent += event_cost_usd({"type": "chunk_result", "usage_metadata": attempt["usage_metadata"]}) \
                if attempt["usage_metadata"] else REFINE_PLACEHOLDER_USD
            decision = decide_refinement_acceptance(attempt)
            yield {
                "type": "refine_result", "call_kind": "refine", "match_id": target.match_id,
                "highlight_index": h_index, "attempt": 1, "action_confirmed": attempt["action_confirmed"],
                "accepted": decision["accept"], "reject_reason": decision["reject_reason"],
                "prompt_text": attempt["prompt_text"], "raw_response_text": attempt["raw_response_text"],
                "usage_metadata": attempt["usage_metadata"], "timing": attempt["timing"],
                "global_calls_spent": state.calls_spent, "global_dollars_spent": round(state.dollars_spent, 4),
            }

            if decision["accept"]:
                enriched_highlights.append(_enriched_highlight(
                    h, refined_start_s=attempt["start_s"], refined_end_s=attempt["end_s"],
                    refined=True, action_confirmed=True, refinement_failed=False,
                ))
                continue

            if attempt["action_confirmed"]:
                # Confirmed but rejected on geometry (edge/OOB) — frozen
                # contract §B: the widen-retry triggers ONLY on "not
                # confirmed", never on a confirmed-but-suspicious result.
                enriched_highlights.append(_enriched_highlight(
                    h, action_confirmed=True, refinement_failed=True,
                ))
                continue

            # Not confirmed -> the ONE conditional widen-retry (frozen
            # contract §B: pad_before doubled backward only, then HARD
            # STOP) — its own cap/stop check, since it is a SEPARATE
            # Gemini call.
            if state.stop_requested or state.calls_spent >= state.cap:
                enriched_highlights.append(_unattempted_enriched_highlight(h))
                cap_or_stop_fired = True
                continue

            try:
                retry_attempt = await refine_highlight_timestamp(
                    match_id=target.match_id, video_url=target.video_url, match_window=window,
                    original_start_s=h["start_s"], original_end_s=h["end_s"], reason=reason,
                    gemini_api_key=gemini_api_key, request_timeout_ms=request_timeout_ms,
                    model=model, thinking_level=thinking_level, refine_fps=refine_fps,
                    pad_before_s=WIDEN_PAD_BEFORE_S, pad_after_s=refine_pad_after_s,
                )
            except RefinementPreCallError as e:
                # AC1-7: never reached generate_content -- zero calls/dollars billed.
                yield {
                    "type": "error", "stage_id": "refine_highlight_timestamp", "call_kind": "refine",
                    "match_id": target.match_id, "highlight_index": h_index, "message": str(e), "billed": False,
                    "global_calls_spent": state.calls_spent, "global_dollars_spent": round(state.dollars_spent, 4),
                }
                enriched_highlights.append(_enriched_highlight(h, refinement_failed=True))
                continue
            except RefinementError as e:
                state.calls_spent += 1
                state.dollars_spent += REFINE_PLACEHOLDER_USD
                yield {
                    "type": "error", "stage_id": "refine_highlight_timestamp", "call_kind": "refine",
                    "match_id": target.match_id, "highlight_index": h_index, "message": str(e), "billed": True,
                    "global_calls_spent": state.calls_spent, "global_dollars_spent": round(state.dollars_spent, 4),
                }
                enriched_highlights.append(_enriched_highlight(h, refinement_failed=True))
                continue

            state.calls_spent += 1
            state.dollars_spent += (
                event_cost_usd({"type": "chunk_result", "usage_metadata": retry_attempt["usage_metadata"]})
                if retry_attempt["usage_metadata"] else REFINE_PLACEHOLDER_USD
            )
            retry_decision = decide_refinement_acceptance(retry_attempt)
            yield {
                "type": "refine_result", "call_kind": "refine", "match_id": target.match_id,
                "highlight_index": h_index, "attempt": 2, "action_confirmed": retry_attempt["action_confirmed"],
                "accepted": retry_decision["accept"], "reject_reason": retry_decision["reject_reason"],
                "prompt_text": retry_attempt["prompt_text"], "raw_response_text": retry_attempt["raw_response_text"],
                "usage_metadata": retry_attempt["usage_metadata"], "timing": retry_attempt["timing"],
                "global_calls_spent": state.calls_spent, "global_dollars_spent": round(state.dollars_spent, 4),
            }

            if retry_decision["accept"]:
                enriched_highlights.append(_enriched_highlight(
                    h, refined_start_s=retry_attempt["start_s"], refined_end_s=retry_attempt["end_s"],
                    refined=True, action_confirmed=True, refinement_failed=False,
                ))
            else:
                # HARD STOP — one pass + one widen-retry, no fabrication.
                enriched_highlights.append(_enriched_highlight(
                    h, action_confirmed=retry_attempt["action_confirmed"], refinement_failed=True,
                ))

        enriched_result = dict(result)
        enriched_result["highlights"] = enriched_highlights
        results[target.match_id] = enriched_result
        yield {"type": "match_complete", "match_id": target.match_id, "highlight_count": len(enriched_highlights)}

        if cap_or_stop_fired:
            # This match's PARTIAL result (already in ``results`` above,
            # unrefined highlights included and tagged) is included in the
            # terminal event so the caller can persist it — frozen contract
            # §C: "emit the PARTIAL result... NEVER re-attempt or drop the
            # remainder."
            if state.stop_requested:
                yield {
                    "type": "run_stopped", "reason": state.stop_reason or "stop requested",
                    "calls_spent": state.calls_spent, "dollars_spent": round(state.dollars_spent, 4),
                    "results": results,
                }
            else:
                yield {
                    "type": "cap_reached", "calls_spent": state.calls_spent, "cap": state.cap, "results": results,
                }
            return

    yield {
        "type": "batch_complete", "calls_spent": state.calls_spent,
        "dollars_spent": round(state.dollars_spent, 4), "results": results,
    }
