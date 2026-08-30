"""LeCun's dry-run $ cost model for the ADCC eval cost-control dashboard.

    $_batch = 1.5 x SUM_runs[ scope_s*0.0000335 + 0.00165 + round(scope_min*2.0)*0.00285 ]

One "run" is one ``chunk-segment-tags`` pipeline invocation — i.e. one
``scope_policy.Slice`` (a match that needed no over-cap slicing is exactly
one run; an over-cap match sliced into N pieces is N runs). Pure arithmetic,
no Gemini import, no I/O — safe to call from a dry-run endpoint that must
NEVER spend a call.

Reference point (LeCun): the full 258-match / 344-slice real batch prices to
roughly $17-24 pre-margin (``estimate_batch_cost_usd(..., apply_margin=False)``)
and ~$30-36 with the 1.5x margin applied (the default, and the number the
dashboard should show as the headline "you might spend this much" figure —
cost-safety dominant, per Jobs's spec, so the MORE conservative number is the
one always show).
"""
from __future__ import annotations

from typing import Sequence

__all__ = [
    "MARGIN_MULTIPLIER",
    "PER_SECOND_USD",
    "PER_RUN_FLAT_USD",
    "PER_CHUNK_USD",
    "CHUNKS_PER_MINUTE",
    "estimate_run_cost_usd",
    "estimate_batch_cost_usd",
    "estimate_run_calls_realistic",
    "estimate_batch_calls_realistic",
    "DEFAULT_TOKENS_PER_FRAME",
    "HIGH_RES_TOKENS_PER_FRAME",
    "MEDIA_RESOLUTION_TOKEN_MULTIPLIER",
    "AUDIO_FLOOR_USD_PER_SEC",
    "video_cost_per_sec_usd",
    "highlight_call_rate_usd_per_sec",
    "estimate_highlight_call_cost_usd",
]

MARGIN_MULTIPLIER: float = 1.5
PER_SECOND_USD: float = 0.0000335
PER_RUN_FLAT_USD: float = 0.00165
PER_CHUNK_USD: float = 0.00285
CHUNKS_PER_MINUTE: float = 2.0


# --------------------------------------------------------------------------- #
# Highlight Eval feature — LeCun's cost DECOMPOSITION (frozen contract).
#
# ``PER_SECOND_USD`` above secretly bundles TWO physically different costs:
# fps/resolution-INVARIANT audio-track tokenization, and fps/resolution-
# SCALING video-frame tokenization. A naive "x fps" or "x resolution" on that
# single blended constant is WRONG — it would scale the audio cost too, which
# Gemini does not actually do (audio is sampled at its own fixed rate
# regardless of the video fps/media_resolution the caller requests). This
# decomposition separates the two so ``estimate_highlight_call_cost_usd``
# below can price an arbitrary (fps, media_resolution) pair correctly.
#
# ``_INPUT_USD_PER_TOKEN`` is the published Gemini input-token rate
# ($0.25/1M) — the SAME rate ``global_cap_driver.INPUT_USD_PER_TOKEN`` prices
# PASS-1/PASS-2 calls at. Duplicated here (never imported) so this module
# stays the zero-dependency pure-arithmetic module its own docstring
# promises — importing ``global_cap_driver`` would transitively pull in
# ``service.pipelines.executors`` (INS-090's barrel-entanglement risk),
# which a dry-run pricing module must never touch.
# --------------------------------------------------------------------------- #
_INPUT_USD_PER_TOKEN: float = 0.25 / 1_000_000

# Gemini's published per-frame input-token cost at default/low/medium
# media_resolution is 70 tokens/frame; MEDIA_RESOLUTION_HIGH is exactly 4x
# that (280) — frozen contract constants, not independently derived here.
DEFAULT_TOKENS_PER_FRAME: int = 70
HIGH_RES_TOKENS_PER_FRAME: int = 280

MEDIA_RESOLUTION_TOKEN_MULTIPLIER: dict[str, int] = {
    "default": DEFAULT_TOKENS_PER_FRAME,
    "low": DEFAULT_TOKENS_PER_FRAME,
    "medium": DEFAULT_TOKENS_PER_FRAME,
    "high": HIGH_RES_TOKENS_PER_FRAME,
}

# The audio-only remainder of PER_SECOND_USD once fps=1/default's OWN
# video-frame cost (1 frame/sec x 70 tokens/frame x the per-token rate) is
# subtracted out. Derived ONCE from the two numbers above — never a second,
# independently-eyeballed constant that could silently drift from
# PER_SECOND_USD if either changes.
AUDIO_FLOOR_USD_PER_SEC: float = PER_SECOND_USD - (1 * DEFAULT_TOKENS_PER_FRAME * _INPUT_USD_PER_TOKEN)


def video_cost_per_sec_usd(fps: float, media_resolution: str) -> float:
    """The fps/resolution-SCALING half of LeCun's decomposition:
    ``fps x tokens_per_frame(media_resolution) x per-token rate``. Raises
    ``ValueError`` for a ``media_resolution`` outside
    ``MEDIA_RESOLUTION_TOKEN_MULTIPLIER`` — never silently coerced to a
    default tier."""
    if media_resolution not in MEDIA_RESOLUTION_TOKEN_MULTIPLIER:
        raise ValueError(
            f"media_resolution must be one of {sorted(MEDIA_RESOLUTION_TOKEN_MULTIPLIER)}, "
            f"got {media_resolution!r}"
        )
    return fps * MEDIA_RESOLUTION_TOKEN_MULTIPLIER[media_resolution] * _INPUT_USD_PER_TOKEN


def highlight_call_rate_usd_per_sec(fps: float, media_resolution: str) -> float:
    """Video (fps/resolution-scaling) + audio (fps/resolution-INVARIANT
    floor) per-second rate for one highlight-discovery Gemini call. At
    fps=1/"default" this reduces to EXACTLY ``PER_SECOND_USD`` (the
    decomposition is a re-derivation of the existing baseline, not a new
    number at the baseline point)."""
    return video_cost_per_sec_usd(fps, media_resolution) + AUDIO_FLOOR_USD_PER_SEC


def estimate_highlight_call_cost_usd(duration_s: float, fps: float, media_resolution: str) -> float:
    """ONE highlight-discovery Gemini call's $ estimate: ``duration_s x
    rate(fps, media_resolution) + PER_RUN_FLAT_USD`` — ONE flat per-call
    overhead, deliberately NO chunk multiplier (``PER_CHUNK_USD`` never
    applies here: this feature is 1 Gemini call == 1 match, no PASS-2
    per-chunk loop, unlike ``estimate_run_cost_usd`` above which prices a
    ``chunk-segment-tags`` run). This is the dedicated pricing path for
    ``/highlight/plan``/``/highlight/run`` — the tag-batch cost path
    (``estimate_run_cost_usd``/``estimate_batch_cost_usd``) is untouched and
    still used for ``/plan``/``/run``."""
    return duration_s * highlight_call_rate_usd_per_sec(fps, media_resolution) + PER_RUN_FLAT_USD


def estimate_run_cost_usd(scope_s: float) -> float:
    """Pre-margin $ estimate for ONE run (one slice) of the given scope
    duration, per LeCun's formula. Never negative (a non-positive scope_s is
    a caller bug — scope_policy.Slice.duration_s is always > 0 by
    construction — but this function does not itself validate/raise; callers
    feed it real Slice durations)."""
    scope_min = scope_s / 60.0
    chunk_count = round(scope_min * CHUNKS_PER_MINUTE)
    return scope_s * PER_SECOND_USD + PER_RUN_FLAT_USD + chunk_count * PER_CHUNK_USD


def estimate_batch_cost_usd(scope_seconds: Sequence[float], *, apply_margin: bool = True) -> float:
    """Sums ``estimate_run_cost_usd`` over every run's scope duration, then
    applies LeCun's 1.5x margin by default (``apply_margin=False`` returns
    the "realistic" pre-margin figure for a secondary/reference display —
    the dashboard's HEADLINE number should always be the margin-applied
    one)."""
    total = sum(estimate_run_cost_usd(s) for s in scope_seconds)
    return total * MARGIN_MULTIPLIER if apply_margin else total


def estimate_run_calls_realistic(scope_s: float) -> int:
    """LeCun's realistic Gemini CALL-COUNT estimate for ONE run (one slice):
    ``1 + round(scope_minutes * CHUNKS_PER_MINUTE)`` -- one PASS-1 rough-scan
    call plus one PASS-2 call per estimated chunk. This is the exact same
    ``chunk_count`` arithmetic ``estimate_run_cost_usd`` already prices above
    (never a second, drifting formula) -- just returned as an integer call
    count instead of a dollar figure. This is the dashboard's PRIMARY
    (headline) call estimate, as distinct from the pathological worst-case
    ceiling computed by ``service.pipelines.executors.estimate_run_plan``
    (``1 + max_chunks`` off the theoretical MINIMUM chunk size), which stays
    a secondary "cap suggestion" number only."""
    scope_min = scope_s / 60.0
    chunk_count = round(scope_min * CHUNKS_PER_MINUTE)
    return 1 + chunk_count


def estimate_batch_calls_realistic(scope_seconds: Sequence[float]) -> int:
    """Sums ``estimate_run_calls_realistic`` over every run's scope
    duration -- the dashboard's headline "you will likely spend about N
    calls" figure."""
    return sum(estimate_run_calls_realistic(s) for s in scope_seconds)
