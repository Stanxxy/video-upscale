"""``eval/adcc/predict/cost_model.py`` — LeCun's dry-run $ formula. Pure
arithmetic, no I/O, no Gemini import at all."""
from __future__ import annotations

import pytest

from eval.adcc.predict.cost_model import (
    AUDIO_FLOOR_USD_PER_SEC,
    CHUNKS_PER_MINUTE,
    DEFAULT_TOKENS_PER_FRAME,
    HIGH_RES_TOKENS_PER_FRAME,
    MARGIN_MULTIPLIER,
    PER_CHUNK_USD,
    PER_RUN_FLAT_USD,
    PER_SECOND_USD,
    estimate_batch_calls_realistic,
    estimate_batch_cost_usd,
    estimate_highlight_call_cost_usd,
    estimate_run_calls_realistic,
    estimate_run_cost_usd,
    highlight_call_rate_usd_per_sec,
    video_cost_per_sec_usd,
)


def test_estimate_run_cost_usd_matches_hand_computed_formula():
    scope_s = 600.0  # 10 minutes -> round(10*2)=20 chunks
    expected = 600.0 * PER_SECOND_USD + PER_RUN_FLAT_USD + 20 * PER_CHUNK_USD
    assert estimate_run_cost_usd(scope_s) == pytest.approx(expected)


def test_estimate_run_cost_usd_rounds_chunk_count_to_nearest_int():
    # 630s = 10.5 min * 2 = 21.0 chunks exactly -> round() = 21 (banker's
    # rounding never applies here since 21.0 isn't a .5 boundary itself).
    scope_s = 630.0
    chunk_count = round((scope_s / 60.0) * CHUNKS_PER_MINUTE)
    expected = scope_s * PER_SECOND_USD + PER_RUN_FLAT_USD + chunk_count * PER_CHUNK_USD
    assert estimate_run_cost_usd(scope_s) == pytest.approx(expected)


def test_estimate_batch_cost_usd_sums_across_runs_before_applying_margin():
    runs = [100.0, 200.0, 300.0]
    realistic = estimate_batch_cost_usd(runs, apply_margin=False)
    with_margin = estimate_batch_cost_usd(runs, apply_margin=True)
    assert with_margin == pytest.approx(realistic * MARGIN_MULTIPLIER)
    assert with_margin == pytest.approx(sum(estimate_run_cost_usd(s) for s in runs) * MARGIN_MULTIPLIER)


def test_estimate_batch_cost_usd_defaults_to_margin_applied():
    runs = [500.0]
    assert estimate_batch_cost_usd(runs) == estimate_batch_cost_usd(runs, apply_margin=True)


def test_estimate_batch_cost_usd_empty_batch_is_zero():
    assert estimate_batch_cost_usd([]) == 0.0


def test_full_batch_ballpark_matches_lecun_headline_reference():
    """Sanity check against LeCun's stated headline reference: ~163,590s
    total scoped seconds across 344 real slices prices to roughly $17-24
    pre-margin / ~$30-36 with the 1.5x margin (CALL_BUDGET_ESTIMATE.md §8).
    This test uses a SYNTHETIC 344-slice distribution with the same total
    scoped seconds (not the real gold-snapshot slice plan — that's exercised
    end-to-end via the /plan endpoint tests) purely to catch a gross formula
    regression (e.g. an accidental order-of-magnitude error), not to
    reproduce the exact real-corpus number."""
    avg_scope_s = 163_590.0 / 344
    runs = [avg_scope_s] * 344
    realistic = estimate_batch_cost_usd(runs, apply_margin=False)
    with_margin = estimate_batch_cost_usd(runs, apply_margin=True)
    assert 15.0 <= realistic <= 30.0
    assert 25.0 <= with_margin <= 45.0


# --------------------------------------------------------------------------- #
# estimate_run_calls_realistic / estimate_batch_calls_realistic — the
# PRIMARY (headline) realistic call-count estimate for the /matches catalog
# and the /plan preview, as distinct from the pathological worst-case
# ceiling (service.pipelines.executors.estimate_run_plan).
# --------------------------------------------------------------------------- #
def test_estimate_run_calls_realistic_matches_hand_computed_formula():
    scope_s = 600.0  # 10 minutes -> round(10*2)=20 chunks -> 1+20=21 calls
    assert estimate_run_calls_realistic(scope_s) == 21


def test_estimate_run_calls_realistic_is_one_plus_the_same_chunk_count_cost_uses():
    """Never a second, drifting formula: the "+1" flat term in
    estimate_run_cost_usd corresponds 1:1 to the PASS-1 call counted here,
    and PER_CHUNK_USD's chunk_count corresponds 1:1 to the PASS-2 calls
    counted here."""
    for scope_s in (37.0, 600.0, 719.0, 720.0, 1234.5):
        chunk_count = round((scope_s / 60.0) * CHUNKS_PER_MINUTE)
        assert estimate_run_calls_realistic(scope_s) == 1 + chunk_count


def test_estimate_batch_calls_realistic_sums_across_runs():
    runs = [100.0, 200.0, 300.0]
    expected = sum(estimate_run_calls_realistic(s) for s in runs)
    assert estimate_batch_calls_realistic(runs) == expected


def test_estimate_batch_calls_realistic_empty_batch_is_zero():
    assert estimate_batch_calls_realistic([]) == 0


# --------------------------------------------------------------------------- #
# Highlight Eval feature — LeCun's cost DECOMPOSITION (this increment).
# PER_SECOND_USD bundles fps/resolution-invariant audio cost with fps/
# resolution-scaling video-frame cost; a naive "x fps" on that single
# constant is wrong. These tests pin the decomposition and the frozen
# sanity-check multiplier table (fps=2 ~1.52x, HIGH ~2.57x, fps=2xHIGH
# ~4.66x, all relative to the fps=1/default baseline).
# --------------------------------------------------------------------------- #
def test_audio_floor_is_the_remainder_of_per_second_usd_at_baseline():
    # Derived ONCE from PER_SECOND_USD/DEFAULT_TOKENS_PER_FRAME -- never a
    # second, independently-eyeballed constant.
    expected = PER_SECOND_USD - (1 * DEFAULT_TOKENS_PER_FRAME * 0.25e-6)
    assert AUDIO_FLOOR_USD_PER_SEC == pytest.approx(expected)


def test_high_res_tokens_per_frame_is_exactly_4x_default():
    assert HIGH_RES_TOKENS_PER_FRAME == 4 * DEFAULT_TOKENS_PER_FRAME


def test_baseline_fps1_default_rate_equals_per_second_usd_exactly():
    """The decomposition is a re-derivation of the EXISTING baseline, not a
    new number at the baseline point (fps=1, media_resolution=default)."""
    assert highlight_call_rate_usd_per_sec(1.0, "default") == pytest.approx(PER_SECOND_USD)


def test_low_and_medium_price_the_same_as_default():
    base = highlight_call_rate_usd_per_sec(1.0, "default")
    assert highlight_call_rate_usd_per_sec(1.0, "low") == pytest.approx(base)
    assert highlight_call_rate_usd_per_sec(1.0, "medium") == pytest.approx(base)


def test_fps2_default_is_approximately_1_52x_baseline():
    base = highlight_call_rate_usd_per_sec(1.0, "default")
    fps2 = highlight_call_rate_usd_per_sec(2.0, "default")
    assert fps2 / base == pytest.approx(1.52, abs=0.01)


def test_fps1_high_is_approximately_2_57x_baseline():
    base = highlight_call_rate_usd_per_sec(1.0, "default")
    high = highlight_call_rate_usd_per_sec(1.0, "high")
    assert high / base == pytest.approx(2.57, abs=0.01)


def test_fps2_high_is_approximately_4_66x_baseline():
    base = highlight_call_rate_usd_per_sec(1.0, "default")
    fps2_high = highlight_call_rate_usd_per_sec(2.0, "high")
    assert fps2_high / base == pytest.approx(4.66, abs=0.01)


def test_video_cost_per_sec_usd_rejects_unknown_media_resolution():
    with pytest.raises(ValueError):
        video_cost_per_sec_usd(1.0, "ultra")


def test_estimate_highlight_call_cost_usd_has_no_chunk_multiplier():
    """1 Gemini call == 1 match for this feature -- PER_CHUNK_USD must never
    appear in this formula, unlike estimate_run_cost_usd above."""
    duration_s = 312.0
    expected = duration_s * highlight_call_rate_usd_per_sec(1.0, "default") + PER_RUN_FLAT_USD
    assert estimate_highlight_call_cost_usd(duration_s, 1.0, "default") == pytest.approx(expected)
    # Sanity: this must NOT match the old chunk-ballpark reuse (which
    # fabricated a PASS-2 chunk cost that never applies to this 1-call
    # feature) -- the whole point of this increment.
    old_ballpark = estimate_run_cost_usd(duration_s)
    assert estimate_highlight_call_cost_usd(duration_s, 1.0, "default") < old_ballpark


def test_estimate_highlight_call_cost_usd_312s_default_price_delta():
    """Pins the reported default-price delta for the representative 312s
    match: OLD chunk-ballpark reuse vs. NEW dedicated highlight formula."""
    duration_s = 312.0
    old_realistic = estimate_run_cost_usd(duration_s)  # chunk-ballpark reuse (pre-increment /highlight/plan path)
    new_realistic = estimate_highlight_call_cost_usd(duration_s, 1.0, "default")
    assert old_realistic == pytest.approx(0.040602)
    assert new_realistic == pytest.approx(0.012102)
    assert new_realistic / old_realistic == pytest.approx(0.298, abs=0.01)
