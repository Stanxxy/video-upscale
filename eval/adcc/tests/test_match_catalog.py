"""``eval/adcc/predict/match_catalog.py`` — the /matches menu's zero-Gemini
per-match catalog (name + length + dual realistic/worst-case call estimates).
"""
from __future__ import annotations

import pytest

from eval.adcc.predict.cost_model import estimate_batch_calls_realistic, estimate_batch_cost_usd
from eval.adcc.predict.dashboard_scope import GoldCorpus
from eval.adcc.predict.match_catalog import build_match_catalog
from eval.adcc.predict.scope_policy import (
    MULTI_MATCH_POST_MARGIN_S,
    MULTI_MATCH_PRE_MARGIN_S,
    slice_scope,
    single_match_scope,
)


def _corpus_records():
    """One single-match video (short, under the 720s slice cap), one
    multi-match VOD with two sibling matches, and one deliberately-long
    single-match video (over the slice cap, so it must slice into >1 run)."""
    records = [
        {
            "match_id": "gordon_v_marcus", "name": "Gordon Ryan vs Marcus Almeida",
            "video_url": "https://x/gordon_marcus", "video_url_md5": "stem_gordon_marcus",
            "events": [{"t_s": 10}],
        },
    ]
    records += [
        {
            "match_id": "multi_0", "name": "PJ Barch vs Kade Ruotolo",
            "video_url": "https://x/multi", "video_url_md5": "stem_multi",
            "events": [{"t_s": 100}, {"t_s": 400}],
        },
        {
            "match_id": "multi_1", "name": "Andre Galvao vs Buchecha",
            "video_url": "https://x/multi", "video_url_md5": "stem_multi",
            "events": [{"t_s": 900}],
        },
    ]
    records += [
        {
            "match_id": "long_single", "name": "AAA Marathon Match",
            "video_url": "https://x/long_single", "video_url_md5": "stem_long_single",
            "events": [{"t_s": 5}],
        },
    ]
    return records


def _corpus():
    return GoldCorpus(gold_records=_corpus_records(), split_assignment=None)


def _corpus_with_one_no_duration_target():
    """Single-match corpus whose ffprobe duration is intentionally absent —
    used to assert catalog fields that must still be present when
    has_duration=False (e.g. video_url for Load match video)."""
    return GoldCorpus(
        gold_records=[
            {
                "match_id": "no_dur_match",
                "name": "No Duration Match",
                "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "video_url_md5": "stem_no_dur",
                "events": [{"t_s": 10}],
            },
        ],
        split_assignment=None,
    )


def _durations():
    return {"stem_gordon_marcus": 1471.0, "stem_long_single": 1800.0}  # 24:31, 30:00


class TestBuildMatchCatalog:
    def test_returns_one_entry_per_match_id(self):
        catalog = build_match_catalog(_corpus(), _durations())
        assert {e["match_id"] for e in catalog} == {"gordon_v_marcus", "multi_0", "multi_1", "long_single"}

    def test_sorted_by_name(self):
        catalog = build_match_catalog(_corpus(), _durations())
        names = [e["name"] for e in catalog]
        assert names == sorted(names)

    def test_single_match_length_is_the_ffprobe_duration(self):
        catalog = build_match_catalog(_corpus(), _durations())
        entry = next(e for e in catalog if e["match_id"] == "gordon_v_marcus")
        assert entry["name"] == "Gordon Ryan vs Marcus Almeida"
        assert entry["is_multi_match"] is False
        assert entry["video_md5"] == "stem_gordon_marcus"
        assert entry["has_duration"] is True
        assert entry["length_seconds"] == pytest.approx(1471.0)
        assert entry["length_mmss"] == "24:31"

    def test_multi_match_length_is_the_gold_event_span_window(self):
        catalog = build_match_catalog(_corpus(), _durations())
        entry = next(e for e in catalog if e["match_id"] == "multi_0")
        assert entry["is_multi_match"] is True
        expected_start = max(0.0, 100.0 - MULTI_MATCH_PRE_MARGIN_S)
        expected_end = 400.0 + MULTI_MATCH_POST_MARGIN_S
        assert entry["length_seconds"] == pytest.approx(expected_end - expected_start)

    def test_est_calls_realistic_matches_cost_model_over_the_real_slice_list(self):
        """gordon_v_marcus's 1471s (24:31) duration exceeds the 720s slice
        cap, so it slices into 3 runs (720, 720, 121) -- the estimate must be
        summed over THOSE real slices, not derived from the raw window
        duration in one shot (which would undercount)."""
        catalog = build_match_catalog(_corpus(), _durations())
        entry = next(e for e in catalog if e["match_id"] == "gordon_v_marcus")
        window = single_match_scope("gordon_v_marcus", 1471.0)
        slices = slice_scope(window)
        assert len(slices) == 3
        expected = estimate_batch_calls_realistic([sl.duration_s for sl in slices])
        assert entry["est_calls_realistic"] == expected

    def test_est_dollars_realistic_matches_cost_model_over_the_real_slice_list(self):
        catalog = build_match_catalog(_corpus(), _durations())
        entry = next(e for e in catalog if e["match_id"] == "gordon_v_marcus")
        window = single_match_scope("gordon_v_marcus", 1471.0)
        slices = slice_scope(window)
        expected = round(estimate_batch_cost_usd([sl.duration_s for sl in slices], apply_margin=False), 2)
        assert entry["est_dollars_realistic"] == expected

    def test_over_cap_match_sums_calls_across_its_slices_not_one_flat_window_call(self):
        """long_single (1800s) exceeds the 720s slice cap -> >1 slice -> the
        realistic/worst-case call counts must be the SUM over every slice,
        not a single formula application to the whole 1800s window (which
        would undercount -- each slice restarts its own PASS-1 rough scan)."""
        catalog = build_match_catalog(_corpus(), _durations())
        entry = next(e for e in catalog if e["match_id"] == "long_single")
        window = single_match_scope("long_single", 1800.0)
        slices = slice_scope(window)
        assert len(slices) > 1
        expected_realistic = estimate_batch_calls_realistic([sl.duration_s for sl in slices])
        assert entry["est_calls_realistic"] == expected_realistic
        # length_seconds stays the raw window, NOT the slice-duration sum
        # (which would double count the SLICE_OVERLAP_S seam overlap).
        assert entry["length_seconds"] == pytest.approx(window.duration_s)
        assert entry["length_seconds"] < sum(sl.duration_s for sl in slices)

    def test_worst_case_calls_is_greater_or_equal_to_realistic(self):
        """The pathological ceiling must never be BELOW the realistic
        estimate for any real (non-degenerate) scope -- it's derived from the
        theoretical MINIMUM chunk size, a stricter upper bound."""
        catalog = build_match_catalog(_corpus(), _durations())
        for entry in catalog:
            if entry["has_duration"]:
                assert entry["est_calls_worst_case"] >= entry["est_calls_realistic"]

    def test_missing_duration_is_flagged_not_guessed(self):
        """gordon_v_marcus's duration is deliberately withheld here -- the
        catalog must flag has_duration=False, not crash the whole /matches
        response or guess a fabricated length."""
        catalog = build_match_catalog(_corpus(), {"stem_long_single": 1800.0})
        entry = next(e for e in catalog if e["match_id"] == "gordon_v_marcus")
        assert entry["has_duration"] is False
        assert entry["length_seconds"] is None
        assert entry["length_mmss"] is None
        assert entry["est_calls_realistic"] is None
        assert entry["est_calls_worst_case"] is None
        assert entry["est_dollars_realistic"] is None
        # Every OTHER match with a resolvable duration is still present and
        # fully computed -- one bad entry never takes down the whole catalog.
        multi_entry = next(e for e in catalog if e["match_id"] == "multi_0")
        assert multi_entry["has_duration"] is True

    def test_missing_name_field_raises_loudly_not_a_silent_hex_id_default(self):
        records = _corpus_records()
        del records[0]["name"]
        corpus = GoldCorpus(gold_records=records, split_assignment=None)
        with pytest.raises(KeyError):
            build_match_catalog(corpus, _durations())

    def test_catalog_entry_exposes_video_url_even_without_a_known_duration(self):
        # AC2-1: Load match video must resolve video_url from /matches alone,
        # never from /highlight/plan (that would spend the priced-flow gate).
        corpus = _corpus_with_one_no_duration_target()
        catalog = build_match_catalog(corpus, video_durations={})
        assert catalog[0]["has_duration"] is False
        assert catalog[0]["video_url"] == corpus.all_targets()[catalog[0]["match_id"]].video_url
