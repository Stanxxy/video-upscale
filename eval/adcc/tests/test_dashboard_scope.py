"""``eval/adcc/predict/dashboard_scope.py`` — the ADCC eval dashboard's scope
resolution (Task 3): exactly four modes (single/list/smoke/split), no "all",
plus the plan-token staleness helper."""
from __future__ import annotations

import json

import pytest

from eval.adcc.predict.dashboard_scope import (
    GoldCorpus,
    ScopeError,
    ScopeSpec,
    compute_plan_token,
    load_gold_corpus,
    resolve_scope_targets,
)


def _corpus_records():
    records = [
        {"match_id": "s_small", "video_url": "https://x/s_small", "video_url_md5": "stem_s_small",
         "events": [{"t_s": 10}]},
    ]
    records += [
        {"match_id": f"multi_small_{i}", "video_url": "https://x/multi_small", "video_url_md5": "stem_multi_small",
         "events": [{"t_s": 100 * i}]}
        for i in range(2)
    ]
    records += [
        {"match_id": f"multi_big_{i}", "video_url": "https://x/multi_big", "video_url_md5": "stem_multi_big",
         "events": [{"t_s": 1000 * i}]}
        for i in range(6)
    ]
    return records


def _corpus(split_assignment=None) -> GoldCorpus:
    return GoldCorpus(gold_records=_corpus_records(), split_assignment=split_assignment)


def _split_assignment():
    return {
        "assignment": {
            "stem_s_small": {"split": "dev", "match_ids": ["s_small"], "event_count": 1},
            "stem_multi_small": {"split": "test", "match_ids": ["multi_small_0", "multi_small_1"], "event_count": 2},
            "stem_multi_big": {
                "split": "sandbox", "match_ids": [f"multi_big_{i}" for i in range(6)], "event_count": 6,
            },
        },
        "ratios": {}, "seed": 0, "video_counts": {},
    }


class TestModeSingle:
    def test_resolves_exactly_one_target(self):
        targets = resolve_scope_targets(ScopeSpec(mode="single", match_id="s_small"), _corpus())
        assert [t.match_id for t in targets] == ["s_small"]

    def test_missing_match_id_raises(self):
        with pytest.raises(ScopeError):
            resolve_scope_targets(ScopeSpec(mode="single"), _corpus())

    def test_unknown_match_id_raises(self):
        with pytest.raises(ScopeError):
            resolve_scope_targets(ScopeSpec(mode="single", match_id="ghost"), _corpus())


class TestModeList:
    def test_resolves_the_hand_picked_ids_in_order(self):
        targets = resolve_scope_targets(
            ScopeSpec(mode="list", match_ids=("multi_small_1", "s_small")), _corpus(),
        )
        assert [t.match_id for t in targets] == ["multi_small_1", "s_small"]

    def test_empty_list_raises(self):
        with pytest.raises(ScopeError):
            resolve_scope_targets(ScopeSpec(mode="list", match_ids=()), _corpus())

    def test_unknown_id_in_list_raises(self):
        with pytest.raises(ScopeError):
            resolve_scope_targets(ScopeSpec(mode="list", match_ids=("s_small", "ghost")), _corpus())


class TestModeSmoke:
    def test_resolves_the_fixed_3_targets(self):
        targets = resolve_scope_targets(ScopeSpec(mode="smoke"), _corpus())
        ids = {t.match_id for t in targets}
        assert len(targets) == 3
        assert "s_small" in ids


class TestModeSplit:
    def test_resolves_the_named_split_only(self):
        targets = resolve_scope_targets(ScopeSpec(mode="split", split="sandbox"), _corpus(_split_assignment()))
        assert {t.match_id for t in targets} == {f"multi_big_{i}" for i in range(6)}

    def test_invalid_split_name_raises(self):
        with pytest.raises(ScopeError):
            resolve_scope_targets(ScopeSpec(mode="split", split="all"), _corpus(_split_assignment()))

    def test_missing_split_assignment_raises(self):
        with pytest.raises(ScopeError):
            resolve_scope_targets(ScopeSpec(mode="split", split="dev"), _corpus(None))


class TestNoAllMode:
    def test_all_is_not_a_valid_mode(self):
        with pytest.raises(ScopeError):
            resolve_scope_targets(ScopeSpec(mode="all"), _corpus())


class TestLoadGoldCorpus:
    def test_loads_jsonl_and_split_json(self, tmp_path):
        gold_path = tmp_path / "gold.jsonl"
        with open(gold_path, "w") as f:
            for rec in _corpus_records():
                f.write(json.dumps(rec) + "\n")
        split_path = tmp_path / "split.json"
        split_path.write_text(json.dumps(_split_assignment()))

        corpus = load_gold_corpus(gold_path, split_path)
        assert len(corpus.gold_records) == 9
        assert corpus.split_assignment is not None

    def test_missing_split_assignment_file_is_none_not_an_error(self, tmp_path):
        gold_path = tmp_path / "gold.jsonl"
        with open(gold_path, "w") as f:
            f.write(json.dumps(_corpus_records()[0]) + "\n")
        corpus = load_gold_corpus(gold_path, tmp_path / "does-not-exist.json")
        assert corpus.split_assignment is None


class TestComputePlanToken:
    def test_deterministic_for_the_same_scope_and_cap(self):
        spec = ScopeSpec(mode="single", match_id="s_small")
        assert compute_plan_token(spec.canonical(), 100) == compute_plan_token(spec.canonical(), 100)

    def test_changes_when_cap_changes(self):
        spec = ScopeSpec(mode="single", match_id="s_small")
        assert compute_plan_token(spec.canonical(), 100) != compute_plan_token(spec.canonical(), 101)

    def test_changes_when_scope_changes(self):
        spec_a = ScopeSpec(mode="single", match_id="s_small")
        spec_b = ScopeSpec(mode="single", match_id="multi_small_0")
        assert compute_plan_token(spec_a.canonical(), 100) != compute_plan_token(spec_b.canonical(), 100)

    def test_match_ids_order_independent(self):
        spec_a = ScopeSpec(mode="list", match_ids=("a", "b"))
        spec_b = ScopeSpec(mode="list", match_ids=("b", "a"))
        assert compute_plan_token(spec_a.canonical(), 5) == compute_plan_token(spec_b.canonical(), 5)
