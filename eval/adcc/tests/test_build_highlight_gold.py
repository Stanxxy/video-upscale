import json

from eval.adcc.data_prep.build_highlight_gold import (
    PREDICATE_VERSION,
    build_highlight_gold,
    is_highlight,
    sha256_of_file,
)


def _event(action_class, outcome="successful", t_s=10):
    return {"t_s": t_s, "action_class": action_class, "outcome": outcome}


class TestPredicate:
    def test_submission_choke_is_highlight(self):
        assert is_highlight(_event("submission_choke")) is True

    def test_submission_arm_lock_is_highlight(self):
        assert is_highlight(_event("submission_arm_lock")) is True

    def test_submission_leg_lock_is_highlight(self):
        assert is_highlight(_event("submission_leg_lock")) is True

    def test_submission_unclear_is_highlight(self):
        assert is_highlight(_event("submission_unclear")) is True

    def test_back_take_is_highlight_regardless_of_outcome_field(self):
        assert is_highlight(_event("back_take", outcome="successful")) is True

    def test_sweep_is_highlight(self):
        assert is_highlight(_event("sweep")) is True

    def test_guard_pass_successful_is_highlight(self):
        assert is_highlight(_event("guard_pass", outcome="successful")) is True

    def test_guard_pass_unsuccessful_is_not_highlight(self):
        assert is_highlight(_event("guard_pass", outcome="unsuccessful")) is False

    def test_escape_is_never_highlight(self):
        assert is_highlight(_event("escape")) is False

    def test_transition_is_never_highlight(self):
        assert is_highlight(_event("transition", outcome="unclear")) is False

    def test_takedown_attempt_excluded_by_default(self):
        assert is_highlight(_event("takedown_attempt", outcome="successful")) is False
        assert is_highlight(_event("takedown_attempt", outcome="successful"), include_takedowns=False) is False

    def test_takedown_attempt_included_when_flag_set(self):
        assert is_highlight(_event("takedown_attempt", outcome="successful"), include_takedowns=True) is True

    def test_unsuccessful_takedown_attempt_never_highlight_even_with_flag(self):
        assert is_highlight(_event("takedown_attempt", outcome="unsuccessful"), include_takedowns=True) is False


def _gold_record(match_id, events):
    return {"match_id": match_id, "video_url": f"https://x/{match_id}", "video_url_md5": f"stem_{match_id}", "name": f"{match_id} name", "events": events}


class TestBuildHighlightGold:
    def _corpus(self):
        return [
            _gold_record("m_with_highlights", [
                _event("submission_choke", t_s=10),
                _event("escape", t_s=20),
                _event("sweep", t_s=30),
                _event("guard_pass", outcome="unsuccessful", t_s=40),
                _event("guard_pass", outcome="successful", t_s=50),
            ]),
            _gold_record("m_zero_highlights", [
                _event("escape", t_s=5),
                _event("transition", outcome="unclear", t_s=15),
            ]),
        ]

    def test_filters_to_only_highlight_events_per_match(self):
        records = build_highlight_gold(self._corpus())
        by_id = {r["match_id"]: r for r in records}
        assert [e["action_class"] for e in by_id["m_with_highlights"]["events"]] == [
            "submission_choke", "sweep", "guard_pass",
        ]

    def test_zero_highlight_match_is_kept_with_empty_events_not_dropped(self):
        records = build_highlight_gold(self._corpus())
        by_id = {r["match_id"]: r for r in records}
        assert "m_zero_highlights" in by_id
        assert by_id["m_zero_highlights"]["events"] == []

    def test_every_match_id_from_source_is_represented(self):
        records = build_highlight_gold(self._corpus())
        assert {r["match_id"] for r in records} == {"m_with_highlights", "m_zero_highlights"}

    def test_identity_fields_preserved_verbatim(self):
        records = build_highlight_gold(self._corpus())
        rec = next(r for r in records if r["match_id"] == "m_with_highlights")
        assert rec["video_url"] == "https://x/m_with_highlights"
        assert rec["video_url_md5"] == "stem_m_with_highlights"
        assert rec["name"] == "m_with_highlights name"

    def test_deterministic_across_repeated_calls(self):
        corpus = self._corpus()
        first = build_highlight_gold(corpus)
        second = build_highlight_gold(corpus)
        assert first == second

    def test_include_takedowns_flag_adds_successful_takedowns(self):
        corpus = [_gold_record("m1", [_event("takedown_attempt", outcome="successful", t_s=1)])]
        without = build_highlight_gold(corpus, include_takedowns=False)
        with_td = build_highlight_gold(corpus, include_takedowns=True)
        assert without[0]["events"] == []
        assert len(with_td[0]["events"]) == 1


class TestRealCorpusCounts:
    """Regression pin against the frozen contract's stated real-corpus
    numbers (557 highlights / 258 matches, median 2, 17 zero-highlight) —
    proves this module's predicate matches what the CEO/founder already
    verified against the real gold snapshot, not just synthetic fixtures."""

    def _real_gold_records(self):
        import pathlib
        path = pathlib.Path(__file__).resolve().parents[1] / "gold_snapshot" / "normalized_gold.jsonl"
        if not path.exists():
            import pytest
            pytest.skip("real gold_snapshot/normalized_gold.jsonl not present in this checkout")
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_real_corpus_predicate_counts_match_frozen_contract(self):
        records = self._real_gold_records()
        highlight_records = build_highlight_gold(records)
        assert len(highlight_records) == 258
        total = sum(len(r["events"]) for r in highlight_records)
        zero = sum(1 for r in highlight_records if not r["events"])
        assert total == 557
        assert zero == 17
        counts = sorted(len(r["events"]) for r in highlight_records)
        assert counts[len(counts) // 2] == 2

    def test_include_takedowns_adds_136(self):
        records = self._real_gold_records()
        without = sum(len(r["events"]) for r in build_highlight_gold(records, include_takedowns=False))
        with_td = sum(len(r["events"]) for r in build_highlight_gold(records, include_takedowns=True))
        assert with_td - without == 136


def test_sha256_of_file_is_stable(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    assert sha256_of_file(p) == sha256_of_file(p)
    p2 = tmp_path / "b.txt"
    p2.write_text("hello!")
    assert sha256_of_file(p) != sha256_of_file(p2)


def test_predicate_version_is_a_nonempty_string():
    assert isinstance(PREDICATE_VERSION, str) and PREDICATE_VERSION
