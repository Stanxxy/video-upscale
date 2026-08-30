import json

import pytest

from eval.adcc.predict import smoke_test
from eval.adcc.predict.run_predictions import RunPredictionsError, build_match_targets
from eval.adcc.predict.smoke_test import main, select_smoke_targets


def _corpus_records():
    records = [
        # single-match videos, varying event counts (smallest = fewest events)
        {"match_id": "s_big", "video_url": "https://x/s_big", "video_url_md5": "stem_s_big",
         "events": [{"t_s": 10}, {"t_s": 20}, {"t_s": 30}]},
        {"match_id": "s_small", "video_url": "https://x/s_small", "video_url_md5": "stem_s_small",
         "events": [{"t_s": 10}]},
    ]
    # small multi-match VOD: 2 matches
    records += [
        {"match_id": f"multi_small_{i}", "video_url": "https://x/multi_small", "video_url_md5": "stem_multi_small",
         "events": [{"t_s": 100 * i}]}
        for i in range(2)
    ]
    # worst-case VOD: 6 matches (the biggest match-count video in this synthetic corpus)
    records += [
        {"match_id": f"multi_big_{i}", "video_url": "https://x/multi_big", "video_url_md5": "stem_multi_big",
         "events": [{"t_s": 1000 * i}]}
        for i in range(6)
    ]
    return records


class TestSelectSmokeTargets:
    def test_picks_smallest_single_smallest_multi_and_worst_case(self):
        targets = build_match_targets(_corpus_records())
        chosen = select_smoke_targets(targets)
        chosen_ids = {t.match_id for t in chosen}

        assert "s_small" in chosen_ids  # single-match, fewest events
        assert any(mid.startswith("multi_small_") for mid in chosen_ids)  # small multi-match VOD
        assert any(mid.startswith("multi_big_") for mid in chosen_ids)  # worst-case VOD
        assert len(chosen) == 3

    def test_raises_without_a_single_match_video(self):
        records = [r for r in _corpus_records() if r["video_url_md5"] not in ("stem_s_big", "stem_s_small")]
        targets = build_match_targets(records)
        with pytest.raises(RunPredictionsError):
            select_smoke_targets(targets)

    def test_raises_without_two_distinct_multi_match_videos(self):
        records = [r for r in _corpus_records() if r["video_url_md5"] != "stem_multi_big"]
        targets = build_match_targets(records)
        with pytest.raises(RunPredictionsError):
            select_smoke_targets(targets)


class TestSmokeTestMain:
    def test_dry_run_never_touches_gemini(self, tmp_path, monkeypatch, capsys):
        async def _boom(*args, **kwargs):
            raise AssertionError("run_pipeline must NEVER be called in smoke-test dry-run mode")
            yield  # pragma: no cover

        # smoke_test imports capture_match_predictions from run_predictions, which
        # itself references run_pipeline at call time -- patch it at the source module.
        import eval.adcc.predict.run_predictions as run_predictions_module
        monkeypatch.setattr(run_predictions_module, "run_pipeline", _boom)

        gold_path = tmp_path / "gold.jsonl"
        with open(gold_path, "w") as f:
            for rec in _corpus_records():
                f.write(json.dumps(rec) + "\n")
        durations_path = tmp_path / "durations.json"
        durations_path.write_text(json.dumps({"stem_s_small": 300.0}))

        main([
            "--gold-snapshot", str(gold_path),
            "--video-durations", str(durations_path),
        ])

        out = capsys.readouterr().out
        assert "SMOKE TEST DRY RUN ONLY" in out
        assert "zero Gemini calls spent" in out

    def test_execute_without_api_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        gold_path = tmp_path / "gold.jsonl"
        with open(gold_path, "w") as f:
            for rec in _corpus_records():
                f.write(json.dumps(rec) + "\n")
        durations_path = tmp_path / "durations.json"
        durations_path.write_text(json.dumps({"stem_s_small": 300.0}))

        with pytest.raises(RunPredictionsError):
            main([
                "--gold-snapshot", str(gold_path),
                "--video-durations", str(durations_path),
                "--execute",
            ])
