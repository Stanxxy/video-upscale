import json

import pytest

from eval.adcc.predict import run_predictions
from eval.adcc.predict.run_predictions import (
    RunPredictionsError,
    build_dry_run_plan,
    build_match_targets,
    capture_match_predictions,
    estimate_slice_calls,
    load_video_durations_json,
    main,
    plan_match_slices,
)
from eval.adcc.predict.scope_policy import MAX_SLICE_S, Slice


def _gold_records():
    return [
        # single-match video, short (single slice)
        {
            "match_id": "m_single", "video_url": "https://x/1", "video_url_md5": "stem_single",
            "events": [{"t_s": 30, "action_class": "guard_pass"}],
        },
        # two matches sharing one VOD (multi-match)
        {
            "match_id": "m_multi_a", "video_url": "https://x/2", "video_url_md5": "stem_multi",
            "events": [{"t_s": 500}, {"t_s": 620}],
        },
        {
            "match_id": "m_multi_b", "video_url": "https://x/2", "video_url_md5": "stem_multi",
            "events": [{"t_s": 3000}, {"t_s": 3050}],
        },
        # single-match video, LONG (needs slicing)
        {
            "match_id": "m_long", "video_url": "https://x/3", "video_url_md5": "stem_long",
            "events": [{"t_s": 100}],
        },
    ]


class TestBuildMatchTargets:
    def test_single_match_video_is_not_multi(self):
        targets = build_match_targets(_gold_records())
        assert targets["m_single"].is_multi_match is False

    def test_shared_vod_matches_are_multi(self):
        targets = build_match_targets(_gold_records())
        assert targets["m_multi_a"].is_multi_match is True
        assert targets["m_multi_b"].is_multi_match is True

    def test_gold_t_s_values_sorted(self):
        targets = build_match_targets(_gold_records())
        assert targets["m_multi_b"].gold_t_s_values == (3000, 3050)


class TestLoadVideoDurations:
    def test_loads_json_mapping(self, tmp_path):
        path = tmp_path / "durations.json"
        path.write_text(json.dumps({"stem_single": 633.0, "stem_long": 2000.5}))
        durations = load_video_durations_json(path)
        assert durations["stem_single"] == 633.0
        assert durations["stem_long"] == 2000.5


class TestPlanMatchSlices:
    def test_single_match_missing_duration_raises(self):
        targets = build_match_targets(_gold_records())
        with pytest.raises(RunPredictionsError):
            plan_match_slices(targets["m_single"], video_durations={})

    def test_single_match_with_duration_produces_one_slice_when_short(self):
        targets = build_match_targets(_gold_records())
        slices = plan_match_slices(targets["m_single"], video_durations={"stem_single": 400.0})
        assert len(slices) == 1
        assert slices[0].end_s == 400.0

    def test_single_match_long_video_slices(self):
        targets = build_match_targets(_gold_records())
        slices = plan_match_slices(targets["m_long"], video_durations={"stem_long": MAX_SLICE_S * 2.2})
        assert len(slices) > 1

    def test_multi_match_uses_gold_derived_window_ignores_durations_dict(self):
        targets = build_match_targets(_gold_records())
        slices = plan_match_slices(targets["m_multi_a"], video_durations={})
        assert len(slices) == 1
        assert slices[0].start_s == 500 - 120  # MULTI_MATCH_PRE_MARGIN_S
        assert slices[0].end_s == 620 + 180  # MULTI_MATCH_POST_MARGIN_S


class TestEstimateSliceCalls:
    def test_returns_planned_gemini_calls(self):
        from service.pipelines import registry
        pipeline_def = registry.get_default("chunk-segment-tags")
        sl = Slice(match_id="m1", slice_index=0, start_s=0.0, end_s=300.0)
        planned = estimate_slice_calls(pipeline_def, sl)
        assert planned["planned_gemini_calls"] > 0


class TestBuildDryRunPlan:
    def test_dry_run_plan_zero_network_and_correct_totals(self):
        targets = build_match_targets(_gold_records())
        video_durations = {"stem_single": 400.0, "stem_long": MAX_SLICE_S * 2.2}
        plan = build_dry_run_plan(list(targets.values()), video_durations)

        assert plan["pipeline_id"] == "chunk-segment-tags"
        assert plan["match_count"] == 4
        assert plan["over_cap_match_count"] == 1  # only m_long needed slicing
        assert plan["total_slices"] >= 4  # >=1 per match, more for m_long
        assert plan["total_planned_gemini_calls_worst_case"] > 0

        by_id = {m["match_id"]: m for m in plan["per_match"]}
        assert by_id["m_single"]["slice_count"] == 1
        assert by_id["m_long"]["slice_count"] > 1
        assert by_id["m_multi_a"]["is_multi_match"] is True

    def test_missing_duration_for_single_match_raises(self):
        targets = build_match_targets(_gold_records())
        with pytest.raises(RunPredictionsError):
            build_dry_run_plan(list(targets.values()), video_durations={})  # m_single/m_long missing


class TestExecuteGuard:
    """C2: capture_match_predictions must refuse to run without an explicit
    execute=True, even when called directly (not just via the CLI's
    --execute flag) -- defense in depth against accidental Gemini spend."""

    def test_capture_raises_without_execute_true(self):
        targets = build_match_targets(_gold_records())
        target = targets["m_single"]
        slices = plan_match_slices(target, video_durations={"stem_single": 400.0})
        with pytest.raises(RunPredictionsError):
            import asyncio
            asyncio.run(
                capture_match_predictions(target, slices, gemini_api_key="fake", execute=False)
            )

    def test_main_default_dry_run_never_touches_run_pipeline(self, tmp_path, monkeypatch, capsys):
        """The module-level guard: with no --execute flag, main() must never
        invoke run_pipeline (i.e. must never attempt a real Gemini call).
        Monkeypatch run_pipeline to raise if called at all."""
        async def _boom(*args, **kwargs):
            raise AssertionError("run_pipeline must NEVER be called in dry-run mode")
            yield  # pragma: no cover — unreachable, keeps this an async generator

        monkeypatch.setattr(run_predictions, "run_pipeline", _boom)

        gold_path = tmp_path / "gold.jsonl"
        with open(gold_path, "w") as f:
            for rec in _gold_records():
                f.write(json.dumps(rec) + "\n")
        split_path = tmp_path / "split.json"
        split_path.write_text(json.dumps({"assignment": {}}))
        durations_path = tmp_path / "durations.json"
        durations_path.write_text(json.dumps({"stem_single": 400.0, "stem_long": MAX_SLICE_S * 2.2}))

        main([
            "--gold-snapshot", str(gold_path),
            "--split-assignment", str(split_path),
            "--video-durations", str(durations_path),
            "--split", "all",
        ])

        out = capsys.readouterr().out
        assert "DRY RUN ONLY" in out
        assert "zero Gemini calls spent" in out

    def test_main_execute_without_api_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        gold_path = tmp_path / "gold.jsonl"
        with open(gold_path, "w") as f:
            f.write(json.dumps(_gold_records()[0]) + "\n")
        split_path = tmp_path / "split.json"
        split_path.write_text(json.dumps({"assignment": {}}))
        durations_path = tmp_path / "durations.json"
        durations_path.write_text(json.dumps({"stem_single": 400.0}))

        with pytest.raises(RunPredictionsError):
            main([
                "--gold-snapshot", str(gold_path),
                "--split-assignment", str(split_path),
                "--video-durations", str(durations_path),
                "--execute",
            ])


class TestVideoUrlNormalization:
    """Live-bug fix regression (2026-07-18): the tag-batch prediction-capture
    path threaded the ADCC gold snapshot's raw /embed/<id> YouTube URL
    (INS-084) straight into RunContext.youtube_url, which
    service/pipelines/executors.py then sent unnormalized to Gemini's
    native file_uri ingestion -- every one of the 258 real matches would 400
    on the first real run. capture_match_predictions must now normalize to
    the canonical watch?v=<id> form BEFORE constructing RunContext, so
    executors.py (co-tenant, byte-untouchable) needs zero changes and always
    receives an already-good URL. Zero real Gemini calls here -- run_pipeline
    itself is monkeypatched, never invoked for real."""

    def test_run_context_receives_canonical_watch_url(self, monkeypatch):
        captured_ctxs = []

        async def _fake_run_pipeline(pipeline_def, ctx, planned, budget_cap=None):
            captured_ctxs.append(ctx)
            return
            yield  # pragma: no cover — unreachable, keeps this an async generator

        monkeypatch.setattr(run_predictions, "run_pipeline", _fake_run_pipeline)

        records = [{
            "match_id": "m_embed", "video_url": "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "video_url_md5": "stem_embed", "events": [{"t_s": 30, "action_class": "guard_pass"}],
        }]
        targets = build_match_targets(records)
        target = targets["m_embed"]
        slices = plan_match_slices(target, video_durations={"stem_embed": 100.0})

        import asyncio
        asyncio.run(capture_match_predictions(target, slices, gemini_api_key="fake", execute=True))

        assert len(captured_ctxs) == 1
        assert captured_ctxs[0].youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_already_canonical_watch_url_passes_through_unchanged(self, monkeypatch):
        captured_ctxs = []

        async def _fake_run_pipeline(pipeline_def, ctx, planned, budget_cap=None):
            captured_ctxs.append(ctx)
            return
            yield  # pragma: no cover

        monkeypatch.setattr(run_predictions, "run_pipeline", _fake_run_pipeline)

        records = [{
            "match_id": "m_canonical", "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "video_url_md5": "stem_canonical", "events": [{"t_s": 30, "action_class": "guard_pass"}],
        }]
        targets = build_match_targets(records)
        target = targets["m_canonical"]
        slices = plan_match_slices(target, video_durations={"stem_canonical": 100.0})

        import asyncio
        asyncio.run(capture_match_predictions(target, slices, gemini_api_key="fake", execute=True))

        assert captured_ctxs[0].youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_malformed_video_url_raises_before_any_run_pipeline_call(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise AssertionError("run_pipeline must NEVER be called when the video URL is malformed")
            yield  # pragma: no cover — unreachable

        monkeypatch.setattr(run_predictions, "run_pipeline", _boom)

        records = [{
            "match_id": "m_bad", "video_url": "https://example.com/not-a-youtube-url",
            "video_url_md5": "stem_bad", "events": [{"t_s": 30, "action_class": "guard_pass"}],
        }]
        targets = build_match_targets(records)
        target = targets["m_bad"]
        slices = plan_match_slices(target, video_durations={"stem_bad": 100.0})

        import asyncio
        from eval.adcc.predict.youtube_url import YoutubeUrlError
        with pytest.raises(YoutubeUrlError):
            asyncio.run(capture_match_predictions(target, slices, gemini_api_key="fake", execute=True))

    def test_normalizes_against_every_real_corpus_url(self):
        """Regression pin against the REAL 103-video corpus -- proves the fix
        covers every URL shape the gold snapshot actually contains."""
        import json
        import pathlib
        from eval.adcc.predict.youtube_url import normalize_youtube_watch_url

        path = pathlib.Path(__file__).resolve().parents[1] / "gold_snapshot" / "normalized_gold.jsonl"
        if not path.exists():
            pytest.skip("real gold_snapshot/normalized_gold.jsonl not present in this checkout")
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        urls = {r["video_url"] for r in records}
        assert len(urls) == 103
        for url in urls:
            canonical = normalize_youtube_watch_url(url)
            assert canonical.startswith("https://www.youtube.com/watch?v=")
