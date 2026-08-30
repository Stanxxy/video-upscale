"""``eval/adcc/predict/highlight_discovery.py`` — ZERO real Gemini calls in
any test here: ``build_gemini_client`` (this module's OWN client
constructor, never ``RunContext.gemini_client``) is monkeypatched with a fake
client whose ``generate_content`` is a plain async stub.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eval.adcc.predict import highlight_discovery as hd
from eval.adcc.predict.global_cap_driver import BatchRunState
from eval.adcc.predict.run_predictions import MatchTarget
from eval.adcc.predict.scope_policy import ScopeWindow
from eval.adcc.score import highlight_scoring as hs


def _target(match_id="m1", video_url_md5="stem1", is_multi=False, gold_t_s=(10.0,)):
    # video_url is a real-SHAPED (fake) YouTube /embed/<11-char-id> URL --
    # live-bug fix regression: discover_match_highlights now rejects
    # unparseable video URLs (normalize_youtube_watch_url), so every fixture
    # video_url here must be a shape it actually recognizes, mirroring the
    # real ADCC gold snapshot's own embed-URL field (INS-084).
    fake_video_id = (video_url_md5 + "00000000000")[:11]
    return MatchTarget(
        match_id=match_id, video_url=f"https://www.youtube.com/embed/{fake_video_id}",
        video_url_md5=video_url_md5, is_multi_match=is_multi, gold_t_s_values=gold_t_s,
    )


def _window(match_id="m1", start_s=0.0, end_s=300.0):
    return ScopeWindow(match_id=match_id, start_s=start_s, end_s=end_s, is_multi_match=False)


def _hl(start_s, end_s, reason="submission — armbar — arm fully extended and secured"):
    """A discovery-schema highlight item WITH the now-required 'reason'
    field (frozen contract §A) -- every synthetic Gemini discovery JSON
    payload in this file must include it or discover_match_highlights
    correctly rejects the response."""
    return {"start_s": start_s, "end_s": end_s, "reason": reason}


def _fake_client_returning(highlights_json_text: str):
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=highlights_json_text, usage_metadata=None)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    return fake_client


def _capturing_fake_client(highlights_json_text: str = "[]"):
    """Like ``_fake_client_returning`` but also stashes the real
    ``config``/``contents`` kwargs the call site passed, so a test can
    inspect the exact Gemini call SHAPE (media_resolution, video_metadata.fps)
    without ever making a real request."""
    captured: dict = {}

    async def _fake_generate(*, model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return SimpleNamespace(text=highlights_json_text, usage_metadata=None)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    return fake_client, captured


class TestIsLongWindowRisk:
    """Bezos fast-follow (non-blocking): ``LONG_WINDOW_RISK_THRESHOLD_S``
    reuses production's ``chunk_segment.MAX_SCOPE_SEC`` (720s) as a WARNING
    threshold only -- never a gate, never new slicing."""

    def test_threshold_reuses_production_max_scope_sec(self):
        from service.pipelines.chunk_segment import MAX_SCOPE_SEC
        assert hd.LONG_WINDOW_RISK_THRESHOLD_S == float(MAX_SCOPE_SEC)

    def test_well_under_threshold_is_not_a_risk(self):
        assert hd.is_long_window_risk(60.0) is False

    def test_exactly_at_threshold_is_not_a_risk(self):
        # Strictly greater-than: AT the production cap is still the same
        # size a legal single chunk-segment-tags run would use.
        assert hd.is_long_window_risk(hd.LONG_WINDOW_RISK_THRESHOLD_S) is False

    def test_just_over_threshold_is_a_risk(self):
        assert hd.is_long_window_risk(hd.LONG_WINDOW_RISK_THRESHOLD_S + 0.001) is True

    def test_well_over_threshold_is_a_risk(self):
        assert hd.is_long_window_risk(5000.0) is True


class TestResolveHighlightWindow:
    def test_single_match_uses_ffprobe_duration(self):
        target = _target(is_multi=False)
        window = hd.resolve_highlight_window(target, {"stem1": 300.0})
        assert window.start_s == 0.0
        assert window.end_s == 300.0

    def test_single_match_missing_duration_raises(self):
        target = _target(is_multi=False)
        with pytest.raises(hd.HighlightDiscoveryError):
            hd.resolve_highlight_window(target, {})

    def test_multi_match_uses_gold_derived_window(self):
        target = _target(is_multi=True, gold_t_s=(200.0, 400.0))
        window = hd.resolve_highlight_window(target, {})
        assert window.start_s == pytest.approx(80.0)  # 200 - 120 margin
        assert window.end_s == pytest.approx(580.0)  # 400 + 180 margin


class TestThinkingConfig:
    def test_off_maps_to_zero_budget_for_ordinary_model(self):
        cfg = hd.highlight_thinking_config("gemini-3.5-flash", "off")
        assert cfg.thinking_budget == 0

    def test_off_maps_to_low_level_for_pro_preview(self):
        cfg = hd.highlight_thinking_config("gemini-3.1-pro-preview", "off")
        assert str(cfg.thinking_level).lower().endswith("low")

    def test_explicit_level_is_honored(self):
        cfg = hd.highlight_thinking_config("gemini-3.5-flash", "high")
        assert str(cfg.thinking_level).lower().endswith("high")

    def test_invalid_level_raises(self):
        with pytest.raises(hd.HighlightDiscoveryError):
            hd.highlight_thinking_config("gemini-3.5-flash", "extreme")


class TestModelAllowlist:
    @pytest.mark.asyncio
    async def test_out_of_family_model_is_rejected(self, monkeypatch):
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("[]"))
        with pytest.raises(hd.HighlightDiscoveryError):
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model="gemini-2.0-flash", thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
            )


class TestRefinementCostConstants:
    """Frozen contract §C (Brooks): a DEDICATED refine cost, never borrowed
    chunk-segment vocab/rates."""

    def test_refine_placeholder_usd_is_derived_via_the_dedicated_formula(self):
        from eval.adcc.predict.cost_model import estimate_highlight_call_cost_usd
        expected = estimate_highlight_call_cost_usd(
            hd.DEFAULT_PAD_BEFORE_S + hd.DEFAULT_PAD_AFTER_S, hd.DEFAULT_REFINE_FPS, hd.REFINE_FRAME_QUALITY,
        )
        assert hd.REFINE_PLACEHOLDER_USD == pytest.approx(expected)

    def test_refine_placeholder_usd_differs_from_chunk_segment_placeholders(self):
        from eval.adcc.predict.global_cap_driver import PASS1_PLACEHOLDER_USD, PASS2_PLACEHOLDER_USD
        assert hd.REFINE_PLACEHOLDER_USD != PASS1_PLACEHOLDER_USD
        assert hd.REFINE_PLACEHOLDER_USD != PASS2_PLACEHOLDER_USD

    def test_pad_constants_are_asymmetric_front_loaded(self):
        # bias is LATE -> more padding goes backward than forward.
        assert hd.DEFAULT_PAD_BEFORE_S > hd.DEFAULT_PAD_AFTER_S
        assert hd.WIDEN_PAD_BEFORE_S == 2 * hd.DEFAULT_PAD_BEFORE_S

    def test_refine_frame_quality_is_fixed_high(self):
        assert hd.REFINE_FRAME_QUALITY == "high"


class TestValidateFpsAndMediaResolution:
    """Pure-function unit tests (no Gemini client at all)."""

    def test_every_fps_option_validates_without_raising(self):
        for fps in hd.FPS_OPTIONS:
            hd.validate_fps(fps)  # must not raise

    def test_out_of_family_fps_raises(self):
        with pytest.raises(hd.HighlightDiscoveryError):
            hd.validate_fps(3.0)

    def test_default_quality_resolves_to_none(self):
        assert hd.resolve_media_resolution("default") is None

    def test_low_medium_high_map_to_the_gemini_enum(self):
        from google.genai import types
        assert hd.resolve_media_resolution("low") == types.MediaResolution.MEDIA_RESOLUTION_LOW
        assert hd.resolve_media_resolution("medium") == types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
        assert hd.resolve_media_resolution("high") == types.MediaResolution.MEDIA_RESOLUTION_HIGH

    def test_out_of_family_frame_quality_raises(self):
        with pytest.raises(hd.HighlightDiscoveryError):
            hd.resolve_media_resolution("ultra")


class TestNormalizeYoutubeWatchUrl:
    """Live-bug regression (2026-07-18): a real user's ``/highlight/run``
    click 400'd with a bare, no-named-field ``google.genai.errors.
    ClientError: 400 INVALID_ARGUMENT``. Root cause: the ADCC gold
    snapshot's ``video_url`` is an ``/embed/<id>`` URL (INS-084), which
    Gemini's native ``file_uri`` YouTube ingestion does not accept — only
    the canonical ``watch?v=<id>`` form (confirmed by the ONE calling
    context that has ever actually exercised a real, successful Gemini
    video call: the QA VLM Studio client, which always normalizes to
    ``watch?v=`` before the backend sees a URL). These tests pin the fix,
    ``normalize_youtube_watch_url``, against the REAL corpus URL shape."""

    def test_embed_url_normalizes_to_canonical_watch_form(self):
        assert (
            hd.normalize_youtube_watch_url("https://www.youtube.com/embed/dQw4w9WgXcQ")
            == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

    def test_every_real_corpus_url_normalizes_without_raising(self):
        """Regression pin against the REAL 103-video corpus (not just a
        synthetic fixture) -- proves the fix actually covers every URL shape
        the gold snapshot contains, not just one hand-picked example."""
        import json
        import pathlib
        path = pathlib.Path(__file__).resolve().parents[1] / "gold_snapshot" / "normalized_gold.jsonl"
        if not path.exists():
            pytest.skip("real gold_snapshot/normalized_gold.jsonl not present in this checkout")
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        urls = {r["video_url"] for r in records}
        assert len(urls) == 103
        for url in urls:
            canonical = hd.normalize_youtube_watch_url(url)
            assert canonical.startswith("https://www.youtube.com/watch?v=")
            assert len(canonical) == len("https://www.youtube.com/watch?v=") + 11

    def test_already_canonical_watch_url_is_a_no_op(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert hd.normalize_youtube_watch_url(url) == url

    def test_youtu_be_short_form_normalizes(self):
        assert (
            hd.normalize_youtube_watch_url("https://youtu.be/dQw4w9WgXcQ")
            == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

    def test_shorts_form_normalizes(self):
        assert (
            hd.normalize_youtube_watch_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
            == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

    def test_non_youtube_url_raises_no_silent_fallback(self):
        with pytest.raises(hd.HighlightDiscoveryError):
            hd.normalize_youtube_watch_url("https://example.com/not-a-youtube-url")

    def test_malformed_embed_path_raises(self):
        with pytest.raises(hd.HighlightDiscoveryError):
            hd.normalize_youtube_watch_url("https://www.youtube.com/embed/tooshort")


class TestVideoUrlNormalizationReachesTheGeminiCall:
    """Proves the fix is actually WIRED IN -- not just a standalone helper
    function nobody calls. Asserts the exact ``file_data.file_uri`` Gemini
    receives is the canonical form, even though the caller supplied the raw
    embed URL (mirrors what ``MatchTarget.video_url`` actually contains for
    every real match)."""

    @pytest.mark.asyncio
    async def test_embed_url_is_normalized_before_reaching_generate_content(self, monkeypatch):
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dQw4w9WgXcQ", window=_window(),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        video_part = captured["contents"][0].parts[1]
        assert video_part.file_data.file_uri == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    @pytest.mark.asyncio
    async def test_unparseable_video_url_raises_before_any_gemini_call(self, monkeypatch):
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        with pytest.raises(hd.HighlightDiscoveryError):
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://example.com/not-a-youtube-url", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
            )
        assert "config" not in captured  # zero Gemini spend on a malformed URL


class TestVideoMetadataOffsetConstruction:
    """Pins the EXACT ``video_metadata.start_offset``/``end_offset`` strings
    sent to Gemini for a NON-integer window — a regression lock on the
    ``int(window.start_s)``/``int(round(window.end_s))`` construction (a
    cost-relevant path: a wrong offset changes what's actually billed and
    analyzed) against future drift. There is no committed git baseline for
    ``eval/`` to diff this against, so this test IS the baseline. Also
    guards against the co-tenant ``service/`` stream's own recently-fixed
    sibling bug (raw-float arithmetic serializing a 15-fractional-digit
    protobuf Duration string, e.g. ``28.299999999999997s``) -- this module
    is NOT exposed to that failure mode because it always casts through
    ``int(...)`` before string-formatting, never f-string-formats a raw
    float directly; this test's non-integer input would catch a regression
    that removed that cast."""

    @pytest.mark.asyncio
    async def test_non_integer_window_offsets_are_quantized_to_whole_seconds(self, monkeypatch):
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
            window=_window(start_s=42.4, end_s=443.6),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        video_part = captured["contents"][0].parts[1]
        # int(42.4) == 42; int(round(443.6)) == 444 -- exact values, not a
        # tolerance/approx check, since these are the LITERAL strings Gemini
        # receives.
        assert video_part.video_metadata.start_offset == "42s"
        assert video_part.video_metadata.end_offset == "444s"


class TestFpsAndFrameQualityThreadedIntoTheGeminiCall:
    """CEO-frozen spec: fps + frame_quality must reach the ACTUAL Gemini call
    kwargs (VideoMetadata.fps / GenerateContentConfig.media_resolution), and
    the default path must be BYTE-IDENTICAL to the pre-this-increment call
    (fps=1, media_resolution left entirely unset)."""

    @pytest.mark.asyncio
    async def test_default_path_leaves_media_resolution_unset(self, monkeypatch):
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert captured["config"].media_resolution is None

    @pytest.mark.asyncio
    async def test_default_path_sends_fps_1(self, monkeypatch):
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        video_part = captured["contents"][0].parts[1]
        assert video_part.video_metadata.fps == 1.0

    @pytest.mark.asyncio
    async def test_custom_fps_reaches_video_metadata(self, monkeypatch):
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
            fps=2.0, frame_quality="default",
        )
        video_part = captured["contents"][0].parts[1]
        assert video_part.video_metadata.fps == 2.0

    @pytest.mark.asyncio
    async def test_high_quality_sets_media_resolution_high(self, monkeypatch):
        from google.genai import types
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
            fps=1.0, frame_quality="high",
        )
        assert captured["config"].media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH

    @pytest.mark.asyncio
    async def test_out_of_family_fps_raises_no_call_made(self, monkeypatch):
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        with pytest.raises(hd.HighlightDiscoveryError):
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
                fps=3.0,
            )
        assert "config" not in captured  # never reached the Gemini call

    @pytest.mark.asyncio
    async def test_out_of_family_frame_quality_raises_no_call_made(self, monkeypatch):
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        with pytest.raises(hd.HighlightDiscoveryError):
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
                frame_quality="ultra",
            )
        assert "config" not in captured


class TestGenuineGeminiErrorIsWrappedAndCounted:
    """Live-bug regression (2026-07-18, IP 100.77.77.23 crash): a REAL
    ``google.genai.errors.ClientError`` (e.g. 400 INVALID_ARGUMENT) raised by
    ``generate_content`` itself — as opposed to a malformed/unparseable
    response body — must be wrapped as ``HighlightDiscoveryError`` (never
    left to escape this module's own error contract and only be caught by
    ``dashboard_app.py``'s broad outer exception handler, which drops it from
    the per-call cap/spend bookkeeping and aborts the whole batch instead of
    just this match). Uses the REAL SDK exception class (not a generic
    stand-in) so this test fails if the wrapping regresses to catching only
    a narrower exception type."""

    def _client_error(self, message: str = "Invalid value at 'generation_config.thinking_config.thinking_level'"):
        from google.genai import errors
        return errors.ClientError(400, {"error": {"message": message, "status": "INVALID_ARGUMENT"}})

    def _fake_client_raising(self, exc: Exception):
        async def _fake_generate(*, model, contents, config):
            raise exc

        fake_client = MagicMock()
        fake_client.aio.models.generate_content = _fake_generate
        return fake_client

    @pytest.mark.asyncio
    async def test_client_error_is_wrapped_as_highlight_discovery_error(self, monkeypatch):
        exc = self._client_error()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: self._fake_client_raising(exc))
        with pytest.raises(hd.HighlightDiscoveryError) as excinfo:
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
            )
        # The REAL Gemini error text (which names the offending field for a
        # 400 INVALID_ARGUMENT) must be preserved verbatim, not replaced with
        # a generic message -- this is the whole point of the fix.
        assert "thinking_config.thinking_level" in str(excinfo.value)
        assert "INVALID_ARGUMENT" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_client_error_chains_the_original_exception(self, monkeypatch):
        exc = self._client_error()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: self._fake_client_raising(exc))
        with pytest.raises(hd.HighlightDiscoveryError) as excinfo:
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
            )
        assert excinfo.value.__cause__ is exc

    @pytest.mark.asyncio
    async def test_batch_driver_counts_the_failed_call_against_the_cap(self, monkeypatch):
        """Restores global_cap_driver's own invariant ("a run of failing
        calls cannot spend unboundedly without ever tripping the cap") for
        THIS driver too -- previously a raw ClientError escaped uncounted."""
        exc = self._client_error()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: self._fake_client_raising(exc))
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window())], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "thinking_config.thinking_level" in error_events[0]["message"]
        assert state.calls_spent == 1  # the real, already-billed call IS counted
        assert state.dollars_spent > 0
        assert events[-1]["type"] == "batch_complete"  # batch continues, never aborts on one bad response

    @pytest.mark.asyncio
    async def test_second_match_still_runs_after_the_first_ones_client_error(self, monkeypatch):
        """The documented "batch continues to the next match" contract,
        proven with a genuine SDK-raised error (not just malformed JSON)."""
        call_count = {"n": 0}

        async def _fake_generate(*, model, contents, config):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise self._client_error()
            return SimpleNamespace(text=json.dumps([_hl(1.0, 2.0)]), usage_metadata=None)

        fake_client = MagicMock()
        fake_client.aio.models.generate_content = _fake_generate
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)

        state = BatchRunState(cap=10)
        targets_and_windows = [(_target("m1"), _window("m1")), (_target("m2", "stem2"), _window("m2"))]
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                targets_and_windows, gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
                refine=False,  # scoped to the discovery-level "batch continues" contract only
            )
        ]
        assert call_count["n"] == 2  # both matches were attempted
        result_match_ids = [e["match_id"] for e in events if e["type"] == "highlight_result"]
        assert result_match_ids == ["m2"]
        assert events[-1]["type"] == "batch_complete"


class TestDiscoverMatchHighlightsNormalization:
    @pytest.mark.asyncio
    async def test_clip_relative_timestamps_normalized_to_vod_absolute(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(10.0, 15.0)])),
        )
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(start_s=100.0, end_s=400.0),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="sys", followup_prompt="find highlights",
        )
        assert result["highlights"] == [_hl(110.0, 115.0)]

    @pytest.mark.asyncio
    async def test_point_highlight_start_equals_end(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(5.0, 5.0)])),
        )
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(start_s=0.0, end_s=100.0),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert result["highlights"] == [_hl(5.0, 5.0)]

    @pytest.mark.asyncio
    async def test_non_json_response_raises_no_fallback(self, monkeypatch):
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("not json at all"))
        with pytest.raises(hd.HighlightDiscoveryError):
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
            )

    @pytest.mark.asyncio
    async def test_non_array_response_raises(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client", lambda *a, **k: _fake_client_returning(json.dumps({"not": "a list"})),
        )
        with pytest.raises(hd.HighlightDiscoveryError):
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
            )

    @pytest.mark.asyncio
    async def test_item_missing_start_s_raises(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client", lambda *a, **k: _fake_client_returning(json.dumps([{"end_s": 5.0}])),
        )
        with pytest.raises(hd.HighlightDiscoveryError):
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
            )

    @pytest.mark.asyncio
    async def test_end_before_start_raises(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([{"start_s": 10.0, "end_s": 5.0}])),
        )
        with pytest.raises(hd.HighlightDiscoveryError):
            await hd.discover_match_highlights(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights",
            )

    @pytest.mark.asyncio
    async def test_founder_followup_prompt_text_is_preserved_in_prompt(self, monkeypatch):
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("[]"))
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="FIND THE COOLEST MOMENTS",
        )
        assert "FIND THE COOLEST MOMENTS" in result["prompt_text"]


class TestRawHighlightsAndOutOfBoundsDetection:
    """Live investigation (2026-07-18, LONG-video out-of-range timestamps):
    (1) the raw, pre-normalization clip-relative spans + raw_response_text
    must be returned for diagnosability; (2) a span past the clip's own
    bounds must be DETECTED and counted, never dropped/clamped (that
    scoring-policy call is LeCun's, not this harness's) — ``highlights``
    (VOD-absolute) must still contain EVERY span, in-bounds or not."""

    @pytest.mark.asyncio
    async def test_raw_response_text_and_raw_highlights_are_returned(self, monkeypatch):
        raw_text = json.dumps([_hl(10.0, 15.0)])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning(raw_text))
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(start_s=100.0, end_s=400.0),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert result["raw_response_text"] == raw_text
        # raw_highlights are the UNMODIFIED clip-relative values (never
        # window.start_s-shifted) -- distinct from the VOD-absolute
        # "highlights" ([110.0, 115.0] for this same input, per the sibling
        # normalization test above).
        assert result["raw_highlights"] == [_hl(10.0, 15.0)]
        assert result["highlights"] == [_hl(110.0, 115.0)]

    @pytest.mark.asyncio
    async def test_in_bounds_span_is_not_flagged(self, monkeypatch):
        # window is 300s long (0..300); a span fully inside it.
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(10.0, 20.0)])),
        )
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(start_s=0.0, end_s=300.0),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert result["out_of_bounds_count"] == 0
        assert result["out_of_bounds_spans"] == []

    @pytest.mark.asyncio
    async def test_span_past_clip_end_is_detected_and_counted_not_dropped(self, monkeypatch):
        # window is 401s long (42..443, mirroring the real incident's
        # window) -- Gemini returns a span whose end_s (416) is well past
        # the 401s clip duration.
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(400.0, 416.0)])),
        )
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(start_s=42.0, end_s=443.0),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert result["out_of_bounds_count"] == 1
        assert result["out_of_bounds_spans"] == [{"start_s": 400.0, "end_s": 416.0}]
        # NEVER dropped/clamped -- the VOD-absolute highlights list still
        # carries this span, unmodified (42 + 400 = 442, 42 + 416 = 458).
        assert result["highlights"] == [_hl(442.0, 458.0)]

    @pytest.mark.asyncio
    async def test_span_before_clip_start_is_also_detected(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(-5.0, 1.0)])),
        )
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(start_s=0.0, end_s=300.0),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert result["out_of_bounds_count"] == 1

    @pytest.mark.asyncio
    async def test_span_within_tolerance_of_the_exact_boundary_is_not_flagged(self, monkeypatch):
        # window is 300s; a span ending at 300.5s is within
        # OUT_OF_BOUNDS_TOLERANCE_S (1.0s) of the exact clip end -- ordinary
        # float/rounding noise, not a real drift finding.
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(295.0, 300.5)])),
        )
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(start_s=0.0, end_s=300.0),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert result["out_of_bounds_count"] == 0

    @pytest.mark.asyncio
    async def test_multiple_spans_only_the_offending_ones_are_counted(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([
                _hl(10.0, 20.0),   # in-bounds
                _hl(290.0, 310.0),  # past the 300s clip end
                _hl(50.0, 60.0),   # in-bounds
            ])),
        )
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234", window=_window(start_s=0.0, end_s=300.0),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert result["out_of_bounds_count"] == 1
        assert result["out_of_bounds_spans"] == [{"start_s": 290.0, "end_s": 310.0}]
        assert len(result["highlights"]) == 3  # all three kept, none dropped


class TestRefineHighlightTimestamp:
    """Timestamp critique/refinement (frozen contract §B) — ONE Gemini call
    per attempt. ZERO real Gemini calls: ``build_gemini_client`` is
    monkeypatched throughout."""

    def _refine_response(self, confirmed, start_s=None, end_s=None):
        return json.dumps({"action_confirmed": confirmed, "start_s": start_s, "end_s": end_s})

    @pytest.mark.asyncio
    async def test_confirmed_result_is_normalized_to_vod_absolute(self, monkeypatch):
        # padded window: [100-12, 130+5] clamped to match_window [0,300] -> [88, 135]
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(self._refine_response(True, start_s=20.0, end_s=25.0)),
        )
        result = await hd.refine_highlight_timestamp(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
            match_window=_window(start_s=0.0, end_s=300.0),
            original_start_s=100.0, original_end_s=130.0, reason="submission — armbar — lock secured",
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
        )
        assert result["action_confirmed"] is True
        assert result["padded_window"] == (88.0, 135.0)
        # 88 (padded_start) + 20 clip-relative == 108 VOD-absolute
        assert result["start_s"] == pytest.approx(108.0)
        assert result["end_s"] == pytest.approx(113.0)

    @pytest.mark.asyncio
    async def test_unconfirmed_result_has_null_start_end(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(self._refine_response(False)),
        )
        result = await hd.refine_highlight_timestamp(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
            match_window=_window(start_s=0.0, end_s=300.0),
            original_start_s=100.0, original_end_s=130.0, reason="submission — armbar — lock secured",
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
        )
        assert result["action_confirmed"] is False
        assert result["start_s"] is None
        assert result["end_s"] is None

    @pytest.mark.asyncio
    async def test_unconfirmed_but_non_null_times_raises_no_fallback(self, monkeypatch):
        """The model violating its own schema contract (action_confirmed=
        false but start_s/end_s non-null) must raise, never be silently
        trusted or coerced."""
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(self._refine_response(False, start_s=5.0, end_s=6.0)),
        )
        with pytest.raises(hd.RefinementError):
            await hd.refine_highlight_timestamp(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
                match_window=_window(start_s=0.0, end_s=300.0),
                original_start_s=100.0, original_end_s=130.0, reason="x",
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            )

    @pytest.mark.asyncio
    async def test_confirmed_but_missing_times_raises(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps({"action_confirmed": True})),
        )
        with pytest.raises(hd.RefinementError):
            await hd.refine_highlight_timestamp(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
                match_window=_window(start_s=0.0, end_s=300.0),
                original_start_s=100.0, original_end_s=130.0, reason="x",
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            )

    @pytest.mark.asyncio
    async def test_non_json_response_raises(self, monkeypatch):
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("not json"))
        with pytest.raises(hd.RefinementError):
            await hd.refine_highlight_timestamp(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
                match_window=_window(start_s=0.0, end_s=300.0),
                original_start_s=100.0, original_end_s=130.0, reason="x",
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            )

    @pytest.mark.asyncio
    async def test_padded_window_clamps_to_match_window(self, monkeypatch):
        # original_start_s - pad_before_s would go negative; match_window
        # starts at 0 -> padded_start clamps to 0.0, never negative.
        captured_windows = []

        async def _fake_generate(*, model, contents, config):
            return SimpleNamespace(text=self._refine_response(False), usage_metadata=None)

        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning(self._refine_response(False)))
        result = await hd.refine_highlight_timestamp(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
            match_window=_window(start_s=0.0, end_s=300.0),
            original_start_s=5.0, original_end_s=10.0, reason="x",  # pad_before=12 would push to -7
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
        )
        assert result["padded_window"][0] == 0.0

    @pytest.mark.asyncio
    async def test_uses_the_real_gemini_call_shape_fps4_high_quality(self, monkeypatch):
        from google.genai import types
        fake_client, captured = _capturing_fake_client(self._refine_response(False))
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        await hd.refine_highlight_timestamp(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
            match_window=_window(start_s=0.0, end_s=300.0),
            original_start_s=100.0, original_end_s=130.0, reason="x",
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
        )
        video_part = captured["contents"][0].parts[1]
        assert video_part.video_metadata.fps == hd.DEFAULT_REFINE_FPS == 4.0
        assert captured["config"].media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH

    @pytest.mark.asyncio
    async def test_out_of_family_model_raises(self, monkeypatch):
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("[]"))
        with pytest.raises(hd.RefinementError):
            await hd.refine_highlight_timestamp(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
                match_window=_window(start_s=0.0, end_s=300.0),
                original_start_s=100.0, original_end_s=130.0, reason="x",
                gemini_api_key="fake", request_timeout_ms=1000,
                model="gemini-2.0-flash", thinking_level="off",
            )


class TestDecideRefinementAcceptance:
    """Pure accept/reject decision (frozen contract §B) — no Gemini call at
    all, just dict-in/dict-out."""

    def _attempt(self, confirmed, start_s, end_s, padded_window):
        return {"action_confirmed": confirmed, "start_s": start_s, "end_s": end_s, "padded_window": padded_window}

    def test_confirmed_and_safely_interior_is_accepted(self):
        # padded_window is [88, 135] (47s clip); a time comfortably inside.
        attempt = self._attempt(True, 108.0, 113.0, (88.0, 135.0))
        decision = hd.decide_refinement_acceptance(attempt)
        assert decision["accept"] is True
        assert decision["reject_reason"] is None

    def test_not_confirmed_is_rejected(self):
        attempt = self._attempt(False, None, None, (88.0, 135.0))
        decision = hd.decide_refinement_acceptance(attempt)
        assert decision["accept"] is False
        assert decision["reject_reason"] == "not_confirmed"

    def test_confirmed_but_out_of_bounds_is_rejected(self):
        # end_s=140 is past the padded window's own end (135), well beyond tolerance.
        attempt = self._attempt(True, 108.0, 140.0, (88.0, 135.0))
        decision = hd.decide_refinement_acceptance(attempt)
        assert decision["accept"] is False
        assert decision["reject_reason"] == "out_of_bounds"

    def test_confirmed_but_at_the_padded_edge_is_rejected_as_suspicious(self):
        # start_s == padded_start exactly -- "at the edge", never auto-accepted.
        attempt = self._attempt(True, 88.0, 90.0, (88.0, 135.0))
        decision = hd.decide_refinement_acceptance(attempt)
        assert decision["accept"] is False
        assert decision["reject_reason"] == "near_edge"

    def test_confirmed_but_near_the_end_edge_is_rejected(self):
        attempt = self._attempt(True, 130.0, 135.0, (88.0, 135.0))
        decision = hd.decide_refinement_acceptance(attempt)
        assert decision["accept"] is False
        assert decision["reject_reason"] == "near_edge"


class TestRunHighlightBatchWithGlobalCap:
    @pytest.mark.asyncio
    async def test_fps_and_frame_quality_are_threaded_through_to_the_gemini_call(self, monkeypatch):
        from google.genai import types
        fake_client, captured = _capturing_fake_client()
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        async for _ in hd.run_highlight_batch_with_global_cap(
            [(_target(), _window())], gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights", state=state,
            fps=2.0, frame_quality="high",
        ):
            pass
        video_part = captured["contents"][0].parts[1]
        assert video_part.video_metadata.fps == 2.0
        assert captured["config"].media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH

    @pytest.mark.asyncio
    async def test_single_match_success_yields_highlight_result_and_batch_complete(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(1.0, 2.0)])),
        )
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window())], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
                refine=False,  # scoped to the discovery-level contract only
            )
        ]
        types_seen = [e["type"] for e in events]
        assert "highlight_result" in types_seen
        assert types_seen[-1] == "batch_complete"
        assert state.calls_spent == 1

    @pytest.mark.asyncio
    async def test_out_of_bounds_count_is_surfaced_in_the_highlight_result_event(self, monkeypatch):
        """Live investigation (2026-07-18): the founder must SEE 'Gemini
        returned N highlights past the clip end' live in the stream, not
        just in a persisted file."""
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(290.0, 310.0)])),  # window is 300s
        )
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
                refine=False,  # scoped to discovery-side OOB surfacing only
            )
        ]
        result_event = next(e for e in events if e["type"] == "highlight_result")
        assert result_event["out_of_bounds_count"] == 1
        assert result_event["out_of_bounds_spans"] == [{"start_s": 290.0, "end_s": 310.0}]

    @pytest.mark.asyncio
    async def test_cap_reached_before_second_match_stops_the_batch(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([{"start_s": 1.0, "end_s": 2.0}])),
        )
        state = BatchRunState(cap=1)
        targets_and_windows = [(_target("m1"), _window("m1")), (_target("m2", "stem2"), _window("m2"))]
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                targets_and_windows, gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        assert events[-1]["type"] == "cap_reached"
        match_ids_started = [e["match_id"] for e in events if e["type"] == "match_start"]
        assert match_ids_started == ["m1"]

    @pytest.mark.asyncio
    async def test_stop_requested_before_loop_yields_run_stopped(self, monkeypatch):
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("[]"))
        state = BatchRunState(cap=10, stop_requested=True)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window())], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        assert events[-1]["type"] == "run_stopped"

    @pytest.mark.asyncio
    async def test_malformed_response_is_a_spent_call_surfaced_as_error_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("not json"))
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window())], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert state.calls_spent == 1


def _sequenced_fake_client(responses: list[str]):
    """A fake client whose ``generate_content`` returns ``responses[n]`` on
    its n-th call (0-based) -- lets a test script discovery's response, then
    each subsequent refinement attempt's response, deterministically."""
    call_count = {"n": 0}

    async def _fake_generate(*, model, contents, config):
        idx = call_count["n"]
        call_count["n"] += 1
        return SimpleNamespace(text=responses[idx], usage_metadata=None)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    return fake_client, call_count


def _refine_json(confirmed, start_s=None, end_s=None):
    return json.dumps({"action_confirmed": confirmed, "start_s": start_s, "end_s": end_s})


class TestRunHighlightBatchWithGlobalCapRefinementOrchestration:
    """Timestamp critique/refinement orchestration (frozen contract §B/§C/
    §H) — ``run_highlight_batch_with_global_cap`` drives the per-highlight
    refine loop, cap/stop checks BEFORE every refinement call (including the
    widen-retry), and ``call_kind`` tagging. ZERO real Gemini calls."""

    @pytest.mark.asyncio
    async def test_call_kind_tags_discovery_and_refine_events(self, monkeypatch):
        fake_client, _ = _sequenced_fake_client([
            json.dumps([_hl(100.0, 110.0)]),
            _refine_json(True, start_s=20.0, end_s=25.0),  # comfortably interior of the padded window
        ])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        discovery_events = [e for e in events if e["type"] == "highlight_result"]
        refine_events = [e for e in events if e["type"] == "refine_result"]
        assert all(e["call_kind"] == "discovery" for e in discovery_events)
        assert all(e["call_kind"] == "refine" for e in refine_events)
        assert len(refine_events) == 1
        assert state.calls_spent == 2  # 1 discovery + 1 refine

    @pytest.mark.asyncio
    async def test_accepted_refinement_produces_refined_true_enriched_dict(self, monkeypatch):
        fake_client, _ = _sequenced_fake_client([
            json.dumps([_hl(100.0, 110.0)]),
            _refine_json(True, start_s=20.0, end_s=25.0),
        ])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        results = next(e for e in events if e["type"] == "batch_complete")["results"]
        h = results["m1"]["highlights"][0]
        assert h["refined"] is True
        assert h["refinement_failed"] is False
        assert h["original_start_s"] == 100.0 and h["original_end_s"] == 110.0
        # padded_start = max(0, 100-12) = 88 -> 88+20=108, 88+25=113
        assert h["refined_start_s"] == pytest.approx(108.0)
        assert h["refined_end_s"] == pytest.approx(113.0)
        assert h["action_confirmed"] is True

    @pytest.mark.asyncio
    async def test_widen_retry_confirmed_on_second_attempt_is_accepted(self, monkeypatch):
        fake_client, call_count = _sequenced_fake_client([
            json.dumps([_hl(100.0, 110.0)]),
            _refine_json(False),  # first attempt: not confirmed
            _refine_json(True, start_s=20.0, end_s=25.0),  # widened (pad_before=24) retry: confirmed
        ])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        assert call_count["n"] == 3  # discovery + 2 refine attempts
        refine_events = [e for e in events if e["type"] == "refine_result"]
        assert [e["attempt"] for e in refine_events] == [1, 2]
        results = next(e for e in events if e["type"] == "batch_complete")["results"]
        h = results["m1"]["highlights"][0]
        assert h["refined"] is True
        # widened padded_start = max(0, 100-24) = 76 -> 76+20=96, 76+25=101
        assert h["refined_start_s"] == pytest.approx(96.0)
        assert h["refined_end_s"] == pytest.approx(101.0)
        assert state.calls_spent == 3

    @pytest.mark.asyncio
    async def test_widen_retry_still_unconfirmed_keeps_original_hard_stop(self, monkeypatch):
        fake_client, call_count = _sequenced_fake_client([
            json.dumps([_hl(100.0, 110.0)]),
            _refine_json(False),  # first attempt: not confirmed
            _refine_json(False),  # widened retry: STILL not confirmed -> HARD STOP, no third attempt
        ])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        assert call_count["n"] == 3  # never a third attempt -- hard stop
        results = next(e for e in events if e["type"] == "batch_complete")["results"]
        h = results["m1"]["highlights"][0]
        assert h["refined"] is False
        assert h["refinement_failed"] is True
        # No fabrication -- original kept, never a guessed refined time.
        assert h["refined_start_s"] is None and h["refined_end_s"] is None
        assert h["original_start_s"] == 100.0 and h["original_end_s"] == 110.0

    @pytest.mark.asyncio
    async def test_confirmed_but_at_edge_is_rejected_without_a_widen_retry(self, monkeypatch):
        """Frozen contract §B: the widen-retry triggers ONLY on "not
        confirmed" -- a confirmed-but-suspicious (near-edge) result must NOT
        trigger a second attempt."""
        fake_client, call_count = _sequenced_fake_client([
            json.dumps([_hl(100.0, 110.0)]),
            _refine_json(True, start_s=0.0, end_s=2.0),  # confirmed, but AT the padded window's own start edge
        ])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        assert call_count["n"] == 2  # discovery + exactly ONE refine attempt, no widen-retry
        results = next(e for e in events if e["type"] == "batch_complete")["results"]
        h = results["m1"]["highlights"][0]
        assert h["refined"] is False
        assert h["refinement_failed"] is True
        assert h["action_confirmed"] is True  # confirmed, just rejected on geometry

    @pytest.mark.asyncio
    async def test_cap_fires_mid_refinement_emits_partial_with_unrefined_remainder(self, monkeypatch):
        """Frozen contract §C, the crux: cap check INSIDE the per-highlight
        loop, not once per match. cap=2 -> discovery(1) + one refine
        call(2) exhausts the cap; the SECOND highlight's refinement is never
        attempted -- partial result emitted, unrefined remainder tagged,
        NEVER re-attempted, NEVER dropped."""
        fake_client, call_count = _sequenced_fake_client([
            json.dumps([_hl(100.0, 110.0), _hl(200.0, 210.0)]),  # discovery finds TWO highlights
            _refine_json(True, start_s=20.0, end_s=25.0),  # highlight #0's refine call (accepted)
        ])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=2)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        assert call_count["n"] == 2  # discovery + exactly one refine call -- cap enforced, no overspend
        assert state.calls_spent == 2
        assert events[-1]["type"] == "cap_reached"
        assert "results" in events[-1]  # the PARTIAL result travels with the terminal event
        highlights = events[-1]["results"]["m1"]["highlights"]
        assert len(highlights) == 2  # NEITHER highlight dropped
        assert highlights[0]["refined"] is True  # highlight #0 got its real outcome
        assert highlights[1]["refined"] is False  # highlight #1 never reached -- unrefined, not fabricated
        assert highlights[1]["refinement_failed"] is False  # "not attempted" != "failed"
        assert highlights[1]["refined_start_s"] is None

    @pytest.mark.asyncio
    async def test_refine_false_skips_refinement_entirely_no_extra_calls(self, monkeypatch):
        fake_client, call_count = _sequenced_fake_client([json.dumps([_hl(100.0, 110.0)])])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
                refine=False,
            )
        ]
        assert call_count["n"] == 1  # discovery ONLY -- no zero-effect refine call
        assert state.calls_spent == 1
        assert not any(e["type"] == "refine_result" for e in events)
        results = next(e for e in events if e["type"] == "batch_complete")["results"]
        h = results["m1"]["highlights"][0]
        assert h["refined"] is False
        assert h["refinement_failed"] is False
        assert h["action_confirmed"] is None
        assert h["refined_start_s"] is None and h["refined_end_s"] is None
        assert h["original_start_s"] == 100.0 and h["original_end_s"] == 110.0
        assert events[-1]["type"] == "batch_complete"


class TestValidateHighlightCandidate:
    """AC1-1/AC1-2/AC1-3 — pure function, no Gemini client, no ScopeWindow
    dependency (takes the already-resolved sent-window bounds directly)."""

    def test_valid_interior_candidate_returns_none(self):
        assert hd.validate_highlight_candidate(10.0, 20.0, 0.0, 300.0) is None

    def test_exactly_at_both_boundaries_is_valid_zero_tolerance_inclusive(self):
        assert hd.validate_highlight_candidate(0.0, 300.0, 0.0, 300.0) is None

    def test_nan_start_is_rejected_non_finite(self):
        assert hd.validate_highlight_candidate(float("nan"), 5.0, 0.0, 300.0) == "non_finite"

    def test_inf_end_is_rejected_non_finite(self):
        assert hd.validate_highlight_candidate(5.0, float("inf"), 0.0, 300.0) == "non_finite"

    def test_end_before_start_is_rejected_invalid_order(self):
        assert hd.validate_highlight_candidate(20.0, 10.0, 0.0, 300.0) == "invalid_order"

    def test_start_past_window_end_is_rejected_out_of_range(self):
        # The exact live-regression shape (§ AC1-8): window sent as [0, 131]
        # (int(round(130.901))), candidate [148, 152] -- entirely past the end.
        assert hd.validate_highlight_candidate(148.0, 152.0, 0.0, 131.0) == "out_of_range"

    def test_end_slightly_past_window_end_is_rejected_out_of_range_zero_tolerance(self):
        # No fuzz band -- 0.01s past the sent bound still rejects.
        assert hd.validate_highlight_candidate(10.0, 300.01, 0.0, 300.0) == "out_of_range"

    def test_start_before_window_start_is_rejected_out_of_range(self):
        assert hd.validate_highlight_candidate(-0.5, 5.0, 0.0, 300.0) == "out_of_range"

    def test_non_finite_checked_before_order_a_nan_never_masquerades_as_valid_order(self):
        # nan < x is always False in Python -- if order were checked first,
        # a NaN start could silently evade both checks. non_finite must win.
        assert hd.validate_highlight_candidate(float("nan"), 5.0, 0.0, 300.0) == "non_finite"

    def test_rejection_reasons_tuple_has_exactly_the_three_named_values(self):
        assert hd.REJECTION_REASONS == ("non_finite", "invalid_order", "out_of_range")


class TestSentWindowExposedFromDiscoveryAndRefinement:
    """The gate's bound-source contract: discover_match_highlights/
    refine_highlight_timestamp must report the INTEGER-SECOND bound they
    actually sent to Gemini, distinct from the pre-rounding float window."""

    @pytest.mark.asyncio
    async def test_discovery_reports_the_integer_rounded_sent_window(self, monkeypatch):
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("[]"))
        result = await hd.discover_match_highlights(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
            window=_window(start_s=0.0, end_s=130.901),
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
            system_prompt="", followup_prompt="find highlights",
        )
        assert result["sent_window_s"] == (0.0, 131.0)

    @pytest.mark.asyncio
    async def test_refinement_reports_the_integer_rounded_padded_sent_window(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps({"action_confirmed": False, "start_s": None, "end_s": None})),
        )
        result = await hd.refine_highlight_timestamp(
            match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
            match_window=_window(start_s=0.0, end_s=300.0),
            original_start_s=100.0, original_end_s=130.0, reason="x",
            gemini_api_key="fake", request_timeout_ms=1000,
            model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
        )
        # padded [88, 135] -- both already whole numbers, sent bound == padded bound.
        assert result["sent_window_s"] == (88.0, 135.0)


class TestRefinementPreCallErrorDistinction:
    """AC1-7: a pre-call validation failure (window collapsed before any
    Gemini client was built) must be a DISTINCT, more-specific exception
    type from a genuine post-call failure -- so a caller can bill them
    differently. Direct unit test against refine_highlight_timestamp,
    independent of the gate (this collapse case is unreachable through the
    gated batch driver once Task 3 lands -- see design note above -- but
    the function itself must still defend against its own degenerate
    inputs, regardless of caller discipline)."""

    def test_precall_error_is_a_refinement_error_subclass(self):
        assert issubclass(hd.RefinementPreCallError, hd.RefinementError)

    @pytest.mark.asyncio
    async def test_window_collapse_raises_the_precall_subclass_before_any_client_is_built(self, monkeypatch):
        boom_called = {"n": 0}

        def _boom(*a, **k):
            boom_called["n"] += 1
            raise AssertionError("refine_highlight_timestamp must not build a Gemini client pre-collapse-check")

        monkeypatch.setattr(hd, "build_gemini_client", _boom)
        # original [5,10] with pad_before=0, pad_after=-20 (a deliberately
        # degenerate direct call -- bypasses the route-level >=0 guard on
        # purpose, to exercise the function's own defensive raise):
        # padded_end = min(10, 10-20=-10) = -10 <= padded_start = max(0,5) = 5.
        with pytest.raises(hd.RefinementPreCallError):
            await hd.refine_highlight_timestamp(
                match_id="m1", video_url="https://www.youtube.com/embed/dummyID1234",
                match_window=_window(start_s=0.0, end_s=10.0),
                original_start_s=5.0, original_end_s=10.0, reason="x",
                gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                pad_before_s=0.0, pad_after_s=-20.0,
            )
        assert boom_called["n"] == 0


class TestEnrichedHighlightShapeHasRejectedFields:
    """Every enriched highlight dict, in EVERY branch, now carries
    rejected/rejection_reason -- default False/None for every non-rejected
    outcome (unattempted, refine-error, refine-accepted, refine-rejected-
    on-geometry, widen-retry-accepted, widen-retry-hard-stop)."""

    @pytest.mark.asyncio
    async def test_unattempted_highlight_has_rejected_false_and_none_reason(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(100.0, 110.0)])),
        )
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state, refine=False,
            )
        ]
        h = next(e for e in events if e["type"] == "batch_complete")["results"]["m1"]["highlights"][0]
        assert h["rejected"] is False
        assert h["rejection_reason"] is None


class TestValidationGateRejectsBeforeRefinement:
    """AC1-4/AC1-5/AC1-6/AC1-7/AC1-8 -- the crux of Item 1."""

    @pytest.mark.asyncio
    async def test_out_of_range_highlight_is_rejected_zero_refine_calls_zero_cost(self, monkeypatch):
        fake_client, call_count = _sequenced_fake_client([
            json.dumps([_hl(148.0, 152.0)]),  # window [0, 130.901] -> sent bound [0, 131]
        ])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=130.901))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        assert call_count["n"] == 1  # discovery ONLY -- zero refinement calls attempted
        assert state.calls_spent == 1  # the ONE real discovery call, nothing more
        rejected_events = [e for e in events if e["type"] == "highlight_rejected"]
        assert len(rejected_events) == 1
        assert rejected_events[0]["rejection_reason"] == "out_of_range"
        assert not any(e["type"] == "refine_result" for e in events)
        results = next(e for e in events if e["type"] == "batch_complete")["results"]
        h = results["m1"]["highlights"][0]
        assert h["rejected"] is True
        assert h["rejection_reason"] == "out_of_range"
        assert h["refined"] is False
        assert h["original_start_s"] == 148.0 and h["original_end_s"] == 152.0
        # AC1-5: never scoreable.
        assert hs.effective_pred_events(results["m1"]["highlights"]) == []

    @pytest.mark.asyncio
    async def test_rejection_fires_even_with_refine_false(self, monkeypatch):
        # AC1-4: rejection is independent of the refine toggle -- an invalid
        # candidate must never be silently waved through just because
        # refinement itself is off.
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(590.0, 610.0)])),  # window [0,600]
        )
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=600.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state, refine=False,
            )
        ]
        assert any(e["type"] == "highlight_rejected" and e["rejection_reason"] == "out_of_range" for e in events)
        h = next(e for e in events if e["type"] == "batch_complete")["results"]["m1"]["highlights"][0]
        assert h["rejected"] is True

    @pytest.mark.asyncio
    async def test_valid_highlight_alongside_a_rejected_one_still_refines_normally(self, monkeypatch):
        fake_client, call_count = _sequenced_fake_client([
            json.dumps([_hl(10.0, 20.0), _hl(148.0, 152.0)]),  # highlight 0 valid, highlight 1 invalid
            _refine_json(True, start_s=15.0, end_s=18.0),  # highlight 0's refine call only
        ])
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=130.901))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        assert call_count["n"] == 2  # discovery + exactly one refine call (for highlight 0 only)
        results = next(e for e in events if e["type"] == "batch_complete")["results"]
        h0, h1 = results["m1"]["highlights"]
        assert h0["rejected"] is False and h0["refined"] is True
        assert h1["rejected"] is True and h1["rejection_reason"] == "out_of_range"

    @pytest.mark.asyncio
    async def test_rejected_highlight_event_carries_the_raw_candidate_and_reason(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client",
            lambda *a, **k: _fake_client_returning(json.dumps([_hl(148.0, 152.0, reason="submission — RNC — hooks locked")])),
        )
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=130.901))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        rejected = next(e for e in events if e["type"] == "highlight_rejected")
        assert rejected["start_s"] == 148.0 and rejected["end_s"] == 152.0
        assert rejected["reason"] == "submission — RNC — hooks locked"
        assert rejected["match_id"] == "m1"
        assert "highlight_index" in rejected


class TestRejectionReasonsTableDriven:
    @pytest.mark.asyncio
    async def test_non_finite_via_raw_nan_json_literal(self, monkeypatch):
        # Python's json module parses the non-standard "NaN"/"Infinity"
        # literals by default (allow_nan=True) -- this is the EXACT way a
        # non-finite value can slip through Gemini's response undetected by
        # naive type coercion. Constructed as a raw string on purpose (not
        # json.dumps(float('nan')), to prove the harness's OWN json.loads
        # call, not a test artifact, is what parses it this way).
        raw_text = '[{"start_s": NaN, "end_s": 5.0, "reason": "x"}]'
        monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning(raw_text))
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        rejected = next(e for e in events if e["type"] == "highlight_rejected")
        assert rejected["rejection_reason"] == "non_finite"

    @pytest.mark.asyncio
    async def test_before_window_start_is_out_of_range(self, monkeypatch):
        monkeypatch.setattr(
            hd, "build_gemini_client", lambda *a, **k: _fake_client_returning(json.dumps([_hl(-5.0, 1.0)])),
        )
        state = BatchRunState(cap=10)
        events = [
            e async for e in hd.run_highlight_batch_with_global_cap(
                [(_target(), _window(start_s=0.0, end_s=300.0))], gemini_api_key="fake", request_timeout_ms=1000,
                model=hd.DEFAULT_HIGHLIGHT_MODEL, thinking_level="off",
                system_prompt="", followup_prompt="find highlights", state=state,
            )
        ]
        rejected = next(e for e in events if e["type"] == "highlight_rejected")
        assert rejected["rejection_reason"] == "out_of_range"
