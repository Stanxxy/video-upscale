"""``eval/adcc/predict/global_cap_driver.py`` — the ADCC eval dashboard's
GLOBAL cross-match/slice Gemini call cap (Task 2). ZERO real Gemini calls:
every test here monkeypatches ``RunContext.gemini_client`` with a fake client
whose ``generate_content`` is a plain counting stub — no network, no API key
needed, safe to run anywhere.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval.adcc.predict.global_cap_driver import (
    PASS1_PLACEHOLDER_USD,
    PASS2_PLACEHOLDER_USD,
    BatchRunState,
    event_cost_usd,
    is_call_event,
    run_batch_with_global_cap,
)
from eval.adcc.predict.run_predictions import MatchTarget
from eval.adcc.predict.scope_policy import Slice
from service.pipelines.executors import RunContext


def _target(match_id: str = "m1", video_url_md5: str = "stem1") -> MatchTarget:
    # video_url is a real-SHAPED (fake) YouTube /embed/<11-char-id> URL --
    # live-bug fix regression: the driver now normalizes target.video_url
    # (youtube_url.normalize_youtube_watch_url) before building RunContext,
    # mirroring the real ADCC gold snapshot's own embed-URL field (INS-084).
    fake_video_id = (video_url_md5 + "00000000000")[:11]
    return MatchTarget(
        match_id=match_id, video_url=f"https://www.youtube.com/embed/{fake_video_id}",
        video_url_md5=video_url_md5, is_multi_match=False, gold_t_s_values=(10.0,),
    )


def _one_slice(match_id: str, start_s: float = 0.0, end_s: float = 80.0) -> list[Slice]:
    return [Slice(match_id=match_id, slice_index=0, start_s=start_s, end_s=end_s)]


def _four_worthy_chunk_map_response() -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps({"chunks": [
        {"start_s": 0, "end_s": 20, "reason": "match_action"},
        {"start_s": 20, "end_s": 40, "reason": "match_action"},
        {"start_s": 40, "end_s": 60, "reason": "match_action"},
        {"start_s": 60, "end_s": 80, "reason": "match_action"},
    ]}))


def _empty_chunk_result_response() -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))


# --------------------------------------------------------------------------- #
# Pure helper unit tests
# --------------------------------------------------------------------------- #
def test_event_cost_usd_uses_real_usage_metadata_when_present():
    event = {"type": "chunk_result", "usage_metadata": {
        "prompt_token_count": 1_000_000, "candidates_token_count": 1_000_000,
    }}
    assert event_cost_usd(event) == pytest.approx(0.25 + 1.50)


def test_event_cost_usd_falls_back_to_placeholder_when_usage_metadata_absent():
    assert event_cost_usd({"type": "chunk_map", "usage_metadata": None}) == PASS1_PLACEHOLDER_USD
    assert event_cost_usd({"type": "chunk_result", "usage_metadata": None}) == PASS2_PLACEHOLDER_USD


def test_is_call_event_counts_chunk_map_chunk_result_and_call_errors_only():
    assert is_call_event({"type": "chunk_map"}) is True
    assert is_call_event({"type": "chunk_result"}) is True
    assert is_call_event({"type": "error", "stage_id": "chunk_segment"}) is True
    assert is_call_event({"type": "error", "stage_id": "chunk_analyze"}) is True
    assert is_call_event({"type": "error", "stage_id": "dedup"}) is False
    assert is_call_event({"type": "progress"}) is False
    assert is_call_event({"type": "run_complete"}) is False


# --------------------------------------------------------------------------- #
# THE core guarantee — mocked Gemini, cap=3 over a plan needing 5 calls
# (1 PASS-1 + 4 worthy-chunk PASS-2), asserts <= 3 real calls happened and
# the run ends in an explicit CAP REACHED state.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_global_cap_aborts_batch_at_exactly_the_cap_mocked_gemini(monkeypatch):
    call_log: list[int] = []

    async def _fake_generate(*, model, contents, config):
        call_log.append(1)
        if len(call_log) == 1:
            return _four_worthy_chunk_map_response()  # PASS-1: 4 worthy chunks queued
        return _empty_chunk_result_response()  # every PASS-2 call after that

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    target = _target()
    state = BatchRunState(cap=3)
    events = [
        event
        async for event in run_batch_with_global_cap(
            [target], {target.match_id: _one_slice(target.match_id)},
            gemini_api_key="fake-key-never-used", request_timeout_ms=1000, state=state,
        )
    ]

    # THE guarantee: a plan whose worst case needs 5 Gemini calls (1 PASS-1 +
    # 4 worthy PASS-2 chunks) never makes more than `cap` real calls.
    assert len(call_log) == 3
    assert len(call_log) <= 3
    assert state.calls_spent == 3

    # Explicit terminal CAP REACHED state — never a silent stop / partial
    # success masquerading as a normal completion.
    assert events[-1]["type"] == "cap_reached"
    assert events[-1]["cap"] == 3
    assert events[-1]["calls_spent"] == 3

    # No batch_complete/match_complete anywhere — the batch was genuinely
    # aborted, not "completed a little early".
    assert not any(e["type"] in ("batch_complete", "match_complete") for e in events)

    # PASS-1's chunk_map really streamed (the real chunk map, not a stub) and
    # exactly 2 of the 4 worthy chunks got a real chunk_result before abort.
    assert sum(1 for e in events if e["type"] == "chunk_map") == 1
    assert sum(1 for e in events if e["type"] == "chunk_result") == 2


@pytest.mark.asyncio
async def test_global_cap_running_totals_are_tagged_on_every_call_event(monkeypatch):
    call_log: list[int] = []

    async def _fake_generate(*, model, contents, config):
        call_log.append(1)
        if len(call_log) == 1:
            return _four_worthy_chunk_map_response()
        return _empty_chunk_result_response()

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    target = _target()
    state = BatchRunState(cap=100)  # high cap — let the single slice run to completion
    events = [
        event
        async for event in run_batch_with_global_cap(
            [target], {target.match_id: _one_slice(target.match_id)},
            gemini_api_key="fake-key-never-used", request_timeout_ms=1000, state=state,
        )
    ]

    assert events[-1]["type"] == "batch_complete"
    assert events[-1]["calls_spent"] == 5  # 1 PASS-1 + 4 PASS-2, all under cap
    call_events = [e for e in events if e["type"] in ("chunk_map", "chunk_result")]
    assert [e["global_calls_spent"] for e in call_events] == [1, 2, 3, 4, 5]
    assert all(e["global_dollars_spent"] > 0 for e in call_events)
    # match_complete's stitched result really landed in batch_complete's results.
    assert target.match_id in events[-1]["results"]


@pytest.mark.asyncio
async def test_global_cap_never_exceeded_even_when_a_slice_would_need_many_more_calls(monkeypatch):
    """cap=1: the very FIRST call (PASS-1) already reaches the cap — the
    entire PASS-2 deep loop (4 worthy chunks) must NEVER start."""
    call_log: list[int] = []

    async def _fake_generate(*, model, contents, config):
        call_log.append(1)
        return _four_worthy_chunk_map_response()

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    target = _target()
    state = BatchRunState(cap=1)
    events = [
        event
        async for event in run_batch_with_global_cap(
            [target], {target.match_id: _one_slice(target.match_id)},
            gemini_api_key="fake-key-never-used", request_timeout_ms=1000, state=state,
        )
    ]

    assert len(call_log) == 1
    assert events[-1]["type"] == "cap_reached"
    assert sum(1 for e in events if e["type"] == "chunk_result") == 0


@pytest.mark.asyncio
async def test_global_cap_persists_across_multiple_matches_not_reset_per_match(monkeypatch):
    """Two single-slice matches, 1 PASS-1 call each (no worthy chunks — an
    all-broadcast chunk map). cap=1 must abort BEFORE the second match's
    PASS-1 call, proving the counter is a genuinely GLOBAL/cross-match total,
    not a per-match budget that resets."""
    call_log: list[int] = []

    async def _fake_generate(*, model, contents, config):
        call_log.append(1)
        return SimpleNamespace(text=json.dumps({"chunks": [
            {"start_s": 0, "end_s": 80, "reason": "broadcast"},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    t1, t2 = _target("m1", "stem1"), _target("m2", "stem2")
    state = BatchRunState(cap=1)
    slices_by_match = {t1.match_id: _one_slice(t1.match_id), t2.match_id: _one_slice(t2.match_id)}
    events = [
        event
        async for event in run_batch_with_global_cap(
            [t1, t2], slices_by_match,
            gemini_api_key="fake-key-never-used", request_timeout_ms=1000, state=state,
        )
    ]

    assert len(call_log) == 1  # NOT 2 — second match's PASS-1 call never happened
    assert events[-1]["type"] == "cap_reached"
    assert not any(e["type"] == "match_start" and e["match_id"] == "m2" for e in events)


@pytest.mark.asyncio
async def test_stop_requested_mid_batch_prevents_any_new_call(monkeypatch):
    """Simulates the operator clicking STOP between chunk_result #1 and #2 —
    the 3rd/4th worthy-chunk calls must never happen."""
    call_log: list[int] = []
    state = BatchRunState(cap=100)

    async def _fake_generate(*, model, contents, config):
        call_log.append(1)
        if len(call_log) == 1:
            return _four_worthy_chunk_map_response()
        if len(call_log) == 2:
            state.stop_requested = True  # operator hits STOP right after call #2 lands
            state.stop_reason = "operator stop"
        return _empty_chunk_result_response()

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    target = _target()
    events = [
        event
        async for event in run_batch_with_global_cap(
            [target], {target.match_id: _one_slice(target.match_id)},
            gemini_api_key="fake-key-never-used", request_timeout_ms=1000, state=state,
        )
    ]

    assert len(call_log) == 2  # PASS-1 + exactly 1 PASS-2 call (the in-flight one), never a 3rd
    assert events[-1]["type"] == "run_stopped"
    assert events[-1]["reason"] == "operator stop"


# --------------------------------------------------------------------------- #
# Live-bug fix regression (2026-07-18): the tag-batch path threaded the ADCC
# gold snapshot's raw /embed/<id> YouTube URL (INS-084) straight into
# RunContext.youtube_url, which service/pipelines/executors.py then sent
# unnormalized to Gemini's native file_uri ingestion -- every one of the 258
# real matches would 400 on the first real run. The driver must now
# normalize to the canonical watch?v=<id> form BEFORE constructing
# RunContext, so executors.py (co-tenant, byte-untouchable) needs zero
# changes. Zero real Gemini calls -- RunContext.gemini_client is
# monkeypatched throughout this file.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_context_receives_canonical_watch_url_not_the_raw_embed_url(monkeypatch):
    captured_ctxs: list = []

    def _fake_gemini_client(self):
        captured_ctxs.append(self)
        fake_client = MagicMock()

        async def _fake_generate(*, model, contents, config):
            return _four_worthy_chunk_map_response() if len(captured_ctxs) == 1 else _empty_chunk_result_response()

        fake_client.aio.models.generate_content = _fake_generate
        return fake_client

    monkeypatch.setattr(RunContext, "gemini_client", _fake_gemini_client)

    target = _target(video_url_md5="stem1")  # -> https://www.youtube.com/embed/stem1000000
    state = BatchRunState(cap=100)
    events = [
        event
        async for event in run_batch_with_global_cap(
            [target], {target.match_id: _one_slice(target.match_id)},
            gemini_api_key="fake-key-never-used", request_timeout_ms=1000, state=state,
        )
    ]

    assert events[-1]["type"] == "batch_complete"
    assert captured_ctxs, "gemini_client() was never called — the slice never ran"
    assert all(ctx.youtube_url == "https://www.youtube.com/watch?v=stem1000000" for ctx in captured_ctxs)


@pytest.mark.asyncio
async def test_malformed_video_url_raises_before_any_gemini_call():
    from eval.adcc.predict.youtube_url import YoutubeUrlError

    bad_target = MatchTarget(
        match_id="m_bad", video_url="https://example.com/not-a-youtube-url", video_url_md5="stem_bad",
        is_multi_match=False, gold_t_s_values=(10.0,),
    )
    state = BatchRunState(cap=100)
    with pytest.raises(YoutubeUrlError):
        async for _ in run_batch_with_global_cap(
            [bad_target], {bad_target.match_id: _one_slice(bad_target.match_id)},
            gemini_api_key="fake-key-never-used", request_timeout_ms=1000, state=state,
        ):
            pass
    assert state.calls_spent == 0  # zero spend — raised before any Gemini call
