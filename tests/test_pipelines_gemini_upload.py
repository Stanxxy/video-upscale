"""Unit tests for service/pipelines/gemini_upload.py (S12 Phase 1b, item 1).

All Gemini Files API calls are mocked — no live network access.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from service.pipelines import gemini_upload


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("service.pipelines.gemini_upload.asyncio.sleep", AsyncMock())


def _fake_client_with_states(*states: "types.FileState"):
    """Build a MagicMock genai.Client whose ``aio.files.get`` yields the given
    states in order (repeating the last one if polled further)."""
    client = MagicMock()
    calls = {"n": 0}

    async def _get(*, name):
        idx = min(calls["n"], len(states) - 1)
        calls["n"] += 1
        return types.File(name=name, state=states[idx])

    client.aio.files.get = _get
    return client, calls


@pytest.mark.asyncio
async def test_upload_video_to_gemini_delegates_to_sdk():
    client = MagicMock()
    uploaded = types.File(name="files/abc", state=types.FileState.PROCESSING)
    client.aio.files.upload = AsyncMock(return_value=uploaded)

    result = await gemini_upload.upload_video_to_gemini(client, "/tmp/video.mp4")

    assert result is uploaded
    client.aio.files.upload.assert_awaited_once_with(file="/tmp/video.mp4")


@pytest.mark.asyncio
async def test_poll_until_active_processing_then_active():
    client, calls = _fake_client_with_states(
        types.FileState.PROCESSING, types.FileState.PROCESSING, types.FileState.ACTIVE,
    )

    result = await gemini_upload.poll_until_active(
        client, "files/abc", poll_interval_sec=0.01, timeout_sec=5.0,
    )

    assert result.state == types.FileState.ACTIVE
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_poll_until_active_failed_state_raises_runtime_error():
    client, _ = _fake_client_with_states(types.FileState.PROCESSING, types.FileState.FAILED)

    with pytest.raises(RuntimeError, match="FAILED"):
        await gemini_upload.poll_until_active(
            client, "files/abc", poll_interval_sec=0.01, timeout_sec=5.0,
        )


@pytest.mark.asyncio
async def test_poll_until_active_timeout_raises_timeout_error_never_fabricates_ready():
    """A file stuck in PROCESSING forever must raise on timeout — NEVER
    silently proceed as if it were ACTIVE."""
    client = MagicMock()

    async def _always_processing(*, name):
        return types.File(name=name, state=types.FileState.PROCESSING)

    client.aio.files.get = _always_processing

    real_monotonic = __import__("time").monotonic
    calls = {"n": 0}

    def _fake_monotonic():
        # Advance time on every call after the first (deadline computation)
        # so the loop's `time.monotonic() >= deadline` check fires quickly
        # without a real sleep-based timeout.
        calls["n"] += 1
        return real_monotonic() + calls["n"] * 100.0

    import service.pipelines.gemini_upload as mod
    real_time_monotonic = mod.time.monotonic
    mod.time.monotonic = _fake_monotonic
    try:
        with pytest.raises(TimeoutError, match="did not reach ACTIVE"):
            await gemini_upload.poll_until_active(
                client, "files/abc", poll_interval_sec=0.01, timeout_sec=1.0,
            )
    finally:
        mod.time.monotonic = real_time_monotonic


@pytest.mark.asyncio
async def test_delete_gemini_file_success():
    client = MagicMock()
    client.aio.files.delete = AsyncMock(return_value=None)

    await gemini_upload.delete_gemini_file(client, "files/abc")

    client.aio.files.delete.assert_awaited_once_with(name="files/abc")


@pytest.mark.asyncio
async def test_delete_gemini_file_best_effort_swallows_failure():
    """Cleanup failure must NEVER raise — the 48h server TTL is the backstop."""
    client = MagicMock()
    client.aio.files.delete = AsyncMock(side_effect=RuntimeError("network blip"))

    await gemini_upload.delete_gemini_file(client, "files/abc")  # must not raise
