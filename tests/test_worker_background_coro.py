"""Tests for the ``_schedule_background_coro`` helper.

Bug context:
- ``service.worker`` schedules best-effort heartbeat coroutines (lifecycle
  progress writes, upscale progress writes) from worker threads via
  ``asyncio.run_coroutine_threadsafe`` without awaiting the resulting Future.
- If the coroutine raised (Keyspaces timeout, serialization error, network
  blip), the failure was silently swallowed.
- The fix wraps every fire-and-forget call in ``_schedule_background_coro``
  which attaches an ``add_done_callback`` that funnels the exception through
  ``_log_progress_future_failure`` -> ``logger.error``.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from service.worker import (
    _log_progress_future_failure,
    _schedule_background_coro,
)


@pytest.mark.asyncio
async def test_schedule_background_coro_logs_coroutine_exception(caplog):
    """A coroutine that raises should NOT propagate the exception; instead
    the failure must land in the engine log at ERROR level."""
    caplog.set_level(logging.ERROR, logger="service.worker")
    loop = asyncio.get_event_loop()

    async def _boom():
        raise RuntimeError("simulated keyspaces timeout")

    future = _schedule_background_coro(_boom(), loop, context="unit-test write")

    # Wait until the future is done (it will have an exception).
    for _ in range(50):
        if future.done():
            break
        await asyncio.sleep(0.01)
    assert future.done(), "background coroutine never completed"
    # The exception is recorded on the Future but our callback should have
    # consumed it via .exception() so the test does not raise.
    assert future.exception() is not None
    # Yield once so the done-callback (scheduled via call_soon_threadsafe)
    # runs before we assert on caplog.
    await asyncio.sleep(0)

    error_records = [
        r for r in caplog.records
        if r.levelno == logging.ERROR
        and "unit-test write" in r.getMessage()
    ]
    assert error_records, (
        f"Expected ERROR log mentioning 'unit-test write'; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_schedule_background_coro_success_is_silent(caplog):
    """Successful coroutines must not log anything at ERROR level."""
    caplog.set_level(logging.ERROR, logger="service.worker")
    loop = asyncio.get_event_loop()

    async def _ok():
        return "fine"

    future = _schedule_background_coro(_ok(), loop, context="happy path")
    for _ in range(50):
        if future.done():
            break
        await asyncio.sleep(0.01)
    assert future.done()
    assert future.exception() is None
    await asyncio.sleep(0)

    assert not [
        r for r in caplog.records
        if r.levelno == logging.ERROR and "happy path" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_log_progress_future_failure_safe_on_cancelled_future():
    """``_log_progress_future_failure`` must tolerate a cancelled Future
    without raising (cancellation has no exception to inspect)."""
    loop = asyncio.get_event_loop()

    async def _never_finishes():
        await asyncio.sleep(10)

    future = asyncio.run_coroutine_threadsafe(_never_finishes(), loop)
    future.cancel()

    # Must not raise on a cancelled future.
    _log_progress_future_failure(future, context="cancelled write")
