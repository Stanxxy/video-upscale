"""``service/pipelines/gemini_retry.py`` — the centralized transient-retry
helper for QA-layer Gemini calls.

Pins the resilience contract root-caused against the 503 UNAVAILABLE bug
report (frame 1116 detect failure): transient errors (503/429/500/502/504,
connection/timeout) retry with exponential backoff + jitter up to
``max_attempts``; a 429's ``Retry-After`` header is honored over the computed
delay; non-transient errors (400/404/permission) never retry; exhausted
retries re-raise the REAL last exception (nothing fabricated). All sleeps are
mocked — these tests never actually sleep.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from google.genai import errors as genai_errors

from service.pipelines.gemini_retry import (
    GeminiRetryConfig,
    call_with_retry,
    is_transient_error,
)


def _server_error(code: int, headers: dict | None = None) -> genai_errors.ServerError:
    response = None
    if headers is not None:
        response = httpx.Response(status_code=code, headers=headers, request=httpx.Request("POST", "http://x"))
    return genai_errors.ServerError(code, {"message": f"{code} error", "status": "UNAVAILABLE"}, response)


def _client_error(code: int, headers: dict | None = None) -> genai_errors.ClientError:
    response = None
    if headers is not None:
        response = httpx.Response(status_code=code, headers=headers, request=httpx.Request("POST", "http://x"))
    return genai_errors.ClientError(code, {"message": f"{code} error", "status": "RESOURCE_EXHAUSTED"}, response)


FAST_CONFIG = GeminiRetryConfig(max_attempts=5, initial_delay_s=1.0, max_delay_s=30.0, exp_base=2.0, jitter_s=1.0)


# --------------------------------------------------------------------------- #
# is_transient_error classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_is_transient_error_true_for_transient_http_codes(code):
    assert is_transient_error(_server_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_is_transient_error_false_for_non_transient_http_codes(code):
    assert is_transient_error(_client_error(code)) is False


def test_is_transient_error_true_for_httpx_connect_error():
    assert is_transient_error(httpx.ConnectError("boom")) is True


def test_is_transient_error_true_for_httpx_timeout_exception():
    assert is_transient_error(httpx.ConnectTimeout("boom")) is True


def test_is_transient_error_false_for_generic_value_error():
    assert is_transient_error(ValueError("not JSON")) is False


# --------------------------------------------------------------------------- #
# call_with_retry — success after transient failures
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_retries_transient_503_then_succeeds(monkeypatch):
    sleeps: list[float] = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", _fake_sleep)

    fn = AsyncMock(side_effect=[_server_error(503), _server_error(503), "real-result"])
    result = await call_with_retry(fn, op_name="test-op", retry_config=FAST_CONFIG)

    assert result == "real-result"
    assert fn.call_count == 3
    assert len(sleeps) == 2  # one sleep between each of the 2 failed attempts and the next


@pytest.mark.asyncio
async def test_retry_count_matches_number_of_transient_failures(monkeypatch):
    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", AsyncMock())
    fn = AsyncMock(side_effect=[_server_error(429), "ok"])
    result = await call_with_retry(fn, op_name="test-op", retry_config=FAST_CONFIG)
    assert result == "ok"
    assert fn.call_count == 2


# --------------------------------------------------------------------------- #
# Retry-After honored over computed backoff
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_429_with_retry_after_header_is_honored(monkeypatch):
    sleeps: list[float] = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", _fake_sleep)

    err = _client_error(429, headers={"retry-after": "12"})
    fn = AsyncMock(side_effect=[err, "ok"])
    result = await call_with_retry(fn, op_name="test-op", retry_config=FAST_CONFIG)

    assert result == "ok"
    assert sleeps == [12.0]  # exact header value, not the computed exponential-backoff delay


@pytest.mark.asyncio
async def test_retry_after_missing_falls_back_to_computed_backoff(monkeypatch):
    sleeps: list[float] = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr("service.pipelines.gemini_retry.random.uniform", lambda lo, hi: 0.0)

    err = _server_error(503)  # no response/headers at all
    fn = AsyncMock(side_effect=[err, "ok"])
    cfg = GeminiRetryConfig(max_attempts=5, initial_delay_s=1.0, max_delay_s=30.0, exp_base=2.0, jitter_s=0.0)
    result = await call_with_retry(fn, op_name="test-op", retry_config=cfg)

    assert result == "ok"
    assert sleeps == [1.0]  # initial_delay_s * exp_base**0, zero jitter


@pytest.mark.asyncio
async def test_backoff_grows_exponentially_and_is_capped(monkeypatch):
    sleeps: list[float] = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr("service.pipelines.gemini_retry.random.uniform", lambda lo, hi: 0.0)

    cfg = GeminiRetryConfig(max_attempts=6, initial_delay_s=1.0, max_delay_s=3.5, exp_base=2.0, jitter_s=0.0)
    fn = AsyncMock(side_effect=[_server_error(503)] * 4 + ["ok"])
    result = await call_with_retry(fn, op_name="test-op", retry_config=cfg)

    assert result == "ok"
    # 1.0, 2.0, capped 3.5 (would be 4.0), capped 3.5 (would be 8.0)
    assert sleeps == [1.0, 2.0, 3.5, 3.5]


# --------------------------------------------------------------------------- #
# Exhausted retries — honest failure, nothing fabricated
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_exhausted_retries_reraises_the_real_last_exception(monkeypatch):
    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", AsyncMock())
    persistent_error = _server_error(503)
    fn = AsyncMock(side_effect=[persistent_error] * 10)  # always fails
    cfg = GeminiRetryConfig(max_attempts=3, initial_delay_s=0.01, max_delay_s=0.01, exp_base=2.0, jitter_s=0.0)

    with pytest.raises(genai_errors.ServerError) as exc_info:
        await call_with_retry(fn, op_name="test-op", retry_config=cfg)

    assert exc_info.value is persistent_error
    assert fn.call_count == 3  # exactly max_attempts, no more no less


# --------------------------------------------------------------------------- #
# Non-transient errors — fail fast, never retried
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_non_transient_400_fails_fast_without_retry(monkeypatch):
    sleep_mock = AsyncMock()
    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", sleep_mock)
    fn = AsyncMock(side_effect=[_client_error(400)])

    with pytest.raises(genai_errors.ClientError) as exc_info:
        await call_with_retry(fn, op_name="test-op", retry_config=FAST_CONFIG)

    assert exc_info.value.code == 400
    assert fn.call_count == 1
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_non_transient_404_fails_fast_without_retry(monkeypatch):
    sleep_mock = AsyncMock()
    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", sleep_mock)
    fn = AsyncMock(side_effect=[_client_error(404)])

    with pytest.raises(genai_errors.ClientError):
        await call_with_retry(fn, op_name="test-op", retry_config=FAST_CONFIG)

    assert fn.call_count == 1
    sleep_mock.assert_not_called()


# --------------------------------------------------------------------------- #
# GeminiRetryConfig.from_service_config
# --------------------------------------------------------------------------- #
def test_from_service_config_reads_configured_knobs():
    from service.config import ServiceConfig

    config = ServiceConfig(
        gemini_retry_max_attempts=7,
        gemini_retry_initial_delay_s=2.0,
        gemini_retry_max_delay_s=45.0,
        gemini_retry_exp_base=3.0,
        gemini_retry_jitter_s=0.5,
    )
    retry_cfg = GeminiRetryConfig.from_service_config(config)
    assert retry_cfg == GeminiRetryConfig(
        max_attempts=7, initial_delay_s=2.0, max_delay_s=45.0, exp_base=3.0, jitter_s=0.5,
    )
