"""Tests for optional Gemini-based athlete pair hints (service/vllm_selector)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from service.config import ServiceConfig


@pytest.mark.asyncio
async def test_suggest_athletes_returns_none_without_api_key():
    from service.vllm_selector import suggest_athletes

    cfg = ServiceConfig(gemini_api_key="")
    candidates = [{"candidate_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9}]
    assert await suggest_athletes(b"\xff\xd8", candidates, cfg) is None


@pytest.mark.asyncio
async def test_suggest_athletes_parses_gemini_response(tmp_path):
    from service.vllm_selector import suggest_athletes

    cfg = ServiceConfig(
        temp_dir=str(tmp_path),
        gemini_api_key="k",
        gemini_athlete_suggest_model="gemini-test",
    )
    candidates = [
        {"candidate_id": 0, "box": [1.0, 1.0, 50.0, 50.0], "confidence": 0.9},
        {"candidate_id": 1, "box": [60.0, 1.0, 100.0, 50.0], "confidence": 0.8},
        {"candidate_id": 2, "box": [200.0, 1.0, 250.0, 50.0], "confidence": 0.7},
    ]

    mock_resp = MagicMock()
    mock_resp.text = '{"athlete_a": 0, "athlete_b": 2}'

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    with patch("google.genai.Client", return_value=mock_client):
        result = await suggest_athletes(b"\xff\xd8fakejpeg", candidates, cfg)

    assert result is not None
    assert result["athlete_a"] == [1.0, 1.0, 50.0, 50.0]
    assert result["athlete_b"] == [200.0, 1.0, 250.0, 50.0]
    assert result["suggestion_model"] == "gemini-test"
