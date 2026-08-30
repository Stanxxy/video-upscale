"""Gemini contract guards — no API calls."""
from __future__ import annotations

import pytest

from eval.adcc.sam3_scout.gemini_arms import ScoutGeminiError, assert_scout_gemini_contract


def test_accepts_locked_contract():
    assert_scout_gemini_contract("gemini-3.5-flash", 10.0, "high")


def test_rejects_flash_lite():
    with pytest.raises(ScoutGeminiError, match="forbidden"):
        assert_scout_gemini_contract("gemini-3.1-flash-lite", 10.0, "high")


def test_rejects_wrong_fps():
    with pytest.raises(ScoutGeminiError, match="fps"):
        assert_scout_gemini_contract("gemini-3.5-flash", 5.0, "high")


def test_rejects_wrong_quality():
    with pytest.raises(ScoutGeminiError, match="frame_quality"):
        assert_scout_gemini_contract("gemini-3.5-flash", 10.0, "medium")
