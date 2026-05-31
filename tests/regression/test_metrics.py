"""Pytest wrapper for tests/regression/metrics.py oracle fixtures."""
from __future__ import annotations

import json
from pathlib import Path

from tests.regression.metrics import Event, parse_events, score, score_major_events

FIXTURE_DIR = Path(__file__).resolve().parent / "oracle"
ORACLE = FIXTURE_DIR / "fixture1.json"
FAST = Path(__file__).resolve().parent / "test_fast_fixture1.json"


def test_parse_events_reads_clips_schema():
    data = {
        "clips": [
            {
                "role": "athlete_a",
                "action": "takedown",
                "technique": "single_leg",
                "start_frame": 10,
                "end_frame": 100,
            }
        ]
    }
    events = parse_events(data)
    assert len(events) == 1
    assert events[0].role == "athlete_a"
    assert events[0].action == "takedown"


def test_event_temporal_iou():
    a = Event("athlete_a", "takedown", "single_leg", 0, 100)
    b = Event("athlete_a", "takedown", "single_leg", 50, 150)
    assert 0.0 < a.iou(b) < 1.0
    c = Event("athlete_b", "takedown", "", 200, 300)
    assert a.iou(c) == 0.0


def test_score_perfect_match():
    events = [Event("athlete_a", "takedown", "single_leg", 0, 100)]
    result = score(events, events)
    assert result["recall"] == 1.0
    assert result["score"] == 1.0


def test_score_major_events_empty_both():
    result = score_major_events([], [])
    assert result["score"] == 1.0


def test_oracle_fixture_files_exist():
    assert ORACLE.is_file()
    assert FAST.is_file()


def test_oracle_vs_fast_fixture_score_runs():
    with ORACLE.open() as f:
        oracle_json = json.load(f)
    with FAST.open() as f:
        fast_json = json.load(f)
    oracle_events = parse_events(oracle_json)
    fast_events = parse_events(fast_json)
    assert oracle_events
    assert fast_events
    result = score(oracle_events, fast_events)
    assert 0.0 <= result["score"] <= 1.0
    major = score_major_events(oracle_events, fast_events)
    assert 0.0 <= major["score"] <= 1.0
