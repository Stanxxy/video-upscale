"""Characterization tests for S3 artifact key layout used by the worker."""
from __future__ import annotations

import os


def _base_key(input_key: str) -> str:
    return os.path.splitext(input_key)[0]


def test_checkpoint_frame_jpg_key():
    job_id = "job-abc"
    frame_idx = 42
    key = f"checkpoints/{job_id}/frame_{frame_idx}.jpg"
    assert key == "checkpoints/job-abc/frame_42.jpg"


def test_partial_tracking_key():
    job_id = "job-abc"
    key = f"checkpoints/{job_id}/partial_tracking.json"
    assert key == "checkpoints/job-abc/partial_tracking.json"


def test_tracked_json_key_from_input_video():
    input_key = "videos/match.mp4"
    assert f"{_base_key(input_key)}_tracked.json" == "videos/match_tracked.json"


def test_analysis_json_key_from_input_video():
    input_key = "folder/v.mp4"
    assert f"{_base_key(input_key)}_analysis.json" == "folder/v_analysis.json"


def test_annotated_mp4_key_from_input_video():
    input_key = "folder/v.mp4"
    assert f"{_base_key(input_key)}_annotated.mp4" == "folder/v_annotated.mp4"


def test_skip_upscale_tracked_mp4_key():
    input_key = "folder/v.mp4"
    assert f"{_base_key(input_key)}_tracked.mp4" == "folder/v_tracked.mp4"
