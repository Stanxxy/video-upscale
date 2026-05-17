"""Shared helpers for worker checkpoint integration tests."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

from service.models import TrackRequest

V1_KEYS = {"schema_version", "pending_detection", "artifacts", "worker_state"}


def track_request(**overrides) -> TrackRequest:
    base = dict(bucket="b", key="v.mp4")
    base.update(overrides)
    return TrackRequest(**base)


def stub_s3() -> MagicMock:
    s3 = MagicMock()
    s3.ensure_bucket = MagicMock()
    s3.download_file = MagicMock(side_effect=lambda b, k, p: p)
    s3.put_object = MagicMock()
    s3.upload_file = MagicMock()
    s3.upload_json = MagicMock()
    s3.get_object = MagicMock()
    s3.download_json = MagicMock()
    return s3


def assert_envelope(data: dict) -> None:
    assert V1_KEYS <= data.keys(), f"missing envelope keys: {V1_KEYS - data.keys()}"
    assert data["schema_version"] == 1
    assert isinstance(data["artifacts"], dict)
    ws = data["worker_state"]
    assert {
        "progress_percent",
        "current_frame",
        "total_frames",
        "stage_progress_fraction",
    } <= ws.keys()


async def invoke_detection_cb(cb, **kwargs):
    """Drive detection_cb from an executor (mirrors run_tracking_job thread usage)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: cb(
            kwargs.pop("reason", "tracking_lost"),
            kwargs.pop("frame_jpeg", b""),
            **kwargs,
        ),
    )


def stub_run_tracking_job(*args, **kwargs):
    """Replacement for run_tracking_job that writes a tiny tracking.json."""
    import json

    output_dir = kwargs.get("tracking_output_dir") or args[3]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tracking.json")
    # >100 bytes so failure finally-block partial upload guard passes in tests.
    payload = {
        "start_frame": 0,
        "frames": [
            {"frame_idx": i, "athletes": [{"box": [0, 0, 10, 10], "id": i}]}
            for i in range(8)
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def leaf_run_tracking_job_with_overlap(*args, **kwargs):
    """Leaf segment: frame 0 overlaps parent (last-writer), frame 1 is new."""
    import json

    output_dir = kwargs.get("tracking_output_dir") or args[3]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tracking.json")
    with open(path, "w") as f:
        json.dump(
            {
                "start_frame": 1,
                "frames": [
                    {"frame_idx": 0, "athlete": "leaf_wins"},
                    {"frame_idx": 1, "athlete": "B"},
                ],
            },
            f,
        )
    return path
