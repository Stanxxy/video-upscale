"""Worker-integration tests asserting service.worker writes V1-envelope checkpoints.

ML/IO collaborators are stubbed via unittest.mock.patch — the goal of these
tests is to lock in the *checkpoint shape* the worker emits at each stage,
not to exercise SAM2/RTMPose/Gemini.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.models import TrackRequest


V1_KEYS = {"schema_version", "pending_detection", "artifacts", "worker_state"}


def _track_request(**overrides) -> TrackRequest:
    base = dict(bucket="b", key="v.mp4")
    base.update(overrides)
    return TrackRequest(**base)


def _stub_s3() -> MagicMock:
    s3 = MagicMock()
    s3.ensure_bucket = MagicMock()
    s3.download_file = MagicMock(side_effect=lambda b, k, p: p)
    s3.put_object = MagicMock()
    s3.upload_file = MagicMock()
    s3.upload_json = MagicMock()
    s3.get_object = MagicMock()
    s3.download_json = MagicMock()
    return s3


def _assert_envelope(data: dict) -> None:
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


# ---------------------------------------------------------------------------
# Mid-track detection callback bug fix
# ---------------------------------------------------------------------------


async def _invoke_detection_cb(cb, **kwargs):
    """Drive ``detection_cb`` from an executor so its internal
    ``run_coroutine_threadsafe(...).result()`` does not deadlock the running
    event loop. This mirrors how ``run_tracking_job`` calls the callback in
    production (from a thread off the event loop)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: cb(
            kwargs.pop("reason", "tracking_lost"),
            kwargs.pop("frame_jpeg", b""),
            **kwargs,
        ),
    )


@pytest.mark.asyncio
async def test_make_detection_cb_uses_put_object_not_upload_file(
    mock_jobs_store, tmp_path,
):
    """The mid-track detection callback uploads the checkpoint frame via
    put_object (raw bytes) — never upload_file (which expects a local path)."""
    from service.worker import _make_detection_cb

    loop = asyncio.get_event_loop()
    config = ServiceConfig(temp_dir=str(tmp_path), s3_endpoint_url="http://x")
    request = _track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = _stub_s3()
    work_dir = tmp_path / "wd"
    work_dir.mkdir()

    cb = _make_detection_cb(
        "job-x", loop, mock_jobs_store, s3, config, request, str(work_dir),
    )

    await _invoke_detection_cb(
        cb,
        reason="tracking_lost",
        frame_jpeg=b"\xff\xd8jpeg-bytes",
        yolo_detections=[{"box": [10, 20, 100, 200], "confidence": 0.9}],
        frame_idx=512,
    )

    # The frame MUST go through put_object, never upload_file (which would
    # treat the bytes blob as a local file path and fail at runtime).
    assert s3.put_object.called, "mid-track must call put_object"
    assert not s3.upload_file.called, "upload_file is the buggy code path"

    # Argument layout: put_object(bucket, key, body, content_type)
    args, _ = s3.put_object.call_args
    assert args[0] == request.bucket
    assert args[1].startswith("checkpoints/job-x/")
    assert args[2] == b"\xff\xd8jpeg-bytes"
    assert args[3] == "image/jpeg"


@pytest.mark.asyncio
async def test_make_detection_cb_writes_track_mid_loss_envelope(
    mock_jobs_store, tmp_path,
):
    """The mid-track detection callback writes a V1 track checkpoint with
    pending_detection.reason == 'tracking_lost' and partial-tracking artifacts."""
    from service.worker import _make_detection_cb

    loop = asyncio.get_event_loop()
    config = ServiceConfig(temp_dir=str(tmp_path), s3_endpoint_url="http://x")
    request = _track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = _stub_s3()

    work_dir = tmp_path / "wd"
    (work_dir / "tracking").mkdir(parents=True)
    # A tiny but parseable tracking.json so _load_partial_tracking_dict
    # can succeed.
    (work_dir / "tracking" / "tracking.json").write_text(
        '{"start_frame":0,"frames":[{"frame_idx":7,"athletes":[]}]}'
    )

    cb = _make_detection_cb(
        "job-mid", loop, mock_jobs_store, s3, config, request, str(work_dir),
    )
    await _invoke_detection_cb(
        cb,
        reason="tracking_lost",
        frame_jpeg=b"\xff\xd8jpeg",
        yolo_detections=[{"box": [10, 20, 100, 200], "confidence": 0.9}],
        frame_idx=512,
    )

    cp = mock_jobs_store._checkpoints[("job-mid", PipelineStage.TRACK.value)]
    data = cp["checkpoint_data"]
    _assert_envelope(data)
    assert data["pending_detection"]["reason"] == "tracking_lost"
    assert data["pending_detection"]["frame_idx"] == 512
    assert data["artifacts"]["resume_from_frame"] == 512
    assert data["artifacts"]["partial_tracking_s3_key"].endswith("partial_tracking.json")


# ---------------------------------------------------------------------------
# Download + initial detect checkpoint envelopes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Periodic track_progress helper (Task 4: 30-second cadence)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_track_progress_helper_uploads_partial_and_writes_checkpoint(
    mock_jobs_store, tmp_path,
):
    """When upload_partial is true and tracking.json exists, the helper
    uploads to S3 and writes a V1 track checkpoint pointing at the upload."""
    from service import worker

    config = ServiceConfig(temp_dir=str(tmp_path), s3_endpoint_url="http://x")
    job_store = InMemoryJobStore()
    request = _track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8], output_bucket="out")
    s3 = _stub_s3()

    work_dir = tmp_path / "wd"
    (work_dir / "tracking").mkdir(parents=True)
    (work_dir / "tracking" / "tracking.json").write_text(
        '{"start_frame":0,"frames":[{"frame_idx":1199,"athletes":[]}]}'
    )

    await worker._update_tracking_progress_with_partial(
        "job-track",
        1200, 3600, 35.0,
        job_store, mock_jobs_store,
        request, str(work_dir), s3,
        write_lifecycle=True,
        upload_partial=True,
    )

    # Partial tracking JSON was uploaded to the output bucket.
    assert s3.upload_json.called
    upload_args, _ = s3.upload_json.call_args
    assert upload_args[1] == "out"  # output_bucket
    assert upload_args[2].endswith("partial_tracking.json")

    cp = mock_jobs_store._checkpoints[("job-track", PipelineStage.TRACK.value)]
    data = cp["checkpoint_data"]
    _assert_envelope(data)
    assert data["reason"] == "tracking_progress"
    assert data["artifacts"]["partial_tracking_s3_key"].endswith("partial_tracking.json")
    assert data["artifacts"]["resume_from_frame"] == 1200
    assert data["worker_state"]["current_frame"] == 1200
    assert data["worker_state"]["total_frames"] == 3600
    assert data["worker_state"]["progress_percent"] == 35.0
    assert data["resume_cursor"] == {"frame_idx": 1200}


@pytest.mark.asyncio
async def test_track_progress_helper_no_op_when_neither_flag_set(
    mock_jobs_store, tmp_path,
):
    """write_lifecycle=False and upload_partial=False → no S3 upload, no checkpoint."""
    from service import worker

    config = ServiceConfig(temp_dir=str(tmp_path), s3_endpoint_url="http://x")
    job_store = InMemoryJobStore()
    request = _track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = _stub_s3()

    work_dir = tmp_path / "wd"
    work_dir.mkdir()

    await worker._update_tracking_progress_with_partial(
        "job-track",
        500, 1000, 35.0,
        job_store, mock_jobs_store,
        request, str(work_dir), s3,
        write_lifecycle=False,
        upload_partial=False,
    )

    assert not s3.upload_json.called
    assert ("job-track", PipelineStage.TRACK.value) not in mock_jobs_store._checkpoints


@pytest.mark.asyncio
async def test_track_progress_helper_skips_partial_when_file_missing(
    mock_jobs_store, tmp_path,
):
    """upload_partial=True but no tracking.json on disk → no upload, no checkpoint."""
    from service import worker

    config = ServiceConfig(temp_dir=str(tmp_path), s3_endpoint_url="http://x")
    job_store = InMemoryJobStore()
    request = _track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = _stub_s3()

    work_dir = tmp_path / "wd"
    work_dir.mkdir()  # no tracking subdir / tracking.json

    await worker._update_tracking_progress_with_partial(
        "job-track",
        500, 1000, 35.0,
        job_store, mock_jobs_store,
        request, str(work_dir), s3,
        write_lifecycle=False,
        upload_partial=True,
    )

    assert not s3.upload_json.called
    assert ("job-track", PipelineStage.TRACK.value) not in mock_jobs_store._checkpoints


# ---------------------------------------------------------------------------
# Existing integration test: download + initial-detect envelope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_job_writes_download_then_detect_checkpoints(
    mock_jobs_store, tmp_path,
):
    """Driving the no-boxes path writes both DOWNLOAD and DETECT V1 envelopes,
    then suspends without raising."""
    from service import worker

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        gemini_api_key="",  # disable analysis
    )
    job_store = InMemoryJobStore()
    request = _track_request(box_a=None, box_b=None)
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = _stub_s3()

    # Stub heavy collaborators.
    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch(
             "service.tracking_runner.run_detect",
             return_value=[{"box": [0, 0, 10, 10], "confidence": 0.9}],
         ), \
         patch(
             "service.tracking_runner.capture_frame_jpeg",
             return_value=b"\xff\xd8jpeg",
         ):
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    # DOWNLOAD checkpoint must exist with V1 envelope.
    download = mock_jobs_store._checkpoints.get(
        (job.job_id, PipelineStage.DOWNLOAD.value)
    )
    assert download is not None, "download checkpoint must be written"
    assert download["completed"] is False
    _assert_envelope(download["checkpoint_data"])
    assert download["checkpoint_data"]["reason"] == "download_completed"
    assert download["checkpoint_data"]["worker_state"]["progress_percent"] == 10.0

    # DETECT checkpoint must exist with V1 envelope and pending_detection.
    detect = mock_jobs_store._checkpoints.get(
        (job.job_id, PipelineStage.DETECT.value)
    )
    assert detect is not None, "detect checkpoint must be written"
    _assert_envelope(detect["checkpoint_data"])
    pd = detect["checkpoint_data"]["pending_detection"]
    assert pd is not None
    assert pd["reason"] == "initial"
    assert pd["candidates"][0]["box"] == [0, 0, 10, 10]
    assert pd["frame_s3_key"].startswith("checkpoints/")

    # Lifecycle should have transitioned to AWAITING_CORRECTION.
    lc = await mock_jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.AWAITING_CORRECTION.value

    # The detection frame should have been uploaded via put_object (initial
    # detection already used put_object — guard against regression).
    assert s3.put_object.called
