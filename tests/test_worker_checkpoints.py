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
# Upscale analysis flush helper (Task 6: every 5 windows + final)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_analysis_checkpoint_uploads_and_writes_v1(
    mock_jobs_store, tmp_path,
):
    """The async flush helper uploads analysis_raw.json and writes a V1
    upscale_analyze checkpoint with cursor + artifacts populated."""
    from service import worker

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    s3 = _stub_s3()
    analysis_results = [
        {"window": 1, "frames": [0, 15], "analysis": {"clips": []}},
        {"window": 2, "frames": [16, 30], "analysis": {"clips": []}},
    ]

    raw_key = await worker._flush_analysis_checkpoint(
        job_id="job-up",
        jobs_store=mock_jobs_store,
        s3=s3,
        output_bucket="out",
        output_dir=str(output_dir),
        tracking_s3_key="checkpoints/job-up/tracking.json",
        analysis_results=analysis_results,
        current_context="ctx-after-window-2",
        next_frame_idx=31,
        progress_percent=58.5,
        total_tracking_frames=120,
        stage_progress_fraction=0.14,
    )

    # Local file written
    assert (output_dir / "analysis_raw.json").is_file()
    # S3 upload happened to the supplied bucket and the returned key.
    assert s3.upload_json.called
    upload_args, _ = s3.upload_json.call_args
    assert upload_args[1] == "out"
    assert upload_args[2] == raw_key
    assert raw_key.endswith("analysis_raw.json")

    cp = mock_jobs_store._checkpoints[("job-up", PipelineStage.UPSCALE_ANALYZE.value)]
    data = cp["checkpoint_data"]
    _assert_envelope(data)
    assert data["reason"] == "analysis_window_completed"
    assert data["resume_cursor"]["frame_idx"] == 31
    assert data["resume_cursor"]["analysis_window_count"] == 2
    assert data["analysis_current_context"] == "ctx-after-window-2"
    assert data["artifacts"]["tracking_s3_key"].endswith("tracking.json")
    assert data["artifacts"]["analysis_raw_s3_key"] == raw_key
    assert data["worker_state"]["progress_percent"] == 58.5
    assert data["worker_state"]["current_frame"] == 31
    assert data["worker_state"]["total_frames"] == 120
    assert data["worker_state"]["stage_progress_fraction"] == 0.14


# ---------------------------------------------------------------------------
# skip_upscale path — track post-upload re-write + upload terminal write
# ---------------------------------------------------------------------------


def _stub_run_tracking_job(*args, **kwargs):
    """Replacement for service.tracking_runner.run_tracking_job that writes
    a tiny tracking.json and returns its path."""
    import json
    output_dir = kwargs.get("tracking_output_dir") or args[3]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tracking.json")
    with open(path, "w") as f:
        json.dump({
            "start_frame": 0,
            "frames": [
                {"frame_idx": 0, "athletes": []},
                {"frame_idx": 1, "athletes": []},
            ],
        }, f)
    return path


@pytest.mark.asyncio
async def test_skip_upscale_writes_post_upload_track_artifact_and_upload_terminal(
    mock_jobs_store, tmp_path,
):
    """Skip-upscale path: after the tracking JSON lands in S3, the worker
    re-writes the track row with artifacts.tracking_s3_key, writes an upload
    row with completed=True, and finishes the job."""
    from service import worker

    config = ServiceConfig(
        temp_dir=str(tmp_path), s3_endpoint_url="http://x", gemini_api_key="",
    )
    job_store = InMemoryJobStore()
    request = _track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        skip_upscale=True,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = _stub_s3()

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch(
             "service.tracking_runner.run_tracking_job",
             side_effect=_stub_run_tracking_job,
         ):
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    # Track row carries the post-upload tracking_s3_key.
    track = mock_jobs_store._checkpoints[(job.job_id, PipelineStage.TRACK.value)]
    track_data = track["checkpoint_data"]
    _assert_envelope(track_data)
    assert track_data["reason"] == "track_completed"
    assert track_data["artifacts"]["tracking_s3_key"].endswith("_tracked.json")
    # Tracking JSON must be in the output bucket.
    assert any(
        call.args[1] == "out" and call.args[2].endswith("_tracked.json")
        for call in s3.upload_json.call_args_list
    )

    # Upload row exists, completed=True (skip_upscale terminates at upload).
    upload = mock_jobs_store._checkpoints[(job.job_id, PipelineStage.UPLOAD.value)]
    upload_data = upload["checkpoint_data"]
    _assert_envelope(upload_data)
    assert upload["completed"] is True
    assert upload_data["reason"] == "tracking_uploaded"
    assert upload_data["artifacts"]["tracking_s3_key"].endswith("_tracked.json")
    assert upload_data["worker_state"]["progress_percent"] == 100.0

    # Lifecycle should be COMPLETED.
    lc = await mock_jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.COMPLETED.value


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
