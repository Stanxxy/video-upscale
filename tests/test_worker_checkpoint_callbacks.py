"""Worker checkpoint callback helper tests."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from service.analysis_keyspaces_enums import PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend

from tests.worker_checkpoint_helpers import (
    assert_envelope,
    invoke_detection_cb,
    stub_s3,
    track_request,
)

# ---------------------------------------------------------------------------
# Mid-track detection callback bug fix
# ---------------------------------------------------------------------------


async def invoke_detection_cb(cb, **kwargs):
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
    request = track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = stub_s3()
    work_dir = tmp_path / "wd"
    work_dir.mkdir()

    cb = _make_detection_cb(
        "job-x", loop, mock_jobs_store, s3, config, request, str(work_dir),
        clip_start_frame=0,
        clip_total_frames=3600,
        progress_floor=10.0,
    )

    with pytest.raises(HumanVerificationSuspend):
        await invoke_detection_cb(
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
    request = track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = stub_s3()

    work_dir = tmp_path / "wd"
    (work_dir / "tracking").mkdir(parents=True)
    # A tiny but parseable tracking.json so _load_partial_tracking_dict
    # can succeed.
    (work_dir / "tracking" / "tracking.json").write_text(
        '{"start_frame":0,"frames":[{"frame_idx":7,"athletes":[]}]}'
    )

    cb = _make_detection_cb(
        "job-mid", loop, mock_jobs_store, s3, config, request, str(work_dir),
        clip_start_frame=0,
        clip_total_frames=3600,
        progress_floor=10.0,
    )
    with pytest.raises(HumanVerificationSuspend):
        await invoke_detection_cb(
            cb,
            reason="tracking_lost",
            frame_jpeg=b"\xff\xd8jpeg",
            yolo_detections=[{"box": [10, 20, 100, 200], "confidence": 0.9}],
            frame_idx=512,
        )

    cp = mock_jobs_store._checkpoints[("job-mid", PipelineStage.TRACK.value)]
    data = cp["checkpoint_data"]
    assert_envelope(data)
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
    request = track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8], output_bucket="out")
    s3 = stub_s3()

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
        resume_next_global=1201,
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
    assert_envelope(data)
    assert data["reason"] == "tracking_progress"
    assert data["artifacts"]["partial_tracking_s3_key"].endswith("partial_tracking.json")
    assert data["artifacts"]["resume_from_frame"] == 1201
    assert data["worker_state"]["current_frame"] == 1200
    assert data["worker_state"]["total_frames"] == 3600
    assert data["worker_state"]["progress_percent"] == 35.0
    assert data["resume_cursor"] == {"frame_idx": 1201}


@pytest.mark.asyncio
async def test_track_progress_helper_no_op_when_neither_flag_set(
    mock_jobs_store, tmp_path,
):
    """write_lifecycle=False and upload_partial=False → no S3 upload, no checkpoint."""
    from service import worker

    config = ServiceConfig(temp_dir=str(tmp_path), s3_endpoint_url="http://x")
    job_store = InMemoryJobStore()
    request = track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = stub_s3()

    work_dir = tmp_path / "wd"
    work_dir.mkdir()

    await worker._update_tracking_progress_with_partial(
        "job-track",
        500, 1000, 35.0,
        job_store, mock_jobs_store,
        request, str(work_dir), s3,
        resume_next_global=501,
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
    request = track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = stub_s3()

    work_dir = tmp_path / "wd"
    work_dir.mkdir()  # no tracking subdir / tracking.json

    await worker._update_tracking_progress_with_partial(
        "job-track",
        500, 1000, 35.0,
        job_store, mock_jobs_store,
        request, str(work_dir), s3,
        resume_next_global=501,
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
    s3 = stub_s3()
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
    assert_envelope(data)
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
# Tracking cadence flag computation (Task 4: 1s lifecycle + 30s partial)
# ---------------------------------------------------------------------------


def test_tracking_progress_flags_below_thresholds_returns_false_pair():
    from service.worker import _tracking_progress_flags
    write_lc, upload_partial = _tracking_progress_flags(
        now=0.5, last_ks_write=0.0, last_partial_upload=0.0,
    )
    assert write_lc is False
    assert upload_partial is False


def test_tracking_progress_flags_crosses_1s_only():
    from service.worker import _tracking_progress_flags
    write_lc, upload_partial = _tracking_progress_flags(
        now=1.5, last_ks_write=0.0, last_partial_upload=0.0,
    )
    assert write_lc is True
    assert upload_partial is False


def test_tracking_progress_flags_crosses_30s_threshold():
    from service.worker import _tracking_progress_flags
    write_lc, upload_partial = _tracking_progress_flags(
        now=30.0, last_ks_write=29.5, last_partial_upload=0.0,
    )
    assert write_lc is False  # only 0.5s since last lifecycle write
    assert upload_partial is True


def test_tracking_progress_flags_crosses_both_thresholds():
    from service.worker import _tracking_progress_flags
    write_lc, upload_partial = _tracking_progress_flags(
        now=100.0, last_ks_write=0.0, last_partial_upload=32.0,
    )
    assert write_lc is True
    assert upload_partial is True


def test_tracking_progress_flags_respects_custom_intervals():
    from service.worker import _tracking_progress_flags
    write_lc, upload_partial = _tracking_progress_flags(
        now=2.0, last_ks_write=0.0, last_partial_upload=0.0,
        ks_interval=5.0, partial_interval=60.0,
    )
    assert write_lc is False
    assert upload_partial is False


# ---------------------------------------------------------------------------
# _run_upscale_analysis orchestration (Task 6 wiring)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_upscale_analysis_writes_started_and_final_flush(
    mock_jobs_store, tmp_path,
):
    """End-to-end exercise of _run_upscale_analysis with stubbed cv2 /
    restorer / analyzer / PIL / utils. With 30 sampled frames and an
    analyzer present, the function must emit:
      - exactly one 'analysis_started' upscale_analyze checkpoint at the top,
      - at least one 'analysis_window_completed' (final flush) at the end.
    """
    import json as _json
    import sys
    from unittest.mock import patch as _patch

    # Build a tracking JSON with 30 frames each carrying a single athlete box.
    tracking = {
        "start_frame": 0,
        "frames": [
            {"frame_idx": i, "athletes": [{"box": [0, 0, 100, 100]}]}
            for i in range(30)
        ],
    }
    tracking_json_path = tmp_path / "tracking.json"
    tracking_json_path.write_text(_json.dumps(tracking))

    work_dir = tmp_path / "wd"
    work_dir.mkdir()

    config = ServiceConfig(
        temp_dir=str(tmp_path), s3_endpoint_url="http://x",
        gemini_api_key="fake", model_path="ignored",
    )
    request = track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = stub_s3()

    # cv2 stub — VideoCapture returns (True, fake-frame) reads.
    cv2_mock = MagicMock()
    cv2_mock.IMWRITE_JPEG_QUALITY = 1
    cv2_mock.CAP_PROP_FPS = 0
    cv2_mock.CAP_PROP_POS_FRAMES = 0
    cv2_mock.COLOR_BGR2RGB = 0
    cv2_mock.INTER_LANCZOS4 = 0

    class FakeFrame:
        shape = (720, 1280, 3)

        def __getitem__(self, _slice):
            return FakeFrame()

        @property
        def size(self):
            return 720 * 1280 * 3

    cap_mock = MagicMock()
    cap_mock.read.return_value = (True, FakeFrame())
    cap_mock.get.return_value = 30.0
    cv2_mock.VideoCapture.return_value = cap_mock
    cv2_mock.imwrite.return_value = True
    cv2_mock.imencode.return_value = (True, b"fake-jpeg-bytes")
    cv2_mock.cvtColor.side_effect = lambda frame, _code: frame
    cv2_mock.resize.side_effect = lambda frame, *_a, **_kw: frame

    # PIL stub — Image.fromarray returns a sentinel placeholder.
    pil_image_mock = MagicMock()
    pil_image_mock.fromarray.return_value = MagicMock(name="PILImage")
    pil_mock = MagicMock(Image=pil_image_mock)

    # utils stub — geometry helpers return canned boxes.
    utils_mock = MagicMock(
        get_union_box=lambda boxes: boxes[0],
        get_padded_square_box=lambda *_a, **_kw: (0, 0, 100, 100),
    )

    # restorer stub — RealESRGANRestorer.enhance returns the input frame.
    restorer_class = MagicMock()
    restorer_instance = MagicMock()
    restorer_instance.enhance.side_effect = lambda crop, **_kw: crop
    restorer_class.return_value = restorer_instance
    restorer_mod_mock = MagicMock(RealESRGANRestorer=restorer_class)

    # analyzer stub — analyze_sequence returns one valid clips JSON.
    analyzer_instance = MagicMock()
    analyzer_instance.analyze_sequence.return_value = _json.dumps({
        "clips": [],
        "current_context_summary": "ctx-after-window",
    })
    analyzer_class = MagicMock(return_value=analyzer_instance)
    analyzer_mod_mock = MagicMock(BJJTechniqueAnalyzer=analyzer_class)

    sys_modules_patches = {
        "cv2": cv2_mock,
        "PIL": pil_mock,
        "PIL.Image": pil_image_mock,
        "utils": utils_mock,
        "restorer": restorer_mod_mock,
        "analyzer": analyzer_mod_mock,
    }

    # ``upscale_batch``/``upscale_jpeg``/``upscale_loop`` bind ``cv2`` (and
    # ``upscale_batch`` binds ``PIL.Image``) at module scope, so earlier
    # tests can leave them pointing at the REAL libraries regardless of the
    # ``sys.modules`` patch above. Patch the module attributes directly to
    # make this test order-independent.
    from service.worker.stages import upscale_batch, upscale_jpeg, upscale_loop

    loop = asyncio.get_event_loop()

    from service import worker

    with _patch.dict(sys.modules, sys_modules_patches), \
         _patch.object(upscale_batch, "cv2", cv2_mock), \
         _patch.object(upscale_batch, "Image", pil_image_mock), \
         _patch.object(upscale_jpeg, "cv2", cv2_mock), \
         _patch.object(upscale_loop, "cv2", cv2_mock), \
         _patch("service.worker.stages.upscale_setup._make_s3", return_value=s3):
        await loop.run_in_executor(
            None,
            lambda: worker._run_upscale_analysis(
                video_path=str(tmp_path / "video.mp4"),
                tracking_json_path=str(tracking_json_path),
                config=config,
                request=request,
                work_dir=str(work_dir),
                job_id="job-up",
                jobs_store=mock_jobs_store,
                loop=loop,
                tracking_s3_key="checkpoints/job-up/tracking.json",
                progress_cb=None,
            ),
        )

    cp = mock_jobs_store._checkpoints.get(
        ("job-up", PipelineStage.UPSCALE_ANALYZE.value)
    )
    assert cp is not None, "expected at least one upscale_analyze checkpoint"
    data = cp["checkpoint_data"]
    assert_envelope(data)
    # Latest write should be the final flush after the one analysis window.
    assert data["reason"] == "analysis_window_completed"
    assert data["resume_cursor"]["analysis_window_count"] >= 1
    assert data["analysis_current_context"] == "ctx-after-window"
    assert data["artifacts"]["tracking_s3_key"].endswith("tracking.json")
    assert data["artifacts"]["analysis_raw_s3_key"].endswith("analysis_raw.json")
    # 30 frames → buffer hits WINDOW_SIZE once mid-loop, then a leftover
    # buffer of STRIDE=15 frames is analyzed by the final-drain step. So
    # exactly 2 analyses run.
    assert analyzer_instance.analyze_sequence.call_count == 2
    assert data["resume_cursor"]["analysis_window_count"] == 2
    # analysis_raw.json was uploaded to the output bucket at least once.
    assert s3.upload_json.called
