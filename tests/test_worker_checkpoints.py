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
from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend


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
        clip_start_frame=0,
        clip_total_frames=3600,
        progress_floor=10.0,
    )

    with pytest.raises(HumanVerificationSuspend):
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
        clip_start_frame=0,
        clip_total_frames=3600,
        progress_floor=10.0,
    )
    with pytest.raises(HumanVerificationSuspend):
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
    _assert_envelope(data)
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
    request = _track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = _stub_s3()

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
    request = _track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = _stub_s3()

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
    request = _track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = _stub_s3()

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

    loop = asyncio.get_event_loop()

    from service import worker

    with _patch.dict(sys.modules, sys_modules_patches), \
         _patch.object(worker, "_make_s3", return_value=s3):
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
    _assert_envelope(data)
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
# Full-path integration: annotate + upload incremental + publish (Task 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_path_writes_annotate_upload_publish_envelopes(
    mock_jobs_store, tmp_path,
):
    """Full path with analysis writes annotate, incremental upload (tracking
    -> analysis -> annotated_video), and a terminal publish checkpoint."""
    from service import worker

    config = ServiceConfig(
        temp_dir=str(tmp_path), s3_endpoint_url="http://x",
        gemini_api_key="fake", sns_topic_arn="arn:aws:sns:test",
    )
    job_store = InMemoryJobStore()
    request = _track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        skip_upscale=False,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = _stub_s3()
    work_root = tmp_path / job.job_id / "tracking"
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "tracked_output.mp4").write_bytes(b"fake-mp4")

    sns_publisher = MagicMock()
    sns_publisher.publish_events = MagicMock(return_value=3)

    def _annotate_stub(tracked, analysis, out, fps, sf):
        with open(out, "wb") as f:
            f.write(b"annotated")
        return out

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch.object(
             worker, "_run_upscale_analysis",
             return_value=({"clips": [], "fps": 30.0}, 30.0),
         ), \
         patch(
             "service.tracking_runner.run_tracking_job",
             side_effect=_stub_run_tracking_job,
         ), \
         patch(
             "service.video_annotator.annotate_video",
             side_effect=_annotate_stub,
         ), \
         patch("service.worker.SNSPublisher", return_value=sns_publisher):
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    # ANNOTATE row exists with annotated_video_s3_key.
    annotate = mock_jobs_store._checkpoints.get(
        (job.job_id, PipelineStage.ANNOTATE.value)
    )
    assert annotate is not None
    annotate_data = annotate["checkpoint_data"]
    _assert_envelope(annotate_data)
    assert annotate_data["reason"] == "annotate_completed"
    assert annotate_data["artifacts"]["annotated_video_s3_key"].endswith("_annotated.mp4")

    # TRACK row was re-written after the pre-upscale tracking JSON upload.
    track = mock_jobs_store._checkpoints[(job.job_id, PipelineStage.TRACK.value)]
    track_data = track["checkpoint_data"]
    _assert_envelope(track_data)
    assert track_data["reason"] == "track_completed"
    assert track_data["artifacts"]["tracking_s3_key"].endswith("_tracked.json")
    # The track row history must include both the pre-upload write (no
    # tracking_s3_key) and the post-upload re-write (with tracking_s3_key).
    track_history = mock_jobs_store._checkpoint_history[
        (job.job_id, PipelineStage.TRACK.value)
    ]
    track_artifacts_seq = [
        rec["checkpoint_data"]["artifacts"].get("tracking_s3_key")
        for rec in track_history
        if rec["checkpoint_data"].get("reason") == "track_completed"
    ]
    assert track_artifacts_seq[0] is None  # pre-upload write
    assert track_artifacts_seq[-1].endswith("_tracked.json")  # post-upload re-write

    # UPLOAD row carries all three artifacts after the annotated video lands.
    upload = mock_jobs_store._checkpoints[(job.job_id, PipelineStage.UPLOAD.value)]
    upload_data = upload["checkpoint_data"]
    _assert_envelope(upload_data)
    arts = upload_data["artifacts"]
    assert arts["tracking_s3_key"].endswith("_tracked.json")
    assert arts["analysis_s3_key"].endswith("_analysis.json")
    assert arts["annotated_video_s3_key"].endswith("_annotated.mp4")
    assert upload_data["reason"] == "annotated_video_uploaded"

    # The upload row was written incrementally: tracking_uploaded first
    # (before upscale), then analysis_uploaded, then annotated_video_uploaded.
    upload_history = mock_jobs_store._checkpoint_history[
        (job.job_id, PipelineStage.UPLOAD.value)
    ]
    reasons = [rec["checkpoint_data"]["reason"] for rec in upload_history]
    assert reasons == ["tracking_uploaded", "analysis_uploaded", "annotated_video_uploaded"]
    # Each step's artifacts is a strict superset of the previous.
    arts_seq = [rec["checkpoint_data"]["artifacts"] for rec in upload_history]
    assert "tracking_s3_key" in arts_seq[0]
    assert {"tracking_s3_key", "analysis_s3_key"} <= arts_seq[1].keys()
    assert {
        "tracking_s3_key", "analysis_s3_key", "annotated_video_s3_key",
    } <= arts_seq[2].keys()

    # PUBLISH terminal row, completed=True with SNS metadata.
    publish = mock_jobs_store._checkpoints[(job.job_id, PipelineStage.PUBLISH.value)]
    publish_data = publish["checkpoint_data"]
    _assert_envelope(publish_data)
    assert publish["completed"] is True
    assert publish_data["reason"] == "publish_completed"
    assert publish_data["artifacts"]["sns_event_count"] == 3
    assert publish_data["artifacts"]["sns_completion_sent"] is True
    assert publish_data["artifacts"]["sns_topic_arn"] == "arn:aws:sns:test"
    assert publish_data["worker_state"]["progress_percent"] == 100.0

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


@pytest.mark.asyncio
async def test_run_job_skips_run_tracking_when_sentinel_resume(mock_jobs_store, tmp_path):
    """Recovery loads durable tracking JSON from S3 — never invokes SAM2."""
    from service import worker
    from service.checkpoints import END_OF_TRACKING_SENTINEL

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        gemini_api_key="fake",
        sns_topic_arn="arn:aws:sns:test",
    )
    job_store = InMemoryJobStore()
    tracking_blob = {
        "video": "/tmp/v.mp4",
        "fps": 30.0,
        "frames": [{"frame_idx": 0, "athletes": []}],
    }
    request = _track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        resume_from_frame=END_OF_TRACKING_SENTINEL,
        resume_tracking_s3_key="folder/v_tracked.json",
        skip_upscale=False,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = _stub_s3()
    s3.download_json = MagicMock(return_value=tracking_blob)

    tr_dir = tmp_path / job.job_id / "tracking"
    tr_dir.mkdir(parents=True, exist_ok=True)
    (tr_dir / "tracked_output.mp4").write_bytes(b"mp4")

    mock_run_track = MagicMock()

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch("service.tracking_runner.run_tracking_job", mock_run_track), \
         patch.object(
             worker, "_run_upscale_analysis",
             return_value=({"clips": [], "fps": 30.0}, 30.0),
         ), \
         patch(
             "service.video_annotator.annotate_video",
             return_value=str(tmp_path / "annotated_output.mp4"),
         ), \
         patch("service.worker.SNSPublisher") as sns_cls:
        sns_cls.return_value.publish_events = MagicMock(return_value=2)
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    mock_run_track.assert_not_called()
    assert s3.download_json.called
