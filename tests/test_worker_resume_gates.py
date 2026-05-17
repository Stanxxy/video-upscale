"""Worker resume-gate and recovery regression tests."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.models import JobStatus

from tests.worker_checkpoint_helpers import (
    assert_envelope,
    stub_run_tracking_job,
    stub_s3,
    track_request,
)

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
    request = track_request(box_a=None, box_b=None)
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = stub_s3()

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
    assert_envelope(download["checkpoint_data"])
    assert download["checkpoint_data"]["reason"] == "download_completed"
    assert download["checkpoint_data"]["worker_state"]["progress_percent"] == 10.0

    # DETECT checkpoint must exist with V1 envelope and pending_detection.
    detect = mock_jobs_store._checkpoints.get(
        (job.job_id, PipelineStage.DETECT.value)
    )
    assert detect is not None, "detect checkpoint must be written"
    assert_envelope(detect["checkpoint_data"])
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
    request = track_request(
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

    s3 = stub_s3()
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


@pytest.mark.asyncio
async def test_run_job_skips_analysis_and_annotated_reupload_when_keys_match(
    mock_jobs_store, tmp_path,
):
    """Recovery hints skip duplicate S3 uploads but still write upload checkpoints."""
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
        "start_frame": 0,
        "fps": 30.0,
        "frames": [{"frame_idx": 0, "athletes": []}],
    }
    request = track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        resume_from_frame=END_OF_TRACKING_SENTINEL,
        resume_tracking_s3_key="folder/v_tracked.json",
        resume_existing_upload_analysis_key="folder/v_analysis.json",
        resume_existing_upload_annotated_key="folder/v_annotated.mp4",
        skip_upscale=False,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = stub_s3()
    s3.download_json = MagicMock(return_value=tracking_blob)

    tr_dir = tmp_path / job.job_id / "tracking"
    tr_dir.mkdir(parents=True, exist_ok=True)
    (tr_dir / "tracked_output.mp4").write_bytes(b"mp4")
    annotated_path = tmp_path / "annotated_output.mp4"
    annotated_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch("service.tracking_runner.run_tracking_job", stub_run_tracking_job), \
         patch.object(
             worker, "_run_upscale_analysis",
             return_value=({"clips": [], "fps": 30.0}, 30.0),
         ), \
         patch(
             "service.video_annotator.annotate_video",
             return_value=str(annotated_path),
         ), \
         patch("service.worker.SNSPublisher") as sns_cls:
        sns_cls.return_value.publish_events = MagicMock(return_value=2)
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    analysis_uploads = [
        c for c in s3.upload_json.call_args_list
        if len(c.args) >= 3 and str(c.args[2]).endswith("_analysis.json")
    ]
    assert analysis_uploads == []
    annotated_uploads = [
        c for c in s3.upload_file.call_args_list
        if len(c.args) >= 3 and str(c.args[2]).endswith("_annotated.mp4")
    ]
    assert annotated_uploads == []

    upload = mock_jobs_store._checkpoints[(job.job_id, PipelineStage.UPLOAD.value)]
    arts = upload["checkpoint_data"]["artifacts"]
    assert arts["analysis_s3_key"] == "folder/v_analysis.json"
    assert arts["annotated_video_s3_key"] == "folder/v_annotated.mp4"


@pytest.mark.asyncio
async def test_run_job_progress_never_regresses_below_lifecycle_floor(
    mock_jobs_store, tmp_path,
):
    """Replacement jobs seed progress_floor from lifecycle — early stages must not dip."""
    from service import worker
    from service.checkpoints import END_OF_TRACKING_SENTINEL

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        gemini_api_key="fake",
        sns_topic_arn="arn:aws:sns:test",
    )
    job_store = InMemoryJobStore()
    floor = 67.5
    tracking_blob = {
        "start_frame": 0,
        "fps": 30.0,
        "frames": [{"frame_idx": 0, "athletes": []}],
    }
    request = track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        resume_from_frame=END_OF_TRACKING_SENTINEL,
        resume_tracking_s3_key="folder/v_tracked.json",
        skip_upscale=True,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(
        job.job_id, "vid", "u", progress_percent=floor,
    )

    progress_writes: list[float] = []

    async def record_update_progress(job_id, stage, pct, **kwargs):
        progress_writes.append(float(pct))

    orig_update_job = job_store.update_job

    async def record_update_job(job_id, **kwargs):
        if "progress_percent" in kwargs:
            progress_writes.append(float(kwargs["progress_percent"]))
        return await orig_update_job(job_id, **kwargs)

    job_store.update_job = record_update_job  # type: ignore[method-assign]
    mock_jobs_store.update_progress = record_update_progress  # type: ignore[method-assign]

    s3 = stub_s3()
    s3.download_json = MagicMock(return_value=tracking_blob)

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch("service.tracking_runner.run_tracking_job", stub_run_tracking_job):
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    assert progress_writes, "expected progress updates during run"
    assert all(p >= floor for p in progress_writes), (
        f"progress regressed below floor {floor}: {progress_writes}"
    )


@pytest.mark.asyncio
async def test_run_job_partial_resume_merges_frames_and_preserves_start_frame(
    mock_jobs_store, tmp_path,
):
    """Non-sentinel partial resume merges JSON and keeps partial start_frame."""
    from service import worker

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        gemini_api_key="",
    )
    job_store = InMemoryJobStore()
    partial = {
        "start_frame": 100,
        "frames": [
            {"frame_idx": 100, "athletes": []},
            {"frame_idx": 101, "athletes": []},
        ],
    }
    resume_frame = 102
    request = track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        resume_tracking_s3_key="checkpoints/job/partial_tracking.json",
        resume_from_frame=resume_frame,
        skip_upscale=True,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = stub_s3()
    s3.download_json = MagicMock(return_value=partial)
    captured: dict = {}

    def partial_resume_stub(*args, **kwargs):
        captured["start_frame"] = kwargs["start_frame"]
        output_dir = kwargs.get("output_dir") or args[3]
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "tracking.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "start_frame": resume_frame,
                    "frames": [
                        {"frame_idx": 102, "athletes": []},
                        {"frame_idx": 103, "athletes": []},
                    ],
                },
                f,
            )
        return path

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch(
             "service.tracking_runner.run_tracking_job",
             side_effect=partial_resume_stub,
         ):
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    assert captured["start_frame"] == resume_frame
    uploaded = [
        call.args[0]
        for call in s3.upload_json.call_args_list
        if len(call.args) >= 3 and str(call.args[2]).endswith("_tracked.json")
    ]
    assert len(uploaded) == 1
    merged = uploaded[0]
    assert merged["start_frame"] == 100
    assert len(merged["frames"]) == 4
    assert [f["frame_idx"] for f in merged["frames"]] == [100, 101, 102, 103]


@pytest.mark.asyncio
async def test_run_job_failed_writes_track_progress_partial_checkpoint(
    mock_jobs_store, tmp_path,
):
    """Failed jobs upload partial tracking in finally and write track_progress row."""
    from service import worker

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        gemini_api_key="fake",
        sns_topic_arn="arn:aws:sns:test",
    )
    job_store = InMemoryJobStore()
    request = track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        skip_upscale=False,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = stub_s3()

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch("service.tracking_runner.run_tracking_job", stub_run_tracking_job), \
         patch.object(
             worker, "_run_upscale_analysis",
             side_effect=RuntimeError("upscale boom"),
         ):
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    job = await job_store.get_job(job.job_id)
    assert job.status == JobStatus.FAILED

    partial_uploads = [
        c for c in s3.upload_json.call_args_list
        if len(c.args) >= 3 and "partial_tracking.json" in str(c.args[2])
    ]
    assert len(partial_uploads) == 1

    track_history = mock_jobs_store._checkpoint_history[
        (job.job_id, PipelineStage.TRACK.value)
    ]
    progress_rows = [
        r for r in track_history
        if r["checkpoint_data"].get("reason") == "tracking_progress"
        and r["checkpoint_data"]["artifacts"].get("partial_tracking_s3_key")
    ]
    assert len(progress_rows) >= 1


@pytest.mark.asyncio
async def test_awaiting_correction_does_not_get_failure_partial_checkpoint(
    mock_jobs_store, tmp_path,
):
    """Suspended jobs must not receive an extra failure-path partial track row."""
    from service import worker

    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        gemini_api_key="",
    )
    job_store = InMemoryJobStore()
    request = track_request(box_a=None, box_b=None)
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = stub_s3()
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

    lc = await mock_jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.AWAITING_CORRECTION.value

    partial_uploads = [
        c for c in s3.upload_json.call_args_list
        if len(c.args) >= 3 and "partial_tracking.json" in str(c.args[2])
    ]
    assert partial_uploads == []

    track_history = mock_jobs_store._checkpoint_history.get(
        (job.job_id, PipelineStage.TRACK.value), [],
    )
    failure_partials = [
        r for r in track_history
        if r["checkpoint_data"].get("reason") == "tracking_progress"
        and "partial_tracking_s3_key" in (r["checkpoint_data"].get("artifacts") or {})
    ]
    assert failure_partials == []


@pytest.mark.asyncio
async def test_run_job_skips_sns_when_resume_terminal_publish_done(
    mock_jobs_store, tmp_path,
):
    """Terminal publish recovery completes without duplicate SNS publish."""
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
        "start_frame": 0,
        "fps": 30.0,
        "frames": [{"frame_idx": 0, "athletes": []}],
    }
    request = track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        resume_from_frame=END_OF_TRACKING_SENTINEL,
        resume_tracking_s3_key="folder/v_tracked.json",
        resume_terminal_publish_done=True,
        skip_upscale=False,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = stub_s3()
    s3.download_json = MagicMock(return_value=tracking_blob)

    tr_dir = tmp_path / job.job_id / "tracking"
    tr_dir.mkdir(parents=True, exist_ok=True)
    (tr_dir / "tracked_output.mp4").write_bytes(b"mp4")

    mock_publish = MagicMock(return_value=3)

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch("service.tracking_runner.run_tracking_job", stub_run_tracking_job), \
         patch.object(
             worker, "_run_upscale_analysis",
             return_value=({"clips": [], "fps": 30.0}, 30.0),
         ), \
         patch(
             "service.video_annotator.annotate_video",
             return_value=str(tmp_path / "annotated_output.mp4"),
         ), \
         patch("service.worker.SNSPublisher") as sns_cls:
        sns_cls.return_value.publish_events = mock_publish
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    mock_publish.assert_not_called()
    job = await job_store.get_job(job.job_id)
    assert job.status == JobStatus.COMPLETED
    lc = await mock_jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.COMPLETED.value
