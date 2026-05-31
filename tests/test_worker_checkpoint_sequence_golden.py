"""Golden checkpoint sequence characterization for worker stage ordering."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore

from tests.worker_checkpoint_helpers import (
    assert_envelope,
    stub_run_tracking_job,
    stub_s3,
    track_request,
)


def _stage_reasons(history: dict, job_id: str) -> list[tuple[str, str | None, bool]]:
    out: list[tuple[str, str | None, bool]] = []
    for (jid, stage), records in history.items():
        if jid != job_id:
            continue
        for rec in records:
            data = rec["checkpoint_data"]
            out.append((stage, data.get("reason"), rec["completed"]))
    return out


@pytest.mark.asyncio
async def test_skip_upscale_golden_checkpoint_sequence(mock_jobs_store, tmp_path):
    from service import worker

    config = ServiceConfig(
        temp_dir=str(tmp_path), s3_endpoint_url="http://x", gemini_api_key="",
    )
    job_store = InMemoryJobStore()
    request = track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        skip_upscale=True,
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")
    s3 = stub_s3()

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch(
             "service.tracking_runner.run_tracking_job",
             side_effect=stub_run_tracking_job,
         ):
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    seq = _stage_reasons(mock_jobs_store._checkpoint_history, job.job_id)
    stages = [s for s, _, _ in seq]
    assert PipelineStage.DOWNLOAD.value in stages
    assert PipelineStage.TRACK.value in stages
    assert PipelineStage.UPLOAD.value in stages

    upload_rows = [r for s, r, c in seq if s == PipelineStage.UPLOAD.value]
    assert upload_rows
    assert upload_rows[-1][1] == "tracking_uploaded"
    assert upload_rows[-1][2] is True

    track_rows = [r for s, r, c in seq if s == PipelineStage.TRACK.value]
    assert any(r == "track_completed" for r in track_rows)

    lc = await mock_jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.COMPLETED.value


@pytest.mark.asyncio
async def test_full_path_golden_includes_annotate_upload_publish(
    mock_jobs_store, tmp_path,
):
    from service import worker

    config = ServiceConfig(
        temp_dir=str(tmp_path), s3_endpoint_url="http://x",
        gemini_api_key="fake", sns_topic_arn="arn:aws:sns:test",
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
    work_root = tmp_path / job.job_id / "tracking"
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "tracked_output.mp4").write_bytes(b"fake-mp4")

    sns_publisher = MagicMock()
    sns_publisher.publish_events = MagicMock(return_value=3)

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch.object(
             worker, "_run_upscale_analysis",
             return_value=({"clips": [], "fps": 30.0}, 30.0),
         ), \
         patch(
             "service.tracking_runner.run_tracking_job",
             side_effect=stub_run_tracking_job,
         ), \
         patch(
             "service.video_annotator.annotate_video",
             side_effect=lambda *a, **k: (open(k.get("output_path", a[2]), "wb").write(b"x") or a[2]),
         ), \
         patch("service.worker.SNSPublisher", return_value=sns_publisher):
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    seq = _stage_reasons(mock_jobs_store._checkpoint_history, job.job_id)
    stages = {s for s, _, _ in seq}
    assert PipelineStage.ANNOTATE.value in stages
    assert PipelineStage.UPLOAD.value in stages
    assert PipelineStage.PUBLISH.value in stages

    publish_rows = [
        (r, c) for s, r, c in seq if s == PipelineStage.PUBLISH.value
    ]
    assert publish_rows
    assert publish_rows[-1][0] == "publish_completed"
    assert publish_rows[-1][1] is True

    upload_history = mock_jobs_store._checkpoint_history[
        (job.job_id, PipelineStage.UPLOAD.value)
    ]
    upload_reasons = [rec["checkpoint_data"]["reason"] for rec in upload_history]
    assert "tracking_uploaded" in upload_reasons
    assert "analysis_uploaded" in upload_reasons
    assert "annotated_video_uploaded" in upload_reasons
    for rec in upload_history:
        assert_envelope(rec["checkpoint_data"])


@pytest.mark.asyncio
async def test_suspend_path_golden_track_mid_loss_not_completed(
    mock_jobs_store, tmp_path,
):
    import asyncio

    from service.worker import _make_detection_cb

    loop = asyncio.get_event_loop()
    config = ServiceConfig(temp_dir=str(tmp_path), s3_endpoint_url="http://x")
    request = track_request(box_a=[1, 2, 3, 4], box_b=[5, 6, 7, 8])
    s3 = stub_s3()
    work_dir = tmp_path / "wd"
    (work_dir / "tracking").mkdir(parents=True)
    (work_dir / "tracking" / "tracking.json").write_text(
        '{"start_frame":0,"frames":[{"frame_idx":7,"athletes":[]}]}'
    )

    from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend

    cb = _make_detection_cb(
        "job-suspend", loop, mock_jobs_store, s3, config, request, str(work_dir),
        clip_start_frame=0,
        clip_total_frames=3600,
        progress_floor=10.0,
    )

    from tests.worker_checkpoint_helpers import invoke_detection_cb

    with pytest.raises(HumanVerificationSuspend):
        await invoke_detection_cb(
            cb,
            reason="tracking_lost",
            frame_jpeg=b"\xff\xd8jpeg",
            yolo_detections=[{"box": [10, 20, 100, 200], "confidence": 0.9}],
            frame_idx=512,
        )

    cp = mock_jobs_store._checkpoints[("job-suspend", PipelineStage.TRACK.value)]
    assert cp["completed"] is False
    data = cp["checkpoint_data"]
    assert data["reason"] == "tracking_lost"
    assert data["pending_detection"]["reason"] == "tracking_lost"
