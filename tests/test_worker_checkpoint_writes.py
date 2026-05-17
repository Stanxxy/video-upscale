"""Worker happy-path checkpoint write integration tests."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore

from tests.worker_checkpoint_helpers import (
    assert_envelope,
    leaf_run_tracking_job_with_overlap,
    stub_run_tracking_job,
    stub_s3,
    track_request,
)

# ---------------------------------------------------------------------------
# skip_upscale path — track post-upload re-write + upload terminal write
# ---------------------------------------------------------------------------


def stub_run_tracking_job(*args, **kwargs):
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

    # Track row carries the post-upload tracking_s3_key.
    track = mock_jobs_store._checkpoints[(job.job_id, PipelineStage.TRACK.value)]
    track_data = track["checkpoint_data"]
    assert_envelope(track_data)
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
    assert_envelope(upload_data)
    assert upload["completed"] is True
    assert upload_data["reason"] == "tracking_uploaded"
    assert upload_data["artifacts"]["tracking_s3_key"].endswith("_tracked.json")
    assert upload_data["worker_state"]["progress_percent"] == 100.0

    # Lifecycle should be COMPLETED.
    lc = await mock_jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.COMPLETED.value


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


@pytest.mark.asyncio
async def test_run_job_merges_ancestor_tracking_chain(
    mock_jobs_store, tmp_path,
):
    """Replacement leaf run merges ancestor tracking JSON before S3 upload."""
    from service import worker

    parent_id = "parent-job-id"
    parent_tracking_key = "ancestors/p_tracked.json"
    parent_doc = {
        "start_frame": 0,
        "frames": [{"frame_idx": 0, "athlete": "A"}],
    }

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
    leaf_job = await job_store.create_job(request)

    await mock_jobs_store.create_lifecycle(parent_id, "vid", "u")
    await mock_jobs_store.write_checkpoint(
        parent_id,
        PipelineStage.TRACK,
        False,
        {
            "schema_version": 1,
            "pending_detection": None,
            "reason": "track_completed",
            "artifacts": {"tracking_s3_key": parent_tracking_key},
            "worker_state": {
                "progress_percent": 55.0,
                "current_frame": 1,
                "total_frames": 100,
                "stage_progress_fraction": 0.5,
            },
        },
    )
    await mock_jobs_store.create_lifecycle(
        leaf_job.job_id, "vid", "u", parent_job_id=parent_id,
    )

    s3 = stub_s3()

    def _download_json(bucket, key):
        if bucket == "out" and key == parent_tracking_key:
            return dict(parent_doc)
        raise AssertionError(f"unexpected download_json: {bucket}/{key}")

    s3.download_json = MagicMock(side_effect=_download_json)

    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, None)), \
         patch(
             "service.tracking_runner.run_tracking_job",
             side_effect=leaf_run_tracking_job_with_overlap,
         ):
        await worker.run_job(
            leaf_job.job_id, request, config, job_store, mock_jobs_store,
        )

    tracked_uploads = [
        call.args[0]
        for call in s3.upload_json.call_args_list
        if len(call.args) >= 3 and str(call.args[2]).endswith("_tracked.json")
    ]
    assert len(tracked_uploads) == 1
    uploaded = tracked_uploads[0]
    by_idx = {f["frame_idx"]: f for f in uploaded["frames"]}
    assert set(by_idx) == {0, 1}
    assert by_idx[0]["athlete"] == "leaf_wins"
    assert by_idx[1]["athlete"] == "B"

    lc = await mock_jobs_store.get_lifecycle(leaf_job.job_id)
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
             side_effect=stub_run_tracking_job,
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
    assert_envelope(annotate_data)
    assert annotate_data["reason"] == "annotate_completed"
    assert annotate_data["artifacts"]["annotated_video_s3_key"].endswith("_annotated.mp4")

    # TRACK row was re-written after the pre-upscale tracking JSON upload.
    track = mock_jobs_store._checkpoints[(job.job_id, PipelineStage.TRACK.value)]
    track_data = track["checkpoint_data"]
    assert_envelope(track_data)
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
    assert_envelope(upload_data)
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
    assert_envelope(publish_data)
    assert publish["completed"] is True
    assert publish_data["reason"] == "publish_completed"
    assert publish_data["artifacts"]["sns_event_count"] == 3
    assert publish_data["artifacts"]["sns_completion_sent"] is True
    assert publish_data["artifacts"]["sns_topic_arn"] == "arn:aws:sns:test"
    assert publish_data["worker_state"]["progress_percent"] == 100.0

    lc = await mock_jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.COMPLETED.value


