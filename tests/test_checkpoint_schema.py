"""Unit tests for service.checkpoints builders — V1 envelope conformance.

Every builder MUST emit the V1 envelope shape:
{
    "schema_version": 1,
    "pending_detection": null | dict,
    "artifacts": dict,
    "worker_state": {
        "progress_percent": float,
        "current_frame": int,
        "total_frames": int,
        "stage_progress_fraction": float,
    },
    ... stage-specific scalar fields ...
}
"""
from service.analysis_keyspaces_enums import PipelineStage
from datetime import datetime, timezone

from service.checkpoints import (
    WorkerStateSnapshot,
    make_envelope,
    build_download_completed,
    build_detect_initial_pending,
    build_track_progress,
    build_track_mid_loss,
    build_track_completed,
    build_upscale_started,
    build_upscale_window_progress,
    build_annotate_completed,
    build_upload_incremental,
    build_publish_completed,
    build_replaced_by_new_job,
    build_verified_boxes_checkpoint,
    build_cancellation_checkpoint,
    build_resume_overrides,
    build_resume_plan,
    latest_checkpoint_data_by_stage,
    worker_state_from,
    END_OF_TRACKING_SENTINEL,
)


V1_KEYS = {"schema_version", "pending_detection", "artifacts", "worker_state"}


def _ws(progress_percent=10.0, current_frame=0, total_frames=0, stage_progress_fraction=1.0):
    return WorkerStateSnapshot(
        progress_percent=progress_percent,
        current_frame=current_frame,
        total_frames=total_frames,
        stage_progress_fraction=stage_progress_fraction,
    )


def _assert_envelope(cp: dict) -> None:
    assert V1_KEYS <= cp.keys(), f"missing envelope keys: {V1_KEYS - cp.keys()}"
    assert cp["schema_version"] == 1
    assert isinstance(cp["artifacts"], dict)
    ws = cp["worker_state"]
    assert {
        "progress_percent",
        "current_frame",
        "total_frames",
        "stage_progress_fraction",
    } <= ws.keys()
    assert isinstance(ws["progress_percent"], (int, float))
    assert isinstance(ws["current_frame"], int)
    assert isinstance(ws["total_frames"], int)
    assert isinstance(ws["stage_progress_fraction"], (int, float))
    assert cp["pending_detection"] is None or isinstance(cp["pending_detection"], dict)


# ---------------------------------------------------------------------------
# Envelope helper
# ---------------------------------------------------------------------------


def test_make_envelope_requires_worker_state():
    cp = make_envelope(worker_state=_ws())
    _assert_envelope(cp)
    assert cp["pending_detection"] is None
    assert cp["artifacts"] == {}


def test_make_envelope_passes_through_extras():
    cp = make_envelope(worker_state=_ws(), reason="custom", resume_cursor={"frame_idx": 7})
    _assert_envelope(cp)
    assert cp["reason"] == "custom"
    assert cp["resume_cursor"] == {"frame_idx": 7}


# ---------------------------------------------------------------------------
# Per-stage builders
# ---------------------------------------------------------------------------


def test_download_completed_envelope_shape():
    cp = build_download_completed(worker_state=_ws(progress_percent=10.0))
    _assert_envelope(cp)
    assert cp["reason"] == "download_completed"
    assert cp["worker_state"]["progress_percent"] == 10.0


def test_detect_initial_pending_includes_required_fields():
    cp = build_detect_initial_pending(
        frame_idx=0,
        frame_s3_key="checkpoints/job-1/frame_0.jpg",
        frame_bucket="bjj-video-analysis",
        candidates=[{"candidate_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9}],
        suggested_boxes=None,
        worker_state=_ws(progress_percent=10.0, stage_progress_fraction=0.0),
    )
    _assert_envelope(cp)
    pd = cp["pending_detection"]
    assert pd["reason"] == "initial"
    assert pd["frame_idx"] == 0
    assert pd["frame_s3_key"] == "checkpoints/job-1/frame_0.jpg"
    assert pd["frame_bucket"] == "bjj-video-analysis"
    assert pd["candidates"][0]["candidate_id"] == 0
    assert pd["suggested_boxes"] is None


def test_track_progress_envelope_carries_partial_artifacts():
    cp = build_track_progress(
        partial_tracking_s3_key="checkpoints/job-1/partial_tracking.json",
        resume_from_frame=1200,
        worker_state=_ws(
            progress_percent=35.0,
            current_frame=1200,
            total_frames=3600,
            stage_progress_fraction=0.5,
        ),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "tracking_progress"
    assert cp["artifacts"]["partial_tracking_s3_key"].endswith("partial_tracking.json")
    assert cp["artifacts"]["resume_from_frame"] == 1200
    assert cp["resume_cursor"] == {"frame_idx": 1200}
    assert cp["worker_state"]["current_frame"] == 1200


def test_track_progress_omits_partial_key_when_none():
    cp = build_track_progress(
        partial_tracking_s3_key=None,
        resume_from_frame=100,
        worker_state=_ws(progress_percent=20.0, current_frame=100, total_frames=1000),
    )
    _assert_envelope(cp)
    assert "partial_tracking_s3_key" not in cp["artifacts"]
    assert cp["artifacts"]["resume_from_frame"] == 100


def test_track_mid_loss_places_partial_tracking_in_artifacts():
    cp = build_track_mid_loss(
        frame_idx=512,
        frame_s3_key="checkpoints/job-1/frame_512.jpg",
        frame_bucket="bjj-video-analysis",
        candidates=[],
        suggested_boxes=[[10, 20, 100, 200], [300, 20, 400, 200]],
        partial_tracking_s3_key="checkpoints/job-1/partial_tracking.json",
        resume_from_frame=512,
        worker_state=_ws(
            progress_percent=20.0,
            current_frame=512,
            total_frames=1024,
            stage_progress_fraction=0.5,
        ),
    )
    _assert_envelope(cp)
    assert cp["pending_detection"]["reason"] == "tracking_lost"
    assert cp["pending_detection"]["frame_idx"] == 512
    assert cp["artifacts"]["partial_tracking_s3_key"].endswith("partial_tracking.json")
    assert cp["artifacts"]["resume_from_frame"] == 512
    assert cp["resume_cursor"] == {"frame_idx": 512}


def test_track_mid_loss_handles_missing_partial_key():
    cp = build_track_mid_loss(
        frame_idx=10,
        frame_s3_key="checkpoints/job-1/frame_10.jpg",
        frame_bucket="b",
        candidates=[],
        suggested_boxes=None,
        partial_tracking_s3_key=None,
        resume_from_frame=10,
        worker_state=_ws(progress_percent=15.5),
    )
    _assert_envelope(cp)
    assert "partial_tracking_s3_key" not in cp["artifacts"]
    assert cp["artifacts"]["resume_from_frame"] == 10


def test_track_completed_carries_tracking_s3_key_when_known():
    cp = build_track_completed(
        start_frame=0,
        frame_count=900,
        tracking_s3_key="job-1_tracked.json",
        worker_state=_ws(
            progress_percent=55.0,
            current_frame=900,
            total_frames=900,
            stage_progress_fraction=1.0,
        ),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "track_completed"
    assert cp["start_frame"] == 0
    assert cp["frame_count"] == 900
    assert cp["artifacts"]["tracking_s3_key"] == "job-1_tracked.json"


def test_track_completed_omits_artifact_when_pre_upload():
    cp = build_track_completed(
        start_frame=0,
        frame_count=900,
        worker_state=_ws(progress_percent=55.0, current_frame=900, total_frames=900),
    )
    _assert_envelope(cp)
    assert cp["artifacts"] == {}


def test_upscale_started_envelope():
    cp = build_upscale_started(
        tracking_s3_key="checkpoints/job-1/tracking.json",
        worker_state=_ws(
            progress_percent=55.0,
            current_frame=0,
            total_frames=21600,
            stage_progress_fraction=0.0,
        ),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "analysis_started"
    assert cp["artifacts"]["tracking_s3_key"].endswith("tracking.json")
    assert cp["resume_cursor"] == {"frame_idx": 0, "analysis_window_count": 0}
    assert cp["analysis_current_context"] == ""


def test_upscale_window_progress_carries_cursor_and_artifacts():
    cp = build_upscale_window_progress(
        frame_idx=9120,
        analysis_window_count=12,
        analysis_current_context="white belt entered guard",
        tracking_s3_key="checkpoints/job-1/tracking.json",
        analysis_raw_s3_key="checkpoints/job-1/analysis_raw.json",
        worker_state=_ws(
            progress_percent=67.5,
            current_frame=9120,
            total_frames=21600,
            stage_progress_fraction=0.5,
        ),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "analysis_window_completed"
    assert cp["resume_cursor"] == {"frame_idx": 9120, "analysis_window_count": 12}
    assert cp["analysis_current_context"] == "white belt entered guard"
    assert cp["artifacts"]["tracking_s3_key"].endswith("tracking.json")
    assert cp["artifacts"]["analysis_raw_s3_key"].endswith("analysis_raw.json")


def test_annotate_completed_with_video_key():
    cp = build_annotate_completed(
        annotated_video_s3_key="job-1_annotated.mp4",
        worker_state=_ws(progress_percent=85.0),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "annotate_completed"
    assert cp["artifacts"]["annotated_video_s3_key"] == "job-1_annotated.mp4"


def test_annotate_completed_omits_video_key_when_failed():
    cp = build_annotate_completed(
        annotated_video_s3_key=None,
        worker_state=_ws(progress_percent=85.0),
    )
    _assert_envelope(cp)
    assert cp["artifacts"] == {}


def test_upload_incremental_is_additive():
    base_ws = _ws(progress_percent=86.6, stage_progress_fraction=0.33)
    cp1 = build_upload_incremental(tracking_s3_key="a_tracked.json", worker_state=base_ws)
    _assert_envelope(cp1)
    assert cp1["artifacts"] == {"tracking_s3_key": "a_tracked.json"}
    assert cp1["reason"] == "tracking_uploaded"

    cp2 = build_upload_incremental(
        tracking_s3_key="a_tracked.json",
        analysis_s3_key="a_analysis.json",
        worker_state=_ws(progress_percent=88.3, stage_progress_fraction=0.66),
    )
    assert cp2["artifacts"] == {
        "tracking_s3_key": "a_tracked.json",
        "analysis_s3_key": "a_analysis.json",
    }
    assert cp2["reason"] == "analysis_uploaded"

    cp3 = build_upload_incremental(
        tracking_s3_key="a_tracked.json",
        analysis_s3_key="a_analysis.json",
        annotated_video_s3_key="a_annotated.mp4",
        worker_state=_ws(progress_percent=90.0, stage_progress_fraction=1.0),
    )
    assert cp3["artifacts"]["annotated_video_s3_key"] == "a_annotated.mp4"
    assert cp3["reason"] == "annotated_video_uploaded"


def test_upload_incremental_with_no_artifacts_yet():
    cp = build_upload_incremental(worker_state=_ws(progress_percent=85.0))
    _assert_envelope(cp)
    assert cp["artifacts"] == {}
    assert cp["reason"] == "upload_started"


def test_publish_completed_records_sns_metadata():
    cp = build_publish_completed(
        sns_topic_arn="arn:aws:sns:us-east-1:000:topic",
        sns_event_count=12,
        sns_completion_sent=True,
        worker_state=_ws(progress_percent=100.0, stage_progress_fraction=1.0),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "publish_completed"
    assert cp["artifacts"]["sns_topic_arn"] == "arn:aws:sns:us-east-1:000:topic"
    assert cp["artifacts"]["sns_event_count"] == 12
    assert cp["artifacts"]["sns_completion_sent"] is True


def test_replaced_by_new_job_records_replacement_id():
    cp = build_replaced_by_new_job(
        replacement_job_id="new-job-uuid",
        worker_state=_ws(
            progress_percent=35.0,
            current_frame=7432,
            total_frames=21600,
            stage_progress_fraction=0.34,
        ),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "replaced_by_new_job"
    assert cp["artifacts"]["replacement_job_id"] == "new-job-uuid"
    assert cp["worker_state"]["current_frame"] == 7432


# ---------------------------------------------------------------------------
# Migrated existing builders
# ---------------------------------------------------------------------------


def test_verified_boxes_envelope_keeps_root_boxes():
    cp = build_verified_boxes_checkpoint(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        PipelineStage.TRACK,
        worker_state=_ws(progress_percent=15.0, stage_progress_fraction=1.0),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "detection_correction_resume"
    assert cp["source_stage"] == "track"
    assert cp["verified_box_a"] == [1, 2, 3, 4]
    assert cp["verified_box_b"] == [5, 6, 7, 8]


def test_cancellation_envelope_keeps_resume_cursor():
    cp = build_cancellation_checkpoint(
        reason="user_cancelled",
        frame_idx=42,
        progress_percent=33.3,
        worker_state=_ws(
            progress_percent=33.3,
            current_frame=42,
            total_frames=100,
            stage_progress_fraction=0.42,
        ),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "user_cancelled"
    assert cp["resume_cursor"] == {"frame_idx": 42}
    assert cp["progress_percent"] == 33.3


# ---------------------------------------------------------------------------
# Resume-override + worker_state_from helpers
# ---------------------------------------------------------------------------


def _cp_record(stage: str, data: dict, *, completed: bool = False) -> dict:
    return {"stage_name": stage, "checkpoint_data": data, "completed": completed}


def test_build_resume_overrides_initial_detection_returns_empty():
    overrides = build_resume_overrides([
        _cp_record("detect", {
            "schema_version": 1,
            "pending_detection": {"reason": "initial", "frame_idx": 0},
            "artifacts": {},
            "worker_state": _ws().to_dict(),
        }),
    ])
    assert "resume_tracking_s3_key" not in overrides
    assert "analysis_raw_s3_key" not in overrides


def test_build_resume_overrides_mid_track_loss():
    overrides = build_resume_overrides([
        _cp_record("track", {
            "schema_version": 1,
            "pending_detection": {"reason": "tracking_lost", "frame_idx": 7432},
            "artifacts": {
                "partial_tracking_s3_key": "checkpoints/orig/partial_tracking.json",
                "resume_from_frame": 7432,
            },
            "worker_state": {
                "progress_percent": 35.0,
                "current_frame": 7432,
                "total_frames": 21600,
                "stage_progress_fraction": 0.34,
            },
        }),
    ])
    assert overrides["resume_tracking_s3_key"] == "checkpoints/orig/partial_tracking.json"
    assert overrides["resume_from_frame"] == 7432


def test_build_resume_overrides_mid_track_progress_no_pending():
    """No pending_detection but a periodic track_progress write — recovery still gets the partials."""
    overrides = build_resume_overrides([
        _cp_record("track", {
            "schema_version": 1,
            "pending_detection": None,
            "reason": "tracking_progress",
            "artifacts": {
                "partial_tracking_s3_key": "checkpoints/orig/partial_tracking.json",
                "resume_from_frame": 1200,
            },
            "resume_cursor": {"frame_idx": 1200},
            "worker_state": {
                "progress_percent": 28.3,
                "current_frame": 1200,
                "total_frames": 3600,
                "stage_progress_fraction": 0.33,
            },
        }),
    ])
    assert overrides["resume_tracking_s3_key"] == "checkpoints/orig/partial_tracking.json"
    assert overrides["resume_from_frame"] == 1200


def test_build_resume_overrides_upscale_crash():
    overrides = build_resume_overrides([
        _cp_record("upscale_analyze", {
            "schema_version": 1,
            "pending_detection": None,
            "reason": "analysis_window_completed",
            "resume_cursor": {"frame_idx": 9120, "analysis_window_count": 12},
            "analysis_current_context": "north-south",
            "artifacts": {
                "tracking_s3_key": "checkpoints/orig/tracking.json",
                "analysis_raw_s3_key": "checkpoints/orig/analysis_raw.json",
            },
            "worker_state": {
                "progress_percent": 67.5,
                "current_frame": 9120,
                "total_frames": 21600,
                "stage_progress_fraction": 0.5,
            },
        }),
    ])
    assert overrides["resume_tracking_s3_key"] == "checkpoints/orig/tracking.json"
    assert overrides["resume_from_frame"] == END_OF_TRACKING_SENTINEL
    assert overrides["analysis_raw_s3_key"] == "checkpoints/orig/analysis_raw.json"
    assert overrides["analysis_window_count"] == 12
    assert overrides["analysis_current_context"] == "north-south"


def test_build_resume_overrides_upscale_overrides_track_for_resume_keys():
    """An upscale crash dominates a track checkpoint — both tracking_s3_key and sentinel apply."""
    overrides = build_resume_overrides([
        _cp_record("track", {
            "schema_version": 1,
            "pending_detection": None,
            "artifacts": {
                "partial_tracking_s3_key": "checkpoints/orig/partial_tracking.json",
                "resume_from_frame": 21600,
            },
            "worker_state": _ws().to_dict(),
        }),
        _cp_record("upscale_analyze", {
            "schema_version": 1,
            "pending_detection": None,
            "resume_cursor": {"frame_idx": 9120, "analysis_window_count": 12},
            "analysis_current_context": "ctx",
            "artifacts": {
                "tracking_s3_key": "checkpoints/orig/tracking.json",
                "analysis_raw_s3_key": "checkpoints/orig/analysis_raw.json",
            },
            "worker_state": _ws().to_dict(),
        }),
    ])
    assert overrides["resume_tracking_s3_key"] == "checkpoints/orig/tracking.json"
    assert overrides["resume_from_frame"] == END_OF_TRACKING_SENTINEL
    assert overrides["analysis_raw_s3_key"] == "checkpoints/orig/analysis_raw.json"


def test_worker_state_from_picks_latest_stage():
    cps = [
        _cp_record("download", {
            "worker_state": {
                "progress_percent": 10.0,
                "current_frame": 0,
                "total_frames": 0,
                "stage_progress_fraction": 1.0,
            },
        }),
        _cp_record("track", {
            "worker_state": {
                "progress_percent": 55.0,
                "current_frame": 21600,
                "total_frames": 21600,
                "stage_progress_fraction": 1.0,
            },
        }),
    ]
    ws = worker_state_from(cps)
    assert ws is not None
    assert ws["progress_percent"] == 55.0
    assert ws["current_frame"] == 21600


def test_worker_state_from_returns_none_when_absent():
    cps = [
        _cp_record("detect", {"pending_detection": {}, "artifacts": {}}),
    ]
    ws = worker_state_from(cps)
    assert ws is None


def test_should_flush_analysis_only_at_multiples_of_n():
    from service.checkpoints import should_flush_analysis
    assert should_flush_analysis(0) is False
    assert should_flush_analysis(1) is False
    assert should_flush_analysis(4) is False
    assert should_flush_analysis(5) is True
    assert should_flush_analysis(9) is False
    assert should_flush_analysis(10) is True
    # Custom every_n
    assert should_flush_analysis(3, every_n=3) is True
    assert should_flush_analysis(2, every_n=3) is False


def test_worker_state_from_skips_stages_without_worker_state():
    """A track row without worker_state is ignored; the download row's worker_state is returned."""
    cps = [
        _cp_record("download", {
            "worker_state": {
                "progress_percent": 10.0,
                "current_frame": 0,
                "total_frames": 0,
                "stage_progress_fraction": 1.0,
            },
        }),
        _cp_record("track", {"artifacts": {}}),
    ]
    ws = worker_state_from(cps)
    assert ws is not None
    assert ws["progress_percent"] == 10.0


def test_build_resume_plan_full_tracking_without_analysis_skips_runner():
    """Crash after tracking JSON uploaded but before raw analysis checkpoint."""
    plan = build_resume_plan([
        _cp_record("track", {
            "schema_version": 1,
            "pending_detection": None,
            "reason": "track_completed",
            "artifacts": {"tracking_s3_key": "folder/v_tracked.json"},
            "worker_state": _ws(55.0).to_dict(),
        }),
    ])
    assert plan.skip_tracking_runner is True
    assert plan.track_request_overrides["resume_tracking_s3_key"] == "folder/v_tracked.json"
    assert plan.track_request_overrides["resume_from_frame"] == END_OF_TRACKING_SENTINEL


def test_build_resume_plan_prefers_full_tracking_when_partial_also_present():
    """Post-upload full key wins over periodic partial when both appear in history."""
    plan = build_resume_plan([
        _cp_record("track", {
            "schema_version": 1,
            "pending_detection": None,
            "artifacts": {
                "tracking_s3_key": "folder/v_tracked.json",
                "partial_tracking_s3_key": "checkpoints/job/partial_tracking.json",
            },
            "worker_state": _ws(35.0).to_dict(),
        }),
    ])
    assert plan.skip_tracking_runner is True
    assert plan.track_request_overrides["resume_tracking_s3_key"] == (
        "folder/v_tracked.json"
    )
    assert plan.track_request_overrides["resume_from_frame"] == END_OF_TRACKING_SENTINEL


def test_build_resume_plan_pipeline_already_complete_publish():
    plan = build_resume_plan([
        _cp_record(
            "publish",
            {"reason": "publish_completed", "worker_state": _ws(100.0).to_dict()},
            completed=True,
        ),
    ])
    assert plan.pipeline_already_complete is True


def test_latest_checkpoint_data_by_stage_prefers_newer_updated_at():
    t_old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t_new = datetime(2026, 6, 1, tzinfo=timezone.utc)
    cps = [
        {
            "stage_name": "track",
            "checkpoint_data": {"reason": "old", "worker_state": {"progress_percent": 10.0}},
            "updated_at": t_old,
        },
        {
            "stage_name": "track",
            "checkpoint_data": {"reason": "new", "worker_state": {"progress_percent": 55.0}},
            "updated_at": t_new,
        },
    ]
    by_stage = latest_checkpoint_data_by_stage(cps)
    assert by_stage["track"]["reason"] == "new"
