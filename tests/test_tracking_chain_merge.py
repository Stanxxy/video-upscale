"""Tests for ``service.tracking_chain_merge``."""

from datetime import datetime, timezone

import pytest

from service.analysis_keyspaces_enums import PipelineStage
from service.tracking_chain_merge import (
    merge_tracking_frames_last_writer,
    resolve_best_tracking_keys_from_checkpoints,
)


def test_merge_last_writer_order():
    a = [{"frame_idx": 0, "v": "root"}, {"frame_idx": 1, "v": "root"}]
    b = [{"frame_idx": 1, "v": "leaf"}, {"frame_idx": 2, "v": "leaf"}]
    out = merge_tracking_frames_last_writer([a, b])
    assert [x["frame_idx"] for x in out] == [0, 1, 2]
    assert out[1]["v"] == "leaf"
    assert out[2]["v"] == "leaf"


def test_resolve_prefers_newest_full_tracked_over_replaced_row():
    """Terminal replaced_by_new_job row must not hide older track_completed keys."""
    old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new_ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    checkpoints = [
        {
            "stage_name": PipelineStage.TRACK.value,
            "updated_at": new_ts,
            "checkpoint_data": {
                "reason": "replaced_by_new_job",
                "artifacts": {"replacement_job_id": "child"},
            },
        },
        {
            "stage_name": PipelineStage.TRACK.value,
            "updated_at": old_ts,
            "checkpoint_data": {
                "reason": "track_completed",
                "artifacts": {
                    "tracking_s3_key": "users/x/video_tracked.json",
                    "partial_tracking_s3_key": "checkpoints/old/partial_tracking.json",
                },
            },
        },
    ]
    full_k, partial_k = resolve_best_tracking_keys_from_checkpoints(checkpoints)
    assert full_k == "users/x/video_tracked.json"
    assert partial_k == "checkpoints/old/partial_tracking.json"


def test_resolve_partial_when_no_full():
    checkpoints = [
        {
            "stage_name": PipelineStage.TRACK.value,
            "updated_at": datetime(2026, 1, 3, tzinfo=timezone.utc),
            "checkpoint_data": {
                "artifacts": {"partial_tracking_s3_key": "checkpoints/j/partial_tracking.json"},
            },
        },
    ]
    full_k, partial_k = resolve_best_tracking_keys_from_checkpoints(checkpoints)
    assert full_k is None
    assert partial_k == "checkpoints/j/partial_tracking.json"


def test_resolve_partial_root_level_fallback_matches_select_correction():
    """Same root-level key as legacy checkpoints / route tests (no artifacts wrap)."""
    checkpoints = [
        {
            "stage_name": PipelineStage.TRACK.value,
            "updated_at": datetime(2026, 1, 3, tzinfo=timezone.utc),
            "checkpoint_data": {
                "partial_tracking_s3_key": "checkpoints/original/partial_tracking.json",
            },
        },
    ]
    full_k, partial_k = resolve_best_tracking_keys_from_checkpoints(checkpoints)
    assert full_k is None
    assert partial_k == "checkpoints/original/partial_tracking.json"


@pytest.mark.asyncio
async def test_walk_chain_stops_on_cycle(monkeypatch):
    from service.tracking_chain_merge import walk_job_chain_leaf_to_root

    class FakeStore:
        async def get_lifecycle(self, job_id: str):
            if job_id == "a":
                return {"parent_job_id": "b"}
            if job_id == "b":
                return {"parent_job_id": "a"}
            return None

    chain = await walk_job_chain_leaf_to_root(FakeStore(), "a")  # type: ignore[arg-type]
    assert chain == ["a", "b"]


def test_preflight_swaps_missing_primary_for_existing_full_key():
    from service.checkpoints import END_OF_TRACKING_SENTINEL
    from service.tracking_chain_merge import preflight_resume_tracking_overrides

    class MS3:
        def object_exists(self, bucket: str, key: str) -> bool:
            return key == "folder/full.json"

    cps = [
        {
            "stage_name": "track",
            "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "checkpoint_data": {
                "artifacts": {
                    "tracking_s3_key": "folder/full.json",
                    "partial_tracking_s3_key": "gone.json",
                },
            },
        },
    ]
    out = preflight_resume_tracking_overrides(
        {"resume_tracking_s3_key": "missing.json", "resume_from_frame": 100},
        cps,
        MS3(),
        "bucket",
    )
    assert out["resume_tracking_s3_key"] == "folder/full.json"
    assert out["resume_from_frame"] == END_OF_TRACKING_SENTINEL


def test_preflight_leaves_overrides_when_head_inconclusive():
    from service.tracking_chain_merge import preflight_resume_tracking_overrides

    class BadS3:
        def object_exists(self, bucket: str, key: str) -> bool:
            raise OSError("no network")

    orig = {"resume_tracking_s3_key": "k.json", "resume_from_frame": 5}
    out = preflight_resume_tracking_overrides(orig, [], BadS3(), "b")
    assert out == orig
