"""Regression test: tracking is ALWAYS sequential.

Design context (post 2026-05-25 refactor):
Even when the clip is large enough that ``compute_segment_ranges`` would
return multiple memory-bounded segments, AND when ``standard_segments > 1``
(both conditions that, before the refactor, would have routed work
through the old ``_run_parallel_segments`` parallel-tracking helper),
``run_job`` MUST drive tracking through the single sequential code path
that wires ``_make_detection_cb``. The sequential path is the only one
that honors the V1 ``track`` checkpoint contract on mid-track loss:

  * ``service/worker.py:_make_detection_cb`` writes ``build_track_mid_loss``
    with ``artifacts.partial_tracking_s3_key`` + ``artifacts.resume_from_frame``
    in the V1 envelope.
  * It transitions lifecycle to ``JobState.AWAITING_CORRECTION``.
  * It raises ``HumanVerificationSuspend`` so SAM2 stops processing
    further frames and the worker releases the semaphore — the
    replacement job from ``POST /jobs/{job_id}/resume`` can start
    immediately.

See:
  * ``working_log/contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md``
    §``stage_name == track`` / "Mid-track detection loss"
  * ``working_log/contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md``
    §4.1
  * ``working_log/knowledge-base/insights/2026-04-25-job-start-resume-workflow-reference.md``
    items 7, 20, 22.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore

from tests.worker_checkpoint_helpers import stub_s3, track_request


def _stub_run_tracking_with_midtrack_loss(*args, **kwargs):
    """Replacement for ``service.tracking_runner.run_tracking_job`` that
    fires the supplied ``detection_cb`` with a synthetic mid-track loss
    on a real frame index, then propagates whatever the callback raises
    (HumanVerificationSuspend in the production path).

    The callback runs on a worker thread (the real ``run_tracking_job``
    uses an executor too — see ``service.worker.run_job`` ``loop.run_in_executor``
    wrapper). Inside the callback, ``asyncio.run_coroutine_threadsafe`` is
    used to push the suspend coroutine back onto the worker loop.
    """
    detection_cb = kwargs.get("detection_cb")
    assert detection_cb is not None, (
        "Tracking MUST receive a non-None detection_cb (regression: "
        "parallel-tracking path was deleted; only sequential remains)"
    )

    output_dir = kwargs.get("output_dir") or args[3]
    os.makedirs(output_dir, exist_ok=True)
    # Write a partial tracking.json so the mid-track suspend can find one
    # to upload as ``artifacts.partial_tracking_s3_key``.
    import json
    partial_path = os.path.join(output_dir, "tracking.json")
    with open(partial_path, "w") as f:
        json.dump(
            {
                "start_frame": 0,
                "frames": [
                    {"frame_idx": 0, "athletes": [{"box": [0, 0, 10, 10], "id": 1}]},
                    {"frame_idx": 1, "athletes": [{"box": [0, 0, 10, 10], "id": 1}]},
                ],
            },
            f,
        )

    # Now invoke the detection_cb to simulate SAM2 losing the athletes at
    # frame 7432 (matching the addendum's mid-track example). The
    # production cb writes the V1 ``build_track_mid_loss`` checkpoint,
    # sets AWAITING_CORRECTION, and raises ``HumanVerificationSuspend``.
    #
    # In production, ``tracking_pipeline.hybrid_tracking`` catches the
    # HumanVerificationSuspend and ``service.tracking_runner.run_tracking_job``
    # returns None to the worker, which then raises JobSuspendedError. We
    # mirror that contract here so the worker's run_job sees None and
    # exits via its ``except JobSuspendedError`` branch — leaving the
    # lifecycle row at AWAITING_CORRECTION (already set by the cb).
    from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend
    try:
        detection_cb(
            "tracking_lost",
            b"fake-jpeg-bytes",
            yolo_detections=[
                {"box": [10.0, 20.0, 100.0, 200.0], "confidence": 0.78},
                {"box": [300.0, 20.0, 400.0, 200.0], "confidence": 0.81},
            ],
            frame_idx=7432,
        )
    except HumanVerificationSuspend:
        # Production behavior: pipeline swallows suspend → runner → None
        return None
    # Should never reach here — the cb always raises.
    raise AssertionError(
        "detection_cb did not raise HumanVerificationSuspend — "
        "the sequential mid-track loss contract is broken"
    )


@pytest.mark.asyncio
async def test_tracking_uses_sequential_path_with_detection_cb_even_when_parallel_conditions_met(
    mock_jobs_store, tmp_path,
):
    """Drive ``run_job`` with conditions that would have triggered the OLD
    parallel-tracking path (multiple memory-bounded segments + standard_segments>1)
    and assert the sequential ``_make_detection_cb`` path is used end-to-end
    on a simulated mid-track loss.

    The assertions cover the three reviewer concerns from the user's
    contract-driven review:
      1. Tracking receives a non-None ``detection_cb``.
      2. A V1 ``build_track_mid_loss`` envelope is persisted on the
         ``track`` row with ``artifacts.partial_tracking_s3_key`` +
         ``artifacts.resume_from_frame`` + a complete ``worker_state``.
      3. Lifecycle reaches ``JobState.AWAITING_CORRECTION`` before any
         further frames are processed (the test fake raises an assertion
         if processing continues past the suspend).
    """
    from service import worker

    # Conditions that would have triggered the old parallel-tracking path:
    #   * standard_segments > 1
    #   * compute_segment_ranges returns multiple segments (we force this
    #     by patching it).
    config = ServiceConfig(
        temp_dir=str(tmp_path),
        s3_endpoint_url="http://x",
        gemini_api_key="",
        standard_segments=3,
        segment_max_frames=100,  # small bound forces multiple segments
    )
    job_store = InMemoryJobStore()
    request = track_request(
        bucket="b",
        key="folder/v.mp4",
        box_a=[1, 2, 3, 4],
        box_b=[5, 6, 7, 8],
        output_bucket="out",
    )
    job = await job_store.create_job(request)
    await mock_jobs_store.create_lifecycle(job.job_id, "vid", "u")

    s3 = stub_s3()

    # Force multiple segments to make sure we hit the
    # "would-have-been-parallel" branch.
    fake_segment_ranges = [(0, 100), (100, 200), (200, 300)]

    # Patch compute_segment_ranges at both import sites:
    #   * service.segment_runner (where _run_parallel_upscale imports it)
    #   * service.worker imports it as _csr at line ~547
    with patch.object(worker, "_make_s3", return_value=s3), \
         patch.object(worker, "_parse_time_range", return_value=(0, 300)), \
         patch.object(worker, "_video_frame_cap", return_value=300), \
         patch(
             "service.segment_runner.compute_segment_ranges",
             return_value=fake_segment_ranges,
         ), \
         patch(
             "service.tracking_runner.run_tracking_job",
             side_effect=_stub_run_tracking_with_midtrack_loss,
         ):
        # The worker's run_job catches JobSuspendedError internally and
        # converts it to a finally-block exit (see run_job's except
        # JobSuspendedError branch). We don't expect an exception to
        # propagate out — but the lifecycle state and checkpoint row
        # changes are observable on the mock store.
        await worker.run_job(
            job.job_id, request, config, job_store, mock_jobs_store,
        )

    # ------------------------------------------------------------------
    # Assertion 1: lifecycle reached AWAITING_CORRECTION (set by
    # _make_detection_cb._async_suspend before HumanVerificationSuspend
    # is raised).
    # ------------------------------------------------------------------
    lifecycle = mock_jobs_store._lifecycles[job.job_id]
    assert lifecycle["job_state"] == JobState.AWAITING_CORRECTION.value, (
        f"Expected AWAITING_CORRECTION; got {lifecycle['job_state']}"
    )

    # ------------------------------------------------------------------
    # Assertion 2: a V1 build_track_mid_loss row was persisted with the
    # required artifacts (partial_tracking_s3_key + resume_from_frame)
    # and a complete worker_state.
    # ------------------------------------------------------------------
    track_history = mock_jobs_store._checkpoint_history.get(
        (job.job_id, PipelineStage.TRACK.value), [],
    )
    assert track_history, (
        "No track checkpoint rows were written — sequential mid-track "
        "loss contract not honored"
    )

    # Find the mid-loss row by its pending_detection.reason
    mid_loss_rows = [
        row for row in track_history
        if (row.get("checkpoint_data") or {}).get("pending_detection")
        and row["checkpoint_data"]["pending_detection"].get("reason")
            == "tracking_lost"
    ]
    assert mid_loss_rows, (
        "No track row with pending_detection.reason=tracking_lost was "
        f"written. Track history: {track_history!r}"
    )

    cp = mid_loss_rows[-1]["checkpoint_data"]
    # V1 envelope shape:
    assert cp.get("schema_version") == 1
    pd = cp["pending_detection"]
    assert pd["frame_idx"] == 7432
    assert pd["frame_s3_key"].endswith("/frame_7432.jpg")
    assert pd["frame_bucket"] == request.bucket
    assert isinstance(pd["candidates"], list) and len(pd["candidates"]) == 2

    artifacts = cp.get("artifacts") or {}
    assert "resume_from_frame" in artifacts, (
        "V1 contract: track.artifacts.resume_from_frame is REQUIRED on "
        "mid-track loss (see CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md §track)"
    )
    assert artifacts["resume_from_frame"] == 7432
    # partial_tracking_s3_key is best-effort but the stub writes a tracking.json
    # to disk before raising, so the upload SHOULD have happened and the key
    # SHOULD be present.
    assert artifacts.get("partial_tracking_s3_key"), (
        "V1 contract: track.artifacts.partial_tracking_s3_key SHOULD be "
        "populated when a partial tracking.json exists at mid-track suspend "
        "(see addendum §track)"
    )

    ws = cp.get("worker_state") or {}
    required_ws_keys = {
        "progress_percent",
        "current_frame",
        "total_frames",
        "stage_progress_fraction",
    }
    assert required_ws_keys <= ws.keys(), (
        f"V1 contract: worker_state must contain {required_ws_keys}; "
        f"got {set(ws.keys())}"
    )

    # ------------------------------------------------------------------
    # Assertion 3: the underlying stub asserts the detection_cb was
    # non-None (failure raises immediately at the top of the stub), so
    # reaching this line implies the sequential path was used. Belt and
    # suspenders: re-state that as an explicit check.
    # ------------------------------------------------------------------
    # If the parallel-tracking path were still wired, run_tracking_job
    # would receive detection_cb=None per the deleted _track_segment
    # closure, and the stub's first assertion would have raised before
    # the mid-loss checkpoint was written. The presence of the
    # checkpoint row above is itself the witness.
    assert True  # explicit no-op for documentation
