---
date: 2026-05-02
revised: 2026-05-03
category: requirement
tags: [service, worker, checkpoints, schema, recovery, tests]
status: active
---

# Worker Checkpoint Standardization, Shape Tests, and Durable Artifact Recovery — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every checkpoint written by `service/worker.py` conform to the V1 envelope (`schema_version`, `pending_detection`, `artifacts`, `worker_state`), back the conformance with worker-integration tests, and persist enough durable artifact state under `artifacts.*` plus enough in-memory progress state under `worker_state.*` for crash recovery to resume tracking, upscale/analysis, upload, and SNS publish without re-doing finished work and without resetting `job_lifecycle.progress_percent`.

**Architecture:** All checkpoint writes go through builder helpers in `service/checkpoints.py` that always emit the V1 envelope including a required `worker_state` block (`progress_percent`, `current_frame`, `total_frames`, `stage_progress_fraction`). The worker writes one checkpoint per stage milestone; the upload stage writes additively as each artifact lands. The upscale/analysis stage uploads `analysis_raw.json` to S3 every 5 windows and persists the cursor in `artifacts.analysis_raw_s3_key`. A single helper `build_resume_overrides(checkpoints)` reads the latest checkpoint state and returns the dict of `TrackRequest` field overrides — used by both `submit_detection_response` (manual resume) and `recover_interrupted_job` (automatic recovery). The recovery path additionally seeds the new `job_lifecycle` row's `progress_percent` / `current_frame` / `total_frames` from `worker_state` so SSE clients do not see progress regress to 0%. The artifact contract is captured in `contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md`.

**Tech Stack:** Python 3.11+, FastAPI, asyncio, Amazon Keyspaces (Cassandra CQL), boto3 S3, pytest, pytest-asyncio, existing `service.jobs_store.JobsStore`, `service.worker.run_job`, `service.routes`, `service.reconciler`.

---

## Decisions Locked In (2026-05-02 + 2026-05-03)

1. `build_verified_boxes_checkpoint` and `build_cancellation_checkpoint` will be migrated to the V1 envelope.
2. `verified_box_a` / `verified_box_b` stay at the **root** of the checkpoint (only meaningful for `detect` / `track` stages).
3. **Anything stored in S3 belongs under `artifacts`.** Scalar progress data may sit at the root, but durable S3 keys never do.
4. `job_stage_checkpoints.completed` means **the whole job is complete**. `completed=true` on terminal-success row OR on the old-job `replaced_by_new_job` row; never on intermediate stages.
5. The mid-track frame upload bug is fixed via `s3.put_object(...)`.
6. Tests cover **both** layers: pure-function unit tests and worker-integration tests with mocked ML deps.
7. Shared `tests/conftest.py` with reusable `MockJobsStore` fixture.
8. ML/IO collaborators are stubbed via `unittest.mock.patch`.
9. The `track` completed checkpoint is updated **after** tracking JSON upload to carry `artifacts.tracking_s3_key`.
10. `_run_upscale_analysis` uploads `analysis_raw.json` every 5 windows + once at final flush.
11. **(NEW 2026-05-03)** A `worker_state` block is REQUIRED on every checkpoint write (`progress_percent`, `current_frame`, `total_frames`, `stage_progress_fraction`). Recovery seeds the new `job_lifecycle` row's progress fields from this block.
12. **(NEW 2026-05-03)** A single helper `build_resume_overrides(checkpoints)` returns the dict of `TrackRequest` field overrides for any latest-checkpoint state. Both `submit_detection_response` and `recover_interrupted_job` use it.
13. **(NEW 2026-05-03)** Recovery from a crash *during or after* `upscale_analyze` MUST forward `analysis_raw_s3_key`, `analysis_window_count`, `analysis_current_context` AND set `resume_tracking_s3_key` from the upscale checkpoint's `artifacts.tracking_s3_key`, plus set `resume_from_frame` past `end_frame` so the tracking pass becomes a no-op.
14. **(NEW 2026-05-03)** `upscale_analyze` cursor `frame_idx` semantics: it is the next *tracking-recorded* video frame index (= `max(last_window["frames"]) + 1`). The worker re-applies `sampling_rate` and `step_size` filters on resume, so callers do not have to align the cursor.
15. The upload stage writes the same `(job_id, "upload")` row three times — additively on `artifacts`.
16. SNS publish is non-idempotent in V1: a crashed job re-publishes from scratch.
17. The `skip_upscale=true` path gets lighter coverage but still V1 envelope conformance.
18. **(NEW 2026-05-03)** `worker_state` is also written on the `replaced_by_new_job` row — it captures the OLD job's progress at handoff for analytics.
19. **(NEW 2026-05-03 — settled)** `END_OF_TRACKING_SENTINEL = 10**9` is the chosen `resume_from_frame` value for upscale-crash recovery. We do not derive `end_frame` from the original `TrackRequest.end_time` in V1.
20. **(NEW 2026-05-03 — settled)** A dedicated `TrackRequest.skip_tracking: bool` field is **deferred to V2**. V1 keeps the sentinel-overload approach.
21. **(NEW 2026-05-03 — settled)** The throttled `track_progress` periodic write MUST include `artifacts.partial_tracking_s3_key` and `artifacts.resume_from_frame` so a no-`pending_detection` mid-track crash (the common recovery-manager case) recovers without re-running tracking from frame 0. Cadence: upload+checkpoint every 30 seconds of wall-clock tracking time (separate from the 1-second-throttled lifecycle write).

## File Structure

- Modify `service/checkpoints.py`:
  - Add `make_envelope(...)` and per-stage builders that all accept a required `worker_state` keyword (a small typed dict `WorkerStateSnapshot`).
  - Migrate existing `build_verified_boxes_checkpoint` / `build_cancellation_checkpoint` to V1 envelope.
  - Add `build_resume_overrides(checkpoints) -> dict[str, Any]` — reads the latest stage checkpoint and returns the `TrackRequest` field overrides for `submit_detection_response` / `recover_interrupted_job`.
  - Add `worker_state_from(checkpoints) -> dict | None` — picks the latest checkpoint's `worker_state` block for lifecycle seeding.
  - Update `select_correction_checkpoint` and `next_unprocessed_frame` to read `artifacts.partial_tracking_s3_key` / `artifacts.resume_from_frame` first with root-level fallback.
- Modify `service/worker.py`:
  - Replace every inline `write_checkpoint(...)` call site with a builder, passing the live `WorkerStateSnapshot`.
  - Add a single small helper `_make_worker_state(...)` near the top of `run_job` that captures the current in-memory state.
  - Insert new writes for `download` completion, `upscale_analyze` periodic windows, `annotate`, incremental `upload`, and the `publish` terminal write.
  - Fix the mid-track frame upload bug.
  - Add a post-upload `track` re-write that populates `artifacts.tracking_s3_key`.
- Modify `service/routes.py`:
  - In `submit_detection_response`, replace ad-hoc resume-param assembly with `build_resume_overrides(checkpoints)`. Before flipping the old job to `CANCELLED`, write the `replaced_by_new_job` checkpoint with `completed=true` and the OLD job's last-known `worker_state`.
  - Update `recover_interrupted_job` likewise — same helper, same `replaced_by_new_job` write.
  - Both routes seed the NEW lifecycle row's `progress_percent` / `current_frame` / `total_frames` from `worker_state_from(old_checkpoints)`.
- Create `tests/conftest.py`: `make_mock_jobs_store()` factory + `mock_jobs_store` / `service_components` / `service_app` / `service_client` fixtures.
- Create `tests/test_checkpoint_schema.py`: pure unit tests over each builder. Every test asserts the V1 envelope (4 required top-level keys: `schema_version`, `pending_detection`, `artifacts`, `worker_state`).
- Create `tests/test_worker_checkpoints.py`: integration tests for `run_job` covering each stage's checkpoint shape including `worker_state` values.
- Modify `tests/test_resume_endpoint.py`: switch to shared mock store; add tests for the `replaced_by_new_job` write, analysis-resume forwarding, and `worker_state`-driven lifecycle seeding.
- Modify `tests/test_job_cancellation.py`: switch to shared mock store; assert V1 envelope including `worker_state`.
- Modify `tests/test_reconciler.py`: add a recovery test for the `replaced_by_new_job` write, analysis artifact forwarding, and lifecycle progress seeding.

---

## Task 1: V1 envelope helper + `worker_state` + per-stage builders

**Files:**
- Modify: `service/checkpoints.py`
- Test: `tests/test_checkpoint_schema.py` (new)

- [ ] **Step 1: Write the failing builder unit tests**

Create `tests/test_checkpoint_schema.py`:

```python
"""Unit tests for service.checkpoints builders — V1 envelope conformance."""
from service.analysis_keyspaces_enums import PipelineStage
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
    worker_state_from,
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
    assert {"progress_percent", "current_frame", "total_frames", "stage_progress_fraction"} <= ws.keys()
    assert isinstance(ws["progress_percent"], (int, float))
    assert isinstance(ws["current_frame"], int)
    assert isinstance(ws["total_frames"], int)
    assert isinstance(ws["stage_progress_fraction"], (int, float))
    assert cp["pending_detection"] is None or isinstance(cp["pending_detection"], dict)


def test_make_envelope_requires_worker_state():
    cp = make_envelope(worker_state=_ws())
    _assert_envelope(cp)
    assert cp["pending_detection"] is None
    assert cp["artifacts"] == {}


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
    assert pd["frame_s3_key"].endswith("frame_0.jpg")


def test_track_progress_envelope():
    cp = build_track_progress(
        worker_state=_ws(progress_percent=35.0, current_frame=1200, total_frames=3600, stage_progress_fraction=0.5),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "tracking_progress"
    assert cp["worker_state"]["current_frame"] == 1200


def test_track_mid_loss_places_partial_tracking_in_artifacts():
    cp = build_track_mid_loss(
        frame_idx=512,
        frame_s3_key="checkpoints/job-1/frame_512.jpg",
        frame_bucket="bjj-video-analysis",
        candidates=[],
        suggested_boxes=[[10, 20, 100, 200], [300, 20, 400, 200]],
        partial_tracking_s3_key="checkpoints/job-1/partial_tracking.json",
        resume_from_frame=512,
        worker_state=_ws(progress_percent=20.0, current_frame=512, total_frames=1024, stage_progress_fraction=0.5),
    )
    _assert_envelope(cp)
    assert cp["pending_detection"]["reason"] == "tracking_lost"
    assert cp["artifacts"]["partial_tracking_s3_key"].endswith("partial_tracking.json")
    assert cp["artifacts"]["resume_from_frame"] == 512


def test_track_completed_carries_tracking_s3_key_when_known():
    cp = build_track_completed(
        start_frame=0,
        frame_count=900,
        tracking_s3_key="job-1_tracked.json",
        worker_state=_ws(progress_percent=55.0, current_frame=900, total_frames=900, stage_progress_fraction=1.0),
    )
    _assert_envelope(cp)
    assert cp["start_frame"] == 0
    assert cp["frame_count"] == 900
    assert cp["artifacts"]["tracking_s3_key"] == "job-1_tracked.json"


def test_track_completed_omits_artifact_when_pre_upload():
    cp = build_track_completed(
        start_frame=0, frame_count=900,
        worker_state=_ws(progress_percent=55.0, current_frame=900, total_frames=900),
    )
    _assert_envelope(cp)
    assert cp["artifacts"] == {}


def test_upscale_window_progress_carries_cursor_and_artifacts():
    cp = build_upscale_window_progress(
        frame_idx=9120,
        analysis_window_count=12,
        analysis_current_context="white belt entered guard",
        tracking_s3_key="checkpoints/job-1/tracking.json",
        analysis_raw_s3_key="checkpoints/job-1/analysis_raw.json",
        worker_state=_ws(progress_percent=67.5, current_frame=9120, total_frames=21600, stage_progress_fraction=0.5),
    )
    _assert_envelope(cp)
    assert cp["resume_cursor"] == {"frame_idx": 9120, "analysis_window_count": 12}
    assert cp["analysis_current_context"] == "white belt entered guard"
    assert cp["artifacts"]["tracking_s3_key"].endswith("tracking.json")
    assert cp["artifacts"]["analysis_raw_s3_key"].endswith("analysis_raw.json")
    assert cp["reason"] == "analysis_window_completed"


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
    assert cp2["artifacts"]["analysis_s3_key"] == "a_analysis.json"
    assert cp2["reason"] == "analysis_uploaded"

    cp3 = build_upload_incremental(
        tracking_s3_key="a_tracked.json",
        analysis_s3_key="a_analysis.json",
        annotated_video_s3_key="a_annotated.mp4",
        worker_state=_ws(progress_percent=90.0, stage_progress_fraction=1.0),
    )
    assert cp3["artifacts"]["annotated_video_s3_key"] == "a_annotated.mp4"
    assert cp3["reason"] == "annotated_video_uploaded"


def test_publish_completed_records_sns_metadata():
    cp = build_publish_completed(
        sns_topic_arn="arn:aws:sns:us-east-1:000:topic",
        sns_event_count=12,
        sns_completion_sent=True,
        worker_state=_ws(progress_percent=100.0, stage_progress_fraction=1.0),
    )
    _assert_envelope(cp)
    assert cp["artifacts"]["sns_event_count"] == 12
    assert cp["artifacts"]["sns_completion_sent"] is True


def test_replaced_by_new_job_records_replacement_id():
    cp = build_replaced_by_new_job(
        replacement_job_id="new-job-uuid",
        worker_state=_ws(progress_percent=35.0, current_frame=7432, total_frames=21600, stage_progress_fraction=0.34),
    )
    _assert_envelope(cp)
    assert cp["artifacts"]["replacement_job_id"] == "new-job-uuid"
    assert cp["worker_state"]["current_frame"] == 7432


def test_verified_boxes_envelope_keeps_root_boxes():
    cp = build_verified_boxes_checkpoint(
        [1, 2, 3, 4], [5, 6, 7, 8], PipelineStage.TRACK,
        worker_state=_ws(progress_percent=15.0, stage_progress_fraction=1.0),
    )
    _assert_envelope(cp)
    assert cp["verified_box_a"] == [1, 2, 3, 4]
    assert cp["verified_box_b"] == [5, 6, 7, 8]
    assert cp["source_stage"] == "track"


def test_cancellation_envelope_keeps_resume_cursor():
    cp = build_cancellation_checkpoint(
        reason="user_cancelled", frame_idx=42, progress_percent=33.3,
        worker_state=_ws(progress_percent=33.3, current_frame=42, total_frames=100, stage_progress_fraction=0.42),
    )
    _assert_envelope(cp)
    assert cp["resume_cursor"] == {"frame_idx": 42}


def test_build_resume_overrides_initial_detection():
    """No prior boxes — overrides only carry the fields the resume body provides (none here)."""
    overrides = build_resume_overrides([
        {"stage_name": "detect", "checkpoint_data": {
            "schema_version": 1, "pending_detection": {"reason": "initial", "frame_idx": 0},
            "artifacts": {}, "worker_state": _ws()._asdict() if hasattr(_ws(), "_asdict") else {},
        }, "completed": False},
    ])
    # Initial detection: no resume_tracking_s3_key, no analysis fields
    assert "resume_tracking_s3_key" not in overrides
    assert "analysis_raw_s3_key" not in overrides


def test_build_resume_overrides_mid_track_loss():
    overrides = build_resume_overrides([
        {"stage_name": "track", "checkpoint_data": {
            "schema_version": 1,
            "pending_detection": {"reason": "tracking_lost", "frame_idx": 7432},
            "artifacts": {
                "partial_tracking_s3_key": "checkpoints/orig/partial_tracking.json",
                "resume_from_frame": 7432,
            },
            "worker_state": {"progress_percent": 35.0, "current_frame": 7432, "total_frames": 21600, "stage_progress_fraction": 0.34},
        }, "completed": False},
    ])
    assert overrides["resume_tracking_s3_key"] == "checkpoints/orig/partial_tracking.json"
    assert overrides["resume_from_frame"] == 7432


def test_build_resume_overrides_upscale_crash():
    overrides = build_resume_overrides([
        {"stage_name": "upscale_analyze", "checkpoint_data": {
            "schema_version": 1, "pending_detection": None,
            "resume_cursor": {"frame_idx": 9120, "analysis_window_count": 12},
            "analysis_current_context": "north-south",
            "artifacts": {
                "tracking_s3_key": "checkpoints/orig/tracking.json",
                "analysis_raw_s3_key": "checkpoints/orig/analysis_raw.json",
            },
            "worker_state": {"progress_percent": 67.5, "current_frame": 9120, "total_frames": 21600, "stage_progress_fraction": 0.5},
        }, "completed": False},
    ])
    assert overrides["resume_tracking_s3_key"] == "checkpoints/orig/tracking.json"
    # resume_from_frame should be set to skip tracking pass; large sentinel or end-frame derivative
    assert overrides.get("resume_from_frame", 0) >= 9120
    assert overrides["analysis_raw_s3_key"] == "checkpoints/orig/analysis_raw.json"
    assert overrides["analysis_window_count"] == 12
    assert overrides["analysis_current_context"] == "north-south"


def test_worker_state_from_picks_latest_stage():
    """Latest pipeline-stage worker_state wins (track completed > download completed)."""
    cps = [
        {"stage_name": "download", "checkpoint_data": {
            "worker_state": {"progress_percent": 10.0, "current_frame": 0, "total_frames": 0, "stage_progress_fraction": 1.0},
        }, "completed": False},
        {"stage_name": "track", "checkpoint_data": {
            "worker_state": {"progress_percent": 55.0, "current_frame": 21600, "total_frames": 21600, "stage_progress_fraction": 1.0},
        }, "completed": False},
    ]
    ws = worker_state_from(cps)
    assert ws is not None
    assert ws["progress_percent"] == 55.0
    assert ws["current_frame"] == 21600
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/test_checkpoint_schema.py -v`
Expected: FAIL — builders not yet defined; existing two builders fail the envelope assertion.

- [ ] **Step 3: Implement `WorkerStateSnapshot` + envelope helper**

In `service/checkpoints.py`, add at the top after the existing imports:

```python
from typing import Any, NamedTuple


class WorkerStateSnapshot(NamedTuple):
    progress_percent: float
    current_frame: int
    total_frames: int
    stage_progress_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "progress_percent": float(self.progress_percent),
            "current_frame": int(self.current_frame),
            "total_frames": int(self.total_frames),
            "stage_progress_fraction": float(self.stage_progress_fraction),
        }


def make_envelope(
    *,
    worker_state: WorkerStateSnapshot,
    pending_detection: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    **extras: Any,
) -> dict[str, Any]:
    cp: dict[str, Any] = {
        "schema_version": 1,
        "pending_detection": pending_detection,
        "artifacts": dict(artifacts or {}),
        "worker_state": worker_state.to_dict(),
    }
    cp.update(extras)
    return cp
```

- [ ] **Step 4: Implement per-stage builders**

Append builders for `download`, `detect_initial_pending`, `track_progress`, `track_mid_loss`, `track_completed`, `upscale_started`, `upscale_window_progress`, `annotate_completed`, `upload_incremental`, `publish_completed`, `replaced_by_new_job`, all taking `worker_state: WorkerStateSnapshot` as a required keyword. (Bodies as in the prior plan revision; each body is a single `make_envelope(...)` call.)

- [ ] **Step 5: Migrate existing builders to envelope (now requiring `worker_state`)**

```python
def build_verified_boxes_checkpoint(
    box_a: list[float],
    box_b: list[float],
    source_stage: PipelineStage | None,
    *,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    return make_envelope(
        worker_state=worker_state,
        reason="detection_correction_resume",
        source_stage=source_stage.value if source_stage else "",
        verified_box_a=box_a,
        verified_box_b=box_b,
    )


def build_cancellation_checkpoint(
    *,
    reason: str,
    frame_idx: int = 0,
    progress_percent: float = 0.0,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    return make_envelope(
        worker_state=worker_state,
        reason=reason,
        resume_cursor={"frame_idx": frame_idx},
        progress_percent=progress_percent,
    )
```

- [ ] **Step 6: Implement `build_resume_overrides` and `worker_state_from`**

```python
END_OF_TRACKING_SENTINEL = 10**9  # past any plausible video frame; turns tracking pass into a no-op


def _checkpoint_for(checkpoints, stage: PipelineStage) -> dict[str, Any]:
    for cp in checkpoints:
        if cp.get("stage_name") == stage.value:
            return cp.get("checkpoint_data", {}) or {}
    return {}


def _latest_stage_with_worker_state(checkpoints):
    """Return the checkpoint for the highest-ordered stage that has a worker_state block."""
    by_stage = {cp["stage_name"]: cp for cp in checkpoints}
    for stage in reversed(STAGE_ORDER):
        cp = by_stage.get(stage.value)
        if cp and (cp.get("checkpoint_data") or {}).get("worker_state"):
            return cp
    return None


def worker_state_from(checkpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    cp = _latest_stage_with_worker_state(checkpoints)
    if not cp:
        return None
    return (cp.get("checkpoint_data") or {}).get("worker_state")


def build_resume_overrides(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Compose TrackRequest overrides from the latest resumable checkpoint state.

    Used by both submit_detection_response (manual resume) and recover_interrupted_job.
    """
    overrides: dict[str, Any] = {}

    # Mid-track loss: forward partial tracking pointers
    track_cp = _checkpoint_for(checkpoints, PipelineStage.TRACK)
    track_artifacts = track_cp.get("artifacts") or {}
    partial_key = (
        track_artifacts.get("partial_tracking_s3_key")
        or track_cp.get("partial_tracking_s3_key")
    )
    if partial_key:
        overrides["resume_tracking_s3_key"] = partial_key
        overrides["resume_from_frame"] = next_unprocessed_frame(track_cp)

    # Crash during/after upscale_analyze: forward analysis fields, set tracking sentinel
    upscale_cp = _checkpoint_for(checkpoints, PipelineStage.UPSCALE_ANALYZE)
    upscale_artifacts = upscale_cp.get("artifacts") or {}
    upscale_cursor = upscale_cp.get("resume_cursor") or {}
    if upscale_artifacts.get("analysis_raw_s3_key"):
        overrides["resume_tracking_s3_key"] = upscale_artifacts.get(
            "tracking_s3_key", overrides.get("resume_tracking_s3_key", "")
        )
        # Push tracking pass past end so it no-ops; worker re-loads tracking from S3 via partial path
        overrides["resume_from_frame"] = END_OF_TRACKING_SENTINEL
        overrides["analysis_raw_s3_key"] = upscale_artifacts["analysis_raw_s3_key"]
        overrides["analysis_window_count"] = upscale_cursor.get("analysis_window_count", 0)
        overrides["analysis_current_context"] = upscale_cp.get("analysis_current_context", "")

    return overrides
```

(Keep `select_correction_checkpoint` and `next_unprocessed_frame` updated to read `artifacts.partial_tracking_s3_key` first with root fallback — see prior plan revision.)

- [ ] **Step 7: Run unit tests to verify all pass**

Run: `source venv/bin/activate && pytest tests/test_checkpoint_schema.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add service/checkpoints.py tests/test_checkpoint_schema.py
git commit -m "feat(checkpoints): V1 envelope + worker_state + per-stage builders + resume_overrides helper"
```

---

## Task 2: Shared `MockJobsStore` fixture

(Identical to prior plan revision — see [details there](#task-2-shared-mockjobsstore-fixture-2026-05-02). Migrates `test_resume_endpoint.py` and `test_job_cancellation.py` onto a `tests/conftest.py` fixture.)

---

## Task 3: Standardize `download` + `detect` worker writes; fix mid-track frame upload

**Files:**
- Modify: `service/worker.py`
- Test: `tests/test_worker_checkpoints.py` (new)

The integration tests follow the prior plan revision shape but ALSO assert `worker_state` values, e.g.:

```python
assert cp_data["worker_state"]["progress_percent"] == pytest.approx(10.0)
assert cp_data["worker_state"]["current_frame"] == 0
assert cp_data["worker_state"]["stage_progress_fraction"] == pytest.approx(1.0)
```

Implementation steps:

- [ ] **Step 1: Add `_make_worker_state(...)` helper near the top of `run_job`**

```python
from service.checkpoints import WorkerStateSnapshot

def _make_worker_state(
    *, progress_percent: float, current_frame: int = 0,
    total_frames: int = 0, stage_progress_fraction: float = 0.0,
) -> WorkerStateSnapshot:
    return WorkerStateSnapshot(
        progress_percent=progress_percent,
        current_frame=current_frame,
        total_frames=total_frames,
        stage_progress_fraction=stage_progress_fraction,
    )
```

(Make this a module-level function so `_make_detection_cb` can use it too.)

- [ ] **Step 2: Insert `download` write after the download finishes**

```python
await jobs_store.write_checkpoint(
    job_id, PipelineStage.DOWNLOAD, False,
    build_download_completed(
        worker_state=_make_worker_state(progress_percent=10.0, stage_progress_fraction=1.0),
    ),
)
```

- [ ] **Step 3: Replace inline initial-detect checkpoint with builder**

```python
await jobs_store.write_checkpoint(
    job_id, PipelineStage.DETECT, False,
    build_detect_initial_pending(
        frame_idx=frame_idx,
        frame_s3_key=frame_s3_key,
        frame_bucket=request.bucket,
        candidates=candidates,
        suggested_boxes=suggested_boxes,
        worker_state=_make_worker_state(progress_percent=10.0, stage_progress_fraction=0.0),
    ),
)
```

- [ ] **Step 4: Fix mid-track frame upload bug (put_object instead of upload_file)**

(Same as prior plan revision Step 5.)

- [ ] **Step 5-7:** Run tests, commit.

---

## Task 4: Standardize `track` worker writes (progress + mid-loss + completed)

**Files:**
- Modify: `service/worker.py`
- Test: `tests/test_worker_checkpoints.py`

Two periodic write cadences inside `tracking_progress_cb`:

- **1-second cadence** — lifecycle row update (`update_progress`) only. No partial-tracking upload, no checkpoint write. Same as today.
- **30-second cadence** — full `track_progress` checkpoint write that uploads the current `tracking.json` to S3 as `checkpoints/{job_id}/partial_tracking.json` and records `artifacts.partial_tracking_s3_key` + `artifacts.resume_from_frame=frames_done` (plus `worker_state`). This is what the recovery manager reads when a worker crashes mid-track without a `pending_detection` event.

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.asyncio
async def test_track_progress_periodic_write_includes_partial_artifacts(...):
    """Throttle expires once -> one track_progress checkpoint with artifacts.partial_tracking_s3_key."""
    # Drive tracking_progress_cb past the 30-second wall-clock threshold (use a fake clock).
    # Assert: exactly one PipelineStage.TRACK checkpoint with reason=tracking_progress;
    #         artifacts.partial_tracking_s3_key endswith "partial_tracking.json";
    #         artifacts.resume_from_frame matches frames_done at the time of the write;
    #         worker_state.current_frame matches frames_done;
    #         s3.upload_json was called once with the parsed tracking JSON.
```

- [ ] **Step 2: Add `build_track_progress(...)` to `service/checkpoints.py`**

```python
def build_track_progress(
    *,
    partial_tracking_s3_key: str | None,
    resume_from_frame: int,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"resume_from_frame": resume_from_frame}
    if partial_tracking_s3_key:
        artifacts["partial_tracking_s3_key"] = partial_tracking_s3_key
    return make_envelope(
        worker_state=worker_state,
        artifacts=artifacts,
        reason="tracking_progress",
        resume_cursor={"frame_idx": resume_from_frame},
    )
```

- [ ] **Step 3: Add the 30-second cadence inside `tracking_progress_cb`**

In `service/worker.py` near the existing `_last_ks_write` block, add a sibling throttle:

```python
_last_partial_upload = 0.0
PARTIAL_UPLOAD_INTERVAL = 30.0  # seconds

def tracking_progress_cb(frames_done: int, total: int):
    nonlocal _last_ks_write, _last_partial_upload
    pct = 15.0 + (frames_done / max(total, 1)) * 40.0
    now = time.monotonic()
    write_lifecycle = (now - _last_ks_write) >= 1.0
    upload_partial = (now - _last_partial_upload) >= PARTIAL_UPLOAD_INTERVAL
    if write_lifecycle:
        _last_ks_write = now
    if upload_partial:
        _last_partial_upload = now
    asyncio.run_coroutine_threadsafe(
        _update_tracking_progress_with_partial(
            job_id, frames_done, total, pct, job_store, jobs_store,
            request, work_dir, s3,
            write_lifecycle=write_lifecycle,
            upload_partial=upload_partial,
        ),
        loop,
    )
```

The async helper `_update_tracking_progress_with_partial`:

```python
async def _update_tracking_progress_with_partial(
    job_id, frames_done, total, pct, job_store, jobs_store,
    request, work_dir, s3,
    *, write_lifecycle: bool, upload_partial: bool,
):
    if write_lifecycle:
        await _update_tracking_progress(
            job_id, frames_done, total, pct, job_store, jobs_store,
            write_ks=True,
        )
    if not upload_partial:
        return
    tracking_json_path = os.path.join(work_dir, "tracking", "tracking.json")
    if not os.path.isfile(tracking_json_path):
        return
    partial_key = f"checkpoints/{job_id}/partial_tracking.json"
    upload_bucket = request.output_bucket or request.bucket
    try:
        partial_data = _load_partial_tracking_dict(tracking_json_path)
        await asyncio.get_event_loop().run_in_executor(
            None, s3.upload_json, partial_data, upload_bucket, partial_key,
        )
    except Exception as e:
        logger.warning("Periodic partial-tracking upload failed: %s", e)
        return
    ws = _make_worker_state(
        progress_percent=pct,
        current_frame=frames_done,
        total_frames=total,
        stage_progress_fraction=(frames_done / max(total, 1)),
    )
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.TRACK, False,
        build_track_progress(
            partial_tracking_s3_key=partial_key,
            resume_from_frame=frames_done,
            worker_state=ws,
        ),
    )
```

- [ ] **Step 4: Replace mid-track checkpoint with `build_track_mid_loss(...)` (worker_state included)**

Same builder, with `worker_state=` populated from the in-flight tracking progress.

- [ ] **Step 5: Replace completed-tracking checkpoint with `build_track_completed(...)` (without `tracking_s3_key` yet — Task 5 adds the post-upload re-write)**

- [ ] **Step 6: Run targeted tests**

Run: `source venv/bin/activate && pytest tests/test_worker_checkpoints.py -k 'track' -v`
Expected: PASS for periodic, mid-loss, and completed tests.

- [ ] **Step 7: Commit**

```bash
git add service/worker.py service/checkpoints.py tests/test_worker_checkpoints.py
git commit -m "feat(worker): periodic track_progress checkpoint with partial-tracking S3 upload"
```

---

## Task 5: Track post-upload re-write + `upload` incremental writes (skip_upscale path)

(Identical structure to prior plan revision; each builder call now passes `worker_state=`. Skip_upscale terminal write uses `completed=True` and `worker_state.progress_percent=100.0`.)

---

## Task 6: `upscale_analyze` checkpoint with periodic raw-analysis upload (every 5 windows)

(Identical structure to prior plan revision; each builder call now passes `worker_state=`. The window-progress write computes `progress_percent = 55.0 + (processed / total) * 25.0` and `stage_progress_fraction = processed / total`.)

---

## Task 7: `annotate` + incremental `upload` + `publish` worker writes

(Identical structure to prior plan revision; each builder call now passes `worker_state=`. Publish terminal write uses `completed=True` and `worker_state.progress_percent=100.0`.)

---

## Task 8: Unified `build_resume_overrides` wired into `submit_detection_response` + `recover_interrupted_job`

**Files:**
- Modify: `service/routes.py`
- Test: `tests/test_resume_endpoint.py`
- Test: `tests/test_reconciler.py`

- [ ] **Step 1: Failing tests for both routes**

Add to `tests/test_resume_endpoint.py`:

```python
@pytest.mark.asyncio
async def test_resume_overrides_forward_analysis_artifacts(
    service_client, service_components,
):
    """If the job has an upscale_analyze checkpoint, manual resume forwards analysis fields."""
    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="v.mp4")
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "vid", "u")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    await jobs_store.set_state(job.job_id, JobState.AWAITING_CORRECTION)
    await jobs_store.write_checkpoint(
        job.job_id, PipelineStage.DETECT, False, {
            "schema_version": 1,
            "pending_detection": {"reason": "initial", "frame_idx": 0},
            "artifacts": {}, "worker_state": {
                "progress_percent": 10.0, "current_frame": 0,
                "total_frames": 0, "stage_progress_fraction": 0.0,
            },
        },
    )
    await jobs_store.write_checkpoint(
        job.job_id, PipelineStage.UPSCALE_ANALYZE, False, {
            "schema_version": 1, "pending_detection": None,
            "resume_cursor": {"frame_idx": 9120, "analysis_window_count": 12},
            "analysis_current_context": "north-south",
            "artifacts": {
                "tracking_s3_key": "checkpoints/orig/tracking.json",
                "analysis_raw_s3_key": "checkpoints/orig/analysis_raw.json",
            },
            "worker_state": {
                "progress_percent": 67.5, "current_frame": 9120,
                "total_frames": 21600, "stage_progress_fraction": 0.5,
            },
        },
    )

    resp = await service_client.post(
        f"/jobs/{job.job_id}/resume",
        json={"box_a": [1, 2, 3, 4], "box_b": [5, 6, 7, 8]},
    )
    assert resp.status_code == 200
    new_job_id = resp.json()["job_id"]
    rec = json.loads(await jobs_store.get_request(new_job_id))
    assert rec["analysis_raw_s3_key"] == "checkpoints/orig/analysis_raw.json"
    assert rec["analysis_window_count"] == 12
    assert rec["analysis_current_context"] == "north-south"
    assert rec["resume_tracking_s3_key"] == "checkpoints/orig/tracking.json"
    assert rec["resume_from_frame"] >= 9120


@pytest.mark.asyncio
async def test_replaced_by_new_job_carries_worker_state(
    service_client, awaiting_job, service_components,
):
    """The 'replaced_by_new_job' row records the old job's last worker_state."""
    _, _, jobs_store = service_components
    job_id = awaiting_job
    # Seed a track checkpoint with worker_state so build_resume_overrides + replaced_by_new_job
    # have something to read.
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.TRACK, False, {
            "schema_version": 1, "pending_detection": None,
            "artifacts": {}, "worker_state": {
                "progress_percent": 35.0, "current_frame": 7432,
                "total_frames": 21600, "stage_progress_fraction": 0.34,
            },
        },
    )

    resp = await service_client.post(
        f"/jobs/{job_id}/detection_response",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )
    new_job_id = resp.json()["job_id"]

    replaced = [
        cp for (jid, _), cp in jobs_store._checkpoints.items()
        if jid == job_id and cp["checkpoint_data"].get("reason") == "replaced_by_new_job"
    ]
    assert len(replaced) == 1
    cp = replaced[0]
    assert cp["completed"] is True
    assert cp["checkpoint_data"]["artifacts"]["replacement_job_id"] == new_job_id
    assert cp["checkpoint_data"]["worker_state"]["current_frame"] == 7432
```

Add a parallel test in `tests/test_reconciler.py` that drives `recover_interrupted_job` with an upscale_analyze checkpoint and asserts the same forwarding + `replaced_by_new_job` write.

- [ ] **Step 2: Run failing tests**

Run: `source venv/bin/activate && pytest tests/test_resume_endpoint.py tests/test_reconciler.py -v`
Expected: FAIL — `build_resume_overrides` not wired in routes; `replaced_by_new_job` write not present.

- [ ] **Step 3: Wire `submit_detection_response` to use `build_resume_overrides`**

```python
from service.checkpoints import (
    build_resume_overrides,
    build_replaced_by_new_job,
    worker_state_from,
    WorkerStateSnapshot,
)

checkpoints = await _jobs_store.get_all_checkpoints(job_id)
correction_stage, _ = select_correction_checkpoint(checkpoints)

resume_params = {
    **orig_request,
    "box_a": body.box_a,
    "box_b": body.box_b,
    **build_resume_overrides(checkpoints),
}
if "resume_tracking_s3_key" in resume_params:
    resume_params["resume_from_job_id"] = job_id
```

- [ ] **Step 4: Add the `replaced_by_new_job` write before flipping to CANCELLED**

```python
old_ws_dict = worker_state_from(checkpoints) or {
    "progress_percent": 0.0, "current_frame": 0,
    "total_frames": 0, "stage_progress_fraction": 0.0,
}
old_ws = WorkerStateSnapshot(**old_ws_dict)
terminal_stage = correction_stage or PipelineStage.TRACK
_require_write(
    await _jobs_store.write_checkpoint(
        job_id, terminal_stage, True,
        build_replaced_by_new_job(replacement_job_id=new_job_id, worker_state=old_ws),
    ),
    "old job replacement checkpoint",
)
```

- [ ] **Step 5: Wire `recover_interrupted_job` identically**

Same `build_resume_overrides` + `replaced_by_new_job` + `worker_state_from` pattern.

- [ ] **Step 6: Run all touched tests**

Run: `source venv/bin/activate && pytest tests/test_resume_endpoint.py tests/test_reconciler.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add service/routes.py tests/test_resume_endpoint.py tests/test_reconciler.py
git commit -m "feat(routes): unified resume_overrides + replaced_by_new_job + worker_state forwarding"
```

---

## Task 9: Seed new lifecycle from `worker_state` so progress_percent does not regress

**Files:**
- Modify: `service/jobs_store.py` (add optional `progress_percent`/`current_frame`/`total_frames` kwargs to `create_lifecycle`)
- Modify: `service/routes.py` (call site for both manual resume and recovery)
- Test: `tests/test_resume_endpoint.py`, `tests/test_reconciler.py`

- [ ] **Step 1: Failing test asserts new lifecycle starts at the old progress, not 0**

```python
new_lc = await jobs_store.get_lifecycle(new_job_id)
assert new_lc["progress_percent"] == pytest.approx(35.0)
assert new_lc["current_frame"] == 7432
assert new_lc["total_frames"] == 21600
```

- [ ] **Step 2: Extend `JobsStore.create_lifecycle` to accept seed values (default 0/0/0.0)**
- [ ] **Step 3: Pass `worker_state_from(checkpoints)` values into `create_lifecycle` from both routes**
- [ ] **Step 4: Tests pass; commit**

---

## Task 10: Cancellation envelope conformance + sweep verification

(Identical to prior plan revision Task 9 — assert envelope including `worker_state` on the cancellation row; run the full suite.)

---

## Task 11: Knowledge-base update + INDEX.md

(Identical to prior plan revision Task 10 — pointer to this file under Requirements; pointer to the addendum under Contracts; insight write-up after merge.)

---

## Verification

- `source venv/bin/activate && pytest tests/test_checkpoint_schema.py tests/test_worker_checkpoints.py tests/test_resume_endpoint.py tests/test_job_cancellation.py tests/test_reconciler.py -v` — all green.
- Manual smoke (LocalStack): submit a job with no boxes → confirm `detect` checkpoint envelope including `worker_state`; let a job fully complete → `download/track/upscale_analyze/annotate/upload(x3)/publish` rows all present with V1 envelope and monotonically-increasing `worker_state.progress_percent`; force a crash mid-`upscale_analyze` after >5 windows → confirm reconciler creates a replacement that resumes from `analysis_window_count`, that the replacement lifecycle's `progress_percent` starts at ~67% (not 0%), and that the OLD job has a `replaced_by_new_job` row with `completed=true` and the OLD `worker_state` recorded.

## Governance Handoff Marker

- architecture_review_required: true
- Impact reasons: [data-model, api-contract]
- Non-negotiable constraints:
  - Every checkpoint write goes through a builder that emits the V1 envelope including `worker_state`.
  - S3 keys live under `artifacts`; scalar progress lives at the root.
  - `completed=true` only on terminal-success row OR on the old-job `replaced_by_new_job` row.
  - SNS publish is non-idempotent in V1.
  - `_make_detection_cb` uses `s3.put_object` for raw bytes, never `upload_file`.
  - Manual resume and crash recovery share `build_resume_overrides`; they MUST stay in sync.
  - New lifecycle rows seeded from `worker_state` so SSE progress never regresses.
- Impacted areas:
  - Backend services: vision engine `worker`, `routes`, `reconciler`; analysis service reads new artifact keys + `worker_state`.
  - Data/storage: Keyspaces `job_stage_checkpoints` row shapes.
  - Infrastructure/runtime: 48-hour artifact retention (covered separately).

## Remaining Decisions Before Code

_All open questions from the previous revision are now settled (see Decisions Locked In #19, #20, #21). No outstanding blockers — implementation may begin once this plan and the addendum are approved._

## References

- Reference contract: `contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md` and `working_log/contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md`.
- Foundational schema (backend-owned): `bjj-vision-backend/contracts/vision_engine/CHECKPOINT_DATA_SCHEMA_V1.md`.
- Workflow reference: [2026-04-25 - Job Start and Resume Workflow Reference](../insights/2026-04-25-job-start-resume-workflow-reference.md).
- Prior batch: [2026-04-26 - Job Lifecycle Resume Refactor Plan](2026-04-26-job-lifecycle-resume-refactor-plan.md).
- Implementation status pre-this-batch: [2026-05-01 - Lifecycle Resume and Recovery Implementation](../insights/2026-05-01-lifecycle-resume-recovery-implementation.md).
