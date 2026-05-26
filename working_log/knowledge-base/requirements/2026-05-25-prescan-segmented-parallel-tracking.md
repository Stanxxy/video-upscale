---
date: 2026-05-25
category: requirement
tags: [tracking, parallel-tracking, prescan, future-upgrade, governance]
status: proposed
---

# Pre-scan segmented parallel tracking (future upgrade)

## Companion documents

- `working_log/contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md` — V1
  envelope, `detect` / `track` artifact shapes, `worker_state` block,
  resume-parameter forwarding rules.
- `working_log/contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md` —
  manual resume / detection handoff sequence (§4) and the mid-track
  "instant rotation" rule (§4.1) the engine MUST keep honoring.
- `working_log/knowledge-base/insights/2026-04-25-job-start-resume-workflow-reference.md` —
  numbered Answered Requirements, especially #7, #20, #22.
- `working_log/knowledge-base/insights/2026-05-25-parallel-upscale-progress-aggregator.md` —
  shipping state of segment-aware parallelism (upscale only) that this
  future upgrade extends to tracking.

---

## 1. Context — why the "first commit" chose sequential tracking

The 2026-05-25 refactor that ships in this PR makes **tracking always
sequential** and parallelizes only the UPSCALE_ANALYZE stage. The user's
direct directive was:

> "Before upscale stage the worker should go with sequential; parallel
> should be applied since upscale stage."

The reason is contractual: parallel tracking with per-segment
`detection_cb=None` violates the mid-track-loss V1 contract
(`CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md` §`stage_name == track`,
`JOB_ROTATION_HANDOFF_AND_RESUME.md` §4.1). When SAM2 drops the athletes
mid-segment, the engine MUST:

1. write a `track` checkpoint with `pending_detection.reason ==
   "tracking_lost"` and `artifacts.{partial_tracking_s3_key,
   resume_from_frame}` populated;
2. set lifecycle state to `AWAITING_CORRECTION`;
3. raise `HumanVerificationSuspend` and **end the current `run_tracking`
   call** in the same process so the worker semaphore releases and the
   replacement job from `POST /jobs/{job_id}/resume` can start
   immediately.

A parallel-tracking helper without `detection_cb` simply cannot do any
of that — it skips re-detection, falls back to the `max_missing_frames`
budget, and (in the worst case) silently produces a tracking JSON whose
identities are wrong from that frame onward. Worse, sibling segments
would have to be cancelled to release the GPU, breaking the "instant
rotation" guarantee.

The simplest correct design is therefore: tracking sequential, upscale
parallel. That is what ships now.

---

## 2. Vision

> "In future upgrade we should also add a way to detect any tracking
> loss before head (referee/people block the athletes) and let human
> correct all detection at the beginning. In that way the video will be
> naturally segmented into parallelizable segments. Please plan
> carefully on this way." — user, 2026-05-25.

A **pre-scan stage** runs a lightweight pass over the entire video
**before** the heavyweight SAM2 tracking pass. It surfaces every frame
where tracking is likely to drop the athletes (occlusion, scene cut,
blackout, third-party crossing) as a **detection candidate**, and the
user confirms / corrects boxes for all of them up-front in a single
batch correction UX. Once those corrections are durable, the video is
**naturally segmented at the correction boundaries** — each segment has
known-good athlete boxes at both its start and its end, so its tracking
is unlikely to drop mid-segment.

Those segments **can be tracked in parallel safely** because:

- Each segment starts from human-confirmed boxes (no parallel
  `detection_cb=None` shortcut required).
- Mid-segment loss falls back to the existing
  `AWAITING_CORRECTION`/replacement flow but only for the single
  affected segment — siblings continue.

The upscale stage retains the parallel pattern shipped in this PR
(per-segment `_run_upscale_analysis` + heartbeat aggregator).

---

## 3. Architecture sketch

### Stage 0 (NEW): Pre-scan

A lightweight pass that produces a list of "probable track-loss frames"
without running full SAM2 propagation:

- Sample every N-th frame (`N = config.prescan_frame_stride`, e.g. 30 or
  60).
- For each sample: run YOLO person detector + scene-cut detector +
  optional fast pose-estimator.
- Heuristics for "drop candidate":
  - Fewer than 2 persons detected (athletes occluded by referee or each
    other).
  - More than `K` persons in the frame (third-party crossing).
  - Scene cut detected (`hist_diff > threshold`).
  - Blackout detected (`mean(frame) < threshold`).
- Emit a list of `(frame_idx, reason, candidate_boxes)` tuples.

Output: a checkpoint row at a NEW pipeline stage,
`PipelineStage.PRESCAN`, with `artifacts.prescan_candidates_s3_key`
pointing to the JSON list.

### Stage 1 (NEW): Batch correction UX

Surface ALL flagged frames in a single multi-frame correction modal.
For each candidate the user can:

- Confirm the suggested boxes.
- Adjust the boxes.
- Mark the candidate as "skip" (not a real drop — keep tracking
  through it).

Each confirmed correction is written as a `detect` checkpoint row in V1
shape (one row per future segment seed). The job lifecycle moves to
`AWAITING_CORRECTION` once and only resumes after the user submits the
batch.

### Stage 2 (modified): Parallel tracking, naturally segmented

After the batch correction, the worker reads N corrected boxes and
builds N tracking segments bounded at the correction frames. Each
segment runs `run_tracking_job` in its own thread executor with:

- `start_frame = correction[i].frame_idx`
- `end_frame   = correction[i+1].frame_idx`
- `box_a, box_b = correction[i].boxes` (human-confirmed)
- `detection_cb` = a real callback (still required as a safety net —
  see Open Questions below).

If a segment **still** experiences mid-segment tracking loss, it falls
back to the existing `AWAITING_CORRECTION` flow but only for that
segment. Sibling segments either complete or are cancelled and re-run
after the correction.

### Stage 3 (existing): Parallel upscale_analyze

Unchanged from the pattern shipped in this PR: segments are sliced from
the merged tracking JSON; each runs `_run_upscale_analysis` in its own
thread executor; the heartbeat aggregator writes
`PipelineStage.UPSCALE_ANALYZE` progress in the 55-80% band.

---

## 4. V1 envelope implications

The current V1 envelope assumes **one** `pending_detection` per stage
row. Pre-scan needs to surface N candidates and let the user resume
with N corrected box sets. Two options:

### Option A: N rows per pre-scan checkpoint

Pre-scan writes N rows to `job_stage_checkpoints` with
`stage_name = "detect"` (or new `"prescan"`), each carrying its own
`pending_detection.{frame_idx, frame_s3_key, candidates, suggested_boxes}`.

Issue: Cassandra/Keyspaces composite key `(job_id, stage_name)` is
unique, so distinct rows need distinct `stage_name` values
(`"detect_0"`, `"detect_1"`, etc.). That breaks the closed-set
`PipelineStage` enum in `service/analysis_keyspaces_enums.py`.

### Option B: One row holding a `pending_detections` array

Pre-scan writes a single `detect` (or `prescan`) row whose
`checkpoint_data` extends the V1 envelope with a `pending_detections:
list[PendingDetection]` field (plural). V1 readers gracefully ignore
unknown top-level keys, but the resume route MUST learn to read the
list when present.

**Preferred:** Option B. Less schema churn, mirrors how
`build_upload_incremental` already accumulates artifacts on a single
row.

### Track-stage extension for per-segment partial uploads

When parallel tracking eventually ships, mid-segment loss on segment
`i` needs its **own** `artifacts.partial_tracking_s3_key` independent
of sibling segments. The `track` stage shape becomes:

```json
{
  "schema_version": 1,
  "pending_detection": { "reason": "tracking_lost", "frame_idx": …, … },
  "artifacts": {
    "partial_tracking_s3_keys": {
      "0": "checkpoints/{job_id}/seg0/partial_tracking.json",
      "1": "checkpoints/{job_id}/seg1/partial_tracking.json",
      …
    },
    "resume_from_frames": {
      "0": 0,                  // seg0 complete
      "1": 7432,               // seg1 stopped here
      "2": null                // seg2 never started
    },
    "active_segment_id": 1
  },
  "worker_state": { … }
}
```

This is a non-trivial schema bump. **Backend coordination required**:
both `bjj-vision-backend/.../checkpoint_data_schema_v1` consumers and
the engine `service/checkpoints.py` builders must be updated together,
and `CHECKPOINT_DATA_SCHEMA_V1.md` (backend source of truth) must lead.

---

## 5. Resume contract impact

Today's resume request body shape (from
`JOB_ROTATION_HANDOFF_AND_RESUME.md` §4) carries a single corrected
`(box_a, box_b)` pair. With batch correction the body needs to carry N
corrections. Sketch:

```json
{
  "corrections": [
    { "frame_idx": 7432, "box_a": [...], "box_b": [...] },
    { "frame_idx": 14200, "box_a": [...], "box_b": [...] },
    …
  ]
}
```

The replacement job is created once (still a single new `job_id`); the
N corrections are written into N future tracking segments via the
extended `track` artifact shape above.

`submit_detection_response` in `service/routes.py` (and its automatic
counterpart `recover_interrupted_job`) must learn this new shape AND
remain backward-compatible with the single-correction shape so non-
pre-scan jobs keep working.

---

## 6. Open questions

These need answers before implementation can begin:

1. **Pre-scan model + frame sampling rate.** What's the cheapest model
   that reliably flags drop frames? Cost vs coverage tradeoff. Initial
   guess: YOLO + scene-cut at every 30th frame on the same GPU stream
   that already loads YOLO during the main pass.
2. **UI design for batch correction.** Surfacing 5-20 candidates at
   once without overwhelming the user. Are they a scrollable list?
   Step-through carousel? Tied to a video timeline?
3. **Segments missed by pre-scan.** If pre-scan misses a real
   tracking-loss point (false negative), how does the per-segment
   tracking pass handle it? Three options from earlier analysis:
   - **A.** Fall back to the current `AWAITING_CORRECTION` flow but
     only for the affected segment (cancel siblings, resume the whole
     job from the new correction).
   - **B.** Cancel only the affected segment, mark it pending, let
     siblings complete; surface a "post-tracking correction needed"
     state to the user.
   - **C.** Drop the segment entirely; fall back to a sequential
     tracking pass for the gap.
   **Likely answer:** A first (simplest), evolve to B if false-negative
   rate is high.
4. **N for upscale-stage parallelism.** Should the upscale stage use
   the same N as the tracking stage (one upscale segment per tracking
   segment) or its own memory-bounded segmentation? Probably **same N**
   for cleanliness, BUT if a tracking segment is very long, the upscale
   segment for it may exceed memory bounds — needs verification.
5. **What stage_name to use for pre-scan?** New `PipelineStage.PRESCAN`
   value or reuse `DETECT` with a `reason: "prescan"`? New value is
   cleaner for analytics and `progress_percent` band assignment.
6. **Pre-scan progress band.** If pre-scan is added at the front, every
   downstream stage's progress band shifts. Options: (a) compress
   download to 0-5% and detect to 5-15% to free up a pre-scan band; (b)
   make pre-scan implicit within `DETECT` so bands don't move. (a) is
   more honest, (b) preserves backward compat.

---

## 7. Migration plan

Phased rollout to keep production stable:

- **Phase A (this PR, 2026-05-25):** Ship sequential tracking +
  parallel upscale. V1 envelope unchanged. Aggregator at upscale stage
  only. **Done.**
- **Phase B (future, shadow mode):** Implement Stage 0 pre-scan but
  do NOT surface candidates to the user. Compute the candidate list,
  log it for offline analysis, compare against actual mid-tracking
  losses observed in production. Validate the heuristics.
- **Phase C (future):** Enable batch correction UX. Pre-scan
  candidates become real `AWAITING_CORRECTION` prompts. Tracking is
  still sequential at this stage, but it now starts from human-
  confirmed boxes at every candidate frame, which should reduce the
  mid-track loss rate substantially even before parallel tracking
  lands.
- **Phase D (future):** Enable parallel tracking gated by completed
  pre-scan corrections. Schema bump for per-segment
  `partial_tracking_s3_keys` and `resume_from_frames`. Resume route
  learns the batch-correction body shape. Backend `CHECKPOINT_DATA_SCHEMA_V1`
  bumps.

Each phase is independently reversible behind a `config.prescan_*`
feature flag.

---

## 8. Acceptance criteria for the future upgrade

When this work eventually ships, the implementer must verify:

1. **V1 envelope compliance.** Every stage write still validates
   against the V1 envelope (`schema_version`, `pending_detection` |
   `pending_detections`, `artifacts`, `worker_state`), even when
   `artifacts` has the per-segment dict shape.
2. **Per-segment AWAITING_CORRECTION satisfies §4.1.** A single
   segment hitting mid-track loss MUST still write a complete `track`
   checkpoint, set `AWAITING_CORRECTION`, release worker capacity, and
   permit `POST /jobs/{job_id}/resume` to create a replacement.
3. **SSE consumers see coherent progress.** During pre-scan, batch
   correction, and parallel tracking the SSE stream (
   `GET /jobs/{job_id}/events`) emits monotonic in-band progress with
   no regression and no >100% values. The aggregator pattern from this
   PR is the model.
4. **Old replay-from-checkpoint scenarios still work.** A pre-existing
   job that was written under the V1-without-pre-scan shape can still
   resume / replay / be picked up by the reconciler. Schema readers
   must treat the absence of `pending_detections` / per-segment
   artifacts as the V1 single-correction shape.
5. **Backend `CHECKPOINT_DATA_SCHEMA_V1` and engine `service/checkpoints.py`
   stay in lockstep.** The duplicate-of-record contract in
   `working_log/contracts/bjj_backend/` is updated atomically with the
   engine builders, with `bjj-vision-backend/.../CHECKPOINT_DATA_SCHEMA_V1.md`
   leading per the addendum's "duplicate of" stanza.

---

## 9. Reference links

- `working_log/contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md`
- `working_log/contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md`
- `working_log/knowledge-base/insights/2026-04-25-job-start-resume-workflow-reference.md`
- `working_log/knowledge-base/insights/2026-05-25-parallel-upscale-progress-aggregator.md`
- PR #2 on `Stanxxy/video-upscale`
  (https://github.com/Stanxxy/video-upscale/pull/2) — the PR whose
  rejected first design motivated this requirement doc.
- Future PR placeholder: the eventual implementation will link back
  here.
