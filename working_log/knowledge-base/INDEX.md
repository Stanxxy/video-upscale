# Knowledge Base Index

Auto-maintained index of design decisions, technical insights, and requirements
for the whole-video-analysis vision engine.

See `bjj-team-meta` skill for how to add entries.

---

## Decisions

- [2026-03-15 - RF-DETR + SAM2 Hybrid Tracking Architecture](decisions/2026-03-15-rfdetr-sam2-hybrid-tracking.md) — tags: tracking, architecture, ml
- [2026-03-15 - Taxonomy Mapper for Frontend Enum Bridging](decisions/2026-03-15-taxonomy-mapper-frontend-bridge.md) — tags: service, api, taxonomy
- [2026-03-15 - Single-Job Concurrency Model](decisions/2026-03-15-single-job-concurrency.md) — tags: service, architecture, concurrency
- [2026-03-15 - tracking Package Shim to Avoid Name Collision](decisions/2026-03-15-tracking-package-shim.md) — tags: python, packaging, tracking
- [2026-03-21 - SAM3 for Mid-Tracking Re-Detection](decisions/2026-03-21-sam3-redetection-strategy.md) — tags: tracking, architecture, ml, sam3, redetection
- [2026-04-19 - AWS Production Infrastructure Baseline (LocalStack Dev-Only)](decisions/2026-04-19-aws-production-infra-baseline.md) — tags: infra, aws, production, service, governance
- [Legacy Draft - Job Pause and Resume Notes](decisions/job-pause-and-resume.md) — tags: service, resume, draft, unstructured

## Insights

- [2026-03-15 - DINOv2 + Color Histogram Re-ID Strategy](insights/2026-03-15-dinov2-color-histogram-reid.md) — tags: tracking, ml, identity
- [2026-04-19 - Service Open Questions and TODO Reality Snapshot](insights/2026-04-19-service-open-questions-and-todos.md) — tags: service, lifecycle, resume, todo, open-questions
- [2026-04-25 - Job Start and Resume Workflow Reference](insights/2026-04-25-job-start-resume-workflow-reference.md) — tags: service, lifecycle, resume, crash-recovery, keyspaces, refactor
- [2026-05-01 - Lifecycle Resume and Recovery Implementation](insights/2026-05-01-lifecycle-resume-recovery-implementation.md) — tags: service, lifecycle, resume, recovery, keyspaces, implementation
- [2026-05-03 - Worker Checkpoint Standardization Implementation](insights/2026-05-03-worker-checkpoint-standardization-implementation.md) — tags: service, worker, checkpoints, schema, recovery, tests, implementation
- [2026-05-10 - Recovery index bucket scan window (overnight stale jobs)](insights/2026-05-10-recovery-index-bucket-scan-window.md) — tags: service, recovery, reconciler, keyspaces, job_recovery_index, ops
- [2026-05-10 - MPS upscale memory: PyTorch cache vs Gemini buffer](insights/2026-05-10-mps-upscale-memory-empty-cache.md) — tags: service, upscale, pytorch, mps, memory, restorer, ops
- [2026-05-10 - Job handoff chain debugging playbook (Keyspaces + S3)](insights/2026-05-10-job-handoff-chain-debug-playbook.md) — tags: service, lifecycle, resume, keyspaces, checkpoints, s3, debugging, ops
- [2026-05-17 - DGX Spark production target host (`ssh gx10`)](insights/2026-05-17-dgx-spark-production-target.md) — tags: infra, ops, dgx-spark, cuda, blackwell, production
- [2026-05-25 - Parallel-upscale progress aggregator pattern](insights/2026-05-25-parallel-upscale-progress-aggregator.md) — tags: service, parallel-upscale, keyspaces, progress, asyncio, regression-prevention
- [2026-05-25 - `run_coroutine_threadsafe` must `add_done_callback`](insights/2026-05-25-run-coroutine-threadsafe-must-add-done-callback.md) — tags: service, asyncio, keyspaces, progress, regression-prevention
- [2026-05-25 - `_detect_and_request_boxes` short-circuit in headless mode](insights/2026-05-25-headless-detect-yolo-short-circuit.md) — tags: service, parallel-tracking, tracking, ml, regression-prevention
- [2026-05-25 - Bootstrap recovery on startup (auto-resume latency fix)](insights/2026-05-25-bootstrap-recovery-on-startup.md) — tags: service, recovery, reconciler, keyspaces, bootstrap-recovery, observability, regression-prevention
- [2026-05-31 - Clean-code refactor package split](insights/2026-05-31-clean-code-refactor-package-split.md) — tags: service, refactor, clean-code, packaging, regression-prevention, tests
- [2026-07-02 - QA VLM Studio: backend proxy + no-download event analysis + server-side frame grab for segmentation](../../working_log/knowledge-base/decisions/2026-07-02-qa-vlm-studio-backend-proxy-architecture.md) — canonical entry lives in the umbrella KB (`bjj-proj/working_log/knowledge-base/`); tags: qa-tooling, vlm, gemini, api-contract. Companion insight: [INS-048 — YouTube IFrame seek/pause poller race](../../working_log/knowledge-base/insights/INS-048-youtube-iframe-seek-pause-poller-race.md).

## Requirements

- [2026-04-19 - Job Pause and Resume Open Questions](requirements/2026-04-19-job-pause-and-resume-open-questions.md) — tags: service, resume, lifecycle, keyspaces
- [2026-04-26 - Job Lifecycle Resume Refactor Plan](requirements/2026-04-26-job-lifecycle-resume-refactor-plan.md) — tags: service, lifecycle, resume, crash-recovery, keyspaces, governance
- [2026-05-01 - Keyspaces Schema Migration — Job Lineage and Recovery Index](requirements/2026-05-01-keyspaces-schema-migration-job-recovery.md) — tags: keyspaces, schema, migration, ops
- [2026-05-02 - Worker Checkpoint Standardization, Shape Tests, Durable Artifact Recovery](requirements/2026-05-02-worker-checkpoint-standardization-plan.md) — tags: service, worker, checkpoints, schema, recovery, tests
- [2026-05-10 - Chain artifact merge and durable tracking resume (plan)](requirements/2026-05-10-chain-artifact-merge-durable-tracking-resume-plan.md) — tags: service, lifecycle, resume, checkpoints, s3, tracking, handoff, merge
- [2026-05-25 - Pre-scan Segmented Parallel Tracking (Future Upgrade)](requirements/2026-05-25-prescan-segmented-parallel-tracking.md) — tags: tracking, parallel-tracking, prescan, future-upgrade, governance

## Contracts (sharable with bjj-vision-backend)

- [Checkpoint Artifacts V1 Addendum](../contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md) — companion to `CHECKPOINT_DATA_SCHEMA_V1.md`; defines `upscale_analyze`/`annotate`/`upload`/`publish` artifact keys.
- [Job rotation, handoff, and resume](../contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md) — `video_id` / `latest_job` / `replacement_job_id`; normative rules for the analysis service after correction or crash recovery.

## Mistakes
