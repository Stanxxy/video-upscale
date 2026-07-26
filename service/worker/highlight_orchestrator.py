"""``run_highlight_job`` — S12 Phase 1b production wiring design §1: the v2
``highlight-scan-critique-analyze`` production job entrypoint, dispatched
unconditionally from ``service/routes/scheduling.py::_schedule_job`` (item 15
— the single call-site swap; ``worker/orchestrator.py::run_job``, the
tracking pipeline, is no longer scheduled from there — decision 2, dormant
in place, not deleted).

**Adapter, not a rewrite** (design §0/§1.4): this module does NOT
reimplement any Gemini-calling logic. It is a thin ``async for event in
executors.run_pipeline(...): dispatch(event)`` loop wrapped in an outer
chunk loop over match duration, reusing ``executors.run_pipeline`` byte-for-
byte (the same function the QA playground drives via
``routes/qa_pipeline.py``) — the budget-cap gate, offset-quantization,
synthetic-seam preroll/postroll clamping, etc. all apply for free.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
from typing import Optional
from uuid import UUID, uuid4

from google import genai
from google.genai import types

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import build_highlight_chunk_completed, build_highlight_publish_completed
from service.checkpoints.highlight_resume import build_highlight_resume_plan
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import JobStatus, TrackRequest
from service.pipelines import executors, gemini_retry, gemini_upload
from service.pipelines.registry import get_default
from service.sns import SNSPublisher, clip_to_axis_only_event
from service.worker import seam_dedup
from service.worker.helpers import _is_cancelled, _make_s3
from service.worker.progress import _make_worker_state
from service.worker.stages.highlight_ingest import run_highlight_ingest_stage

logger = logging.getLogger("service.worker")

# The ONE v2 pipeline this orchestrator drives — see the S12 Phase 1b
# production wiring design's core thesis (§0): one pipeline implementation,
# two callers (QA playground, this production orchestrator).
PIPELINE_ID = "highlight-scan-critique-analyze"

# 2026-07-26-engine13-rescope-single-call-cutover.md AC5 / Gate 5
# (VERDICTS_V2.md) — BACKWARD-ONLY overlap at every outer-chunk seam, sized
# as the derivation the founder-approved spec specifies:
# critique_backpad_s's own max (30s) + preroll_s's own max (15s) = 45s. This
# is NOT a measured-optimal value — Gate 5's own accuracy addendum was ruled
# "too noise-dominated to support a confident directional claim" across two
# model generations (VERDICTS_V2.md §1) — it ships on first-principles
# chunking-boundary grounds: production's real 720s zero-overlap grid can
# genuinely lose setup evidence across a seam, independent of any model.
HIGHLIGHT_OUTER_CHUNK_OVERLAP_S: float = 45.0


def _outer_chunks(
    duration_sec: float, outer_chunk_scope_sec: int, overlap_s: float = HIGHLIGHT_OUTER_CHUNK_OVERLAP_S,
) -> list[tuple[float, float]]:
    """Outer chunks on a contiguous grid, with a BACKWARD-ONLY ``overlap_s``
    read at every seam after the first (2026-07-26 re-scope AC5 — see
    ``HIGHLIGHT_OUTER_CHUNK_OVERLAP_S``). Each chunk's END stays exactly at
    the nominal grid boundary (no forward overlap — none was measured or
    directed, OQ5); each chunk's START (after the first) is pulled
    ``overlap_s`` EARLIER than its nominal grid position, clamped to never go
    below 0. This means adjacent chunks' sent video windows now genuinely
    overlap by up to ``overlap_s`` at every seam — this is a DELIBERATE
    change from the prior zero-overlap grid, and is exactly why
    ``run_highlight_job`` now runs ``seam_dedup`` (AC6) over the highlights
    that land in that overlap band: a real event sitting there can be
    independently discovered and reported by both the trailing highlight_scan
    of chunk k and the leading highlight_scan of chunk k+1."""
    if duration_sec <= 0 or outer_chunk_scope_sec <= 0:
        return []
    n = math.ceil(duration_sec / outer_chunk_scope_sec)
    chunks: list[tuple[float, float]] = []
    for i in range(n):
        nominal_start = float(i * outer_chunk_scope_sec)
        end = float(min((i + 1) * outer_chunk_scope_sec, duration_sec))
        start = max(0.0, nominal_start - overlap_s) if i > 0 else nominal_start
        chunks.append((start, end))
    return chunks


def _chunk_progress_pct(position: int, chunks_total: int) -> float:
    if chunks_total <= 0:
        return 100.0
    return round(100.0 * min(position, chunks_total) / chunks_total, 1)


async def run_highlight_job(
    job_id: str,
    request: TrackRequest,
    config: ServiceConfig,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
) -> None:
    """Run the v2 highlight job: HIGHLIGHT_INGEST -> outer chunk loop
    (``executors.run_pipeline`` per chunk, per-highlight SNS publish,
    ``HIGHLIGHT_CHUNK`` checkpoint) -> HIGHLIGHT_PUBLISH -> COMPLETED.

    Same top-level try/except/finally SHAPE as ``worker/orchestrator.py::
    run_job`` (cancellation -> CANCELLED, uncaught exception -> FAILED,
    ``finally`` cleanup) — reused as a pattern, NOT a fork of that file's
    tracking-specific stage machinery (no SAM2/YOLO/upscale imports at all).
    """
    work_dir = os.path.join(config.temp_dir, job_id)
    os.makedirs(work_dir, exist_ok=True)
    gemini_client: Optional[genai.Client] = None
    gemini_file_name: Optional[str] = None

    logger.info(
        "Job %s: highlight_job starting work_dir=%s input=s3://%s/%s",
        job_id, work_dir, request.bucket, request.key,
    )

    try:
        await jobs_store.set_state(job_id, JobState.RUNNING)

        gemini_client = genai.Client(
            api_key=config.gemini_api_key,
            http_options=types.HttpOptions(timeout=config.gemini_request_timeout_ms),
        )

        ingest = await run_highlight_ingest_stage(
            job_id, request, config, jobs_store, work_dir, gemini_client,
        )
        gemini_file_name = ingest.gemini_file_name

        checkpoints = await jobs_store.get_all_checkpoints(job_id)
        resume_plan = build_highlight_resume_plan(checkpoints)
        resume_from_chunk_index = resume_plan["resume_from_chunk_index"]

        chunks = _outer_chunks(ingest.video_duration_sec, config.outer_chunk_scope_sec)
        chunks_total = len(chunks)
        if resume_from_chunk_index:
            logger.info(
                "Job %s: resuming from chunk_index=%d/%d (already-completed chunks skipped)",
                job_id, resume_from_chunk_index, chunks_total,
            )

        retry_config = gemini_retry.GeminiRetryConfig.from_service_config(config)
        pipeline = get_default(PIPELINE_ID)

        topic_arn = request.sns_topic_arn or config.sns_topic_arn
        sns: Optional[SNSPublisher] = None
        if topic_arn:
            sns = SNSPublisher(
                config.aws_region, topic_arn,
                endpoint_url=config.s3_endpoint_url or None,
                access_key_id=config.aws_access_key_id or None,
                secret_access_key=config.aws_secret_access_key or None,
            )
        else:
            logger.warning("Job %s: no SNS topic configured — highlights will be analyzed but NOT published", job_id)

        video_id: UUID = request.video_id or uuid4()
        output_bucket = request.output_bucket or request.bucket
        base_key = os.path.splitext(request.key)[0]

        published_events: list[dict] = []
        highlights_found_total = 0
        event_index = 0
        # Item 11.5 — pure measurement plumbing. No behavior branches on
        # these numbers anywhere in this function or downstream.
        player_id_counts: dict[str, int] = {}
        sentinel_count = 0
        identity_uncertain_count = 0
        # AC6 seam dedup (see service/worker/seam_dedup.py): the PREVIOUS
        # chunk's own trailing-seam-band clips (already published), carried
        # forward one chunk at a time so the CURRENT chunk can check its own
        # leading-seam-band highlights against them. Empty at job start and
        # after a resume (chunk_index < resume_from_chunk_index chunks are
        # never reprocessed) — a documented, narrow limitation: a resumed
        # job's first processed chunk has no seam-dedup context against the
        # chunk immediately before the resume point.
        prior_seam_clips: list[dict] = []

        for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
            if chunk_index < resume_from_chunk_index:
                continue
            if _is_cancelled(job_id, job_store):
                return

            ctx = executors.RunContext(
                youtube_id=job_id,
                youtube_url=ingest.gemini_file_uri,
                start_sec=int(chunk_start), end_sec=int(chunk_end),
                gemini_api_key=config.gemini_api_key,
                request_timeout_ms=config.gemini_request_timeout_ms,
                retry_config=retry_config,
                video_mime_type=ingest.gemini_file_mime_type,
                player_references=ingest.player_references,
            )
            planned = executors.estimate_run_plan(pipeline, duration_sec=chunk_end - chunk_start)

            highlights_scanned = 0
            highlights_analyzed = 0
            highlights_ditched = 0
            highlights_published = 0
            chunk_error_count = 0
            # This chunk's own clips landing in the seam band it shares with
            # the NEXT chunk — becomes that chunk's `prior_seam_clips`.
            this_chunk_trailing_seam_clips: list[dict] = []

            async for event in executors.run_pipeline(
                pipeline, ctx, planned, budget_cap=config.highlight_pipeline_budget_cap,
            ):
                etype = event.get("type")

                if etype == "highlight_map":
                    highlights_scanned = len(event.get("highlights") or [])

                elif etype == "stage_complete" and event.get("stage_type") == "highlight_scan":
                    highlights_found_total += highlights_scanned
                    await jobs_store.update_highlight_chunk_progress(
                        job_id, PipelineStage.HIGHLIGHT_CHUNK,
                        _chunk_progress_pct(chunk_index, chunks_total),
                        chunk_index=chunk_index, chunks_total=chunks_total,
                        highlights_found_so_far=highlights_found_total,
                    )

                elif etype == "highlight_result":
                    status = event.get("status")
                    if status == "ditched":
                        # Legacy branch — the 2026-07-26 single-call cutover
                        # removed the validator/ditch authority entirely
                        # (executors.highlight_analyze_node's own docstring),
                        # so this never fires in production anymore; kept for
                        # wire-contract stability (a "status" key consumer
                        # should not need a special case if a future analyze
                        # node ever reintroduces a ditch verdict).
                        highlights_ditched += 1
                        continue
                    highlights_analyzed += 1
                    clips = event.get("clips") or []
                    if not clips:
                        continue
                    clip = clips[0]
                    clip_start = clip.get("start_s")
                    clip_end = clip.get("end_s")

                    # AC6 seam dedup: if this highlight lands in the LEADING
                    # overlap band shared with the PREVIOUS chunk, check
                    # whether that chunk already published the same real
                    # event (temporal-proximity + class-compatible, actor
                    # NEVER consulted — seam_dedup.py). The already-published
                    # record is never touched/rewritten; this candidate is
                    # simply never published (AC6: "not re-published").
                    is_seam_duplicate = False
                    if (
                        chunk_index > 0 and clip_start is not None and clip_end is not None
                        and seam_dedup.in_seam_band(clip_start, clip_end, chunk_start, chunks[chunk_index - 1][1])
                    ):
                        match = seam_dedup.find_seam_duplicate(clip, prior_seam_clips)
                        if match is not None:
                            is_seam_duplicate = True
                            logger.info(
                                "Job %s: chunk %d: highlight %s [%.2f-%.2f] suppressed as a seam "
                                "duplicate of a highlight already published by chunk %d",
                                job_id, chunk_index, event.get("highlight_index"),
                                clip_start, clip_end, chunk_index - 1,
                            )

                    # Track this chunk's own trailing-seam-band clips
                    # regardless of the dedup outcome above — chunk k+1 needs
                    # the REAL published record to check against, and a
                    # highlight THIS chunk itself just suppressed as a
                    # duplicate is never a valid anchor (it was never
                    # published).
                    if (
                        not is_seam_duplicate and chunk_index + 1 < chunks_total
                        and clip_start is not None and clip_end is not None
                        and seam_dedup.in_seam_band(clip_start, clip_end, chunks[chunk_index + 1][0], chunk_end)
                    ):
                        this_chunk_trailing_seam_clips.append(clip)

                    if is_seam_duplicate:
                        continue

                    pid = clip.get("player_id")
                    if pid:
                        player_id_counts[pid] = player_id_counts.get(pid, 0) + 1
                    elif clip.get("actor_sentinel"):
                        sentinel_count += 1
                    if clip.get("identity_uncertain"):
                        identity_uncertain_count += 1

                    if sns is None:
                        continue

                    try:
                        candidate_event = clip_to_axis_only_event(clip, video_id)
                        event_index += 1
                        sns.publish_axis_only_event(candidate_event, event_index=event_index)
                        highlights_published += 1
                        published_events.append(candidate_event.model_dump(mode="json"))
                    except Exception as e:  # noqa: BLE001 — real publish failure, logged, non-fatal
                        logger.error(
                            "Job %s: chunk %d: publish failed for highlight %s: %s",
                            job_id, chunk_index, event.get("highlight_index"), e,
                        )

                elif etype == "error":
                    # design §1.4: one bad Gemini call degrades, never
                    # crashes the run — matches the existing QA-layer
                    # discipline this pipeline already follows internally.
                    chunk_error_count += 1
                    logger.warning(
                        "Job %s: chunk %d: pipeline error: %s",
                        job_id, chunk_index, event.get("message"),
                    )

            if chunk_error_count:
                logger.info(
                    "Job %s: chunk %d finished with %d error event(s) (degraded, not aborted)",
                    job_id, chunk_index, chunk_error_count,
                )

            await jobs_store.write_checkpoint(
                job_id, PipelineStage.HIGHLIGHT_CHUNK, True,
                build_highlight_chunk_completed(
                    chunk_index=chunk_index, chunks_total=chunks_total,
                    highlights_scanned=highlights_scanned,
                    highlights_analyzed=highlights_analyzed,
                    highlights_ditched=highlights_ditched,
                    highlights_published=highlights_published,
                    gemini_file_uri=ingest.gemini_file_uri,
                    worker_state=_make_worker_state(
                        progress_percent=_chunk_progress_pct(chunk_index + 1, chunks_total),
                        stage_progress_fraction=1.0,
                    ),
                ),
            )
            # AC6: hand this chunk's own trailing-seam-band clips to the NEXT
            # chunk's dedup check (see the `prior_seam_clips` docstring above).
            prior_seam_clips = this_chunk_trailing_seam_clips

        attribution_metrics = {
            "player_id_counts": player_id_counts,
            "sentinel_count": sentinel_count,
            "identity_uncertain_count": identity_uncertain_count,
            "total_published": len(published_events),
        }

        # design §5.4: ONE consolidated audit JSON of every published event —
        # not load-bearing for the backend (already delivered live via SNS),
        # operational hygiene only.
        s3 = _make_s3(config)
        events_key = f"{base_key}_v2_events.json"
        result_s3_uri: Optional[str] = None
        try:
            loop = asyncio.get_event_loop()
            result_s3_uri = await loop.run_in_executor(
                None, s3.upload_json,
                {
                    "video_id": str(video_id),
                    "job_id": job_id,
                    "events": published_events,
                    "attribution_metrics": attribution_metrics,
                },
                output_bucket, events_key,
            )
        except Exception as e:  # noqa: BLE001 — audit artifact is hygiene, never blocks completion
            logger.warning("Job %s: failed to write consolidated audit JSON: %s", job_id, e)

        sns_completion_sent = False
        if sns is not None:
            try:
                sns.publish_analysis_complete(
                    video_id, job_id, total_event_count=len(published_events),
                    result_s3_uri=result_s3_uri,
                )
                sns_completion_sent = True
            except Exception as e:  # noqa: BLE001 — real publish failure, logged, non-fatal
                logger.error("Job %s: analysis_complete publish failed: %s", job_id, e)

        await jobs_store.write_checkpoint(
            job_id, PipelineStage.HIGHLIGHT_PUBLISH, True,
            build_highlight_publish_completed(
                sns_topic_arn=topic_arn or "",
                sns_event_count=len(published_events),
                sns_completion_sent=sns_completion_sent,
                result_s3_uri=result_s3_uri,
                worker_state=_make_worker_state(progress_percent=100.0, stage_progress_fraction=1.0),
            ),
        )
        await jobs_store.update_highlight_chunk_progress(
            job_id, PipelineStage.HIGHLIGHT_PUBLISH, 100.0,
            chunk_index=chunks_total, chunks_total=chunks_total,
            highlights_found_so_far=highlights_found_total,
            attribution_metrics_json=json.dumps(attribution_metrics),
        )

        await job_store.update_job(
            job_id, status=JobStatus.COMPLETED, progress_percent=100.0,
            result_bucket=output_bucket, result_key=events_key,
        )
        await jobs_store.set_state(job_id, JobState.COMPLETED)
        logger.info(
            "Job %s: highlight_job completed (%d event(s) published across %d chunk(s))",
            job_id, len(published_events), chunks_total,
        )

    except asyncio.CancelledError:
        logger.info("Job %s cancelled (client disconnected)", job_id)
        await job_store.update_job(job_id, status=JobStatus.CANCELLED)
        await jobs_store.set_state(job_id, JobState.CANCELLED)

    except Exception as e:
        logger.exception("Job %s highlight_job failed", job_id)
        await job_store.update_job(job_id, status=JobStatus.FAILED, error_message=str(e))
        await jobs_store.set_state(job_id, JobState.FAILED, error_message=str(e))

    finally:
        if gemini_client is not None and gemini_file_name:
            await gemini_upload.delete_gemini_file(gemini_client, gemini_file_name)
        shutil.rmtree(work_dir, ignore_errors=True)
