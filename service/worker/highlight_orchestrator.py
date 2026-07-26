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

**Batched publish (2026-07-26 CEO re-scope, AC8-11; Brooks's architecture
ruling §2a, correcting his own original §2):** SNS publish is NO LONGER
per-highlight-incremental. The per-chunk loop below only ANALYZES and
COLLECTS each highlight's clip (persisted onto that chunk's own
``HIGHLIGHT_CHUNK`` checkpoint). After the full loop completes, the whole
match's clips are reconstructed from checkpoints (works identically
whether a chunk was freshly analyzed this run or resumed past), seam-
deduped (``service/worker/seam_dedup.py``), majority-vote reconciled
(``service/worker/majority_vote.py`` — the founder-approved per-match actor
consensus), and ONLY THEN published, once, in a single finalize loop that
is itself resumable at candidate granularity (the ``HIGHLIGHT_PUBLISH``
checkpoint stage now carries real per-candidate progress rows, not just a
terminal marker). SSE progress (``service/routes/events.py``) is untouched
— it stays streaming, per-chunk, for job-progress UI visibility; only the
SNS/``VideoEventCandidate`` delivery path moved.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from google import genai
from google.genai import types

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import (
    build_highlight_chunk_completed,
    build_highlight_publish_completed,
    build_highlight_publish_progress,
)
from service.checkpoints.highlight_resume import build_highlight_resume_plan
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import JobStatus, TrackRequest
from service.pipelines import executors, gemini_retry, gemini_upload
from service.pipelines.registry import get_default
from service.sns import SNSPublisher, clip_to_axis_only_event
from service.worker import majority_vote, seam_dedup
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


def _checkpoint_sort_key(cp: dict) -> tuple[float, int]:
    """Rows without a real ``updated_at`` sort first (so an explicit
    timestamp always wins when both exist), ``id(cp)`` breaks ties without
    ever comparing two ``None``s directly (which would raise) — mirrors
    ``service/checkpoints/query.py::_checkpoint_ts``'s own tie-break
    discipline exactly; duplicated intentionally rather than importing a
    private helper across a module-privacy boundary."""
    ts = cp.get("updated_at")
    if isinstance(ts, datetime):
        return (ts.timestamp(), id(cp))
    return (0.0, id(cp))


def _clips_by_chunk_from_checkpoints(checkpoints: list[dict], chunks_total: int) -> list[list[dict]]:
    """2026-07-26 CEO batched-publish re-scope (AC8-11) — reconstructs every
    chunk's OWN collected clip list from its durable ``HIGHLIGHT_CHUNK``
    checkpoint (``build_highlight_chunk_completed``'s additive ``clips``
    field). The single source of truth for the match's full candidate set
    at finalize time, regardless of whether a given chunk was freshly
    re-analyzed THIS run or its clips are being recovered from an earlier,
    resumed-past run's completed checkpoint — this function never
    distinguishes the two, by design (see ``run_highlight_job``).

    An OLDER-format ``HIGHLIGHT_CHUNK`` checkpoint written before the
    ``clips`` field existed (or an in-flight job crossing this exact
    deploy boundary) contributes an empty list for that chunk — additive-
    only, never a crash, never fabricated clips. Multiple rows for the
    SAME ``chunk_index`` (should not normally happen — resume skips
    already-completed chunks — but the schema permits it) resolve to the
    LATEST write by ``updated_at``, same "last INSERT wins" convention as
    ``build_highlight_resume_plan``'s own ingest-checkpoint handling."""
    by_chunk: dict[int, list[dict]] = {}
    for cp in sorted(checkpoints, key=_checkpoint_sort_key):
        if cp.get("stage_name") != PipelineStage.HIGHLIGHT_CHUNK.value:
            continue
        if not cp.get("completed"):
            continue
        data = cp.get("checkpoint_data") or {}
        idx = data.get("chunk_index")
        if not isinstance(idx, int):
            continue
        clips = (data.get("artifacts") or {}).get("clips") or []
        by_chunk[idx] = clips
    return [by_chunk.get(i, []) for i in range(chunks_total)]


def _published_candidate_keys_from_checkpoints(checkpoints: list[dict]) -> frozenset[str]:
    """Every ``candidate_key`` already recorded as published by a
    ``build_highlight_publish_progress`` row from a PRIOR run — the
    resumable-finalize-loop's own "already sent, never re-send" set
    (Brooks's named new requirement, §2a)."""
    keys: set[str] = set()
    for cp in checkpoints:
        if cp.get("stage_name") != PipelineStage.HIGHLIGHT_PUBLISH.value:
            continue
        data = cp.get("checkpoint_data") or {}
        if data.get("reason") != "highlight_publish_candidate":
            continue
        candidate_key = (data.get("artifacts") or {}).get("candidate_key")
        if candidate_key:
            keys.add(candidate_key)
    return frozenset(keys)


def _publish_terminal_done(checkpoints: list[dict]) -> bool:
    """Whether the terminal ``HIGHLIGHT_PUBLISH`` checkpoint
    (``build_highlight_publish_completed``, ``completed=True``) was already
    written by a prior run — mirrors the tracking pipeline's own established
    "resume skips a fully-completed terminal-publish stage" convention
    (``worker/orchestrator.py``'s analogous resume-terminal-publish check)."""
    for cp in checkpoints:
        if cp.get("stage_name") != PipelineStage.HIGHLIGHT_PUBLISH.value:
            continue
        if not cp.get("completed"):
            continue
        data = cp.get("checkpoint_data") or {}
        if data.get("reason") == "highlight_publish_completed":
            return True
    return False


async def run_highlight_job(
    job_id: str,
    request: TrackRequest,
    config: ServiceConfig,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
) -> None:
    """Run the v2 highlight job: HIGHLIGHT_INGEST -> outer chunk loop
    (``executors.run_pipeline`` per chunk, collect-only, ``HIGHLIGHT_CHUNK``
    checkpoint) -> seam dedup + majority-vote reconciliation -> finalize
    publish loop (resumable at candidate granularity) -> HIGHLIGHT_PUBLISH
    -> COMPLETED.

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

        highlights_found_total = 0

        # ------------------------------------------------------------- #
        # Phase 1 — per-chunk analyze loop: ANALYZE + COLLECT only. No
        # publish, no dedup, no attribution counting here anymore (2026-
        # 07-26 CEO batched-publish re-scope) — every one of those is now a
        # match-scope concern computed once, after this loop, over the
        # complete, checkpoint-reconstructed clip set (Phase 2 onward).
        # ------------------------------------------------------------- #
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
            chunk_error_count = 0
            this_chunk_clips: list[dict] = []

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
                    # Stable per-candidate identity (survives a resume —
                    # derived from data persisted on THIS chunk's own
                    # checkpoint, never from in-memory-only run state) —
                    # see build_highlight_publish_progress's docstring.
                    highlight_index = event.get("highlight_index")
                    clip = dict(clips[0])
                    clip["_chunk_index"] = chunk_index
                    clip["_highlight_index"] = highlight_index
                    clip["_candidate_key"] = f"{chunk_index}:{highlight_index}"
                    this_chunk_clips.append(clip)

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
                    # Publish no longer happens per-chunk at all — see
                    # build_highlight_chunk_completed's own docstring.
                    highlights_published=0,
                    gemini_file_uri=ingest.gemini_file_uri,
                    worker_state=_make_worker_state(
                        progress_percent=_chunk_progress_pct(chunk_index + 1, chunks_total),
                        stage_progress_fraction=1.0,
                    ),
                    clips=this_chunk_clips,
                ),
            )

        # ------------------------------------------------------------- #
        # Phase 2 — reconstruct the FULL match's clip set from durable
        # checkpoints (works identically for chunks freshly analyzed this
        # run and chunks recovered from an earlier, resumed-past run).
        # ------------------------------------------------------------- #
        checkpoints_after_chunks = await jobs_store.get_all_checkpoints(job_id)
        clips_by_chunk = _clips_by_chunk_from_checkpoints(checkpoints_after_chunks, chunks_total)

        # Phase 3 — whole-match seam dedup (AC6; service/worker/seam_dedup.py).
        deduped_clips, seam_duplicates_dropped = seam_dedup.dedup_match_clips(clips_by_chunk, chunks)
        if seam_duplicates_dropped:
            logger.info(
                "Job %s: seam dedup dropped %d duplicate highlight(s) across outer-chunk boundaries",
                job_id, seam_duplicates_dropped,
            )

        # Phase 4 — per-match majority-vote actor reconciliation (AC8-11;
        # service/worker/majority_vote.py) — overwrites player_id/
        # player_name/identity_uncertain/actor_sentinel on every clip.
        reconciled_clips, flip_count = majority_vote.reconcile_match_actors(deduped_clips)
        cross_highlight_flip_rate = (flip_count / len(reconciled_clips)) if reconciled_clips else 0.0

        # Phase 5 — attribution metrics, computed ONCE over the final
        # (deduped + reconciled) clip set — never double-counted against a
        # seam duplicate, never measuring pre-reconciliation identities.
        player_id_counts: dict[str, int] = {}
        sentinel_count = 0
        identity_uncertain_count = 0
        for clip in reconciled_clips:
            pid = clip.get("player_id")
            if pid:
                player_id_counts[pid] = player_id_counts.get(pid, 0) + 1
            elif clip.get("actor_sentinel"):
                sentinel_count += 1
            if clip.get("identity_uncertain"):
                identity_uncertain_count += 1

        # ------------------------------------------------------------- #
        # Phase 6 — finalize publish: ONE pass over the reconciled clips,
        # resumable at candidate granularity (Brooks's named new
        # requirement, §2a) — a candidate already recorded as published by
        # an earlier, crashed-mid-batch run is never re-sent.
        # ------------------------------------------------------------- #
        already_published_keys = _published_candidate_keys_from_checkpoints(checkpoints_after_chunks)
        publish_terminal_done = _publish_terminal_done(checkpoints_after_chunks)
        if publish_terminal_done:
            logger.info(
                "Job %s: HIGHLIGHT_PUBLISH already terminal on a prior run — rebuilding the "
                "audit view only, no new SNS calls", job_id,
            )
        elif already_published_keys:
            logger.info(
                "Job %s: resuming finalize publish — %d candidate(s) already published on a "
                "prior run, skipping those, publishing the remainder", job_id, len(already_published_keys),
            )

        published_events: list[dict] = []
        total_candidates = len(reconciled_clips)
        for event_idx, clip in enumerate(reconciled_clips, start=1):
            candidate_key = clip.get("_candidate_key")
            try:
                candidate_event = clip_to_axis_only_event(clip, video_id)
            except Exception as e:  # noqa: BLE001 — real mapping failure, logged, non-fatal
                logger.error(
                    "Job %s: failed to build candidate event for %s: %s", job_id, candidate_key, e,
                )
                continue

            if publish_terminal_done or candidate_key in already_published_keys:
                # Already sent on an earlier run (or the whole stage is
                # already terminal) — NEVER re-publish; just fold it into
                # this run's audit view.
                published_events.append(candidate_event.model_dump(mode="json"))
                continue

            if sns is None:
                continue  # no topic configured — analyzed but never published, same as before

            try:
                sns.publish_axis_only_event(candidate_event, event_index=event_idx)
            except Exception as e:  # noqa: BLE001 — real publish failure, logged, non-fatal
                logger.error(
                    "Job %s: publish failed for candidate %s: %s", job_id, candidate_key, e,
                )
                continue

            published_events.append(candidate_event.model_dump(mode="json"))
            if candidate_key:
                await jobs_store.write_checkpoint(
                    job_id, PipelineStage.HIGHLIGHT_PUBLISH, False,
                    build_highlight_publish_progress(
                        candidate_key=candidate_key, event_index=event_idx,
                        worker_state=_make_worker_state(
                            progress_percent=_chunk_progress_pct(event_idx, total_candidates),
                            stage_progress_fraction=(event_idx / total_candidates) if total_candidates else 1.0,
                        ),
                    ),
                )

        attribution_metrics = {
            "player_id_counts": player_id_counts,
            "sentinel_count": sentinel_count,
            "identity_uncertain_count": identity_uncertain_count,
            "total_published": len(published_events),
            # Additive telemetry (AC10/Gate 4) — audit/SSE-only, never
            # coach-facing (OQ3): fraction of the match's highlights whose
            # own raw attribution disagreed with the match's majority-vote
            # winner. A tied/no-majority match counts as 100% (see
            # majority_vote.reconcile_match_actors's own docstring).
            "cross_highlight_flip_rate": round(cross_highlight_flip_rate, 4),
            "seam_duplicates_dropped": seam_duplicates_dropped,
        }

        s3 = _make_s3(config)
        events_key = f"{base_key}_v2_events.json"
        result_s3_uri: Optional[str] = None
        sns_completion_sent = False

        if publish_terminal_done:
            # Both the consolidated audit JSON and the analysis_complete
            # notification were already sent on the run that wrote the
            # terminal checkpoint — resending analysis_complete here would
            # duplicate the completion email (product memory: "users are
            # notified by email at completion" — Brooks §2a); resending the
            # audit JSON is harmless but pointless. Skip both, matching the
            # tracking pipeline's own established "resume skips a fully-
            # completed terminal-publish stage" convention.
            logger.info(
                "Job %s: skipping consolidated audit re-upload and analysis_complete re-publish "
                "(already sent on a prior run)", job_id,
            )
        else:
            # design §5.4: ONE consolidated audit JSON of every published
            # event — not load-bearing for the backend (already delivered
            # via SNS), operational hygiene only.
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
