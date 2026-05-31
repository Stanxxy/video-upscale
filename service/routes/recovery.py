"""Startup recovery and interrupted-job replacement."""

import json
import logging
from datetime import datetime, timezone

from service.analysis_keyspaces_enums import JobState
from service.checkpoints import build_resume_plan, select_correction_checkpoint
from service.models import TrackRequest
from service.routes import state as route_state
from service.routes.resume_job_factory import build_resume_params, create_replacement_job
import service.routes as routes_pkg

logger = logging.getLogger("service.routes")


async def drain_orphan_pending_jobs_on_startup(
    instance_id: str,
    *,
    heartbeat_bucket_hours: int = 24,
) -> None:
    """Re-schedule ``PENDING`` jobs persisted in Keyspaces without a local worker task.

    ``RecoveryManager`` only revives stale ``RUNNING`` / ``INTERRUPTED`` lifecycles.
    Resume/detection handoff creates a replacement row in ``PENDING`` and then calls
    ``_schedule_job``; if the process exits before ``run_job`` flips the row to
    ``RUNNING``, no asyncio task survives restart — this drain reloads the saved
    ``TrackRequest`` and schedules work again.

    Uses a lightweight CAS on ``(job_state, owner_instance_id)`` so two instances
    cannot both take the same pending row.
    """
    if route_state._jobs_store is None or route_state._job_store is None:
        return

    from service.reconciler import RecoveryManager

    buckets = RecoveryManager.heartbeat_buckets_for_scan(
        datetime.now(timezone.utc),
        hours=max(1, int(heartbeat_bucket_hours)),
    )
    try:
        index_rows = await route_state._jobs_store.list_active_recovery_index_rows_newest_first(
            buckets,
        )
    except Exception as e:
        logger.warning("Startup pending drain: recovery index scan failed: %s", e)
        return

    seen: set[str] = set()
    for row in index_rows:
        job_id = row.get("job_id") or ""
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)

        task = route_state._active_tasks.get(job_id)
        if task is not None and not task.done():
            continue

        lifecycle = await route_state._jobs_store.get_lifecycle(job_id)
        if not lifecycle:
            continue
        if lifecycle.get("job_state") != JobState.PENDING.value:
            continue
        if lifecycle.get("replacement_job_id"):
            continue

        expected_owner = lifecycle.get("owner_instance_id") or ""

        request_json = await route_state._jobs_store.get_request(job_id)
        if not request_json:
            logger.warning(
                "Startup pending drain: job %s has lifecycle but no request row; skipping",
                job_id,
            )
            continue

        claimed = await route_state._jobs_store.claim_pending_job_takeover(
            job_id,
            instance_id,
            expected_owner_instance_id=expected_owner,
        )
        if not claimed:
            continue

        try:
            request = TrackRequest(**json.loads(request_json))
        except Exception as e:
            logger.error(
                "Startup pending drain: invalid request JSON for job %s: %s",
                job_id,
                e,
            )
            continue

        await route_state._job_store.hydrate_job(job_id, request)
        routes_pkg._schedule_job(job_id, request)
        logger.info("Startup pending drain: re-scheduled pending job %s", job_id)


async def bootstrap_recovery_on_startup(
    instance_id: str,
    recover_job,
    *,
    heartbeat_bucket_hours: int = 24,
    max_heartbeat_age_hours: int = 24,
) -> None:
    """Single-shot recovery sweep for orphaned ``RUNNING`` / ``INTERRUPTED`` rows.

    Sibling of ``drain_orphan_pending_jobs_on_startup``. The periodic
    ``RecoveryManager`` defaults ``stale_after=90s``, so when this process
    restarts the previous owner's heartbeat (potentially only seconds old) is
    still considered "fresh" and recovery is delayed by 90-120s. That latency
    was perceived by users as "auto-resume is broken."

    This bootstrap runs **once** at lifespan startup with
    ``stale_after_override=0``: every recovery_index ACTIVE row whose lifecycle
    ``last_heartbeat_at`` is strictly in the past is considered stale. Since
    this process has not heartbeated anything yet, that set is exactly the
    orphaned rows from the previous owner. Periodic reconciler defaults
    (90s / 30s) are unchanged — they govern steady-state peer-instance safety.

    ``max_heartbeat_age_hours`` (default 24) bounds the upper edge: rows older
    than this are treated as garbage from a previous deployment and skipped.
    Set to 0 to disable the upper bound.

    Failures are caught and logged with ``exc_info=True`` so startup never
    fails because of a transient recovery glitch.
    """
    if route_state._jobs_store is None:
        return

    from service.reconciler import RecoveryManager

    bucket_hours = max(1, int(heartbeat_bucket_hours))
    max_age_seconds = (
        None if int(max_heartbeat_age_hours) == 0
        else float(max_heartbeat_age_hours) * 3600.0
    )
    logger.info(
        "Bootstrap recovery: scanning %d buckets, stale_after_override=0, "
        "max_heartbeat_age=%s",
        bucket_hours,
        "unbounded" if max_age_seconds is None else f"{max_age_seconds}s",
    )
    manager = RecoveryManager(
        route_state._jobs_store,
        instance_id,
        heartbeat_bucket_hours=bucket_hours,
        max_heartbeat_age_seconds=max_age_seconds,
        recover_job=recover_job,
    )
    try:
        await manager.reconcile_once(stale_after_override=0.0)
    except Exception as e:
        logger.error(
            "Bootstrap recovery: failed: %s", e, exc_info=True,
        )
        return
    logger.info("Bootstrap recovery: complete")


async def recover_interrupted_job(lifecycle: dict) -> None:
    """Create and schedule a replacement job for an interrupted worker job."""
    job_id = lifecycle.get("job_id", "")
    try:
        if lifecycle.get("replacement_job_id"):
            return

        request_json = await route_state._jobs_store.get_request(job_id)
        if not request_json:
            raise RuntimeError(
                f"Original request not found for interrupted job {job_id}",
            )

        resume_params = json.loads(request_json)
        checkpoints = await route_state._jobs_store.get_all_checkpoints(job_id)
        correction_stage, _ = select_correction_checkpoint(checkpoints)
        resume_plan = build_resume_plan(checkpoints)
        if resume_plan.pipeline_already_complete:
            logger.info(
                "Recovery skip: job %s checkpoints show pipeline already terminal-complete",
                job_id,
            )
            route_state._require_write(
                await route_state._jobs_store.set_state(job_id, JobState.COMPLETED),
                "mark stale job completed",
            )
            return

        resume_params = build_resume_params(
            job_id, resume_params, checkpoints, resume_plan,
        )

        await create_replacement_job(
            job_id,
            lifecycle,
            resume_params,
            checkpoints,
            correction_stage,
            claim_expected_state=JobState.INTERRUPTED,
            on_claim_failure="cancel_and_return",
            lifecycle_operation="recovery replacement lifecycle",
            request_operation="recovery replacement request",
            latest_operation="latest recovery replacement",
            old_checkpoint_operation="interrupted job replacement checkpoint",
        )
    except Exception:
        # Re-raise so callers (RecoveryManager, bootstrap) see the failure,
        # but ensure the full stack lands in service.log even when their
        # outer `except Exception` swallows it.
        logger.error(
            "recover_interrupted_job FAILED for job_id=%s", job_id, exc_info=True,
        )
        raise
