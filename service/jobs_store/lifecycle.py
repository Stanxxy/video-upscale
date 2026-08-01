"""Job lifecycle Keyspaces operations."""

from datetime import datetime, timezone
from typing import Any

from service.analysis_keyspaces_enums import (
    JobState,
    PipelineStage,
    parse_job_state,
    states_that_sync_latest_job_row,
    states_with_completed_at,
)
from service.jobs_store._utils import as_utc_aware


class LifecycleMixin:
    _client: Any
    _ks: str
    owned_jobs: set[str]

    def register_owned_job(self, job_id: str) -> None:
        self.owned_jobs.add(job_id)

    def unregister_owned_job(self, job_id: str) -> None:
        self.owned_jobs.discard(job_id)

    @staticmethod
    def heartbeat_bucket_for(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y%m%d%H")

    async def create_lifecycle(
        self,
        job_id: str,
        video_id: str,
        user_id: str,
        origin_job_id: str | None = None,
        parent_job_id: str | None = None,
        replacement_job_id: str | None = None,
        owner_instance_id: str = "",
        progress_percent: float = 0.0,
        current_frame: int = 0,
        total_frames: int = 0,
        pipeline_kind: str | None = None,
    ) -> bool:
        """``pipeline_kind`` (S12 Phase 1b production wiring, design §6.2):
        ``"tracking"`` or ``"highlight_v2"``, additive Cassandra column.
        ``None`` writes an absent/NULL value — ``get_lifecycle`` below
        treats absent as ``"tracking"`` for back-compat with rows created
        before this column existed (a job created before this change is,
        definitionally, a tracking job)."""
        video_id = "" if video_id is None else str(video_id)
        now = datetime.now(timezone.utc)
        q = (
            f"INSERT INTO {self._ks}.job_lifecycle "
            f"(job_id, video_id, user_id, origin_job_id, parent_job_id, "
            f"replacement_job_id, job_state, stage, "
            f"progress_percent, current_frame, total_frames, stage_message, "
            f"error_message, owner_instance_id, pipeline_kind, last_heartbeat_at, started_at, "
            f"updated_at, completed_at) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        )
        ok = await self._client.execute_write(q, [
            job_id, video_id, user_id, origin_job_id, parent_job_id,
            replacement_job_id, JobState.PENDING.value, "",
            progress_percent, current_frame, total_frames, "",
            "", owner_instance_id, pipeline_kind, now, now, now, None,
        ])
        if ok:
            return await self.upsert_recovery_index({
                "job_id": job_id,
                "video_id": video_id,
                "job_state": JobState.PENDING.value,
                "owner_instance_id": owner_instance_id,
                "last_heartbeat_at": now,
            })
        return ok

    async def get_lifecycle(self, job_id: str) -> dict[str, Any] | None:
        q = f"SELECT * FROM {self._ks}.job_lifecycle WHERE job_id = %s"
        rows = await self._client.execute(q, [job_id])
        if not rows:
            return None
        r = rows[0]
        st = parse_job_state(r.job_state)
        from service.analysis_keyspaces_enums import parse_pipeline_stage_optional
        stg = parse_pipeline_stage_optional(r.stage)
        return {
            "job_id": r.job_id,
            "video_id": r.video_id or "",
            "user_id": r.user_id or "",
            "origin_job_id": r.origin_job_id,
            "parent_job_id": getattr(r, "parent_job_id", None),
            "replacement_job_id": getattr(r, "replacement_job_id", None),
            "job_state": st.value,
            "stage": stg.value if stg else "",
            # Absent/NULL (rows created before this column existed) ->
            # "tracking" back-compat default (design §6.2).
            "pipeline_kind": getattr(r, "pipeline_kind", None) or "tracking",
            # S12 Phase 1b (design §1.2/item 11.5) — additive, only ever
            # populated by run_highlight_job; None for every tracking job
            # (same "stays null for v2-only concepts" symmetry as
            # current_frame/total_frames for a v2 job).
            "chunk_index": getattr(r, "chunk_index", None),
            "chunks_total": getattr(r, "chunks_total", None),
            "highlights_found_so_far": getattr(r, "highlights_found_so_far", None),
            "attribution_metrics_json": getattr(r, "attribution_metrics_json", None),
            "progress_percent": r.progress_percent or 0.0,
            "current_frame": r.current_frame or 0,
            "total_frames": r.total_frames or 0,
            "stage_message": r.stage_message or "",
            "error_message": r.error_message or "",
            "owner_instance_id": r.owner_instance_id or "",
            "last_heartbeat_at": as_utc_aware(r.last_heartbeat_at),
            "started_at": as_utc_aware(r.started_at),
            "updated_at": as_utc_aware(r.updated_at),
            "completed_at": as_utc_aware(r.completed_at),
        }

    async def update_progress(
        self,
        job_id: str,
        stage: PipelineStage,
        progress_percent: float,
        current_frame: int = 0,
        total_frames: int = 0,
        stage_message: str = "",
    ) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"stage = %s, progress_percent = %s, "
            f"current_frame = %s, total_frames = %s, stage_message = %s, "
            f"last_heartbeat_at = %s, updated_at = %s "
            f"WHERE job_id = %s"
        )
        return await self._client.execute_write(q, [
            stage.value, progress_percent,
            current_frame, total_frames, stage_message,
            now, now, job_id,
        ])

    async def update_highlight_progress(
        self,
        job_id: str,
        stage: PipelineStage,
        progress_percent: float,
        *,
        chunk_index: int | None = None,
        chunks_total: int | None = None,
        highlights_found_so_far: int | None = None,
        attribution_metrics_json: str | None = None,
        stage_message: str = "",
    ) -> bool:
        """Persist one v2 progress snapshot supplied by the engine writer.

        S12 Phase 1b (design §1.2, item 11.5) — v2-only progress write.
        A DEDICATED method (not an extension of ``update_progress`` above)
        so the dormant tracking pipeline's own UPDATE statement/call sites
        are byte-untouched. ``current_frame``/``total_frames`` are never
        touched here — v2 has no frame-counting concept (they stay at
        whatever ``create_lifecycle`` seeded, i.e. 0). The engine's
        ``HighlightProgressWriter`` owns validation and monotonic floors;
        this method is intentionally a thin persistence boundary."""
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"stage = %s, progress_percent = %s, stage_message = %s, "
            f"chunk_index = %s, chunks_total = %s, highlights_found_so_far = %s, "
            f"attribution_metrics_json = %s, "
            f"last_heartbeat_at = %s, updated_at = %s "
            f"WHERE job_id = %s"
        )
        return await self._client.execute_write(q, [
            stage.value, progress_percent, stage_message,
            chunk_index, chunks_total, highlights_found_so_far,
            attribution_metrics_json,
            now, now, job_id,
        ])

    async def _sync_lifecycle_after_write(
        self,
        job_id: str,
        state: JobState,
    ) -> bool:
        """Refresh recovery/latest projections after a lifecycle UPDATE."""
        lifecycle = await self.get_lifecycle(job_id)
        if not lifecycle:
            return True
        if not await self.upsert_recovery_index(lifecycle):
            return False
        if state in states_that_sync_latest_job_row() and lifecycle.get("video_id"):
            return await self.set_latest(lifecycle["video_id"], job_id, state)
        return True

    async def _write_highlight_terminal(
        self,
        job_id: str,
        state: JobState,
        stage: PipelineStage,
        progress_percent: float,
        stage_message: str,
        *,
        error_message: str = "",
        highlights_found_so_far: int | None = None,
        attribution_metrics_json: str | None = None,
    ) -> bool:
        """Atomically publish one terminal highlight lifecycle snapshot."""
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"job_state = %s, stage = %s, progress_percent = %s, "
            f"current_frame = %s, total_frames = %s, stage_message = %s, "
            f"chunk_index = %s, chunks_total = %s, "
            f"highlights_found_so_far = %s, attribution_metrics_json = %s, "
            f"error_message = %s, last_heartbeat_at = %s, updated_at = %s, "
            f"completed_at = %s WHERE job_id = %s"
        )
        ok = await self._client.execute_write(q, [
            state.value,
            stage.value,
            progress_percent,
            0,
            0,
            stage_message,
            None,
            None,
            highlights_found_so_far,
            attribution_metrics_json,
            error_message,
            now,
            now,
            now,
            job_id,
        ])
        if not ok:
            return False
        return await self._sync_lifecycle_after_write(job_id, state)

    async def complete_highlight(
        self,
        job_id: str,
        stage: PipelineStage,
        *,
        highlights_found_so_far: int | None = None,
        attribution_metrics_json: str | None = None,
    ) -> bool:
        """Atomically publish ``COMPLETED`` at 100% for a highlight job."""
        return await self._write_highlight_terminal(
            job_id,
            JobState.COMPLETED,
            stage,
            100.0,
            "completed",
            highlights_found_so_far=highlights_found_so_far,
            attribution_metrics_json=attribution_metrics_json,
        )

    async def fail_highlight(
        self,
        job_id: str,
        stage: PipelineStage,
        *,
        progress_percent: float,
        error_message: str,
        highlights_found_so_far: int | None = None,
        attribution_metrics_json: str | None = None,
    ) -> bool:
        """Atomically publish ``FAILED`` with the current progress floor."""
        return await self._write_highlight_terminal(
            job_id,
            JobState.FAILED,
            stage,
            progress_percent,
            "error",
            error_message=error_message,
            highlights_found_so_far=highlights_found_so_far,
            attribution_metrics_json=attribution_metrics_json,
        )

    async def set_state(
        self,
        job_id: str,
        state: JobState,
        error_message: str = "",
        sync_latest: bool = True,
    ) -> bool:
        import logging
        logger = logging.getLogger(__name__)
        now = datetime.now(timezone.utc)
        completed_at = now if state in states_with_completed_at() else None
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"job_state = %s, error_message = %s, "
            f"updated_at = %s, completed_at = %s "
            f"WHERE job_id = %s"
        )
        ok = await self._client.execute_write(q, [
            state.value, error_message, now, completed_at, job_id,
        ])
        lifecycle = None
        if ok:
            lifecycle = await self.get_lifecycle(job_id)
            if lifecycle:
                index_ok = await self.upsert_recovery_index(lifecycle)
                if not index_ok:
                    return False
        if ok and sync_latest and state in states_that_sync_latest_job_row():
            try:
                if lifecycle and lifecycle.get("video_id"):
                    await self.set_latest(lifecycle["video_id"], job_id, state)
            except Exception as e:
                logger.warning("Failed to sync latest_job for %s: %s", job_id, e)
        return ok

    async def set_replacement(
        self, job_id: str, replacement_job_id: str
    ) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"replacement_job_id = %s, updated_at = %s "
            f"WHERE job_id = %s"
        )
        return await self._client.execute_write(q, [
            replacement_job_id, now, job_id,
        ])

    async def claim_replacement(
        self,
        job_id: str,
        replacement_job_id: str,
        expected_state: JobState = JobState.AWAITING_CORRECTION,
    ) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"replacement_job_id = %s, updated_at = %s "
            f"WHERE job_id = %s "
            f"IF replacement_job_id = null "
            f"AND job_state = %s"
        )
        rows = await self._client.execute(q, [
            replacement_job_id,
            now,
            job_id,
            expected_state.value,
        ])
        if not rows:
            return False
        row = rows[0]
        if hasattr(row, "_asdict"):
            row_dict = row._asdict()
            return bool(row_dict.get("[applied]", row_dict.get("applied", False)))
        return bool(getattr(row, "applied", False))

    async def heartbeat(self, job_id: str, owner_instance_id: str) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"last_heartbeat_at = %s, owner_instance_id = %s "
            f"WHERE job_id = %s"
        )
        ok = await self._client.execute_write(q, [now, owner_instance_id, job_id])
        if ok:
            lifecycle = await self.get_lifecycle(job_id)
            if lifecycle:
                return await self.upsert_recovery_index(lifecycle)
        return ok

    async def claim_job_for_recovery(
        self,
        job_id: str,
        owner_instance_id: str,
        expected_owner_instance_id: str,
        expected_state: JobState = JobState.RUNNING,
        expected_last_heartbeat_at: datetime | None = None,
    ) -> bool:
        now = datetime.now(timezone.utc)
        heartbeat_condition = "AND last_heartbeat_at = %s" if expected_last_heartbeat_at else ""
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"owner_instance_id = %s, last_heartbeat_at = %s, updated_at = %s "
            f"WHERE job_id = %s "
            f"IF owner_instance_id = %s AND job_state = %s {heartbeat_condition}"
        )
        params = [
            owner_instance_id,
            now,
            now,
            job_id,
            expected_owner_instance_id,
            expected_state.value,
        ]
        if expected_last_heartbeat_at:
            params.append(expected_last_heartbeat_at)
        rows = await self._client.execute(q, params)
        if not rows:
            return False
        row = rows[0]
        if hasattr(row, "_asdict"):
            row_dict = row._asdict()
            return bool(row_dict.get("[applied]", row_dict.get("applied", False)))
        return bool(getattr(row, "applied", False))

    async def claim_pending_job_takeover(
        self,
        job_id: str,
        new_owner_instance_id: str,
        *,
        expected_owner_instance_id: str,
    ) -> bool:
        """CAS: take ownership of a PENDING row so only one instance re-schedules it."""
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"owner_instance_id = %s, last_heartbeat_at = %s, updated_at = %s "
            f"WHERE job_id = %s "
            f"IF job_state = %s AND owner_instance_id = %s"
        )
        params = [
            new_owner_instance_id,
            now,
            now,
            job_id,
            JobState.PENDING.value,
            expected_owner_instance_id,
        ]
        rows = await self._client.execute(q, params)
        if not rows:
            return False
        row = rows[0]
        if hasattr(row, "_asdict"):
            row_dict = row._asdict()
            return bool(row_dict.get("[applied]", row_dict.get("applied", False)))
        return bool(getattr(row, "applied", False))

    async def list_running_jobs(self, owner_instance_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job_id in list(self.owned_jobs):
            row = await self.get_lifecycle(job_id)
            if row and row.get("job_state") in (
                JobState.PENDING.value,
                JobState.RUNNING.value,
                JobState.AWAITING_CORRECTION.value,
            ):
                results.append(row)
        return results
