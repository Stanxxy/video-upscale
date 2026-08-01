"""Durable progress arithmetic and the single highlight progress writer.

The highlight-v2 worker is the only component that knows how far the
pipeline has advanced.  This module keeps the coach-facing arithmetic in one
place and routes every lifecycle write through the Keyspaces-backed
``JobsStore``.  The writer re-reads the durable row before each write and also
keeps a local floor so a worker cannot regress progress between two writes in
the same event loop.
"""

from __future__ import annotations

import math

from service.analysis_keyspaces_enums import PipelineStage
from service.jobs_store import JobsStore

PREPARING_END = 10.0
DETECTING_END = 90.0
FINALIZING_END = 99.0

PREPARING = "preparing"
DETECTING = "detecting"
FINALIZING = "finalizing"
COMPLETED = "completed"
ERROR = "error"

ACTIVE_PHASES = frozenset({PREPARING, DETECTING, FINALIZING})
TERMINAL_PHASES = frozenset({COMPLETED, ERROR})
COACH_PHASES = ACTIVE_PHASES | TERMINAL_PHASES
_PHASE_RANK = {
    PREPARING: 0,
    DETECTING: 1,
    FINALIZING: 2,
    COMPLETED: 3,
    ERROR: 3,
}
_PHASE_BANDS = {
    PREPARING: (0.0, PREPARING_END),
    DETECTING: (PREPARING_END, DETECTING_END),
    FINALIZING: (DETECTING_END, FINALIZING_END),
}


def detecting_percent(chunk_index: int, chunks_total: int, fraction: float) -> float:
    """Map one chunk's bounded fraction into the whole-job detecting band."""
    if chunks_total <= 0:
        return PREPARING_END
    bounded_fraction = min(1.0, max(0.0, float(fraction)))
    total = int(chunks_total)
    position = min(total, max(0.0, float(chunk_index) + bounded_fraction))
    return round(PREPARING_END + 80.0 * position / total, 1)


def finalizing_percent(published: int, total: int) -> float:
    """Map published candidates into the 90--99 finalizing band."""
    if total <= 0:
        return FINALIZING_END
    bounded_published = min(max(0, int(published)), int(total))
    return round(90.0 + 9.0 * bounded_published / int(total), 1)


def _bounded_percent(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"progress_percent must be finite, got {value!r}")
    return round(min(100.0, max(0.0, value)), 1)


class HighlightProgressWriter:
    """Write one monotonic highlight-v2 progress snapshot to Keyspaces."""

    def __init__(self, job_id: str, jobs_store: JobsStore) -> None:
        self.job_id = job_id
        self.jobs_store = jobs_store
        self._last_percent = 0.0

    async def write(
        self,
        stage: PipelineStage,
        progress_percent: float,
        *,
        phase: str,
        chunk_index: int | None = None,
        chunks_total: int | None = None,
        highlights_found_so_far: int | None = None,
        attribution_metrics_json: str | None = None,
    ) -> float:
        """Persist a snapshot and return the percent actually written.

        ``phase`` is a stable product contract, not raw Gemini vocabulary.
        The durable row is read before every write so a resumed worker starts
        at the persisted floor instead of resetting to its in-memory value.
        """
        if phase not in COACH_PHASES:
            raise ValueError(f"unknown highlight progress phase: {phase!r}")

        _, durable_percent, durable_phase, durable_count, _ = await self._durable_snapshot()
        self._validate_phase_fields(
            phase,
            chunk_index=chunk_index,
            chunks_total=chunks_total,
            highlights_found_so_far=highlights_found_so_far,
        )

        requested = self._phase_percent(phase, progress_percent)
        if durable_phase in ACTIVE_PHASES:
            low, high = _PHASE_BANDS[durable_phase]
            if not low <= durable_percent <= high:
                raise ValueError(
                    f"durable highlight percent {durable_percent} is outside "
                    f"the {durable_phase} phase band",
                )

        if durable_phase in TERMINAL_PHASES:
            self._last_percent = max(self._last_percent, durable_percent)
            return self._last_percent

        if durable_phase and _PHASE_RANK[durable_phase] > _PHASE_RANK[phase]:
            self._last_percent = max(self._last_percent, durable_percent)
            return self._last_percent

        actual = max(self._last_percent, durable_percent, requested)
        if phase in ACTIVE_PHASES:
            _, high = _PHASE_BANDS[phase]
            actual = min(actual, high)
        actual_count = self._count_floor(durable_count, highlights_found_so_far)
        ok = await self.jobs_store.update_highlight_progress(
            self.job_id,
            stage,
            actual,
            chunk_index=chunk_index,
            chunks_total=chunks_total,
            highlights_found_so_far=actual_count,
            attribution_metrics_json=attribution_metrics_json,
            stage_message=phase,
        )
        if not ok:
            raise RuntimeError(f"failed to persist highlight progress for job {self.job_id}")
        self._last_percent = actual
        return actual

    async def complete(
        self,
        stage: PipelineStage,
        *,
        highlights_found_so_far: int | None = None,
        attribution_metrics_json: str | None = None,
    ) -> float:
        """Atomically publish terminal completion at 100%."""
        _, durable_percent, durable_phase, durable_count, durable_metrics = (
            await self._durable_snapshot()
        )
        if durable_phase in TERMINAL_PHASES:
            self._last_percent = max(self._last_percent, durable_percent)
            return self._last_percent
        actual_count = self._count_floor(durable_count, highlights_found_so_far)
        if attribution_metrics_json is None:
            attribution_metrics_json = durable_metrics
        ok = await self.jobs_store.complete_highlight(
            self.job_id,
            stage,
            highlights_found_so_far=actual_count,
            attribution_metrics_json=attribution_metrics_json,
        )
        if not ok:
            raise RuntimeError(f"failed to persist completed highlight job {self.job_id}")
        self._last_percent = 100.0
        return 100.0

    async def fail(
        self,
        stage: PipelineStage,
        error_message: str,
        *,
        highlights_found_so_far: int | None = None,
        attribution_metrics_json: str | None = None,
    ) -> float:
        """Atomically publish terminal failure at the current progress floor."""
        _, durable_percent, durable_phase, durable_count, durable_metrics = (
            await self._durable_snapshot()
        )
        if durable_phase in TERMINAL_PHASES:
            self._last_percent = max(self._last_percent, durable_percent)
            return self._last_percent
        actual_percent = min(99.0, max(self._last_percent, durable_percent))
        actual_count = self._count_floor(durable_count, highlights_found_so_far)
        if attribution_metrics_json is None:
            attribution_metrics_json = durable_metrics
        ok = await self.jobs_store.fail_highlight(
            self.job_id,
            stage,
            progress_percent=actual_percent,
            error_message=error_message,
            highlights_found_so_far=actual_count,
            attribution_metrics_json=attribution_metrics_json,
        )
        if not ok:
            raise RuntimeError(f"failed to persist failed highlight job {self.job_id}")
        self._last_percent = actual_percent
        return actual_percent

    async def _durable_snapshot(self):
        lifecycle = await self.jobs_store.get_lifecycle(self.job_id)
        if lifecycle is None:
            return None, 0.0, "", None, None
        durable_percent = _bounded_percent(lifecycle.get("progress_percent", 0.0) or 0.0)
        durable_phase = lifecycle.get("stage_message", "") or ""
        if durable_phase and durable_phase not in COACH_PHASES:
            raise ValueError(
                f"unknown durable highlight progress phase: {durable_phase!r}",
            )
        durable_count = lifecycle.get("highlights_found_so_far")
        if durable_count is not None:
            durable_count = max(0, int(durable_count))
        return (
            lifecycle,
            durable_percent,
            durable_phase,
            durable_count,
            lifecycle.get("attribution_metrics_json"),
        )

    @staticmethod
    def _count_floor(durable_count: int | None, requested_count: int | None) -> int | None:
        if requested_count is None:
            return durable_count
        requested_count = max(0, int(requested_count))
        if durable_count is None:
            return requested_count
        return max(durable_count, requested_count)

    @staticmethod
    def _phase_percent(phase: str, requested: float) -> float:
        if phase == COMPLETED:
            if float(requested) != 100.0:
                raise ValueError("completed highlight progress must be exactly 100")
            return 100.0
        if phase in ACTIVE_PHASES:
            low, high = _PHASE_BANDS[phase]
            return round(min(high, max(low, _bounded_percent(requested))), 1)
        return min(99.0, _bounded_percent(requested))

    @staticmethod
    def _validate_phase_fields(
        phase: str,
        *,
        chunk_index: int | None,
        chunks_total: int | None,
        highlights_found_so_far: int | None,
    ) -> None:
        if highlights_found_so_far is not None and highlights_found_so_far < 0:
            raise ValueError("phase-specific highlight count must be non-negative")
        if phase == PREPARING:
            if any(value is not None for value in (chunk_index, chunks_total, highlights_found_so_far)):
                raise ValueError("phase-specific preparing fields must be null")
            return
        if phase == DETECTING:
            if chunk_index is None or chunks_total is None:
                raise ValueError("phase-specific detecting fields require chunk_index and chunks_total")
            if chunk_index < 0 or chunks_total <= 0 or chunk_index >= chunks_total:
                raise ValueError("phase-specific detecting chunk fields are out of range")
            return
        if phase in (FINALIZING, COMPLETED, ERROR):
            if chunk_index is not None or chunks_total is not None:
                raise ValueError(f"phase-specific {phase} fields must be null")


__all__ = [
    "PREPARING_END",
    "DETECTING_END",
    "FINALIZING_END",
    "PREPARING",
    "DETECTING",
    "FINALIZING",
    "COMPLETED",
    "ERROR",
    "HighlightProgressWriter",
    "detecting_percent",
    "finalizing_percent",
]
