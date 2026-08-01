"""Durable progress arithmetic and the single highlight progress writer.

The highlight-v2 worker is the only component that knows how far the
pipeline has advanced.  This module keeps the coach-facing arithmetic in one
place and routes every lifecycle write through the Keyspaces-backed
``JobsStore``.  The store re-reads the durable row before writing, while this
writer also keeps a local floor so a worker cannot regress progress between
two writes in the same event loop.
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

        lifecycle = await self.jobs_store.get_lifecycle(self.job_id)
        durable_percent = 0.0
        durable_phase = ""
        if lifecycle is not None:
            durable_percent = _bounded_percent(lifecycle.get("progress_percent", 0.0) or 0.0)
            durable_phase = lifecycle.get("stage_message", "") or ""
            if durable_phase and durable_phase not in COACH_PHASES:
                raise ValueError(
                    f"unknown durable highlight progress phase: {durable_phase!r}",
                )

        requested = _bounded_percent(progress_percent)
        if phase in ACTIVE_PHASES:
            requested = min(requested, FINALIZING_END)

        if durable_phase and _PHASE_RANK[durable_phase] > _PHASE_RANK[phase]:
            self._last_percent = max(self._last_percent, durable_percent)
            return self._last_percent

        actual = max(self._last_percent, durable_percent, requested)
        ok = await self.jobs_store.update_highlight_chunk_progress(
            self.job_id,
            stage,
            actual,
            chunk_index=chunk_index,
            chunks_total=chunks_total,
            highlights_found_so_far=highlights_found_so_far,
            attribution_metrics_json=attribution_metrics_json,
            stage_message=phase,
        )
        if not ok:
            raise RuntimeError(f"failed to persist highlight progress for job {self.job_id}")
        self._last_percent = actual
        return actual


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
