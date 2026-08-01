"""Pure tests for the coach-facing highlight progress contract."""

from __future__ import annotations

import pytest

from service.analysis_keyspaces_enums import PipelineStage
from service.worker.highlight_progress import (
    DETECTING_END,
    FINALIZING_END,
    PREPARING_END,
    HighlightProgressWriter,
    detecting_percent,
    finalizing_percent,
)


def test_detecting_percent_maps_chunk_fraction_into_whole_job_band():
    assert detecting_percent(0, 4, 0.0) == PREPARING_END
    assert detecting_percent(1, 4, 0.0) == 30.0
    assert detecting_percent(1, 4, 0.5) == 40.0
    assert detecting_percent(4, 4, 1.0) == DETECTING_END


@pytest.mark.parametrize("fraction", [-1.0, 2.0])
def test_detecting_percent_clamps_fraction(fraction):
    assert PREPARING_END <= detecting_percent(1, 4, fraction) <= DETECTING_END


def test_detecting_percent_handles_empty_chunk_plan_without_regressing():
    assert detecting_percent(0, 0, 0.0) == PREPARING_END
    assert detecting_percent(999, 0, 1.0) == PREPARING_END


def test_finalizing_percent_clamps_published_candidates():
    assert finalizing_percent(0, 0) == FINALIZING_END
    assert finalizing_percent(0, 4) == 90.0
    assert finalizing_percent(2, 4) == 94.5
    assert finalizing_percent(99, 4) == FINALIZING_END
    assert finalizing_percent(-1, 4) == 90.0


class _FakeJobsStore:
    def __init__(self, percent: float = 0.0, phase: str = ""):
        self.lifecycle = {"progress_percent": percent, "stage_message": phase}
        self.writes: list[dict] = []

    async def get_lifecycle(self, job_id: str):
        return dict(self.lifecycle)

    async def update_highlight_progress(self, job_id, stage, percent, **kwargs):
        self.lifecycle["progress_percent"] = percent
        self.writes.append({"stage": stage, "percent": percent, **kwargs})
        return True

    async def complete_highlight(self, job_id, stage, **kwargs):
        self.writes.append({"terminal": "completed", "stage": stage, **kwargs})
        return True

    async def fail_highlight(self, job_id, stage, **kwargs):
        self.writes.append({"terminal": "failed", "stage": stage, **kwargs})
        return True


@pytest.mark.asyncio
async def test_writer_rejects_regressions_against_durable_resume_value():
    store = _FakeJobsStore(percent=64.0)
    writer = HighlightProgressWriter("job-1", store)

    await writer.write(
        PipelineStage.HIGHLIGHT_CHUNK,
        20.0,
        phase="detecting",
        chunk_index=0,
        chunks_total=2,
    )
    await writer.write(
        PipelineStage.HIGHLIGHT_CHUNK,
        70.0,
        phase="detecting",
        chunk_index=1,
        chunks_total=2,
    )

    assert [write["percent"] for write in store.writes] == [64.0, 70.0]
    assert all(write["stage_message"] == "detecting" for write in store.writes)


@pytest.mark.asyncio
async def test_writer_preserves_monotonicity_when_local_calls_decrease():
    store = _FakeJobsStore()
    writer = HighlightProgressWriter("job-1", store)

    await writer.write(PipelineStage.HIGHLIGHT_INGEST, 8.0, phase="preparing")
    await writer.write(PipelineStage.HIGHLIGHT_INGEST, 4.0, phase="preparing")

    assert [write["percent"] for write in store.writes] == [8.0, 8.0]


@pytest.mark.asyncio
async def test_writer_skips_lower_phase_after_resume_and_preserves_durable_snapshot():
    store = _FakeJobsStore(percent=99.0, phase="finalizing")
    store.lifecycle["highlights_found_so_far"] = 12
    writer = HighlightProgressWriter("job-1", store)

    actual = await writer.write(
        PipelineStage.HIGHLIGHT_INGEST,
        4.0,
        phase="preparing",
    )

    assert actual == 99.0
    assert store.writes == []
    assert store.lifecycle == {
        "progress_percent": 99.0,
        "stage_message": "finalizing",
        "highlights_found_so_far": 12,
    }


@pytest.mark.asyncio
async def test_writer_rejects_unknown_durable_phase():
    store = _FakeJobsStore(percent=42.0, phase="legacy_tracking")
    writer = HighlightProgressWriter("job-1", store)

    with pytest.raises(ValueError, match="unknown durable highlight progress phase"):
        await writer.write(
            PipelineStage.HIGHLIGHT_CHUNK,
            50.0,
            phase="detecting",
            chunk_index=0,
            chunks_total=1,
        )


@pytest.mark.asyncio
async def test_writer_allows_same_rank_phase_update():
    store = _FakeJobsStore(percent=30.0, phase="detecting")
    writer = HighlightProgressWriter("job-1", store)

    await writer.write(
        PipelineStage.HIGHLIGHT_CHUNK,
        35.0,
        phase="detecting",
        chunk_index=0,
        chunks_total=1,
    )

    assert store.writes[-1]["stage_message"] == "detecting"
    assert store.writes[-1]["percent"] == 35.0


@pytest.mark.asyncio
async def test_writer_does_not_reopen_a_terminal_durable_phase():
    store = _FakeJobsStore(percent=100.0, phase="completed")
    writer = HighlightProgressWriter("job-1", store)

    actual = await writer.write(
        PipelineStage.HIGHLIGHT_CHUNK,
        50.0,
        phase="detecting",
        chunk_index=0,
        chunks_total=1,
    )

    assert actual == 100.0
    assert store.writes == []


@pytest.mark.asyncio
async def test_writer_enforces_phase_bands_and_phase_specific_fields():
    store = _FakeJobsStore()
    writer = HighlightProgressWriter("job-1", store)

    await writer.write(PipelineStage.HIGHLIGHT_INGEST, 50.0, phase="preparing")
    await writer.write(
        PipelineStage.HIGHLIGHT_CHUNK,
        0.0,
        phase="detecting",
        chunk_index=0,
        chunks_total=2,
        highlights_found_so_far=0,
    )
    await writer.write(
        PipelineStage.HIGHLIGHT_PUBLISH,
        100.0,
        phase="finalizing",
        highlights_found_so_far=3,
    )

    assert [write["percent"] for write in store.writes] == [10.0, 10.0, 99.0]
    assert store.writes[0]["chunk_index"] is None
    assert store.writes[1]["chunk_index"] == 0
    assert store.writes[1]["chunks_total"] == 2
    assert store.writes[2]["chunk_index"] is None
    assert store.writes[2]["chunks_total"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "stage", "kwargs"),
    [
        ("preparing", PipelineStage.HIGHLIGHT_INGEST, {"chunk_index": 0}),
        ("detecting", PipelineStage.HIGHLIGHT_CHUNK, {"chunks_total": 2}),
        ("finalizing", PipelineStage.HIGHLIGHT_PUBLISH, {"chunk_index": 0}),
    ],
)
async def test_writer_rejects_phase_specific_field_mismatches(phase, stage, kwargs):
    store = _FakeJobsStore()
    writer = HighlightProgressWriter("job-1", store)

    with pytest.raises(ValueError, match="phase-specific"):
        await writer.write(stage, 10.0, phase=phase, **kwargs)


@pytest.mark.asyncio
async def test_writer_complete_uses_atomic_terminal_store_operation():
    store = _FakeJobsStore(percent=98.0, phase="finalizing")
    writer = HighlightProgressWriter("job-1", store)

    await writer.complete(
        PipelineStage.HIGHLIGHT_PUBLISH,
        highlights_found_so_far=4,
        attribution_metrics_json='{"total_published": 4}',
    )

    assert store.writes[-1] == {
        "terminal": "completed",
        "stage": PipelineStage.HIGHLIGHT_PUBLISH,
        "highlights_found_so_far": 4,
        "attribution_metrics_json": '{"total_published": 4}',
    }


@pytest.mark.asyncio
async def test_writer_fail_uses_atomic_terminal_store_operation():
    store = _FakeJobsStore(percent=47.0, phase="detecting")
    writer = HighlightProgressWriter("job-1", store)

    await writer.fail(
        PipelineStage.HIGHLIGHT_CHUNK,
        "Gemini unavailable",
        highlights_found_so_far=2,
    )

    assert store.writes[-1] == {
        "terminal": "failed",
        "stage": PipelineStage.HIGHLIGHT_CHUNK,
        "error_message": "Gemini unavailable",
        "progress_percent": 47.0,
        "highlights_found_so_far": 2,
        "attribution_metrics_json": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("phase, percent", [("completed", 100.0), ("error", 47.0)])
async def test_writer_terminal_methods_do_not_reopen_terminal_snapshot(phase, percent):
    store = _FakeJobsStore(percent=percent, phase=phase)
    writer = HighlightProgressWriter("job-1", store)

    complete_percent = await writer.complete(PipelineStage.HIGHLIGHT_PUBLISH)
    fail_percent = await writer.fail(PipelineStage.HIGHLIGHT_PUBLISH, "late error")

    assert complete_percent == percent
    assert fail_percent == percent
    assert store.writes == []
