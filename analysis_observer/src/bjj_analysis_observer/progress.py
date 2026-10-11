"""Authorized, read-only core0.3 progress reconstruction; no scan/index or UI API.

Checkpoint statuses are last recorded facts, NOT worker liveness. Callers must
separately project lifecycle/heartbeat and delivery readiness. No synthetic percent.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid5

from bjj_analysis_core import CORE_VERSION, WORKFLOW_ID
from bjj_analysis_core.outputs import (
    CandidateBatchV1,
    PositionScanV1,
    TechniqueAnalysisInputV1,
)
from pydantic import TypeAdapter
from shared_lib.models.analysis_contracts import NodeProgressV1

NODES = (
    "provider_upload",
    "position_scan",
    "position_critique",
    "positions_normalize",
    "positions_publish",
    "technique_scan",
    "position_technique_candidates",
    "technique_critique",
    "technique_worklist",
    "technique_analyze",
    "technique_validate",
    "techniques_normalize",
    "techniques_publish",
    "dag_complete",
    "provider_cleanup",
)
HOST_NODES = {"provider_upload", "dag_complete", "provider_cleanup"}
FANOUTS = {
    "position_critique",
    "technique_critique",
    "technique_analyze",
    "technique_validate",
}
SUPPORTED_CORE_VERSION = "0.3.0"


class ProgressEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class NodeView:
    node_id: str
    recorded_status: str
    completed_items: int
    total_items: int | None
    checkpoints: tuple[NodeProgressV1, ...]


@dataclass(frozen=True)
class ProgressView:
    run_id: UUID
    observed_at: datetime
    nodes: tuple[NodeView, ...]


def _step(run_id, node, item=None):
    suffix = f"host:v1:{node}" if node in HOST_NODES else f"step:v1:{node}:{item or ''}"
    return uuid5(run_id, suffix)


async def read_progress(journal, *, guard, max_item_reads=2000):
    """Reconstruct bounded topology from durable discovery, never replay execution.

    The guard must reauthorize the registered run/current reader on every I/O;
    completed runs are readable without granting a worker lease. No raw prompts,
    results, usage prices or private object references leave this projection.
    """
    run = journal.run
    if (
        not callable(guard)
        or type(max_item_reads) is not int
        or not 0 <= max_item_reads <= 2000
    ):
        raise ValueError("live read authority and bounded item budget required")
    if CORE_VERSION != SUPPORTED_CORE_VERSION or (
        run.spec.workflow_id,
        run.spec.core_version,
    ) != (WORKFLOW_ID, SUPPORTED_CORE_VERSION):
        raise ProgressEvidenceError("no progress reader for registered core/workflow")

    async def read(node, item=None):
        await guard()
        step = _step(run.spec.run_id, node, item)
        record = await journal.read(step)
        if record is not None and (
            record.step_id != step
            or record.progress.run_id != run.spec.run_id
            or record.progress.node_id != node
            or record.progress.item_id != item
            or (item is not None and record.request is None)
            or (record.request is not None and record.request.node_id != node)
            or (node in HOST_NODES and record.request is not None)
        ):
            raise ProgressEvidenceError("checkpoint differs from expected execution")
        await guard()
        return record

    # Read topology sources before their dependent summaries. A racing read that
    # cannot justify a summary fails explicitly rather than claiming empty success.
    heads = {node: await read(node) for node in NODES}

    def output(node, value_type):
        record = heads[node]
        if record is None or record.phase != "succeeded":
            return None
        return TypeAdapter(value_type).validate_json(record.result_json)

    scan = output("position_scan", PositionScanV1)
    batch = output("position_technique_candidates", CandidateBatchV1)
    work = output("technique_worklist", tuple[TechniqueAnalysisInputV1, ...])
    items = {
        "position_critique": (
            None if scan is None else [str(i) for i in range(len(scan.transitions))]
        ),
        "technique_critique": (
            None if batch is None else [item.candidate_id for item in batch.candidates]
        ),
        "technique_analyze": (
            None if work is None else [item.candidate_id for item in work]
        ),
        "technique_validate": (
            None if work is None else [item.candidate_id for item in work]
        ),
    }
    if sum(len(ids) for ids in items.values() if ids is not None) > max_item_reads:
        raise ProgressEvidenceError("progress item read budget exceeded")
    views = []
    for node in NODES:
        head = heads[node]
        if node not in FANOUTS:
            progress = None if head is None else head.progress
            views.append(
                NodeView(
                    node,
                    "not_started" if progress is None else progress.status,
                    0 if progress is None else progress.completed_items,
                    None if progress is None else progress.total_items,
                    () if progress is None else (progress,),
                )
            )
            continue
        ids = items[node]
        if ids is None:
            if head is not None:
                raise ProgressEvidenceError(
                    "summary has no committed topology; reread snapshot"
                )
            views.append(NodeView(node, "not_started", 0, None, ()))
            continue
        if len(ids) != len(set(ids)):
            raise ProgressEvidenceError("duplicate fanout execution identities")
        records = [await read(node, item) for item in ids]
        done = sum(
            record is not None and record.phase == "succeeded" for record in records
        )
        if head is not None and (
            head.request is not None
            or head.progress.total_items != len(ids)
            or done != len(ids)
        ):
            raise ProgressEvidenceError(
                "summary disagrees with durable fanout evidence"
            )
        observed = [record.progress for record in records if record is not None]
        if head is not None:
            status = head.progress.status
            observed.append(head.progress)
        elif any(p.status == "failed" for p in observed):
            status = "failed"
        elif any(p.status == "cancelled" for p in observed):
            status = "cancelled"
        else:
            # Even all completed items (including zero) need their durable summary.
            status = "running" if observed else "not_started"
        views.append(NodeView(node, status, done, len(ids), tuple(observed)))
    await guard()
    return ProgressView(run.spec.run_id, datetime.now(timezone.utc), tuple(views))
