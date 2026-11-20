"""Project complete private checkpoint history into truthful progress/usage.

No pricing guesses or customer debits. The host still authorizes/redacts the API;
these pure projections neither publish SSE nor replace durable lifecycle state.
"""

from collections import defaultdict
from uuid import UUID

from bjj_analysis_core.execution import StepCheckpointV1
from shared_lib.models.analysis_contracts import (
    CostSummaryV1,
    Identifier,
    NodeProgressV1,
    VersionedAnalysisContract,
    summarize_costs,
)


class NodeCostV1(VersionedAnalysisContract):
    node_id: Identifier
    costs: CostSummaryV1


class ExecutionReportV1(VersionedAnalysisContract):
    run_id: UUID
    progress: tuple[NodeProgressV1, ...]
    node_costs: tuple[NodeCostV1, ...]
    total_costs: CostSummaryV1


def report_execution(history):
    """Require the full CAS history, not only latest heads (which lose retries).

    Each attempt contributes its latest usage receipt exactly once. Failed and
    uncertain attempts remain in totals after successful recovery. Missing cost is
    explicitly unknown; deterministic zero is model cost only, not infrastructure.
    """
    rows = tuple(StepCheckpointV1.model_validate(r) for r in history)
    if not rows or len({r.progress.run_id for r in rows}) != 1:
        raise ValueError("a report requires exactly one run's checkpoint history")
    steps = defaultdict(list)
    for row in rows:
        steps[row.step_id].append(row)
    heads, calls = [], {}
    for records in steps.values():
        records.sort(key=lambda r: r.revision)
        if [r.revision for r in records] != list(range(1, records[-1].revision + 1)):
            raise ValueError("history has missing or duplicate checkpoint revisions")
        if len({r.input_sha256 for r in records}) != 1:
            raise ValueError("checkpoint input changed during execution")
        heads.append(records[-1])
        for record in records:
            calls[record.usage.call_id] = (record.progress.node_id, record.usage)
    by_node = defaultdict(list)
    for node_id, usage in calls.values():
        by_node[node_id].append(usage)
    return ExecutionReportV1(
        run_id=rows[0].progress.run_id,
        progress=tuple(r.progress for r in sorted(heads, key=lambda r: str(r.step_id))),
        node_costs=tuple(
            NodeCostV1(node_id=node, costs=summarize_costs(tuple(usages)))
            for node, usages in sorted(by_node.items())
        ),
        total_costs=summarize_costs(tuple(u for _, u in calls.values())),
    )
