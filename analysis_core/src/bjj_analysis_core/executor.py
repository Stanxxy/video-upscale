"""Full serial DAG through explicit provider/journal/publication ports.

No service activation or implicit concurrent fan-out. The host supplies verified
media, durable history, authorization/cancellation guard and an idempotent sink.
"""

from typing import Protocol

from bjj_analysis_core.discovery import (
    build_candidates,
    critique_window,
    resolve_temporal_critiques,
)
from bjj_analysis_core.execution import StepRunner
from bjj_analysis_core.normalization import normalize_positions, normalize_techniques
from bjj_analysis_core.outputs import (
    CandidateBatchV1,
    PositionCritiqueV1,
    TechniqueAnalysisInputV1,
    TechniqueAnalysisV1,
    TechniqueCandidateV1,
    TechniqueDecisionV1,
    TemporalCritiqueV1,
)
from bjj_analysis_core.requests import prepare_call
from shared_lib.models.analysis_contracts import TimeRangeMs, VersionedAnalysisContract
from shared_lib.models.analysis_results import AnalysisRevisionV1
from shared_lib.models.analysis_timeline import content_sha256


class ComponentSink(Protocol):
    async def publish(self, run, component: str, revision: AnalysisRevisionV1) -> None:
        """Idempotently prepare/publish this exact revision using existing outbox.

        Return after the host's durable publication boundary, not before. This
        does not mean consumer review-ready; that remains its separate receipt.
        Revalidate current authority even on replay. Never auto-accept annotations.
        """
        ...


class ExecutionResultV1(VersionedAnalysisContract):
    positions: AnalysisRevisionV1
    techniques: AnalysisRevisionV1


async def execute_pipeline(
    run,
    config_bytes,
    *,
    journal,
    provider,
    sink,
    guard,
    retry_failed=False,
    reissue_unknown=False,
    clock=None,
):
    if not callable(guard):
        raise ValueError(
            "executor requires an explicit current-authority/cancellation guard"
        )
    runner = StepRunner(
        run,
        config_bytes,
        journal,
        provider,
        retry_failed=retry_failed,
        reissue_unknown=reissue_unknown,
        guard=guard,
        clock=clock,
    )
    run, cfg = runner.run, runner.config

    async def deterministic(node, inputs, value, output_type, *, total=1):
        async def compute(at):
            return value(at)

        return await runner.deterministic(
            node, inputs, compute, output_type, total=total
        )

    async def summary(node, items, output_type):
        # Explicitly complete zero-item fan-outs; never fake a provider call.
        return await deterministic(
            node,
            [i.model_dump(mode="json") for i in items],
            lambda at: tuple(items),
            output_type,
            total=len(items),
        )

    async def publish(component, revision):
        await guard()  # also reauthorize cached publication without resending

        async def operation(at):
            await sink.publish(run, component, revision)
            return content_sha256(revision)

        await runner.deterministic(
            f"{component}_publish", revision.model_dump(mode="json"), operation, str
        )

    scan = await runner.model(
        prepare_call(
            run, config_bytes, "position_scan", input_window=run.spec.scope, context={}
        )
    )
    decisions, state_before = [], scan.initial_state
    for index, transition in enumerate(scan.transitions):
        at = transition.at_ms
        window = TimeRangeMs(
            start_ms=max(run.spec.scope.start_ms, at - cfg.position_critique_back_ms),
            end_ms=min(run.spec.scope.end_ms, at + cfg.position_critique_forward_ms),
        )
        decision = await runner.model(
            prepare_call(
                run,
                config_bytes,
                "position_critique",
                input_window=window,
                context=dict(
                    transition_index=index,
                    proposed=transition,
                    state_before=state_before,
                ),
            ),
            item=str(index),
        )
        decisions.append(decision)
        # Rejected rough states do not poison the next critique's context.
        if decision.confirmed:
            state_before = decision.state_after
    decisions = await summary(
        "position_critique", decisions, tuple[PositionCritiqueV1, ...]
    )
    positions = await deterministic(
        "positions_normalize",
        dict(
            scan=scan.model_dump(mode="json"),
            critiques=[d.model_dump(mode="json") for d in decisions],
        ),
        lambda at: normalize_positions(run, scan, decisions, completed_at=at),
        AnalysisRevisionV1,
    )
    await publish(
        "positions", positions
    )  # independently available before technique work

    technique_scan = await runner.model(
        prepare_call(
            run, config_bytes, "technique_scan", input_window=run.spec.scope, context={}
        )
    )
    batch = await deterministic(
        "position_technique_candidates",
        dict(
            scan=technique_scan.model_dump(mode="json"),
            positions=positions.model_dump(mode="json"),
        ),
        lambda at: build_candidates(run, config_bytes, technique_scan, positions),
        CandidateBatchV1,
    )
    temporal = []
    for candidate in batch.candidates:
        temporal.append(
            await runner.model(
                prepare_call(
                    run,
                    config_bytes,
                    "technique_critique",
                    input_window=critique_window(run, config_bytes, candidate),
                    context=dict(candidate=candidate),
                    advisory_positions=positions,
                ),
                item=candidate.candidate_id,
            )
        )
    temporal = await summary(
        "technique_critique", temporal, tuple[TemporalCritiqueV1, ...]
    )
    work = await deterministic(
        "technique_worklist",
        dict(
            candidates=batch.model_dump(mode="json"),
            decisions=[d.model_dump(mode="json") for d in temporal],
        ),
        lambda at: resolve_temporal_critiques(
            run, config_bytes, batch.candidates, temporal
        ),
        tuple[TechniqueAnalysisInputV1, ...],
    )
    analyses = []
    for target in work:
        analyses.append(
            await runner.model(
                prepare_call(
                    run,
                    config_bytes,
                    "technique_analyze",
                    input_window=target.bounds,
                    context=target,
                    advisory_positions=positions,
                ),
                item=target.candidate_id,
            )
        )
    analyses = await summary(
        "technique_analyze", analyses, tuple[TechniqueAnalysisV1, ...]
    )
    candidates, verdicts = [], []
    for target, analysis in zip(work, analyses, strict=True):
        candidate = TechniqueCandidateV1(
            candidate_id=target.candidate_id,
            description=target.description,
            observation_window=target.bounds,
            proposed=analysis,
        )
        candidates.append(candidate)
        verdicts.append(
            await runner.model(
                prepare_call(
                    run,
                    config_bytes,
                    "technique_validate",
                    input_window=target.bounds,
                    context=candidate,
                    advisory_positions=positions,
                ),
                item=target.candidate_id,
            )
        )
    verdicts = await summary(
        "technique_validate", verdicts, tuple[TechniqueDecisionV1, ...]
    )
    techniques = await deterministic(
        "techniques_normalize",
        dict(
            candidates=[c.model_dump(mode="json") for c in candidates],
            verdicts=[d.model_dump(mode="json") for d in verdicts],
        ),
        lambda at: normalize_techniques(run, candidates, verdicts, completed_at=at),
        AnalysisRevisionV1,
    )
    await publish("techniques", techniques)
    await guard()
    return ExecutionResultV1(positions=positions, techniques=techniques)
