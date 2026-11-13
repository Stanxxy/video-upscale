"""Full DAG through explicit provider/journal/publication ports.

Serial by default. A host may pass an explicit ``max_parallel_calls`` bound to run
independent model steps concurrently; there is no implicit fan-out. Every request,
step identity and stage input is the same either way, only wall-clock order changes.
The host supplies verified media, durable history, authorization/cancellation guard
and an idempotent sink.
"""

import asyncio
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
    TechniqueAnalysisInputV1,
    TemporalCritiqueV1,
)
from bjj_analysis_core.packs import pack_for
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


# Upper bound a host may request; each in-flight call is one uncertain-spend item
# if the attempt is interrupted, so the bound is deliberately small.
MAX_PARALLEL_CALLS = 16


async def _ordered(steps, limit):
    """Run zero-argument step coroutines, at most ``limit`` at once, results in input order.

    ``limit == 1`` is the plain serial loop. Otherwise, after the first failure no
    further step starts; steps already in flight finish (their dispatch and spend are
    already committed, and their checkpoints are reused on resume) and the earliest
    failure in input order is raised. Cancelling the caller cancels every in-flight
    step, each of which records its own cancelled checkpoint.
    """
    if limit == 1:
        return [await step() for step in steps]
    gate = asyncio.Semaphore(limit)
    stopped = False

    async def bounded(step):
        nonlocal stopped
        async with gate:
            if stopped:
                return None  # never started; a recorded failure is raised below
            try:
                return await step()
            except BaseException:
                stopped = True
                raise

    outcomes = await asyncio.gather(
        *(bounded(step) for step in steps), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return outcomes


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
    max_parallel_calls=1,
):
    if not callable(guard):
        raise ValueError(
            "executor requires an explicit current-authority/cancellation guard"
        )
    if (
        type(max_parallel_calls) is not int
        or not 1 <= max_parallel_calls <= MAX_PARALLEL_CALLS
    ):
        raise ValueError(
            f"max_parallel_calls must be an integer from 1 to {MAX_PARALLEL_CALLS}"
        )
    limit = max_parallel_calls
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
    pack = pack_for(run)

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

    async def position_branch():
        scan = await runner.model(
            prepare_call(
                run,
                config_bytes,
                "position_scan",
                input_window=run.spec.scope,
                context={},
            )
        )
        # Always serial: each critique's context carries the previous confirmed
        # state, a real data dependency between items.
        decisions, state_before = [], scan.initial_state
        for index, transition in enumerate(scan.transitions):
            at = transition.at_ms
            window = TimeRangeMs(
                start_ms=max(
                    run.spec.scope.start_ms, at - cfg.position_critique_back_ms
                ),
                end_ms=min(
                    run.spec.scope.end_ms, at + cfg.position_critique_forward_ms
                ),
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
            "position_critique", decisions, tuple[pack.position_critique, ...]
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
        )  # independently available before technique work completes
        return positions

    async def technique_scan_step():
        # Observes the full scope with no position context, so it is independent
        # of the position branch; serial runs keep it after position publication.
        return await runner.model(
            prepare_call(
                run,
                config_bytes,
                "technique_scan",
                input_window=run.spec.scope,
                context={},
            )
        )

    positions, technique_scan = await _ordered(
        [position_branch, technique_scan_step], limit
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
    def critique_step(candidate):
        async def step():
            return await runner.model(
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

        return step

    temporal = await _ordered([critique_step(c) for c in batch.candidates], limit)
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
    def analyze_step(target):
        async def step():
            return await runner.model(
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

        return step

    analyses = await _ordered([analyze_step(t) for t in work], limit)
    analyses = await summary(
        "technique_analyze", analyses, tuple[pack.technique_analysis, ...]
    )
    candidates = [
        pack.technique_candidate(
            candidate_id=target.candidate_id,
            description=target.description,
            observation_window=target.bounds,
            proposed=analysis,
        )
        for target, analysis in zip(work, analyses, strict=True)
    ]

    def validate_step(target, candidate):
        async def step():
            return await runner.model(
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

        return step

    verdicts = await _ordered(
        [validate_step(t, c) for t, c in zip(work, candidates, strict=True)], limit
    )
    verdicts = await summary(
        "technique_validate", verdicts, tuple[pack.technique_decision, ...]
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
