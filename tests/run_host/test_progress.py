import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from uuid import uuid4

import httpx
import pytest
from bjj_analysis_observer.checkpoints import KeyspacesCheckpointReader
from bjj_analysis_observer.progress import FANOUTS, NODES, ProgressEvidenceError, _step
from shared_lib.infrastructure.annotation_store import AnnotationNotFound
from test_dispatch import dispatch_tables, setup_dispatch
from test_worker import leases

from service.analysis_checkpoint_journal import (
    CheckpointIntegrityError,
)


def nodes(view):
    return {node.node_id: node for node in view.nodes}


@pytest.mark.asyncio
async def test_unstarted_run_progress_does_not_claim_or_touch_provider(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    view = await dispatch.progress(run.spec.run_id)
    assert tuple(nodes(view)) == NODES
    assert all(
        n.recorded_status == "not_started" and n.total_items is None for n in view.nodes
    )
    assert wire.calls == [] and list(dispatch.scratch_root.iterdir()) == []
    assert (
        dispatch.session.execute(
            f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
            [str(run.spec.run_id)],
        ).one()
        is None
    )


@pytest.mark.asyncio
async def test_completed_zero_fanouts_require_and_report_explicit_summaries(
    setup_dispatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    count = len(wire.calls)
    view = await dispatch.progress(run.spec.run_id)
    for node in FANOUTS:
        row = nodes(view)[node]
        assert (row.recorded_status, row.completed_items, row.total_items) == (
            "succeeded",
            0,
            0,
        )
        assert len(row.checkpoints) == 1
    assert nodes(view)["dag_complete"].recorded_status == "succeeded"
    assert nodes(view)["provider_cleanup"].recorded_status == "not_started"
    assert len(wire.calls) == count and list(dispatch.scratch_root.iterdir()) == []
    rendered = str(asdict(view))
    assert not any(
        secret in rendered
        for secret in (
            "system_prompt",
            "initial_prompt",
            "artifact_key",
            "result_json",
            "fixture-secret",
        )
    )


def positive_transport(dispatch, run, wire, entered, release):
    start, end = run.spec.scope.start_ms, run.spec.scope.end_ms
    at = start + (end - start) // 2
    bounds = dict(start_ms=start, end_ms=end)
    standing = dict(
        observation="observed",
        position="standing",
        roles=["neutral", "neutral"],
        notes="",
        evidence="standing",
        uncertainty_note=None,
        interruption_note=None,
    )
    back = {
        **standing,
        "position": "back_control",
        "roles": ["back_controller", "back_defender"],
    }

    async def handler(request):
        if ":generateContent" not in str(request.url):
            return await wire(request)
        payload = json.loads(await request.aread())
        initial = json.loads(payload["contents"][0]["parts"][-1]["text"])
        context = initial["stage_context"]
        schema = payload["generationConfig"]["responseJsonSchema"]["properties"]
        if "initial_state" in schema:
            value = dict(
                initial_state=standing, transitions=[dict(at_ms=at, state_after=back)]
            )
        elif "transition_index" in schema:
            entered.set()
            await release.wait()
            value = dict(
                transition_index=0,
                confirmed=True,
                corrected_at_ms=at,
                state_after=back,
                reason="visible control",
            )
        elif "events" in schema:
            value = dict(events=[])
        elif "confirmed" in schema:
            value = dict(
                candidate_id=context["candidate"]["candidate_id"],
                confirmed=True,
                bounds=bounds,
                end_status="outcome_visible",
                reason="tap seen",
            )
        elif "verdict" in schema:
            value = dict(
                candidate_id=context["candidate_id"],
                verdict="agree",
                intentional_technique_visible=True,
                technique=context["proposed"],
                reason="tap seen",
            )
        else:
            value = dict(
                bounds=context["bounds"],
                action_classes=["submission_choke"],
                actor_player_id=run.spec.athletes[0].player_id,
                actor_sentinel=None,
                outcome="successful",
                technique_shortlist=None,
                technique_guess="choke",
                notes="",
                evidence="grip",
                outcome_evidence="tap",
                outcome_bounds=dict(start_ms=end - 2, end_ms=end - 1),
            )
        wire.result["candidates"][0]["content"]["parts"] = [{"text": json.dumps(value)}]
        return await wire(request)

    dispatch.transport = httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_active_fanout_and_completed_dynamic_nodes_from_real_dag(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    entered, release = asyncio.Event(), asyncio.Event()
    positive_transport(dispatch, run, wire, entered, release)
    task = asyncio.create_task(dispatch.dispatch(run.spec.run_id))
    try:
        wait = asyncio.create_task(entered.wait())
        try:
            done, _ = await asyncio.wait(
                (wait, task), timeout=30, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                await task  # expose an early fixture/executor error, not a timeout
            assert wait in done
        finally:
            wait.cancel()
            await asyncio.gather(wait, return_exceptions=True)
        row = nodes(await dispatch.progress(run.spec.run_id))["position_critique"]
        assert (row.recorded_status, row.completed_items, row.total_items) == (
            "running",
            0,
            1,
        )
        assert row.checkpoints[0].item_id == "0"
        with pytest.raises(ProgressEvidenceError, match="budget"):
            await dispatch.progress(run.spec.run_id, max_item_reads=0)
        release.set()
        await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    view = nodes(await dispatch.progress(run.spec.run_id))
    for node in FANOUTS:
        assert (
            view[node].recorded_status,
            view[node].completed_items,
            view[node].total_items,
        ) == ("succeeded", 1, 1)
        assert len(view[node].checkpoints) == 2
    assert len(wire.generates()) == 6
    # Test-only scan of this disposable fixture keyspace: production uses exact keys.
    actual = dispatch.session.execute(
        f"SELECT run_id,step_id,revision FROM {dispatch.keyspace}.analysis_execution_checkpoints"
    )
    actual_steps = {
        row.step_id
        for row in actual
        if row.run_id == run.spec.run_id and row.revision == 0
    }
    projected_steps = {
        p.execution_id for node in view.values() for p in node.checkpoints
    }
    assert actual_steps == projected_steps
    assert {node.node_id for node in view.values() if node.checkpoints} == set(
        NODES
    ) - {"provider_cleanup"}


@pytest.mark.asyncio
async def test_empty_known_fanout_not_success_until_summary_commit(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    original, entered, release = dispatch._host, asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def stalled(candidate, guard):
        async with original(candidate, guard) as host:
            save = host.journal.compare_and_swap

            async def wait(expected, record):
                if (
                    record.progress.node_id == "position_critique"
                    and record.phase == "running"
                ):
                    entered.set()
                    await release.wait()
                return await save(expected, record)

            host.journal.compare_and_swap = wait
            yield host

    dispatch._host = stalled
    task = asyncio.create_task(dispatch.dispatch(run.spec.run_id))
    try:
        await asyncio.wait_for(entered.wait(), 30)
        row = nodes(await dispatch.progress(run.spec.run_id))["position_critique"]
        assert (row.recorded_status, row.completed_items, row.total_items) == (
            "not_started",
            0,
            0,
        )
        release.set()
        await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_failure_reports_committed_failure_not_success(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    wire.status = 503
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await dispatch.dispatch(run.spec.run_id)
    count = len(wire.calls)
    row = nodes(await dispatch.progress(run.spec.run_id))["position_scan"]
    assert row.recorded_status == "failed" and row.completed_items == 0
    assert row.checkpoints[0].error_code == "DISPATCH_UNCERTAIN"
    assert row.checkpoints[0].heartbeat_at is None
    assert len(wire.calls) == count


@pytest.mark.asyncio
async def test_progress_rechecks_access_without_worker_authority(setup_dispatch):
    dispatch, run, wire, _, _, revoke = setup_dispatch
    revoke()
    with pytest.raises(PermissionError):
        await dispatch.progress(run.spec.run_id)
    assert wire.calls == []


@pytest.mark.asyncio
async def test_corrupt_checkpoint_cannot_be_reported_as_missing_or_complete(
    setup_dispatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    row = dispatch.session.execute(
        f"SELECT artifact_key FROM {dispatch.keyspace}.analysis_execution_checkpoints WHERE run_id=%s AND step_id=%s AND revision=0",
        [run.spec.run_id, _step(run.spec.run_id, "position_scan")],
    ).one()
    dispatch.s3.put_object(Bucket=dispatch.bucket, Key=row.artifact_key, Body=b"{}")
    with pytest.raises(CheckpointIntegrityError):
        await dispatch.progress(run.spec.run_id)


@pytest.mark.asyncio
async def test_missing_run_is_not_an_empty_progress_snapshot(setup_dispatch):
    dispatch, _, wire, _, _, _ = setup_dispatch
    with pytest.raises(AnnotationNotFound):
        await dispatch.progress(uuid4())
    assert wire.calls == []


@pytest.mark.asyncio
async def test_source_summary_race_fails_instead_of_inventing_empty_success(
    setup_dispatch, monkeypatch
):
    dispatch, run, _, _, _, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    read = KeyspacesCheckpointReader.read

    async def before_scan(journal, step):
        if step == _step(run.spec.run_id, "position_scan"):
            return None  # source read precedes commit; summary read sees later commit
        return await read(journal, step)

    monkeypatch.setattr(KeyspacesCheckpointReader, "read", before_scan)
    with pytest.raises(ProgressEvidenceError, match="no committed topology"):
        await dispatch.progress(run.spec.run_id)


@pytest.mark.asyncio
async def test_summary_count_mismatch_fails_closed(setup_dispatch, monkeypatch):
    dispatch, run, _, _, _, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    read = KeyspacesCheckpointReader.read

    async def inconsistent(journal, step):
        record = await read(journal, step)
        if step == _step(run.spec.run_id, "position_critique"):
            record = record.model_copy(
                update={
                    "progress": record.progress.model_copy(
                        update={"total_items": 1, "completed_items": 1}
                    )
                }
            )
        return record

    monkeypatch.setattr(KeyspacesCheckpointReader, "read", inconsistent)
    with pytest.raises(ProgressEvidenceError, match="summary disagrees"):
        await dispatch.progress(run.spec.run_id)


@pytest.mark.asyncio
async def test_revocation_during_read_returns_no_partial_projection(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    authority, count = dispatch._authority, 0

    async def revoked(candidate):
        nonlocal count
        count += 1
        if count == 5:
            raise PermissionError("read access revoked")
        await authority(candidate)

    dispatch._authority = revoked
    with pytest.raises(PermissionError):
        await dispatch.progress(run.spec.run_id)
    assert count == 5 and wire.calls == []


@pytest.mark.asyncio
async def test_unqualified_topology_version_is_rejected(setup_dispatch, monkeypatch):
    from bjj_analysis_observer import progress as analysis_progress

    dispatch, run, wire, _, _, _ = setup_dispatch
    monkeypatch.setattr(analysis_progress, "CORE_VERSION", "0.4.0")
    with pytest.raises(ProgressEvidenceError, match="no progress reader"):
        await dispatch.progress(run.spec.run_id)
    assert wire.calls == []


@pytest.mark.asyncio
async def test_duplicate_dynamic_ids_are_not_counted_twice(setup_dispatch, monkeypatch):
    dispatch, run, _, _, _, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    read = KeyspacesCheckpointReader.read
    candidate = dict(
        candidate_id="duplicate",
        bounds=run.spec.scope.model_dump(),
        description="movement",
        reasoning="visible",
        scan_index=0,
        transition_at_ms=None,
        sources=["technique_scan"],
    )

    async def duplicate(journal, step):
        record = await read(journal, step)
        if step == _step(run.spec.run_id, "position_technique_candidates"):
            record = record.model_copy(
                update={
                    "result_json": json.dumps(
                        dict(
                            schema_version=1,
                            candidates=[candidate, candidate],
                            scan_count=2,
                            transition_count=0,
                            matched_count=0,
                        )
                    )
                }
            )
        return record

    monkeypatch.setattr(KeyspacesCheckpointReader, "read", duplicate)
    with pytest.raises(ProgressEvidenceError, match="duplicate fanout"):
        await dispatch.progress(run.spec.run_id)
