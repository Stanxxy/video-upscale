"""Production parallelism through the real host: Keyspaces CAS journal, S3 artifacts,
GeminiProvider HTTP mapping and durable outbox, with concurrent technique steps."""

import asyncio
import copy
import json
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from bjj_analysis_core.requests import prompt_bundle
from conftest import Wire
from shared_lib.infrastructure.analysis_baseline_artifacts import S3AnalysisBaselines
from shared_lib.infrastructure.annotation_store import DynamoAnnotationStore
from shared_lib.infrastructure.component_result_store import (
    DynamoAIComponentResultStore,
)
from shared_lib.infrastructure.proposal_store import DynamoProposalStore
from shared_lib.infrastructure.result_publication_store import (
    DynamoResultPublicationStore,
)
from shared_lib.models.analysis_metadata import AnalysisMetadataV1
from test_dispatch import dispatch_tables  # noqa: F401 — pytest fixture
from test_worker import leases  # noqa: F401 — pytest fixture

from service.analysis_run_dispatch import (
    PRODUCTION_MAX_PARALLEL_CALLS,
    PositionRunDispatcher,
)

# Inside the media fixture scope [500, 2500).
EVENTS = ((500, 800), (1000, 1300), (1600, 1900), (2100, 2400))
NODES = {text: node for node, text in prompt_bundle()["prompts"].items()}


def state(position="standing", roles=("neutral", "neutral")):
    return dict(
        observation="observed",
        position=position,
        roles=list(roles),
        notes="",
        evidence="visible torso and leg relationship",
        uncertainty_note=None,
        interruption_note=None,
    )


def reply_for(node, context):
    if node == "position_scan":
        return dict(
            initial_state=state(),
            transitions=[
                dict(
                    at_ms=1450,
                    state_after=state("closed_guard", ("guard_player", "guard_opponent")),
                )
            ],
        )
    if node == "position_critique":
        proposed = context["proposed"]
        return dict(
            transition_index=context["transition_index"],
            confirmed=True,
            corrected_at_ms=proposed["at_ms"],
            state_after=proposed["state_after"],
            reason="guard closed",
        )
    if node == "technique_scan":
        return dict(
            events=[
                dict(
                    bounds=dict(start_ms=start, end_ms=end),
                    description=f"attempt at {start}",
                    reasoning="grip visible",
                )
                for start, end in EVENTS
            ]
        )
    if node == "technique_critique":
        candidate = context["candidate"]
        return dict(
            candidate_id=candidate["candidate_id"],
            confirmed=True,
            bounds=candidate["bounds"],
            end_status="outcome_visible",
            reason="outcome visible",
        )
    if node == "technique_analyze":
        bounds = context["bounds"]
        return dict(
            schema_version=1,
            bounds=bounds,
            action_classes=["submission_choke"],
            actor_player_id=None,
            actor_sentinel="unclear",
            outcome="successful",
            notes="rear naked choke",
            evidence="visible intentional choke attempt",
            outcome_evidence="defender taps",
            outcome_bounds=dict(start_ms=bounds["end_ms"] - 50, end_ms=bounds["end_ms"]),
        )
    return dict(
        candidate_id=context["candidate_id"],
        verdict="agree",
        intentional_technique_visible=True,
        technique=context["proposed"],
        reason="outcome visible",
    )


@pytest_asyncio.fixture
async def parallel_dispatch(
    dispatch_tables, inputs, media_bytes, s3, checkpoint_database, leases, tmp_path  # noqa: F811
):
    client, tables, proposal_tables, publication = dispatch_tables
    annotations = DynamoAnnotationStore(client, tables)
    spec = inputs[0].spec
    await annotations.create(
        AnalysisMetadataV1(
            analysis_id=spec.analysis_id,
            scope_id=spec.scope_id,
            owner=spec.owner,
            video_id=spec.video_id,
            source_sha256=spec.source.sha256,
            source_duration_ms=spec.source_duration_ms,
            scope=spec.scope,
            player_ids=tuple(a.player_id for a in spec.athletes),
            created_at=spec.admitted_at - timedelta(seconds=1),
            updated_at=spec.admitted_at - timedelta(seconds=1),
        ),
        actor=spec.owner,
    )
    proposals = DynamoProposalStore(
        annotations, proposal_tables, S3AnalysisBaselines(s3[0], s3[1])
    )
    run = await proposals.register_run(spec, actor=spec.owner)
    components = DynamoAIComponentResultStore(proposals, s3[0], s3[1])
    outbox = DynamoResultPublicationStore(
        proposals, components, publication, "arn:aws:sns:us-east-1:000000000000:fixture-results"
    )
    wire = Wire(SimpleNamespace(run=run, source_size=len(media_bytes[0])))
    flight = SimpleNamespace(active=[], peak=0, nodes=[])

    async def handler(request):
        if ":generateContent" not in str(request.url):
            return await wire(request)
        body = await request.aread()
        wire.calls.append((request.method, str(request.url), body, dict(request.headers)))
        data = json.loads(body)
        node = NODES[data["systemInstruction"]["parts"][0]["text"]]
        context = json.loads(data["contents"][0]["parts"][-1]["text"])["stage_context"]
        flight.nodes.append(node)
        flight.active.append(node)
        flight.peak = max(flight.peak, len(flight.active))
        try:
            await asyncio.sleep(0.05)  # provider latency so independent calls overlap
        finally:
            flight.active.remove(node)
        result = copy.deepcopy(wire.result)  # per-request body: no shared mutation
        result["candidates"][0]["content"]["parts"] = [
            {"text": json.dumps(reply_for(node, context))}
        ]
        return httpx.Response(200, json=result)

    async def authority(candidate):
        assert candidate == run

    dispatcher = PositionRunDispatcher(
        proposals=proposals,
        outbox=outbox,
        session=checkpoint_database[0],
        s3_client=s3[0],
        keyspace=checkpoint_database[1],
        artifact_bucket=s3[1],
        allowed_buckets={s3[1]},
        scratch_root=tmp_path,
        max_source_bytes=10_000_000,
        api_key="fixture-secret",
        qualified_models={"fixture-model": {"fixture-model-r1"}},
        authority=authority,
        lease_seconds=60,
        heartbeat_seconds=1,
        mime_type="video/mp4",
        max_parallel_calls=PRODUCTION_MAX_PARALLEL_CALLS,
        transport=httpx.MockTransport(handler),
    )
    return dispatcher, run, wire, components, flight


@pytest.mark.asyncio
async def test_parallel_production_dispatch_completes_real_dag_on_keyspaces(parallel_dispatch):
    dispatch, run, wire, components, flight = parallel_dispatch
    result = await dispatch.dispatch(run.spec.run_id)

    # Fan-out genuinely overlapped on the real journal, inside the production bound.
    assert 1 < flight.peak <= PRODUCTION_MAX_PARALLEL_CALLS
    technique_items = flight.nodes.count("technique_analyze")
    assert technique_items >= len(EVENTS)
    assert flight.nodes.count("technique_critique") >= technique_items
    assert flight.nodes.count("technique_validate") == technique_items
    assert flight.nodes.count("position_critique") == 1
    assert len(result.techniques.techniques) == technique_items
    for index, lane in enumerate(("positions", "techniques"), 1):
        publication, sent = await dispatch.outbox.read(run.spec.run_id, lane)
        assert sent is None and publication.message.sequence == index
        await components.ingest(run.spec.analysis_id, publication.message, actor=run.spec.owner)

    progress = await dispatch.progress(run.spec.run_id)
    fanned = {v.node_id: v for v in progress.nodes}
    for node in ("technique_critique", "technique_analyze", "technique_validate"):
        assert fanned[node].recorded_status == "succeeded"
        assert fanned[node].completed_items == fanned[node].total_items > 1
    assert list(dispatch.scratch_root.iterdir()) == []


@pytest.mark.parametrize("bound", [0, 17, True])
def test_dispatcher_rejects_unbounded_parallelism(bound):
    with pytest.raises(ValueError, match="parallelism"):
        PositionRunDispatcher(
            proposals=SimpleNamespace(),
            outbox=SimpleNamespace(proposals=None),
            session=None,
            s3_client=None,
            keyspace="k",
            artifact_bucket="b",
            allowed_buckets={"b"},
            scratch_root=None,
            max_source_bytes=1,
            api_key="k",
            qualified_models={},
            authority=lambda run: None,
            lease_seconds=60,
            heartbeat_seconds=10,
            mime_type="video/mp4",
            max_parallel_calls=bound,
        )
