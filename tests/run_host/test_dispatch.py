import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import boto3
import httpx
import pytest
import pytest_asyncio
from botocore.config import Config
from conftest import Wire
from shared_lib.infrastructure.analysis_baseline_artifacts import S3AnalysisBaselines
from shared_lib.infrastructure.annotation_store import (
    AnnotationNotFound,
    DynamoAnnotationStore,
    encode_item,
)
from shared_lib.infrastructure.annotation_tables import AnnotationTables
from shared_lib.infrastructure.component_result_store import (
    DynamoAIComponentResultStore,
)
from shared_lib.infrastructure.proposal_store import DynamoProposalStore
from shared_lib.infrastructure.proposal_tables import ProposalTables
from shared_lib.infrastructure.result_publication_store import (
    DynamoResultPublicationStore,
)
from shared_lib.infrastructure.result_publication_table import ResultPublicationTable
from shared_lib.models.analysis_metadata import AnalysisMetadataV1
from shared_lib.models.analysis_timeline import content_sha256
from test_worker import leases

from service.analysis_run_dispatch import PositionRunDispatcher, UnsupportedRunVersion
from service.analysis_run_worker import LeaseLost
from service.analysis_media import DeterministicPreparationRejected


@pytest.fixture(scope="module")
def dispatch_tables(s3):
    # s3 fixture has already required an explicit loopback endpoint.
    client = boto3.client(
        "dynamodb",
        endpoint_url=os.environ["ANNOTATION_TEST_ENDPOINT"],
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 0}),
    )
    prefix = "s39_dispatch_" + uuid4().hex
    tables, proposals, publication = (
        AnnotationTables.with_prefix(prefix),
        ProposalTables.with_prefix(prefix),
        ResultPublicationTable.with_prefix(prefix),
    )
    created = []
    try:
        for group in (tables, proposals, publication):
            for definition in group.definitions():
                client.create_table(**definition)
                created.append(definition["TableName"])
                client.get_waiter("table_exists").wait(
                    TableName=definition["TableName"]
                )
        yield client, tables, proposals, publication
    finally:
        for name in created:
            client.delete_table(TableName=name)
        client.close()


@pytest_asyncio.fixture
async def setup_dispatch(
    dispatch_tables, inputs, media_bytes, s3, checkpoint_database, leases, tmp_path
):
    client, tables, proposal_tables, publication = dispatch_tables
    annotations = DynamoAnnotationStore(client, tables)
    spec = inputs[0].spec
    metadata = AnalysisMetadataV1(
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
    )
    await annotations.create(metadata, actor=spec.owner)
    proposals = DynamoProposalStore(
        annotations, proposal_tables, S3AnalysisBaselines(s3[0], s3[1])
    )
    run = await proposals.register_run(spec, actor=spec.owner)
    components = DynamoAIComponentResultStore(proposals, s3[0], s3[1])
    outbox = DynamoResultPublicationStore(
        proposals,
        components,
        publication,
        "arn:aws:sns:us-east-1:000000000000:fixture-results",
    )
    wire = Wire(SimpleNamespace(run=run, source_size=len(media_bytes[0])))

    async def handler(request):
        if ":generateContent" in str(request.url):
            data = json.loads(await request.aread())
            schema = data["generationConfig"]["responseJsonSchema"]
            output = dict(events=[])
            if "initial_state" in schema["properties"]:
                output = dict(
                    initial_state=dict(
                        observation="observed",
                        position="standing",
                        roles=["neutral", "neutral"],
                        notes="",
                        evidence="both standing",
                        uncertainty_note=None,
                        interruption_note=None,
                    ),
                    transitions=[],
                )
            wire.result["candidates"][0]["content"]["parts"] = [
                {"text": json.dumps(output)}
            ]
        return await wire(request)

    authorized = True

    async def authority(candidate):
        if not authorized:
            raise PermissionError("fixture account access revoked")
        assert candidate == run

    def revoke():
        nonlocal authorized
        authorized = False

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
        transport=httpx.MockTransport(handler),
    )
    return dispatcher, run, wire, components, annotations, revoke


@pytest.mark.asyncio
async def test_stored_run_dispatches_real_dag_to_durable_outbox(setup_dispatch):
    dispatch, run, wire, components, annotations, _ = setup_dispatch
    result = await dispatch.dispatch(run.spec.run_id)
    assert len(wire.generates()) == 2 and result.techniques.techniques == ()
    for index, lane in enumerate(("positions", "techniques"), 1):
        publication, sent = await dispatch.outbox.read(run.spec.run_id, lane)
        assert sent is None and publication.message.sequence == index
        assert publication.message.payload.item_count == (
            1 if lane == "positions" else 0
        )
        # Real consumer intake can read the exact immutable result artifact.
        await components.ingest(
            run.spec.analysis_id, publication.message, actor=run.spec.owner
        )
    current = await annotations.read(run.spec.analysis_id, actor=run.spec.owner)
    assert current.metadata.edit_version == 0  # AI never silently edits/accepts
    assert list(dispatch.scratch_root.iterdir()) == []
    count = len(wire.calls)
    with pytest.raises(LeaseLost):
        await dispatch.dispatch(run.spec.run_id, recover=True)
    assert len(wire.calls) == count


@pytest.mark.asyncio
async def test_missing_run_never_creates_lifecycle_or_provider_work(setup_dispatch):
    dispatch, _, wire, _, _, _ = setup_dispatch
    with pytest.raises(AnnotationNotFound):
        await dispatch.dispatch(uuid4())
    assert wire.calls == []


@pytest.mark.asyncio
async def test_revoked_access_never_dispatches(setup_dispatch):
    dispatch, run, wire, _, _, revoke = setup_dispatch
    revoke()
    with pytest.raises(PermissionError):
        await dispatch.dispatch(run.spec.run_id)
    assert wire.calls == []
    rows = dispatch.session.execute(
        f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
        [str(run.spec.run_id)],
    )
    assert list(rows) == []


@pytest.mark.asyncio
async def test_unqualified_node_prevents_even_upload(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    dispatch.qualified_models = {"another-model": {"r1"}}
    with pytest.raises(UnsupportedRunVersion, match="unqualified model"):
        await dispatch.dispatch(run.spec.run_id)
    assert wire.calls == [] and list(dispatch.scratch_root.iterdir()) == []
    row = dispatch.session.execute(
        f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
        [str(run.spec.run_id)],
    ).one()
    assert row.job_state == "INTERRUPTED"
    assert row.completed_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["media_digest", "config_validator"])
async def test_corrupt_preparation_interrupts_before_provider_and_requires_recovery(
    setup_dispatch, rejection, monkeypatch
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    config = run.spec.effective_config
    if rejection == "media_digest":
        dispatch.s3.put_object(
            Bucket=config.bucket,
            Key=config.key,
            Body=b'{"schema_version":1,"tampered":true}',
            ContentType="application/json",
        )
    else:
        import service.analysis_media as analysis_media

        def reject_config(_run, _config_bytes):
            raise ValueError("raw parser details must not escape")

        monkeypatch.setattr(analysis_media, "load_config", reject_config)

    with pytest.raises(DeterministicPreparationRejected) as rejection_error:
        await dispatch.dispatch(run.spec.run_id)
    if rejection == "config_validator":
        assert str(rejection_error.value) == "configured analysis settings rejected"

    row = dispatch.session.execute(
        f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
        [str(run.spec.run_id)],
    ).one()
    assert row.job_state == "INTERRUPTED"
    assert row.completed_at is None
    assert wire.calls == []
    assert await dispatch.outbox.read(run.spec.run_id, "positions") is None
    assert await dispatch.outbox.read(run.spec.run_id, "techniques") is None
    checkpoints = list(
        dispatch.session.execute(
            f"SELECT step_id FROM {dispatch.keyspace}.analysis_execution_checkpoints "
            "WHERE run_id=%s ALLOW FILTERING",
            [run.spec.run_id],
        )
    )
    assert checkpoints == []

    with pytest.raises(LeaseLost):
        await dispatch.dispatch(run.spec.run_id)
    with pytest.raises(DeterministicPreparationRejected):
        await dispatch.dispatch(run.spec.run_id, recover=True)
    assert wire.calls == []
    assert (
        dispatch.session.execute(
            f"SELECT job_state FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
            [str(run.spec.run_id)],
        ).one().job_state
        == "INTERRUPTED"
    )


@pytest.mark.parametrize(
    "field,value", [("core_version", "999.0.0"), ("workflow_id", "legacy")]
)
def test_unknown_executor_version_has_no_fallback(field, value, inputs):
    run = inputs[0].model_copy(
        update={"spec": inputs[0].spec.model_copy(update={field: value})}
    )
    with pytest.raises(UnsupportedRunVersion):
        PositionRunDispatcher._version(run)


@pytest.mark.asyncio
async def test_authority_loss_after_generation_keeps_raw_receipt_not_publication(
    setup_dispatch,
):
    dispatch, run, wire, _, _, revoke = setup_dispatch
    wire.after_generate = revoke
    with pytest.raises(PermissionError):
        await dispatch.dispatch(run.spec.run_id)
    assert len(wire.generates()) == 1
    objects = dispatch.s3.list_objects_v2(
        Bucket=dispatch.bucket,
        Prefix=f"analysis-provider/v1/{run.spec.analysis_id}/{run.spec.run_id}/generate/",
    )["Contents"]
    assert len(objects) == 1  # private paid-response evidence survives revocation
    assert await dispatch.outbox.read(run.spec.run_id, "positions") is None
    assert list(dispatch.scratch_root.iterdir()) == []


@pytest.mark.asyncio
async def test_waiting_dispatch_rechecks_access_before_claim(setup_dispatch):
    dispatch, run, wire, _, _, revoke = setup_dispatch
    await dispatch.slots.acquire()
    entered = asyncio.Event()
    authority = dispatch._authority

    async def signaled(candidate):
        await authority(candidate)
        entered.set()

    dispatch._authority = signaled
    task = asyncio.create_task(dispatch.dispatch(run.spec.run_id))
    await asyncio.wait_for(entered.wait(), 5)
    revoke()
    dispatch.slots.release()
    with pytest.raises(PermissionError):
        await task
    assert wire.calls == []


@pytest.mark.asyncio
async def test_outbox_failure_prevents_complete_marker_and_retains_positions(
    setup_dispatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    prepare = dispatch.outbox.prepare

    async def fail_techniques(run_id, component, revision):
        if component == "techniques":
            raise TimeoutError("outbox unavailable")
        return await prepare(run_id, component, revision)

    dispatch.outbox.prepare = fail_techniques
    with pytest.raises(TimeoutError, match="outbox unavailable"):
        await dispatch.dispatch(run.spec.run_id)
    assert await dispatch.outbox.read(run.spec.run_id, "positions") is not None
    assert await dispatch.outbox.read(run.spec.run_id, "techniques") is None
    row = list(
        dispatch.session.execute(
            f"SELECT job_state FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
            [str(run.spec.run_id)],
        )
    )[0]
    assert row.job_state == "RUNNING" and len(wire.generates()) == 2
    dispatch.outbox.prepare = prepare
    # Simulate the terminated attempt's lease expiring; no sleep or live worker.
    dispatch.session.execute(
        f"UPDATE {dispatch.keyspace}.job_lifecycle SET last_heartbeat_at=%s WHERE job_id=%s",
        [datetime.now(timezone.utc) - timedelta(seconds=61), str(run.spec.run_id)],
    )
    count = len(wire.calls)
    await dispatch.dispatch(run.spec.run_id, recover=True, retry_failed=True)
    assert len(wire.calls) == count
    assert await dispatch.outbox.read(run.spec.run_id, "techniques") is not None


@pytest.mark.asyncio
async def test_future_stored_core_is_rejected_before_claim_or_media(setup_dispatch):
    dispatch, run, wire, _, annotations, _ = setup_dispatch
    future = run.model_copy(
        update={"spec": run.spec.model_copy(update={"core_version": "999.0.0"})}
    )
    annotations.client.update_item(
        TableName=dispatch.proposals.tables.runs,
        Key=encode_item({"run_id": str(run.spec.run_id)}),
        UpdateExpression="SET content=:c, content_hash=:h",
        ExpressionAttributeValues=encode_item(
            {":c": future.model_dump_json(), ":h": content_sha256(future)}
        ),
    )
    with pytest.raises(UnsupportedRunVersion):
        await dispatch.dispatch(run.spec.run_id)
    rows = dispatch.session.execute(
        f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
        [str(run.spec.run_id)],
    )
    assert list(rows) == [] and wire.calls == []
