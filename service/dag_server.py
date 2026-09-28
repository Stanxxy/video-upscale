"""Explicit owned engine entrypoint; no DDL, model defaults or legacy recovery changes."""

import asyncio
import json
import logging
import os
import ssl
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from pathlib import Path

from shared_lib.infrastructure.analysis_baseline_artifacts import S3AnalysisBaselines
from shared_lib.infrastructure.annotation_store import DynamoAnnotationStore
from shared_lib.infrastructure.annotation_tables import AnnotationTables
from shared_lib.infrastructure.component_result_store import (
    DynamoAIComponentResultStore,
)
from shared_lib.infrastructure.dag_admission_relay import run_io
from shared_lib.infrastructure.dag_internal_transport import (
    DagInternalClient,
    DagRunAuthority,
    DagServiceKey,
)
from shared_lib.infrastructure.dag_service_runtime import (
    DagListenerSettings,
    _finish,
    serve_dag_host,
)
from shared_lib.infrastructure.proposal_store import DynamoProposalStore
from shared_lib.infrastructure.proposal_tables import ProposalTables
from shared_lib.infrastructure.result_publication_store import (
    DynamoResultPublicationStore,
)
from shared_lib.infrastructure.result_publication_table import ResultPublicationTable

from service.analysis_checkpoint_journal import _target
from service.analysis_gemini import qualified_model_versions
from service.analysis_run_dispatch import PositionRunDispatcher
from service.dag_runtime import partitioned_engine_service_runtime

LOG = logging.getLogger(__name__)
OWNERS = ("account", "session")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate engine configuration key")
        result[key] = value
    return result


def settings_from_env(environ):
    def required(name):
        value = environ[name]
        if not value or value != value.strip():
            raise ValueError(f"explicit {name} required")
        return value

    if required("BJJ_DAG_ENGINE_ENABLED") != "1":
        raise ValueError("explicit DAG engine activation required")
    models = qualified_model_versions(
        json.loads(required("BJJ_DAG_QUALIFIED_MODELS_JSON"), object_pairs_hook=_pairs)
    )
    listener = DagListenerSettings(
        required("BJJ_DAG_INTERNAL_HOST"),
        int(required("BJJ_DAG_INTERNAL_PORT")),
        certificate=required("BJJ_DAG_INTERNAL_CERTIFICATE"),
        private_key=required("BJJ_DAG_INTERNAL_PRIVATE_KEY"),
    )
    lanes = {}
    for owner in OWNERS:
        prefix = f"BJJ_DAG_{owner.upper()}_"
        buckets = json.loads(required(prefix + "SOURCE_BUCKETS_JSON"))
        if (
            not isinstance(buckets, list)
            or not buckets
            or any(not isinstance(b, str) or not b or b != b.strip() for b in buckets)
        ):
            raise ValueError("explicit source bucket list required")
        scratch = Path(required(prefix + "SCRATCH_ROOT"))
        if not scratch.is_absolute() or not scratch.is_dir():
            raise ValueError("existing absolute scratch directory required")
        lanes[owner] = dict(
            annotations=AnnotationTables.with_prefix(
                required(prefix + "ANNOTATION_TABLE_PREFIX")
            ),
            proposals=ProposalTables.with_prefix(
                required(prefix + "PROPOSAL_TABLE_PREFIX")
            ),
            publication=ResultPublicationTable(required(prefix + "PUBLICATION_TABLE")),
            artifact_bucket=required(prefix + "ARTIFACT_BUCKET"),
            allowed_buckets=frozenset(buckets),
            scratch_root=scratch,
            topic_arn=required(prefix + "RESULT_TOPIC_ARN"),
            access_origin=required(prefix + "ACCESS_ORIGIN"),
            access_audience=required(prefix + "ACCESS_AUDIENCE"),
            access_key=DagServiceKey(
                required(prefix + "ACCESS_KEY_ID"),
                required(prefix + "ACCESS_SECRET"),
                owner,
            ),
            enqueue_key=DagServiceKey(
                required(prefix + "ENQUEUE_KEY_ID"),
                required(prefix + "ENQUEUE_SECRET"),
                owner,
            ),
        )
    all_secrets = [
        lane[key].secret
        for lane in lanes.values()
        for key in ("access_key", "enqueue_key")
    ]
    if len(set(all_secrets)) != 4:
        raise ValueError("four distinct host credentials required")
    max_bytes = int(required("BJJ_DAG_MAX_SOURCE_BYTES"))
    if max_bytes <= 0:
        raise ValueError("positive source size bound required")
    keyspace = required("BJJ_DAG_KEYSPACE")
    _target(keyspace, "job_lifecycle")
    return dict(
        lanes=lanes,
        qualified_models=models,
        api_key=required("BJJ_DAG_GEMINI_API_KEY"),
        listener_settings=listener,
        enqueue_audience=required("BJJ_DAG_ENQUEUE_AUDIENCE"),
        region=required("BJJ_DAG_AWS_REGION"),
        keyspace=keyspace,
        cassandra_host=required("BJJ_DAG_CASSANDRA_HOST"),
        cassandra_port=int(required("BJJ_DAG_CASSANDRA_PORT")),
        cassandra_username=required("BJJ_DAG_CASSANDRA_USERNAME"),
        cassandra_password=required("BJJ_DAG_CASSANDRA_PASSWORD"),
        cassandra_ca=required("BJJ_DAG_CASSANDRA_CA_FILE"),
        max_source_bytes=max_bytes,
    )


def _verify_table(client, definition):
    actual = client.describe_table(TableName=definition["TableName"])["Table"]
    if actual.get("TableStatus") != "ACTIVE":
        raise ValueError("DAG table is not active")
    for field in ("KeySchema", "AttributeDefinitions"):
        if sorted(actual[field], key=lambda x: x["AttributeName"]) != sorted(
            definition[field], key=lambda x: x["AttributeName"]
        ):
            raise ValueError("DAG table schema differs")
    for expected in definition.get("GlobalSecondaryIndexes", []):
        index = next(
            (
                i
                for i in actual.get("GlobalSecondaryIndexes", [])
                if i["IndexName"] == expected["IndexName"]
            ),
            None,
        )
        if (
            index is None
            or index.get("IndexStatus") != "ACTIVE"
            or any(
                index[field] != expected[field] for field in ("KeySchema", "Projection")
            )
        ):
            raise ValueError("DAG table index is not ready or differs")


def prepare_dispatchers(
    settings, *, dynamodb, s3, sns, session, authority, transports=None
):
    """Read-only qualification using supplied native clients; never creates resources."""
    names = set()
    for lane in settings["lanes"].values():
        for group in (lane["annotations"], lane["proposals"], lane["publication"]):
            for definition in group.definitions():
                if definition["TableName"] in names:
                    raise ValueError("owner model tables must not overlap")
                names.add(definition["TableName"])
                _verify_table(dynamodb, definition)
        for bucket in lane["allowed_buckets"] | {lane["artifact_bucket"]}:
            s3.head_bucket(Bucket=bucket)
        if (
            sns.get_topic_attributes(TopicArn=lane["topic_arn"])["Attributes"][
                "TopicArn"
            ]
            != lane["topic_arn"]
        ):
            raise ValueError("result topic differs")
        if not lane["scratch_root"].is_dir():
            raise ValueError("scratch directory missing")
    # Prepare validates the real CQL columns/types without scanning history or DDL.
    keyspace = settings["keyspace"]
    for table, query in (
        (
            "job_lifecycle",
            "SELECT job_id,video_id,user_id,pipeline_kind,job_state,owner_instance_id,last_heartbeat_at,started_at,updated_at,completed_at,stage,progress_percent FROM {} WHERE job_id=?",
        ),
        (
            "job_recovery_index",
            "SELECT recovery_state,heartbeat_bucket,last_heartbeat_at,job_id,owner_instance_id,video_id,job_state,updated_at FROM {} WHERE recovery_state=? AND heartbeat_bucket=?",
        ),
        (
            "analysis_execution_checkpoints",
            "SELECT run_id,step_id,revision,record_revision,artifact_key,artifact_sha256,input_sha256 FROM {} WHERE run_id=? AND step_id=?",
        ),
    ):
        session.prepare(query.format(_target(keyspace, table)))
    result = {}
    for owner, lane in settings["lanes"].items():
        annotations = DynamoAnnotationStore(dynamodb, lane["annotations"])
        proposals = DynamoProposalStore(
            annotations,
            lane["proposals"],
            S3AnalysisBaselines(s3, lane["artifact_bucket"]),
        )
        components = DynamoAIComponentResultStore(
            proposals, s3, lane["artifact_bucket"]
        )
        outbox = DynamoResultPublicationStore(
            proposals, components, lane["publication"], lane["topic_arn"]
        )
        result[owner] = PositionRunDispatcher(
            proposals=proposals,
            outbox=outbox,
            session=session,
            s3_client=s3,
            keyspace=keyspace,
            artifact_bucket=lane["artifact_bucket"],
            allowed_buckets=lane["allowed_buckets"],
            scratch_root=lane["scratch_root"],
            max_source_bytes=settings["max_source_bytes"],
            api_key=settings["api_key"],
            qualified_models=settings["qualified_models"],
            authority=authority,
            lease_seconds=60,
            heartbeat_seconds=10,
            mime_type="video/mp4",
            transport=None if transports is None else transports[owner],
        )
    return result


def _native_clients(settings, stack):
    import boto3
    from botocore.config import Config
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.cluster import Cluster

    config = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 0})
    aws = boto3.Session(region_name=settings["region"])
    clients = {}
    for kind in ("dynamodb", "s3", "sns"):
        client = aws.client(kind, config=config)
        stack.callback(client.close)
        clients[kind] = client
    tls = ssl.create_default_context(cafile=settings["cassandra_ca"])
    cluster = Cluster(
        [settings["cassandra_host"]],
        port=settings["cassandra_port"],
        protocol_version=4,
        ssl_context=tls,
        auth_provider=PlainTextAuthProvider(
            settings["cassandra_username"], settings["cassandra_password"]
        ),
        connect_timeout=5,
        control_connection_timeout=10,
    )
    stack.callback(cluster.shutdown)
    clients["session"] = cluster.connect(settings["keyspace"])
    return clients


def runtime_factory(
    settings,
    *,
    native_clients=_native_clients,
    transports=None,
    access_transports=None,
    observe=None,
    interval_seconds=5,
):
    """The owning host must drain requests/runtime before leaving this context."""

    @asynccontextmanager
    async def factory(app, on_failure):
        stack, clients, cleanup = ExitStack(), {}, AsyncExitStack()
        cleanup.push_async_callback(run_io, stack.close)
        try:
            for owner, lane in settings["lanes"].items():
                clients[owner] = DagInternalClient(
                    lane["access_origin"],
                    key=lane["access_key"],
                    audience=lane["access_audience"],
                    transport=(
                        None if access_transports is None else access_transports[owner]
                    ),
                )
                cleanup.push_async_callback(clients[owner].aclose)
            native = await run_io(native_clients, settings, stack)
            dispatchers = await run_io(
                prepare_dispatchers,
                settings,
                **native,
                authority=DagRunAuthority(**clients),
                transports=transports,
            )

            async def schedule(owner, page):
                if observe is not None:
                    await observe("schedule", owner, page)
                else:
                    LOG.info(
                        "DAG schedule owner=%s attempts=%s error=%s",
                        owner,
                        len(page.attempts),
                        page.error_code,
                    )

            async def delivery(owner, page):
                if observe is not None:
                    await observe("delivery", owner, page)
                else:
                    LOG.info(
                        "DAG delivery owner=%s attempts=%s", owner, len(page.attempts)
                    )

            runtime = partitioned_engine_service_runtime(
                dispatchers,
                {owner: native["sns"] for owner in OWNERS},
                listener_settings=settings["listener_settings"],
                enqueue_keys=[
                    settings["lanes"][owner]["enqueue_key"] for owner in OWNERS
                ],
                enqueue_audience=settings["enqueue_audience"],
                observe_schedule=schedule,
                observe_delivery=delivery,
                on_failure=on_failure,
                interval_seconds=interval_seconds,
            )
            yield runtime
        finally:
            # run_io joins driver operations before this native-client owner unwinds.
            await _finish(asyncio.create_task(cleanup.aclose()))

    return factory


def run(app, *, settings=None):
    import uvicorn

    settings = settings_from_env(os.environ) if settings is None else settings
    app.state.dag_owned_entrypoint = True
    try:
        asyncio.run(
            serve_dag_host(
                app,
                public_config=uvicorn.Config(
                    app,
                    host=os.environ.get("BJJ_HOST", "0.0.0.0"),
                    port=int(os.environ.get("BJJ_PORT", "8000")),
                    proxy_headers=False,
                    access_log=False,
                    timeout_graceful_shutdown=30,
                ),
                runtime_factory=runtime_factory(settings),
            )
        )
    finally:
        app.state.dag_owned_entrypoint = False


if __name__ == "__main__":
    # Reject incomplete configuration before importing/starting the legacy host.
    settings = settings_from_env(os.environ)
    from service.app import app

    run(app, settings=settings)
