"""Single-owner DAG runtime composition.

One host embeds one engine lane: the current account host runs the account
lane, the guest trial host runs the session lane. This coexists with
``service.dag_runtime``'s two-lane composition and does not alter the public
engine protocol.
"""

from __future__ import annotations

from pathlib import Path

from shared_lib.infrastructure.dag_internal_transport import (
    DagInternalClient,
    DagInternalEndpoint,
    DagRunAuthority,
    DagServiceKey,
)
from shared_lib.infrastructure.dag_service_runtime import DagServiceRuntime

from service.analysis_gemini import qualified_model_versions
from service.analysis_run_recovery import OwnerRunRecovery
from service.analysis_run_schedule import OwnerPositionRunSchedule
from service.result_delivery_supervisor import ResultDeliverySupervisor


class SingleLaneDagRunAuthority(DagRunAuthority):
    """Signed authority containing exactly one genuine host client."""

    def __init__(self, client: DagInternalClient):
        if not isinstance(client, DagInternalClient):
            raise TypeError("one signed authority client is required")
        # DagServiceKey already binds the client to the account or session owner.
        # Keep the inherited callback and deny-by-absence behavior for every
        # other owner kind. The base constructor is intentionally not called
        # because it requires both lanes.
        self.owner_kind = client.key.owner_kind
        self.clients = {self.owner_kind: client}


def single_lane_authority_client(
    origin, *, access_key: DagServiceKey, audience, local_test=False, transport=None
):
    """Create the one signed host client for a single-lane dispatcher."""
    if not isinstance(access_key, DagServiceKey):
        raise ValueError("owner-bound host access credential required")
    return SingleLaneDagRunAuthority(
        DagInternalClient(
            origin,
            key=access_key,
            audience=audience,
            local_test=local_test,
            transport=transport,
        )
    )


def single_lane_position_dispatcher(
    *,
    proposals,
    components,
    publication_table,
    topic_arn,
    authority,
    session,
    s3_client,
    keyspace,
    source_bucket,
    artifact_bucket,
    reference_bucket,
    scratch_root,
    max_source_bytes,
    api_key,
    qualified_models,
    transport=None,
):
    """Build the one real position dispatcher over one lane's admitted stores."""
    from shared_lib.infrastructure.admitted_dynamo_client import AdmittedDynamoClient
    from shared_lib.infrastructure.result_publication_store import (
        DynamoResultPublicationStore,
    )
    from shared_lib.infrastructure.result_publication_table import ResultPublicationTable
    from service.analysis_run_dispatch import PositionRunDispatcher

    if not isinstance(authority, SingleLaneDagRunAuthority):
        raise ValueError("an explicit single-lane authority is required")
    if not isinstance(source_bucket, str) or not source_bucket.strip():
        raise ValueError("admitted source bucket required")
    if not isinstance(reference_bucket, str) or not reference_bucket.strip():
        raise ValueError("admitted reference bucket required")
    if not isinstance(artifact_bucket, str) or not artifact_bucket.strip():
        raise ValueError("admitted artifact bucket required")
    if not isinstance(scratch_root, Path) or not scratch_root.is_dir():
        raise ValueError("existing private engine scratch directory required")
    if type(max_source_bytes) is not int or max_source_bytes <= 0:
        raise ValueError("positive source byte limit required")
    if getattr(proposals, "annotations", None) is None or getattr(
        components, "proposals", None
    ) is not proposals:
        raise ValueError("proposals and component stores must share authority")
    if not isinstance(publication_table, ResultPublicationTable):
        raise TypeError("the admitted result-publication table is required")
    admitted_client = getattr(proposals.annotations, "client", None)
    if not isinstance(admitted_client, AdmittedDynamoClient):
        raise ValueError("result publication requires the admitted DynamoDB facade")
    admitted_client._read_table(publication_table.name)
    if not callable(getattr(s3_client, "get_object", None)):
        raise ValueError("explicit source/artifact S3 client required")
    if not callable(getattr(session, "execute", None)):
        raise ValueError("explicit Keyspaces session required")
    if not isinstance(keyspace, str) or not keyspace:
        raise ValueError("explicit engine keyspace required")

    publication_store = DynamoResultPublicationStore(
        proposals, components, publication_table, topic_arn
    )
    return PositionRunDispatcher(
        proposals=proposals,
        outbox=publication_store,
        session=session,
        s3_client=s3_client,
        keyspace=keyspace,
        artifact_bucket=artifact_bucket,
        allowed_buckets=frozenset(
            {source_bucket, artifact_bucket, reference_bucket}
        ),
        scratch_root=scratch_root,
        max_source_bytes=max_source_bytes,
        api_key=api_key,
        qualified_models=qualified_models,
        authority=authority,
        lease_seconds=60,
        heartbeat_seconds=10,
        mime_type="video/mp4",
        transport=transport,
    )


def _lane_owner(dispatcher):
    authority = getattr(dispatcher, "authority", None)
    if not isinstance(authority, SingleLaneDagRunAuthority):
        raise ValueError("dispatcher requires single-lane signed authority")
    owner = authority.owner_kind
    if set(authority.clients) != {owner}:
        raise ValueError("single-lane authority must contain only its own lane")
    client = authority.clients[owner]
    if not isinstance(client, DagInternalClient) or client.key.owner_kind != owner:
        raise ValueError("single-lane authority client is invalid")
    return owner, client


def single_lane_run_recovery(dispatcher):
    """Owner-requested recovery queue for the lane of ``dispatcher``."""
    owner, _client = _lane_owner(dispatcher)
    return OwnerRunRecovery(OwnerPositionRunSchedule(dispatcher, owner_kind=owner))


def single_lane_engine_service_runtime(
    dispatcher,
    sns_client,
    *,
    listener_settings,
    enqueue_key: DagServiceKey,
    enqueue_audience,
    observe_schedule,
    observe_delivery,
    on_failure,
    interval_seconds=5,
    recovery=None,
    observe_recovery=None,
):
    """Build supervised scheduling, enqueue and result delivery for one lane.

    ``recovery`` (an ``OwnerRunRecovery`` on this dispatcher's lane) adds the
    worker that resumes owner-requested stalled runs.

    The caller retains ownership of native clients and the dispatcher and must
    drain this runtime before closing them, matching ``DagServiceRuntime``'s
    lifetime contract.
    """
    owner, client = _lane_owner(dispatcher)
    authority = dispatcher.authority
    if (
        not isinstance(enqueue_key, DagServiceKey)
        or enqueue_key.owner_kind != owner
        or enqueue_key.secret == client.key.secret
    ):
        raise ValueError(f"one distinct {owner} enqueue credential is required")
    if not callable(getattr(sns_client, "publish", None)):
        raise ValueError("real result publication client is required")
    proposals = getattr(dispatcher, "proposals", None)
    outbox = getattr(dispatcher, "outbox", None)
    artifact_bucket = getattr(dispatcher, "bucket", None)
    allowed_buckets = getattr(dispatcher, "allowed_buckets", None)
    if (
        proposals is None
        or outbox is None
        or getattr(outbox, "proposals", None) is not proposals
        or not getattr(getattr(proposals, "tables", None), "runs", None)
        or not getattr(getattr(outbox, "table", None), "name", None)
        or not isinstance(artifact_bucket, str)
        or not artifact_bucket.strip()
        or not isinstance(allowed_buckets, (set, frozenset))
        or artifact_bucket not in allowed_buckets
    ):
        raise ValueError(f"{owner} dispatcher resources are incomplete")
    qualified_model_versions(dispatcher.qualified_models)
    if not isinstance(dispatcher.api_key, str) or not dispatcher.api_key.strip():
        raise ValueError("explicit qualified-provider credential is required")
    if not callable(observe_schedule) or not callable(observe_delivery):
        raise ValueError("schedule and delivery observers are required")

    if recovery is not None and (
        not isinstance(recovery, OwnerRunRecovery)
        or recovery.dispatcher is not dispatcher
        or recovery.schedule.owner_kind != owner
        or not callable(observe_recovery)
    ):
        raise ValueError("recovery must share this dispatcher and have an observer")

    schedule = OwnerPositionRunSchedule(dispatcher, owner_kind=owner)

    async def enqueue(run_id, owner_kind):
        if owner_kind != owner:
            raise PermissionError(f"{owner} host cannot enqueue another owner")
        return await schedule.enqueue(run_id)

    endpoint = DagInternalEndpoint(
        action="enqueue",
        audience=enqueue_audience,
        keys=[enqueue_key],
        handler=enqueue,
    )
    delivery = ResultDeliverySupervisor(
        outbox,
        sns_client,
        authority=authority,
    )

    async def schedule_work(stop):
        await schedule.run(
            stop,
            observe=observe_schedule,
            interval_seconds=interval_seconds,
        )

    async def delivery_work(stop):
        await delivery.run(
            stop,
            observe=observe_delivery,
            interval_seconds=interval_seconds,
        )

    workers = {f"{owner}_schedule": schedule_work, f"{owner}_delivery": delivery_work}
    if recovery is not None:

        async def recovery_work(stop):
            await recovery.run(stop, observe=observe_recovery)

        workers[f"{owner}_recovery"] = recovery_work

    return DagServiceRuntime(
        application=endpoint,
        settings=listener_settings,
        workers=workers,
        on_failure=on_failure,
    )
