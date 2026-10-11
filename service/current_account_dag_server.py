"""Account-only current DAG runtime composition.

This opt-in seam deliberately coexists with ``service.dag_runtime``'s existing
two-lane composition. It does not synthesize a session lane or alter the public
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


class AccountOnlyDagRunAuthority(DagRunAuthority):
    """Signed authority containing exactly one genuine account host client."""

    def __init__(self, account: DagInternalClient):
        if not isinstance(account, DagInternalClient):
            raise TypeError("one signed account authority client is required")
        if account.key.owner_kind != "account":
            raise ValueError("account authority client must be account-bound")
        # Keep the inherited callback and deny-by-absence behavior for every
        # unsupported owner kind. The base constructor is intentionally not
        # called because it requires a session lane.
        self.clients = {"account": account}


def current_account_authority_client(
    origin, *, access_key: DagServiceKey, audience, local_test=False, transport=None
):
    """Create the one signed account gateway client for an account dispatcher."""
    if not isinstance(access_key, DagServiceKey) or access_key.owner_kind != "account":
        raise ValueError("account-bound gateway access credential required")
    return AccountOnlyDagRunAuthority(
        DagInternalClient(
            origin,
            key=access_key,
            audience=audience,
            local_test=local_test,
            transport=transport,
        )
    )


def current_account_position_dispatcher(
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
    """Build the one real position dispatcher over admitted current AI stores."""
    from shared_lib.infrastructure.admitted_dynamo_client import AdmittedDynamoClient
    from shared_lib.infrastructure.result_publication_store import (
        DynamoResultPublicationStore,
    )
    from shared_lib.infrastructure.result_publication_table import ResultPublicationTable
    from service.analysis_run_dispatch import PositionRunDispatcher

    if not isinstance(authority, AccountOnlyDagRunAuthority):
        raise ValueError("an explicit current account authority is required")
    if not isinstance(source_bucket, str) or not source_bucket.strip():
        raise ValueError("admitted current source bucket required")
    if not isinstance(reference_bucket, str) or not reference_bucket.strip():
        raise ValueError("admitted current reference bucket required")
    if not isinstance(artifact_bucket, str) or not artifact_bucket.strip():
        raise ValueError("admitted current artifact bucket required")
    if not isinstance(scratch_root, Path) or not scratch_root.is_dir():
        raise ValueError("existing private engine scratch directory required")
    if type(max_source_bytes) is not int or max_source_bytes <= 0:
        raise ValueError("positive current source byte limit required")
    if getattr(proposals, "annotations", None) is None or getattr(
        components, "proposals", None
    ) is not proposals:
        raise ValueError("current proposals and component stores must share authority")
    if not isinstance(publication_table, ResultPublicationTable):
        raise TypeError("the admitted current result-publication table is required")
    admitted_client = getattr(proposals.annotations, "client", None)
    if not isinstance(admitted_client, AdmittedDynamoClient):
        raise ValueError("current result publication requires the admitted DynamoDB facade")
    admitted_client._read_table(publication_table.name)
    if not callable(getattr(s3_client, "get_object", None)):
        raise ValueError("explicit current source/artifact S3 client required")
    if not callable(getattr(session, "execute", None)):
        raise ValueError("explicit current Keyspaces session required")
    if not isinstance(keyspace, str) or not keyspace:
        raise ValueError("explicit current engine keyspace required")

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


def current_account_run_recovery(dispatcher):
    """Owner-requested recovery queue for the account lane of ``dispatcher``."""
    return OwnerRunRecovery(OwnerPositionRunSchedule(dispatcher, owner_kind="account"))


def current_account_engine_service_runtime(
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
    """Build supervised account scheduling, enqueue and result delivery workers.

    ``recovery`` (an account-lane ``OwnerRunRecovery`` on this dispatcher) adds
    the worker that resumes owner-requested stalled runs.

    The caller retains ownership of native clients and the dispatcher and must
    drain this runtime before closing them, matching ``DagServiceRuntime``'s
    lifetime contract.
    """
    if not isinstance(getattr(dispatcher, "authority", None), AccountOnlyDagRunAuthority):
        raise ValueError("current account dispatcher requires account-only signed authority")
    authority = dispatcher.authority
    if set(authority.clients) != {"account"}:
        raise ValueError("current account authority must contain only the account lane")
    account_client = authority.clients["account"]
    if (
        not isinstance(account_client, DagInternalClient)
        or account_client.key.owner_kind != "account"
    ):
        raise ValueError("current account authority client is invalid")
    if (
        not isinstance(enqueue_key, DagServiceKey)
        or enqueue_key.owner_kind != "account"
        or enqueue_key.secret == account_client.key.secret
    ):
        raise ValueError("one distinct account enqueue credential is required")
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
        raise ValueError("current account dispatcher resources are incomplete")
    qualified_model_versions(dispatcher.qualified_models)
    if not isinstance(dispatcher.api_key, str) or not dispatcher.api_key.strip():
        raise ValueError("explicit qualified-provider credential is required")
    if not callable(observe_schedule) or not callable(observe_delivery):
        raise ValueError("schedule and delivery observers are required")

    if recovery is not None and (
        not isinstance(recovery, OwnerRunRecovery)
        or recovery.dispatcher is not dispatcher
        or recovery.schedule.owner_kind != "account"
        or not callable(observe_recovery)
    ):
        raise ValueError("account recovery must share this dispatcher and have an observer")

    schedule = OwnerPositionRunSchedule(dispatcher, owner_kind="account")

    async def enqueue(run_id, owner_kind):
        if owner_kind != "account":
            raise PermissionError("current account host cannot enqueue another owner")
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

    workers = {"account_schedule": schedule_work, "account_delivery": delivery_work}
    if recovery is not None:

        async def recovery_work(stop):
            await recovery.run(stop, observe=observe_recovery)

        workers["account_recovery"] = recovery_work

    return DagServiceRuntime(
        application=endpoint,
        settings=listener_settings,
        workers=workers,
        on_failure=on_failure,
    )
