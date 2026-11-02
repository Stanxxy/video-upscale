"""Account-only current DAG runtime composition.

The current account host's binding of ``service.single_lane_dag_server``: every
entry point here refuses anything but the account lane, and the dispatcher reads
only the account namespace's admitted stores.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path

from shared_lib.infrastructure.dag_internal_transport import (
    DagInternalClient,
    DagServiceKey,
)

from service.single_lane_dag_server import (
    SingleLaneDagRunAuthority,
    single_lane_engine_service_runtime,
    single_lane_run_recovery,
)


LOG = logging.getLogger(__name__)

# A run's guard asks for authority several times per checkpoint, media chunk and
# heartbeat; each answer is a full signed re-verification (~20 store reads, ~1.5 s
# in production). A confirmation is reused for this long, so a revocation takes
# effect within it, and concurrent asks share one check.
AUTHORITY_REUSE_SECONDS = 5.0
# A check that cannot be confirmed (store hiccup, slow reply, or a denial) is
# asked again after these pauses before the run is stopped; a denial still denies.
AUTHORITY_RETRY_DELAYS = (0.25, 1.0)


class AccountOnlyDagRunAuthority(SingleLaneDagRunAuthority):
    """Signed authority containing exactly one genuine account host client."""

    def __init__(
        self,
        account: DagInternalClient,
        *,
        reuse_seconds=AUTHORITY_REUSE_SECONDS,
        retry_delays=AUTHORITY_RETRY_DELAYS,
        clock=time.monotonic,
        sleep=asyncio.sleep,
    ):
        if not isinstance(account, DagInternalClient):
            raise TypeError("one signed account authority client is required")
        if account.key.owner_kind != "account":
            raise ValueError("account authority client must be account-bound")
        if not (
            isinstance(reuse_seconds, (int, float))
            and math.isfinite(reuse_seconds)
            and 0 <= reuse_seconds <= 30
        ) or any(
            not isinstance(d, (int, float)) or not math.isfinite(d) or not 0 <= d <= 5
            for d in retry_delays
        ):
            raise ValueError("bounded authority reuse and retry delays required")
        super().__init__(account)
        self.reuse_seconds, self.retry_delays = reuse_seconds, tuple(retry_delays)
        self.clock, self.sleep = clock, sleep
        self._confirmed = {}  # run_id -> monotonic deadline of reuse
        self._checks = {}  # run_id -> the one in-flight check

    async def __call__(self, run):
        if self.clients.get(run.spec.owner.kind) is None:
            raise PermissionError("unsupported authority host")
        run_id = run.spec.run_id
        if self.clock() < self._confirmed.get(run_id, -math.inf):
            return
        check = self._checks.get(run_id)
        if check is None:
            check = asyncio.ensure_future(self._confirm(run_id))
            self._checks[run_id] = check
            check.add_done_callback(
                lambda done: self._checks.pop(run_id, None)
                if self._checks.get(run_id) is done
                else None
            )
        # A cancelled caller must not cancel the check other callers share.
        await asyncio.shield(check)

    async def _confirm(self, run_id):
        client = self.clients[self.owner_kind]
        for attempt, delay in enumerate((*self.retry_delays, None), 1):
            started = self.clock()
            try:
                await client.call("authorize", run_id)
            except Exception as error:
                if delay is None:
                    raise
                LOG.warning(
                    "account authority for run %s unconfirmed (attempt %d): %s",
                    run_id,
                    attempt,
                    type(error).__name__,
                )
                await self.sleep(delay)
                continue
            # Reuse counts from when the confirming check started, not ended.
            now = self.clock()
            self._confirmed = {
                k: v for k, v in self._confirmed.items() if v > now
            }
            self._confirmed[run_id] = started + self.reuse_seconds
            return


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


def _require_account(dispatcher):
    if not isinstance(getattr(dispatcher, "authority", None), AccountOnlyDagRunAuthority):
        raise ValueError("current account dispatcher requires account-only signed authority")


def current_account_run_recovery(dispatcher):
    """Owner-requested recovery queue for the account lane of ``dispatcher``."""
    _require_account(dispatcher)
    return single_lane_run_recovery(dispatcher)


def current_account_engine_service_runtime(dispatcher, sns_client, **settings):
    """Build supervised account scheduling, enqueue and result delivery workers.

    See ``single_lane_engine_service_runtime`` for the settings and the
    lifetime contract.
    """
    _require_account(dispatcher)
    return single_lane_engine_service_runtime(dispatcher, sns_client, **settings)
