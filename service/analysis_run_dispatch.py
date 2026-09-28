"""Explicit, version-bound dispatch of registered position runs; no public route.

All clients, credentials, model qualifications and host policy are server-owned.
This module never admits a run, debits usage, retries uncertain work or starts a loop.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import get_args
from uuid import UUID

from bjj_analysis_core import CORE_VERSION, WORKFLOW_ID
from bjj_analysis_core.execution import RecoveryRequired
from bjj_analysis_core.requests import NodeId
from bjj_analysis_observer.checkpoints import KeyspacesCheckpointReader
from bjj_analysis_observer.lifecycle import read_status
from bjj_analysis_observer.progress import read_progress
from bjj_analysis_observer.results import read_result_status
from shared_lib.models.analysis_contracts import PrivateArtifactRef

from service.analysis_checkpoint_journal import KeyspacesCheckpointJournal
from service.analysis_gemini import GeminiProvider, S3GeminiArtifacts
from service.analysis_media import (
    DeterministicPreparationRejected,
    S3RunMedia,
)
from service.analysis_run_host import DurableGeminiRun
from service.analysis_run_worker import KeyspacesRunLease, run_position_attempt


class UnsupportedRunVersion(DeterministicPreparationRejected):
    pass


@dataclass(frozen=True)
class ProviderActions:
    """Internal explicit instructions; never admission input or retry authority."""

    cleanup: bool = False
    upload_receipt: PrivateArtifactRef | None = None
    cleanup_receipt: PrivateArtifactRef | None = None

    def __post_init__(self):
        if type(self.cleanup) is not bool:
            raise ValueError("explicit boolean cleanup choice required")
        for field in ("upload_receipt", "cleanup_receipt"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(
                    self, field, PrivateArtifactRef.model_validate(value)
                )
        if self.upload_receipt is not None and self.cleanup_receipt is not None:
            raise ValueError("recover one provider operation at a time")
        if self.cleanup_receipt is not None and not self.cleanup:
            raise ValueError("cleanup receipt requires explicit cleanup")

    def validate_recovery(self, recover):
        if (
            self.upload_receipt is not None or self.cleanup_receipt is not None
        ) and recover is not True:
            raise ValueError("provider receipt recovery requires recover=True")


class _AttemptHost:
    """Fixed host actions inside the worker's existing heartbeat/teardown lifetime."""

    def __init__(self, host, actions):
        self.host, self.actions, self.run = host, actions, host.run

    async def execute(self, *, retry_failed=False):
        host, actions = self.host, self.actions
        if actions.upload_receipt is not None:
            await host.reconcile_upload(actions.upload_receipt)
        if actions.cleanup_receipt is not None:
            await host.reconcile_cleanup(actions.cleanup_receipt)
        result = await host.execute(retry_failed=retry_failed)
        if actions.cleanup:
            await host.cleanup()
        else:
            cleanup = await host._read("provider_cleanup")
            if cleanup is not None and cleanup.phase != "succeeded":
                raise RecoveryRequired(
                    "pending cleanup must be explicitly reconciled before lifecycle completion"
                )
        return result


class PublicationSink:
    """Core completion acknowledges durable intent, not SNS or consumer readiness."""

    def __init__(self, run, outbox, guard):
        self.run, self.outbox, self.guard = run, outbox, guard

    async def publish(self, run, component, revision):
        if run != self.run:
            raise ValueError("publication belongs to another registered run")
        await self.guard()
        await self.outbox.prepare(run.spec.run_id, component, revision)
        await self.guard()


class PositionRunDispatcher:
    def __init__(
        self,
        *,
        proposals,
        outbox,
        session,
        s3_client,
        keyspace,
        artifact_bucket,
        allowed_buckets,
        scratch_root,
        max_source_bytes,
        api_key,
        qualified_models,
        authority,
        lease_seconds,
        heartbeat_seconds,
        mime_type,
        concurrency=1,
        checkpoint_timeout=10,
        provider_timeout=180,
        transport=None,
    ):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("positive worker concurrency required")
        if not callable(authority) or outbox.proposals is not proposals:
            raise ValueError(
                "shared authoritative run store and live authority required"
            )
        self.proposals, self.outbox = proposals, outbox
        self.session, self.s3 = session, s3_client
        self.keyspace, self.bucket = keyspace, artifact_bucket
        self.allowed_buckets = frozenset(allowed_buckets)
        self.scratch_root, self.max_source_bytes = scratch_root, max_source_bytes
        self.api_key = api_key
        self.qualified_models = {
            key: frozenset(value) for key, value in qualified_models.items()
        }
        self.authority = authority
        self.lease_seconds, self.heartbeat_seconds = lease_seconds, heartbeat_seconds
        self.mime_type, self.checkpoint_timeout = mime_type, checkpoint_timeout
        self.provider_timeout, self.transport = provider_timeout, transport
        self.slots = asyncio.Semaphore(concurrency)

    @staticmethod
    def _version(run):
        if (run.spec.workflow_id, run.spec.core_version) != (WORKFLOW_ID, CORE_VERSION):
            raise UnsupportedRunVersion(
                "no installed executor for admitted workflow/core version"
            )

    async def _authority(self, run):
        stored = await self.proposals.resolve_delivery_run(run.spec.run_id)
        if stored != run:
            raise PermissionError("registered run changed during execution")
        self._version(stored)
        await self.authority(stored)

    @asynccontextmanager
    async def _host(self, run, guard):
        async def run_guard(candidate):
            if candidate != run:
                raise PermissionError("resource belongs to another run")
            await guard()

        media_store = S3RunMedia(
            self.s3,
            allowed_buckets=self.allowed_buckets,
            scratch_root=self.scratch_root,
            max_source_bytes=self.max_source_bytes,
            guard=run_guard,
        )
        async with media_store.open(run) as media:
            provider = GeminiProvider(
                media,
                S3GeminiArtifacts(run, self.s3, bucket=self.bucket),
                api_key=self.api_key,
                qualified_models=self.qualified_models,
                guard=run_guard,
                timeout_seconds=self.provider_timeout,
                transport=self.transport,
            )
            try:
                # Check every configured node before any upload/model action.
                if any(
                    getattr(provider.config, node).model not in provider.models
                    for node in get_args(NodeId)
                ):
                    raise UnsupportedRunVersion(
                        "admitted DAG contains an unqualified model"
                    )
                journal = KeyspacesCheckpointJournal(
                    run,
                    self.session,
                    self.s3,
                    bucket=self.bucket,
                    keyspace=self.keyspace,
                    guard=guard,
                    timeout=self.checkpoint_timeout,
                )
                yield DurableGeminiRun(
                    run,
                    media.config_bytes,
                    journal=journal,
                    provider=provider,
                    sink=PublicationSink(run, self.outbox, guard),
                    guard=guard,
                    mime_type=self.mime_type,
                )
            finally:
                await provider.aclose()

    @asynccontextmanager
    async def _attempt_host(self, run, guard, actions):
        async with self._host(run, guard) as host:
            yield _AttemptHost(host, actions)

    async def progress(self, run_id, *, max_item_reads=2000):
        """Internal authorized status read, no lease/claim/media/provider action."""
        run = await self.proposals.resolve_delivery_run(UUID(str(run_id)))
        self._version(run)

        async def guard():
            await self._authority(run)

        await guard()
        journal = KeyspacesCheckpointReader(
            run,
            self.session,
            self.s3,
            bucket=self.bucket,
            keyspace=self.keyspace,
            guard=guard,
            timeout=self.checkpoint_timeout,
        )
        return await read_progress(journal, guard=guard, max_item_reads=max_item_reads)

    async def status(self, run_id, *, max_item_reads=2000):
        """Internal guarded progress plus lifecycle observation; never claims work."""
        run = await self.proposals.resolve_delivery_run(UUID(str(run_id)))
        self._version(run)

        async def guard():
            await self._authority(run)

        await guard()
        journal = KeyspacesCheckpointReader(
            run,
            self.session,
            self.s3,
            bucket=self.bucket,
            keyspace=self.keyspace,
            guard=guard,
            timeout=self.checkpoint_timeout,
        )
        return await read_status(
            journal,
            session=self.session,
            keyspace=self.keyspace,
            guard=guard,
            lease_seconds=self.lease_seconds,
            timeout=self.checkpoint_timeout,
            max_item_reads=max_item_reads,
        )

    async def result_status(self, run_id):
        """Internal delivery/review observations, independently available by lane."""
        run = await self.proposals.resolve_delivery_run(UUID(str(run_id)))
        self._version(run)

        async def guard():
            await self._authority(run)

        return await read_result_status(run, self.outbox, guard=guard)

    async def dispatch(
        self, run_id, *, recover=False, retry_failed=False, provider_actions=None
    ):
        """Trusted internal invocation by stored UUID only, never a browser payload.

        Queue discovery/ACK, admission and cancellation policy are deliberately absent.
        Durable core attempts, not caller flags, decide whether replay is safe.
        """
        actions = ProviderActions() if provider_actions is None else provider_actions
        if not isinstance(actions, ProviderActions):
            raise ValueError("trusted ProviderActions required")
        actions.validate_recovery(recover)
        run_id = UUID(str(run_id))
        run = await self.proposals.resolve_delivery_run(run_id)
        self._version(run)
        await self._authority(run)
        # Capacity wait occurs before claiming a lease or opening source media.
        async with self.slots:
            await self._authority(run)
            lease = KeyspacesRunLease(
                run,
                self.session,
                keyspace=self.keyspace,
                authority=self._authority,
                lease_seconds=self.lease_seconds,
                timeout=self.checkpoint_timeout,
            )
            await lease.register()
            return await run_position_attempt(
                lease,
                lambda guard: self._attempt_host(run, guard, actions),
                heartbeat_seconds=self.heartbeat_seconds,
                recover=recover,
                retry_failed=retry_failed,
            )
