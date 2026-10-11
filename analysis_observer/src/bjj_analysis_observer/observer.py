"""Host adapter: actual viewer identity plus current host authorization on every read.

Hosts establish account/session identity. This adapter never issues credentials,
chooses an owner from an IP address, admits work, or trusts a browser owner field.
"""

from uuid import UUID

from bjj_analysis_observer.checkpoints import KeyspacesCheckpointReader
from bjj_analysis_observer.lifecycle import read_status
from shared_lib.infrastructure.annotation_store import AnnotationNotFound
from shared_lib.models.analysis_contracts import PrincipalRef


class AuthorizedRunObserver:
    def __init__(
        self,
        proposals,
        session,
        s3_client,
        *,
        keyspace,
        bucket,
        authorize,
        lease_seconds,
        timeout=10,
        max_item_reads=2000,
    ):
        if not callable(authorize):
            raise ValueError("current host resource authorization required")
        self.proposals, self.session, self.s3 = proposals, session, s3_client
        self.keyspace, self.bucket = keyspace, bucket
        self.authorize, self.lease_seconds = authorize, lease_seconds
        self.timeout, self.max_item_reads = timeout, max_item_reads

    async def status(self, analysis_id, run_id, *, actor):
        """actor comes from verified host credentials, not the URL/body/run owner."""
        analysis_id, run_id = UUID(str(analysis_id)), UUID(str(run_id))
        actor = PrincipalRef.model_validate(actor)

        async def resolve():
            candidate = await self.proposals._run(analysis_id, run_id, actor)
            if candidate is None or candidate.spec.owner != actor:
                raise AnnotationNotFound("Analysis run not found")
            # Account: recheck video access. Session: recheck session/source expiry.
            await self.authorize(candidate, actor)
            return candidate

        run = await resolve()

        async def guard():
            if await resolve() != run:
                raise AnnotationNotFound("Analysis run changed")

        journal = KeyspacesCheckpointReader(
            run,
            self.session,
            self.s3,
            keyspace=self.keyspace,
            bucket=self.bucket,
            guard=guard,
            timeout=self.timeout,
        )
        return await read_status(
            journal,
            session=self.session,
            keyspace=self.keyspace,
            guard=guard,
            lease_seconds=self.lease_seconds,
            timeout=self.timeout,
            max_item_reads=self.max_item_reads,
        )
