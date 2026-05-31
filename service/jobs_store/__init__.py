"""Persistent jobs store backed by Keyspaces (Cassandra).

Operates on the same four tables as shared_lib's AnalysisJobsStore.
Returns plain dicts (string job_state / stage values) for routes.
"""

from service.jobs_store.checkpoints_store import CheckpointsStoreMixin
from service.jobs_store.latest_job import LatestJobMixin
from service.jobs_store.lifecycle import LifecycleMixin
from service.jobs_store.recovery_index import RecoveryIndexMixin
from service.jobs_store.request_params import RequestParamsMixin
from service.keyspaces_client import KeyspacesClient


class JobsStore(
    LifecycleMixin,
    RecoveryIndexMixin,
    CheckpointsStoreMixin,
    RequestParamsMixin,
    LatestJobMixin,
):
    """Async facade over Keyspaces for the 4 analysis-job tables."""

    def __init__(self, client: KeyspacesClient) -> None:
        self._client = client
        self._ks = client.keyspace
        self.owned_jobs: set[str] = set()


__all__ = ["JobsStore"]
