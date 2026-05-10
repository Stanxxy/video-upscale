import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from service.models import JobResponse, JobStatus, TrackRequest


class InMemoryJobStore:
    def __init__(self):
        self._jobs: dict[str, JobResponse] = {}
        self._requests: dict[str, TrackRequest] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, request: TrackRequest) -> JobResponse:
        async with self._lock:
            job_id = str(uuid4())
            now = datetime.now(timezone.utc).isoformat()
            job = JobResponse(
                job_id=job_id,
                status=JobStatus.PENDING,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
            self._requests[job_id] = request
            self._cancel_events[job_id] = asyncio.Event()
            return job

    async def hydrate_job(self, job_id: str, request: TrackRequest) -> JobResponse:
        """Register a job_id already persisted in Keyspaces (post-restart worker resume).

        Idempotent: returns the existing in-memory row if ``job_id`` is already known.
        """
        async with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                return existing
            now = datetime.now(timezone.utc).isoformat()
            job = JobResponse(
                job_id=job_id,
                status=JobStatus.PENDING,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
            self._requests[job_id] = request
            self._cancel_events[job_id] = asyncio.Event()
            return job

    async def update_job(self, job_id: str, **kwargs) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            now = datetime.now(timezone.utc).isoformat()
            updated = job.model_copy(update={**kwargs, "updated_at": now})
            self._jobs[job_id] = updated

    async def get_job(self, job_id: str) -> JobResponse | None:
        async with self._lock:
            return self._jobs.get(job_id)

    def get_request(self, job_id: str) -> TrackRequest | None:
        return self._requests.get(job_id)

    async def set_cancelled(self, job_id: str) -> None:
        event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
        await self.update_job(job_id, status=JobStatus.CANCELLED)

    def is_cancelled(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        return event is not None and event.is_set()
