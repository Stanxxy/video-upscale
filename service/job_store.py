import asyncio
from datetime import datetime, timezone
from uuid import uuid4
from service.models import JobResponse, JobStatus, JobRequest


class InMemoryJobStore:
    def __init__(self):
        self._jobs: dict[str, JobResponse] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, request: JobRequest) -> JobResponse:
        async with self._lock:
            job_id = str(uuid4())
            now = datetime.now(timezone.utc).isoformat()
            job = JobResponse(
                job_id=job_id,
                status=JobStatus.QUEUED,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
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
