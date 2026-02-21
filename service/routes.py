import asyncio
import torch
from fastapi import APIRouter, HTTPException

from service.models import JobRequest, JobResponse
from service.job_store import InMemoryJobStore
from service.worker import run_job
from service.config import ServiceConfig

router = APIRouter()

# These are set during app lifespan
_config: ServiceConfig | None = None
_job_store: InMemoryJobStore | None = None
_job_semaphore: asyncio.Semaphore | None = None


def init_routes(config: ServiceConfig, job_store: InMemoryJobStore):
    global _config, _job_store, _job_semaphore
    _config = config
    _job_store = job_store
    _job_semaphore = asyncio.Semaphore(config.max_concurrent_jobs)


async def _run_with_semaphore(job_id: str, request: JobRequest):
    async with _job_semaphore:
        await run_job(job_id, request, _config, _job_store)


@router.post("/jobs", status_code=202, response_model=JobResponse)
async def create_job(request: JobRequest):
    if _job_semaphore.locked():
        raise HTTPException(429, "Server is at capacity. Try again later.")
    job = await _job_store.create_job(request)
    asyncio.create_task(_run_with_semaphore(job.job_id, request))
    return job


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job = await _job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/health")
async def health():
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    return {"status": "ok", "gpu_available": gpu_available}
