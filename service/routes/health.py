"""Health, debug, and QA client endpoints."""

from datetime import datetime, timezone

import torch
from fastapi.responses import FileResponse

from service.routes.state import QA_HTML


async def qa_client():
    return FileResponse(QA_HTML, media_type="text/html")


async def health():
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def debug_memory():
    """Return GPU/MPS memory usage for QA verification of model offload."""
    result = {"gpu_available": False, "allocated_mb": 0, "reserved_mb": 0}
    if torch.cuda.is_available():
        result["gpu_available"] = True
        result["allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        result["reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        result["gpu_available"] = True
        try:
            result["allocated_mb"] = round(torch.mps.current_allocated_memory() / 1024 / 1024, 1)
        except Exception:
            pass
    return result
