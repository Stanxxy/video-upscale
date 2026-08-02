"""Health and QA client endpoints."""

from datetime import datetime, timezone

from fastapi.responses import FileResponse

from service.routes.state import QA_HTML


async def qa_client():
    return FileResponse(QA_HTML, media_type="text/html")


async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
