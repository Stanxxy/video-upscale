"""Gemini Files API upload lane (S12 Phase 1b production wiring design, §2).

Pure-ish helper module (mirrors ``gemini_retry.py``'s style — no orchestration,
just the Gemini Files API mechanics). Production videos are S3 objects, not
YouTube URLs — every native-video call in the v2 pipeline already treats
``RunContext.youtube_url`` as "whatever URI Gemini can resolve" (a YouTube URL
today, a Files API ``files/xxxx`` URI for production), so this module's ONLY
job is to produce that URI from a local file and confirm it's ready.

No fallback logic: a non-ACTIVE terminal state (FAILED) or a poll timeout
RAISES — this module never fabricates a "ready" state for a file Gemini has
not actually finished indexing.
"""
from __future__ import annotations

import asyncio
import logging
import time

from google import genai
from google.genai import types

logger = logging.getLogger("service.pipelines.gemini_upload")


async def upload_video_to_gemini(client: genai.Client, local_path: str) -> types.File:
    """Upload a local video file via the Gemini Files API (resumable upload,
    handled internally by the SDK). Returns the ``types.File`` the SDK hands
    back immediately after the upload request completes — this is USUALLY in
    ``PROCESSING`` state (server-side video indexing is asynchronous); callers
    MUST run the result through :func:`poll_until_active` before using
    ``.uri``/``.name`` in a Gemini ``file_data`` call.
    """
    return await client.aio.files.upload(file=local_path)


async def poll_until_active(
    client: genai.Client,
    file_name: str,
    *,
    poll_interval_sec: float,
    timeout_sec: float,
) -> types.File:
    """Poll ``client.aio.files.get(name=file_name)`` in a bounded loop until
    the file reaches ``ACTIVE``.

    - ``FAILED`` -> raises ``RuntimeError`` immediately (never silently
      retried into a fabricated "ready" state).
    - Timeout -> raises ``TimeoutError`` (never proceeds with a non-ACTIVE
      file).
    - Any other non-terminal state (``PROCESSING``/``STATE_UNSPECIFIED``)
      keeps polling until one of the above fires or ``ACTIVE`` is reached.
    """
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    while True:
        attempt += 1
        f = await client.aio.files.get(name=file_name)
        if f.state == types.FileState.ACTIVE:
            logger.info(
                "gemini_upload: file %s reached ACTIVE after %d poll(s)",
                file_name, attempt,
            )
            return f
        if f.state == types.FileState.FAILED:
            raise RuntimeError(
                f"Gemini Files API upload failed for {file_name!r}: state=FAILED "
                f"(error={getattr(f, 'error', None)})"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Gemini Files API upload for {file_name!r} did not reach ACTIVE "
                f"within {timeout_sec}s (last observed state={f.state!r})"
            )
        logger.info(
            "gemini_upload: file %s state=%s, polling again in %.1fs (attempt %d)",
            file_name, f.state, poll_interval_sec, attempt,
        )
        await asyncio.sleep(poll_interval_sec)


async def delete_gemini_file(client: genai.Client, file_name: str) -> None:
    """Best-effort cleanup ("suspenders" — the 48h server-side TTL is the
    "belt"). Log-and-continue on failure; never raises, never blocks job
    completion on a cleanup failure."""
    try:
        await client.aio.files.delete(name=file_name)
        logger.info("gemini_upload: deleted %s", file_name)
    except Exception as e:  # noqa: BLE001 — best-effort cleanup only
        logger.warning("gemini_upload: best-effort delete failed for %s: %s", file_name, e)
