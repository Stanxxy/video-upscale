"""YouTube URL normalization — a small, dependency-free LEAF module (no
imports of any other ``eval/adcc/predict/**`` module), deliberately factored
OUT of ``highlight_discovery.py`` so both that module and the tag-batch
prediction-capture path (``global_cap_driver.py``/``run_predictions.py``) can
import it without a cycle: ``highlight_discovery.py`` already imports
``global_cap_driver`` (for ``BatchRunState``), so ``global_cap_driver``
importing back from ``highlight_discovery`` would be circular. This module
sits below both.

**Live-bug root cause (2026-07-18, see ``highlight_discovery.py``'s module
docstring for the full incident writeup).** The ADCC gold snapshot's
``video_url`` field (``MatchTarget.video_url``) is, by INS-084's own design,
a YouTube ``https://www.youtube.com/embed/<id>`` URL — the corpus's
``md5(embed_url)`` on-disk file-stem join key, never meant to be
Gemini-facing. Gemini's native ``file_uri`` YouTube ingestion only
recognizes the CANONICAL ``https://www.youtube.com/watch?v=<id>`` form
(confirmed by the ONE calling context that has ever actually exercised a
real, successful Gemini video call in this codebase: the QA VLM Studio
client, which always normalizes to ``watch?v=`` before the backend ever sees
a URL). Every eval/-side path that threads a corpus ``video_url`` toward a
Gemini call — highlight-discovery's own client, AND the tag-batch
prediction-capture path's ``RunContext.youtube_url`` — must run it through
``normalize_youtube_watch_url`` below BEFORE the URL reaches a Gemini
request, never send the raw embed form.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

__all__ = ["YoutubeUrlError", "normalize_youtube_watch_url"]

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YoutubeUrlError(ValueError):
    """Raised when a valid 11-character YouTube video ID cannot be
    extracted from a caller-supplied URL — never a silent best-effort guess,
    never a malformed URL passed through to a real Gemini request. A
    NEUTRAL error type (this is a leaf module with no opinion on how its
    callers want to present the failure) — callers that want their OWN
    exception type (e.g. ``highlight_discovery.HighlightDiscoveryError``)
    should catch this and re-raise/rewrap, never suppress it."""


def normalize_youtube_watch_url(video_url: str) -> str:
    """Normalizes ANY of the embed/short/canonical YouTube URL forms to the
    ONE form Gemini's native ``file_uri`` ingestion accepts:
    ``https://www.youtube.com/watch?v=<id>``. Recognizes ``/embed/<id>``,
    ``/shorts/<id>``, ``youtu.be/<id>``, and an already-canonical
    ``watch?v=<id>`` (a no-op in that last case). Raises ``YoutubeUrlError``
    (never a silent best-effort guess) if no valid 11-character YouTube
    video ID can be extracted, rather than sending a malformed URL to Gemini
    and letting it 400."""
    parsed = urlparse(video_url)
    video_id: Optional[str] = None

    if parsed.netloc.endswith("youtu.be"):
        video_id = parsed.path.strip("/")
    elif parsed.netloc.endswith("youtube.com"):
        if parsed.path == "/watch":
            candidates = parse_qs(parsed.query).get("v")
            video_id = candidates[0] if candidates else None
        else:
            for prefix in ("/embed/", "/shorts/"):
                if parsed.path.startswith(prefix):
                    video_id = parsed.path[len(prefix):].strip("/")
                    break

    if not video_id or not _YOUTUBE_ID_RE.match(video_id):
        raise YoutubeUrlError(
            f"could not extract a valid YouTube video ID from {video_url!r} to build the "
            "canonical watch?v= URL Gemini's native file_uri ingestion requires."
        )
    return f"https://www.youtube.com/watch?v={video_id}"
