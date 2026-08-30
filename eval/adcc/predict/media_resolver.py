"""eval/adcc/predict/media_resolver.py — trusted match_id -> on-disk media
resolution (AC2-4/AC2-5/§7). This module is the ONLY code path that ever
reads a real filesystem path derived from a caller-supplied identifier; its
public surface (ResolvedMedia, MediaRootIndex) never returns a raw path to
anything outside this module's own trust boundary -- dashboard_app.py's
route (Task 11) uses ResolvedMedia.source_path only to hand to
DerivativeCache (Task 10)/ffmpeg, never to the browser.

Security discipline (§7):
  - match_id -> stem is a caller-supplied, already-trusted map (the gold
    corpus's own MatchTarget.video_url_md5, never a client-supplied path).
  - stem -> on-disk file is resolved ONCE at index-build time by listing
    the (realpath'd) media root directory -- never a per-request
    directory probe, never a client-supplied filename.
  - A symlink whose real target escapes the real, resolved media root is
    SKIPPED at index-build time (never indexed, never resolvable) --
    checked via Path.is_relative_to on REAL (os.path.realpath) paths, never
    a naive string-startswith (the classic "/root-evil".startswith("/root")
    false-positive trap).
  - Two on-disk files mapping to the same stem is failed loud at
    index-build time (DuplicateStemError) -- this is DISTINCT from, and
    never confused with, multiple match_ids legitimately sharing ONE stem
    (the normal multi-match-VOD case, which is NOT an error).
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from eval.adcc.predict.generate_video_durations import VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS

logger = logging.getLogger("eval.adcc.predict.media_resolver")

# Single caller-facing 404 message for every UnknownMediaError path.
# Must stay path-free and mode-indistinguishable (unknown match_id vs
# missing media vs outside-root) so probes cannot oracle the failure mode.
_UNKNOWN_MEDIA_MSG = "unknown or unavailable match_id"

DERIVATIVE_EXTENSION = ".mp4"

__all__ = [
    "KNOWN_VIDEO_EXTENSIONS",
    "MediaResolutionError",
    "UnknownMediaError",
    "UnsupportedExtensionError",
    "DuplicateStemError",
    "MediaBuildError",
    "ResolvedMedia",
    "MediaRootIndex",
    "DerivativeCache",
    "probe_is_h264",
    "build_mp4_derivative",
    "DERIVATIVE_EXTENSION",
]


class MediaResolutionError(ValueError):
    """Base for a caller-facing, HTTP-mappable resolution failure. Never
    includes a raw filesystem path in its message -- match_id/stem only."""


class UnknownMediaError(MediaResolutionError):
    """Unknown match_id, OR a known match_id whose stem has no indexed
    on-disk file (including one skipped as a symlink escape), OR a
    resolve-time realpath that escapes the media root -- all map to 404
    with the same path-free message. An attacker probing match_ids must
    not learn which failure mode occurred."""


class UnsupportedExtensionError(MediaResolutionError):
    """The resolved on-disk file's extension is outside
    KNOWN_VIDEO_EXTENSIONS -- maps to 415."""


class DuplicateStemError(MediaResolutionError):
    """>1 on-disk file in the media root resolves to the same stem --
    raised at MediaRootIndex.build() time (fail loud at startup/first
    access), never picked-one-silently."""


class MediaBuildError(RuntimeError):
    """A real ffmpeg/ffprobe subprocess failure while building the
    derivative for an already-resolved, already-security-checked source
    file -- maps to 500. The real subprocess stderr is preserved in the
    message, never swallowed (§8)."""


@dataclass(frozen=True)
class ResolvedMedia:
    match_id: str
    stem: str
    source_path: Path
    source_extension: str


@dataclass
class MediaRootIndex:
    real_root: Path
    match_id_to_stem: dict[str, str]
    stem_to_source_path: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def build(cls, media_root: Path, match_id_to_stem: dict[str, str]) -> "MediaRootIndex":
        """Scans media_root ONCE. Fails loud (DuplicateStemError) if two
        on-disk files share a stem. Silently SKIPS (never indexes) an entry
        whose real path escapes the real, resolved media root -- logged as
        a warning, not raised, since one bad symlink should never crash
        indexing of the other ~100 legitimate files."""
        real_root = Path(os.path.realpath(str(media_root)))
        stem_to_source_path: dict[str, Path] = {}
        if real_root.is_dir():
            for entry in sorted(real_root.iterdir()):
                if not entry.is_file():
                    continue
                real_entry = Path(os.path.realpath(str(entry)))
                if not (real_entry == real_root or real_entry.is_relative_to(real_root)):
                    logger.warning(
                        "media_resolver: skipping %s -- its real path %s escapes the configured "
                        "media root %s (symlink pointing outside the root).",
                        entry.name, real_entry, real_root,
                    )
                    continue
                stem = entry.stem
                if stem in stem_to_source_path:
                    raise DuplicateStemError(
                        f"ADCC media root has >1 on-disk file for stem {stem!r}: "
                        f"{stem_to_source_path[stem].name} and {real_entry.name} — cannot pick one silently."
                    )
                stem_to_source_path[stem] = real_entry
        return cls(real_root=real_root, match_id_to_stem=dict(match_id_to_stem), stem_to_source_path=stem_to_source_path)

    def resolve(self, match_id: str) -> ResolvedMedia:
        stem = self.match_id_to_stem.get(match_id)
        if stem is None:
            raise UnknownMediaError(_UNKNOWN_MEDIA_MSG)
        source_path = self.stem_to_source_path.get(stem)
        if source_path is None:
            raise UnknownMediaError(_UNKNOWN_MEDIA_MSG)
        # Re-realpath at handoff (TOCTOU / poisoned-index defense) -- never
        # a naive string-startswith. Same unified message on escape.
        real_source = Path(os.path.realpath(str(source_path)))
        if not (real_source == self.real_root or real_source.is_relative_to(self.real_root)):
            raise UnknownMediaError(_UNKNOWN_MEDIA_MSG)
        ext = real_source.suffix.lower()
        if ext not in KNOWN_VIDEO_EXTENSIONS:
            raise UnsupportedExtensionError(f"match_id {match_id!r}: unsupported source extension {ext!r}")
        return ResolvedMedia(
            match_id=match_id,
            stem=stem,
            source_path=real_source,
            source_extension=ext,
        )


async def probe_is_h264(source_path: Path) -> bool:
    """Metadata-only ffprobe check (mirrors generate_video_durations.py's
    own metadata-only discipline) -- never decodes a single frame."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(source_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise MediaBuildError(
            f"ffprobe failed (exit {proc.returncode}) probing {source_path.name}: "
            f"{stderr.decode(errors='replace')[-2000:]}"
        )
    return stdout.decode().strip().lower() == "h264"


async def build_mp4_derivative(source_path: Path, out_path: Path) -> None:
    """Video passthrough (remux, no re-encode) when the source is already
    H.264; a real transcode to H.264 otherwise. Audio is ALWAYS transcoded
    to AAC (the reported source is Opus, which no <video> element's native
    MP4 demuxer accepts). ``-movflags +faststart`` moves MP4 metadata to
    the front of the file -- required for progressive playback/seeking to
    start before the whole file has downloaded; without it, Range support
    alone does not help a browser locate the moov atom. Writes directly to
    out_path -- the CALLER (DerivativeCache) is responsible for the
    build-to-a-temp-path-then-atomic-rename discipline, since only the
    caller knows the real final cache path."""
    is_h264 = await probe_is_h264(source_path)
    video_args = ["-c:v", "copy"] if is_h264 else ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    cmd = [
        "ffmpeg", "-y", "-i", str(source_path), *video_args,
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out_path),
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise MediaBuildError(
            f"ffmpeg failed (exit {proc.returncode}) building a derivative for {source_path.name}: "
            f"{stderr.decode(errors='replace')[-2000:]}"
        )


class DerivativeCache:
    """Lazy, per-stem MP4 derivative cache. Never pre-generated in bulk
    (§7) -- populated strictly on-demand, one match at a time, triggered
    only by an actual ensure_derivative() call (which the route, Task 11,
    only makes on a real fallback request). A per-stem asyncio.Lock
    ensures a SECOND concurrent request for the SAME stem awaits the SAME
    in-flight build rather than starting a second ffmpeg process (AC2-6's
    "concurrency-safe single generation per stem")."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def cached_path_for(self, stem: str) -> Path:
        return self._cache_dir / f"{stem}{DERIVATIVE_EXTENSION}"

    def _lock_for(self, stem: str) -> asyncio.Lock:
        lock = self._locks.get(stem)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[stem] = lock
        return lock

    async def ensure_derivative(self, resolved: ResolvedMedia) -> Path:
        cached_path = self.cached_path_for(resolved.stem)
        if cached_path.exists():
            return cached_path
        async with self._lock_for(resolved.stem):
            if cached_path.exists():  # a concurrent caller may have finished building while we waited
                return cached_path
            tmp_path = self._cache_dir / f"{resolved.stem}.building{DERIVATIVE_EXTENSION}"
            try:
                await build_mp4_derivative(resolved.source_path, tmp_path)
            except MediaBuildError:
                tmp_path.unlink(missing_ok=True)  # never leave a partial file behind, at ANY path
                raise
            os.replace(tmp_path, cached_path)  # atomic on the same filesystem -- never a partial file at cached_path
            return cached_path
