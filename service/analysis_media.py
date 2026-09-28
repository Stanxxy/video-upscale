"""Opt-in, exact-byte media preparation for admitted position-DAG runs.

No provider uploads, URL fetching, admission, credential discovery or transcoding.
The host owns S3 client timeouts/credentials and current authority/lease checks.
"""

import asyncio
import hashlib
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

from bjj_analysis_core.requests import load_config, strict_json
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

VIDEO_FORMATS = "mov,matroska,webm,avi,mpegts"
IMAGE_FORMATS = "png_pipe,jpeg_pipe"


class MediaIntegrityError(ValueError):
    pass


class DeterministicPreparationRejected(MediaIntegrityError):
    """Sanitized, pre-provider input rejection that can safely interrupt a lease."""


@dataclass(frozen=True)
class VerifiedCrop:
    player_id: str
    display_name: str
    sha256: str
    mime_type: str
    data: bytes
    width: int
    height: int


@dataclass(frozen=True)
class VerifiedRunMedia:
    run: RegisteredAIRunV1
    config_bytes: bytes
    source_path: Path
    source_size: int
    source_format: str
    source_duration_ms: int
    crops: tuple[VerifiedCrop, VerifiedCrop]

    @contextmanager
    def open_source(self):
        """Recheck the same descriptor that the future uploader will consume.

        Valid only inside S3RunMedia.open(). No path-based reopen after hashing.
        Local worker isolation remains the host's responsibility.
        """
        fd = os.open(self.source_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            digest, count = hashlib.sha256(), 0
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                count += len(chunk)
                if count > self.source_size:
                    raise MediaIntegrityError("prepared source grew")
            if (
                count != self.source_size
                or digest.hexdigest() != self.run.spec.source.sha256
            ):
                raise MediaIntegrityError("prepared source bytes changed")
            source.seek(0)
            yield source


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class S3RunMedia:
    def __init__(
        self,
        client,
        *,
        allowed_buckets,
        scratch_root,
        max_source_bytes,
        guard,
        max_crop_bytes=10 * 1024 * 1024,
        max_crop_pixels=4096 * 4096,
        max_config_bytes=2 * 1024 * 1024,
        concurrency=2,
        probe_timeout=30,
    ):
        if (
            not isinstance(allowed_buckets, (tuple, list, set, frozenset))
            or not allowed_buckets
            or any(not isinstance(b, str) or not b for b in allowed_buckets)
        ):
            raise ValueError("explicit allowed artifact buckets required")
        if not callable(guard):
            raise ValueError("live run authority/lease guard required")
        self.client, self.guard = client, guard
        self.allowed_buckets = frozenset(allowed_buckets)
        self.root = Path(scratch_root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("existing private scratch parent required")
        self.max_source = _positive(max_source_bytes, "max_source_bytes")
        self.max_crop = _positive(max_crop_bytes, "max_crop_bytes")
        self.max_pixels = _positive(max_crop_pixels, "max_crop_pixels")
        self.max_config = _positive(max_config_bytes, "max_config_bytes")
        self.timeout = _positive(probe_timeout, "probe_timeout")
        self.slots = asyncio.Semaphore(_positive(concurrency, "concurrency"))

    def _download(self, ref, path, limit):
        # Exact opaque S3 key: never unquote %, split #/?, or resolve as a path/URL.
        response = self.client.get_object(Bucket=ref.bucket, Key=ref.key)
        body = response["Body"]
        try:
            length = response.get("ContentLength")
            if type(length) is not int or not 0 < length <= limit:
                raise MediaIntegrityError("artifact length missing or outside limit")
            digest, count = hashlib.sha256(), 0
            with path.open("xb") as target:
                os.chmod(path, 0o600)
                while chunk := body.read(min(1024 * 1024, limit + 1 - count)):
                    count += len(chunk)
                    if count > limit or count > length:
                        raise MediaIntegrityError("artifact exceeds declared length")
                    digest.update(chunk)
                    target.write(chunk)
            if count != length or digest.hexdigest() != ref.sha256:
                raise MediaIntegrityError("artifact bytes differ from admitted digest")
            os.chmod(path, 0o400)
            return count
        finally:
            body.close()

    def _probe(self, path, formats):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-max_alloc",
                "67108864",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                formats,
                "-show_entries",
                "format=format_name,duration,start_time:stream=codec_type,codec_name,width,height,start_time:stream_disposition=attached_pic",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=self.timeout,
            check=True,
        )
        return strict_json(result.stdout)

    def _video(self, path, run):
        probe = self._probe(path, VIDEO_FORMATS)
        videos = [s for s in probe["streams"] if s.get("codec_type") == "video"]
        if len(videos) != 1 or videos[0].get("disposition", {}).get("attached_pic"):
            raise MediaIntegrityError("source needs one unambiguous video stream")
        stream, container = videos[0], probe["format"]
        if any(
            type(stream.get(k)) is not int or stream[k] <= 0
            for k in ("width", "height")
        ):
            raise MediaIntegrityError("source video dimensions invalid")
        # The registration inspector uses this exact floor-to-ms convention.
        duration = Decimal(container["duration"])
        if not duration.is_finite() or duration <= 0:
            raise MediaIntegrityError("source duration invalid")
        duration_ms = int((duration * 1000).to_integral_value(rounding=ROUND_FLOOR))
        if duration_ms != run.spec.source_duration_ms:
            raise MediaIntegrityError("source duration differs from admitted run")
        # No silent rebasing/remuxing. A nonzero or absent time origin needs a
        # separately qualified conversion contract before native provider use.
        for metadata in (container, stream):
            if "start_time" not in metadata or Decimal(metadata["start_time"]) != 0:
                raise MediaIntegrityError("source time origin is not verified zero")
        return container["format_name"], duration_ms

    def _crop(self, path, athlete):
        probe = self._probe(path, IMAGE_FORMATS)
        streams = probe["streams"]
        if len(streams) != 1 or streams[0].get("codec_type") != "video":
            raise MediaIntegrityError("reference must be a single image")
        stream = streams[0]
        mime = {"png": "image/png", "mjpeg": "image/jpeg"}.get(stream.get("codec_name"))
        width, height = stream.get("width"), stream.get("height")
        if (
            mime is None
            or type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or width * height > self.max_pixels
        ):
            raise MediaIntegrityError("reference format/dimensions outside limits")
        # Decode at most two frames: reject header-only, corrupt and concatenated
        # images. Bytes are retained unchanged, not silently repaired/re-encoded.
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-xerror",
                "-max_alloc",
                "67108864",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                IMAGE_FORMATS,
                "-err_detect",
                "explode",
                "-threads",
                "1",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-frames:v",
                "2",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            timeout=self.timeout,
            check=True,
        )
        if len(decoded.stdout) != width * height * 3:
            raise MediaIntegrityError("reference must decode to exactly one image")
        return VerifiedCrop(
            athlete.player_id,
            athlete.display_name,
            athlete.reference_crop.sha256,
            mime,
            path.read_bytes(),
            width,
            height,
        )

    async def _io(self, operation, *args):
        # Shield worker lifetime, not caller cancellation. Keep scratch and slot
        # until blocking I/O actually exits, including repeated cancellation.
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()  # consume a worker failure; cancellation still wins
            raise

    async def _prepare(self, operation, *args):
        """Translate only explicit media-integrity results, never guard/I/O failures."""
        try:
            return await self._io(operation, *args)
        except MediaIntegrityError as error:
            raise DeterministicPreparationRejected(str(error)) from None

    @asynccontextmanager
    async def open(self, run):
        run = RegisteredAIRunV1.model_validate(run)
        refs = (run.spec.effective_config, run.spec.source) + tuple(
            a.reference_crop for a in run.spec.athletes
        )
        if any(ref.bucket not in self.allowed_buckets for ref in refs):
            raise DeterministicPreparationRejected(
                "artifact outside configured buckets"
            )
        await self.guard(run)
        async with self.slots:
            await self.guard(run)
            with tempfile.TemporaryDirectory(
                prefix="position-media-", dir=self.root
            ) as folder:
                root = Path(
                    folder
                )  # generated local names only, never user object keys
                config_path, source_path = root / "config.json", root / "source.video"
                await self.guard(run)
                await self._prepare(
                    self._download,
                    run.spec.effective_config,
                    config_path,
                    self.max_config,
                )
                config_bytes = config_path.read_bytes()
                try:
                    load_config(run, config_bytes)
                except ValueError:
                    raise DeterministicPreparationRejected(
                        "configured analysis settings rejected"
                    ) from None
                await self.guard(run)
                size = await self._prepare(
                    self._download,
                    run.spec.source,
                    source_path,
                    self.max_source,
                )
                await self.guard(run)
                format_name, duration = await self._prepare(
                    self._video, source_path, run
                )
                crops = []
                for index, athlete in enumerate(run.spec.athletes):
                    await self.guard(run)
                    path = root / f"player-{index}.image"
                    await self._prepare(
                        self._download, athlete.reference_crop, path, self.max_crop
                    )
                    crops.append(await self._prepare(self._crop, path, athlete))
                await self.guard(run)
                yield VerifiedRunMedia(
                    run,
                    config_bytes,
                    source_path,
                    size,
                    format_name,
                    duration,
                    tuple(crops),
                )
