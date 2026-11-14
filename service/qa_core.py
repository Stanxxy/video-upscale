"""QA VLM Studio host for the PRODUCTION analysis core (BJJ or MMA pack).

Runs ``DurableGeminiRun`` -> ``execute_pipeline`` with the real ``S3RunMedia``,
``GeminiProvider`` and ``S3GeminiArtifacts``, so a VLM Studio run exercises the
exact prompts, schemas, windows and validation that production uses. Only the
host ports differ, and only for local QA:

- artifacts live in the shared LocalStack under the caller's namespace (the
  process env from ``scripts/localinfra.sh ns env <nsid>``; bucket ``QA_CORE_BUCKET``),
- checkpoints live in a per-run SQLite journal (resume never re-bills a call that
  already succeeded; same CAS semantics as the Keyspaces journal),
- publication goes to an in-memory sink that keeps the revisions for the page.

Not production code: never imported by ``service.app``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from bjj_analysis_core import CORE_VERSION, MMA_WORKFLOW_ID, WORKFLOW_ID
from bjj_analysis_core.execution import StepCheckpointV1
from bjj_analysis_core.requests import CoreConfigV1, NodeConfigV1
from shared_lib.models.analysis_contracts import AnalysisRunSpecV1
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

from service.analysis_gemini import GeminiProvider, S3GeminiArtifacts
from service.analysis_media import S3RunMedia
from service.analysis_preset import gemini38_mma_preset, gemini38_preset
from service.analysis_run_host import DurableGeminiRun

logger = logging.getLogger("service.qa_core")

NODES = (
    "position_scan",
    "position_critique",
    "technique_scan",
    "technique_critique",
    "technique_analyze",
    "technique_validate",
)
SPORTS = {
    "bjj": (WORKFLOW_ID, "bjj.v1", gemini38_preset),
    "mma": (MMA_WORKFLOW_ID, "mma.v1", gemini38_mma_preset),
}
YOUTUBE_ID = re.compile(r"[A-Za-z0-9_-]{11}")
ENGINE_ROOT = Path(__file__).resolve().parents[1]
QUALIFIED_MODELS = ENGINE_ROOT / "config" / "analysis" / "qualified-models.json"

# Google's published Standard paid-tier rate (https://ai.google.dev/gemini-api/docs/pricing),
# the same table s34 VLM Studio used. Thinking tokens bill as output. An unlisted
# model has no estimate (shown as unknown, never zero).
_PROMO_END = date(2026, 12, 31)


def _rate(model: str | None, on: date):
    if model not in {"gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"}:
        return None
    if on <= _PROMO_END:
        return {"input": 0.75, "output": 3.75, "cached": 0.075}
    return {"input": 1.50, "output": 7.50, "cached": 0.15}


# Approximate video tokens per sampled frame by media resolution (Gemini 3 docs);
# only used for the pre-run estimate, never for reported spend.
_FRAME_TOKENS = {"low": 70, "medium": 70, "high": 280}


class FighterIn(BaseModel):
    name: str = Field(min_length=1, max_length=60, pattern=r"^[^\x00-\x1f]+$")
    frame_ms: int = Field(ge=0)
    # Crop box as fractions of the frame: left, top, width, height.
    box: tuple[float, float, float, float]


class NodeOverride(BaseModel):
    model: Optional[str] = None
    thinking: Optional[Literal["minimal", "low", "medium", "high"]] = None
    media_resolution: Optional[Literal["low", "medium", "high"]] = None
    fps: Optional[int] = Field(default=None, ge=1, le=30)


class CoreRunIn(BaseModel):
    youtube_id: str = Field(pattern=YOUTUBE_ID.pattern)
    sport: Literal["bjj", "mma"]
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    fighters: tuple[FighterIn, FighterIn]
    overrides: dict[str, NodeOverride] = Field(default_factory=dict)
    max_parallel_calls: int = Field(default=4, ge=1, le=16)
    label: Optional[str] = Field(default=None, max_length=80)


def _require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required (run the QA backend under `scripts/localinfra.sh ns env <nsid>`)")
    return value


@dataclass
class Settings:
    root: Path
    media_dir: Path
    bucket: str
    api_key: str

    @classmethod
    def from_env(cls, api_key):
        root = Path(_require_env("QA_CORE_ROOT")).resolve()
        media = Path(os.environ.get("QA_CORE_MEDIA_DIR") or root / "media").resolve()
        root.mkdir(parents=True, exist_ok=True)
        media.mkdir(parents=True, exist_ok=True)
        _require_env("AWS_ENDPOINT_URL")  # local QA only: never real AWS
        return cls(root=root, media_dir=media, bucket=_require_env("QA_CORE_BUCKET"), api_key=api_key)


def s3_client():
    return boto3.client("s3", endpoint_url=_require_env("AWS_ENDPOINT_URL"), region_name=os.environ.get("AWS_REGION", "us-east-1"))


def ensure_bucket(client, bucket):
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration_ms(path: Path):
    """Floor-to-ms container duration, the convention S3RunMedia verifies."""
    from decimal import ROUND_FLOOR, Decimal

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, check=True, timeout=30,
    )
    duration = Decimal(json.loads(out.stdout)["format"]["duration"])
    return int((duration * 1000).to_integral_value(rounding=ROUND_FLOOR))


def media_path(settings: Settings, youtube_id: str, *, download: bool):
    """The local copy of a YouTube video; H.264 MP4 so Gemini can decode it."""
    if not YOUTUBE_ID.fullmatch(youtube_id):
        raise ValueError("invalid YouTube id")
    path = settings.media_dir / f"{youtube_id}.mp4"
    if path.exists() or not download:
        return path
    partial = settings.media_dir / f"{youtube_id}.part.mp4"
    subprocess.run(
        [
            "yt-dlp", "--no-playlist", "--quiet",
            "-f", "bv*[ext=mp4][vcodec^=avc1][height<=720]+ba[ext=m4a]/b[ext=mp4][vcodec^=avc1][height<=720]",
            "--merge-output-format", "mp4", "-o", str(partial),
            f"https://www.youtube.com/watch?v={youtube_id}",
        ],
        check=True, timeout=1800,
    )
    os.replace(partial, path)
    return path


def crop_png(source: Path, at_ms: int, box, *, full=False):
    """One PNG frame at ``at_ms``, optionally cropped to a fractional box."""
    left, top, width, height = box
    if not (0 <= left < 1 and 0 <= top < 1 and 0 < width <= 1 - left + 1e-9 and 0 < height <= 1 - top + 1e-9):
        raise ValueError("crop box must lie inside the frame")
    vf = [] if full else [f"crop=iw*{width:.4f}:ih*{height:.4f}:iw*{left:.4f}:ih*{top:.4f}"]
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{at_ms / 1000:.3f}", "-i", str(source),
         "-frames:v", "1", *(["-vf", ",".join(vf)] if vf else []), "-f", "image2", "-c:v", "png", "pipe:1"],
        capture_output=True, check=True, timeout=60,
    )
    if not out.stdout:
        raise ValueError("no frame at that time")
    return out.stdout


class SqliteJournal:
    """Local CAS checkpoint journal with full history (QA host port only)."""

    def __init__(self, path: Path, run):
        self.run = run
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS heads (id TEXT PRIMARY KEY, revision INTEGER, body TEXT, sha TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS history (id TEXT, revision INTEGER, body TEXT, sha TEXT, PRIMARY KEY(id, revision))")
        self.db.commit()

    async def read(self, step_id):
        row = self.db.execute("SELECT body, sha FROM heads WHERE id=?", (str(step_id),)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("checkpoint checksum mismatch")
        return StepCheckpointV1.model_validate_json(row[0])

    async def compare_and_swap(self, expected_revision, checkpoint):
        body = checkpoint.model_dump_json()
        values = (str(checkpoint.step_id), checkpoint.revision, body, hashlib.sha256(body.encode()).hexdigest())
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT revision FROM heads WHERE id=?", (values[0],)).fetchone()
            if (None if row is None else row[0]) != expected_revision:
                self.db.rollback()
                return False
            if checkpoint.revision != (expected_revision or 0) + 1:
                self.db.rollback()
                raise ValueError("checkpoint revision must advance by one")
            self.db.execute("INSERT INTO history VALUES (?,?,?,?)", values)
            self.db.execute("INSERT OR REPLACE INTO heads VALUES (?,?,?,?)", values)
            self.db.commit()
            return True
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    def history(self):
        return [StepCheckpointV1.model_validate_json(r[0]) for r in self.db.execute("SELECT body FROM history ORDER BY rowid")]

    def close(self):
        self.db.close()


class MemorySink:
    def __init__(self):
        self.revisions = {}

    async def publish(self, run, component, revision):
        self.revisions[component] = revision.model_dump(mode="json")


class Cancelled(RuntimeError):
    pass


def build_config(sport, overrides):
    config = SPORTS[sport][2]()
    for node, change in overrides.items():
        if node not in NODES:
            raise ValueError(f"unknown node {node}")
        values = {k: v for k, v in change.model_dump().items() if v is not None}
        if values:
            current = getattr(config, node)
            config = config.model_copy(update={node: NodeConfigV1.model_validate({**current.model_dump(), **values})})
    return CoreConfigV1.model_validate(config.model_dump())


def estimate(config: CoreConfigV1, start_ms, end_ms):
    """Scan-call input tokens only; per-item calls are unknown until the scans answer."""
    seconds = (end_ms - start_ms) / 1000
    out = {}
    for node in ("position_scan", "technique_scan"):
        cfg = getattr(config, node)
        tokens = int(seconds * cfg.fps * _FRAME_TOKENS[cfg.media_resolution])
        rate = _rate(cfg.model, date.today())
        out[node] = {"frames": int(seconds * cfg.fps), "input_tokens": tokens,
                     "usd": None if rate is None else round(tokens * rate["input"] / 1e6, 4)}
    return out


@dataclass
class Job:
    run_id: str
    folder: Path
    request: dict
    state: str = "queued"
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    cancelled: bool = False
    task: Optional[asyncio.Task] = None
    sink: MemorySink = field(default_factory=MemorySink)


class CoreRuns:
    """Runs keyed by run id; each run's folder holds request, run, config, journal, results."""

    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client or s3_client()
        self.jobs: dict[str, Job] = {}
        ensure_bucket(self.client, settings.bucket)
        self._load_existing()

    # -- persistence ----------------------------------------------------------------
    def _runs_dir(self):
        path = self.settings.root / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_existing(self):
        for folder in sorted(self._runs_dir().iterdir()):
            meta = folder / "job.json"
            if not meta.exists():
                continue
            data = json.loads(meta.read_text())
            job = Job(run_id=folder.name, folder=folder, request=data["request"], state=data["state"],
                      error=data.get("error"), created_at=data["created_at"], finished_at=data.get("finished_at"))
            if job.state in {"queued", "preparing", "running"}:
                job.state, job.error = "interrupted", "the QA backend stopped during this run; resume to continue"
            for component in ("positions", "techniques"):
                result = folder / f"{component}.json"
                if result.exists():
                    job.sink.revisions[component] = json.loads(result.read_text())
            self.jobs[job.run_id] = job

    def _save(self, job: Job):
        tmp = job.folder / "job.json.tmp"
        tmp.write_text(json.dumps(dict(request=job.request, state=job.state, error=job.error,
                                       created_at=job.created_at, finished_at=job.finished_at), indent=2))
        os.replace(tmp, job.folder / "job.json")
        for component, revision in job.sink.revisions.items():
            path = job.folder / f"{component}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(revision))
            os.replace(tmp, path)

    # -- run building ---------------------------------------------------------------
    def _put(self, key, data: bytes):
        digest = hashlib.sha256(data).hexdigest()
        self.client.put_object(Bucket=self.settings.bucket, Key=key, Body=data)
        return dict(bucket=self.settings.bucket, key=key, sha256=digest)

    def _put_file(self, key, path: Path):
        digest = sha256_file(path)
        try:
            head = self.client.head_object(Bucket=self.settings.bucket, Key=key)
            if head.get("Metadata", {}).get("sha256") == digest:
                return dict(bucket=self.settings.bucket, key=key, sha256=digest)
        except ClientError:
            pass
        self.client.upload_file(str(path), self.settings.bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
        return dict(bucket=self.settings.bucket, key=key, sha256=digest)

    def _register(self, job: Job, request: CoreRunIn, source: Path):
        workflow_id, taxonomy_id, _ = SPORTS[request.sport]
        config = build_config(request.sport, request.overrides)
        config_bytes = config.model_dump_json().encode()
        duration = probe_duration_ms(source)
        if request.end_ms > duration or request.start_ms >= request.end_ms:
            raise ValueError(f"scope must lie inside the video (0–{duration} ms)")
        prefix = f"vlm-core/{job.run_id}"
        source_ref = self._put_file(f"vlm-core/sources/{request.youtube_id}.mp4", source)
        config_ref = self._put(f"{prefix}/config.json", config_bytes)
        athletes = []
        for index, fighter in enumerate(request.fighters):
            png = crop_png(source, fighter.frame_ms, fighter.box)
            (job.folder / f"fighter-{index}.png").write_bytes(png)
            athletes.append(dict(player_id=f"fighter-{'ab'[index]}", display_name=fighter.name,
                                 reference_crop=self._put(f"{prefix}/fighter-{index}.png", png)))
        now = datetime.now(timezone.utc)
        spec = AnalysisRunSpecV1(
            run_id=UUID(job.run_id), analysis_id=uuid4(), base_edit_version=0, scope_id=uuid4(),
            owner=dict(kind="account", principal_id="vlm-studio"), video_id=request.youtube_id,
            source=source_ref, source_duration_ms=duration,
            scope=dict(start_ms=request.start_ms, end_ms=request.end_ms), athletes=tuple(athletes),
            workflow_id=workflow_id, taxonomy_id=taxonomy_id, core_version=CORE_VERSION,
            effective_config=config_ref, admission_id="vlm-studio", usage_policy_version="vlm-studio-v1",
            admitted_at=now,
        )
        baseline = self._put(f"{prefix}/baseline.json", b"{}")
        run = RegisteredAIRunV1(spec=spec, baseline=baseline, baseline_revision_id=uuid4(),
                                baseline_revision_sha256=baseline["sha256"], registered_at=now + timedelta(milliseconds=1))
        (job.folder / "run.json").write_text(run.model_dump_json(indent=2))
        (job.folder / "config.json").write_bytes(config_bytes)
        return run, config_bytes

    # -- execution ------------------------------------------------------------------
    def create(self, request: CoreRunIn):
        run_id = str(uuid4())
        folder = self._runs_dir() / run_id
        folder.mkdir()
        job = Job(run_id=run_id, folder=folder, request=request.model_dump(mode="json"))
        self.jobs[run_id] = job
        self._save(job)
        job.task = asyncio.create_task(self._execute(job, fresh=True))
        return job

    def resume(self, run_id):
        job = self.jobs[run_id]
        if job.state not in {"failed", "interrupted", "cancelled"}:
            raise ValueError(f"a {job.state} run cannot be resumed")
        if not (job.folder / "run.json").exists():
            raise ValueError("this run never registered; start a new run")
        job.cancelled, job.error = False, None
        job.task = asyncio.create_task(self._execute(job, fresh=False))
        return job

    def cancel(self, run_id):
        job = self.jobs[run_id]
        job.cancelled = True
        return job

    async def _execute(self, job: Job, *, fresh: bool):
        request = CoreRunIn.model_validate(job.request)

        async def guard(*_):
            if job.cancelled:
                raise Cancelled("cancelled from VLM Studio")

        journal = None
        try:
            job.state = "preparing"
            self._save(job)
            if fresh:
                source = await asyncio.to_thread(media_path, self.settings, request.youtube_id, download=True)
                run, config_bytes = await asyncio.to_thread(self._register, job, request, source)
            else:
                run = RegisteredAIRunV1.model_validate_json((job.folder / "run.json").read_text())
                config_bytes = (job.folder / "config.json").read_bytes()
            journal = SqliteJournal(job.folder / "journal.sqlite", run)
            scratch = self.settings.root / "scratch"
            scratch.mkdir(exist_ok=True)
            media = S3RunMedia(self.client, allowed_buckets={self.settings.bucket}, scratch_root=scratch,
                               max_source_bytes=4 * 1024**3, guard=guard)
            qualified = json.loads(QUALIFIED_MODELS.read_text())
            async with media.open(run) as verified:
                artifacts = S3GeminiArtifacts(run, self.client, bucket=self.settings.bucket)
                provider = GeminiProvider(verified, artifacts, api_key=self.settings.api_key,
                                          qualified_models=qualified, guard=guard)
                try:
                    host = DurableGeminiRun(run, config_bytes, journal=journal, provider=provider, sink=job.sink,
                                            guard=guard, mime_type="video/mp4",
                                            max_parallel_calls=request.max_parallel_calls)
                    job.state = "running"
                    self._save(job)
                    result = await host.execute(retry_failed=not fresh, reissue_unknown=not fresh)
                    job.sink.revisions["positions"] = result.positions.model_dump(mode="json")
                    job.sink.revisions["techniques"] = result.techniques.model_dump(mode="json")
                    await host.cleanup()
                finally:
                    await provider.aclose()
            job.state = "succeeded"
        except Cancelled as error:
            job.state, job.error = "cancelled", str(error)
        except Exception as error:  # surfaced to the page verbatim; the journal keeps the detail
            logger.exception("vlm-studio core run %s failed", job.run_id)
            job.state, job.error = "failed", f"{type(error).__name__}: {error}"
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()
            self._save(job)
            if journal is not None:
                journal.close()

    # -- reporting ------------------------------------------------------------------
    def status(self, run_id):
        job = self.jobs[run_id]
        request = job.request
        config = json.loads((job.folder / "config.json").read_text()) if (job.folder / "config.json").exists() else None
        nodes = {}
        db = job.folder / "journal.sqlite"
        if db.exists():
            conn = sqlite3.connect(db)
            try:
                rows = [StepCheckpointV1.model_validate_json(r[0]) for r in conn.execute("SELECT body FROM history ORDER BY rowid")]
            finally:
                conn.close()
            heads, usage = {}, {}
            for row in rows:
                heads[row.step_id] = row
                usage[row.usage.call_id] = (row.progress.node_id, row.usage, row.phase)
            for head in heads.values():
                node = nodes.setdefault(head.progress.node_id, dict(node_id=head.progress.node_id, items={}, total=None,
                                                                   calls=0, input_tokens=0, output_tokens=0,
                                                                   thinking_tokens=0, usd=0.0, unknown_cost_calls=0,
                                                                   in_flight_calls=0))
                key = head.progress.item_id or "-"
                node["items"][key] = head.progress.status
                if head.progress.total_items is not None:
                    node["total"] = head.progress.total_items
            for node_id, use, phase in usage.values():
                if use.kind != "model":
                    continue
                node = nodes[node_id]
                node["calls"] += 1
                if phase in {"running", "responded"} and use.usage_status == "unknown":
                    node["in_flight_calls"] += 1  # dispatched, no reply yet: not a missing receipt
                    continue
                node["input_tokens"] += use.input_tokens or 0
                node["output_tokens"] += use.output_tokens or 0
                node["thinking_tokens"] += use.thinking_tokens or 0
                rate = _rate(use.model, use.recorded_at.date())
                if rate is None or use.input_tokens is None or use.output_tokens is None:
                    node["unknown_cost_calls"] += 1
                else:
                    cached = use.cached_input_tokens or 0
                    node["usd"] += ((use.input_tokens - cached) * rate["input"] + cached * rate["cached"]
                                    + (use.output_tokens + (use.thinking_tokens or 0)) * rate["output"]) / 1e6
        summary = []
        for node in nodes.values():
            states = list(node.pop("items").values())
            node.update(succeeded=states.count("succeeded"), running=sum(s in {"running", "retrying"} for s in states),
                        failed=states.count("failed"), seen=len(states), usd=round(node["usd"], 4))
            summary.append(node)
        order = {n: i for i, n in enumerate(("provider_upload", "position_scan", "position_critique", "positions_normalize",
                                             "positions_publish", "technique_scan", "position_technique_candidates",
                                             "technique_critique", "technique_worklist", "technique_analyze",
                                             "technique_validate", "techniques_normalize", "techniques_publish",
                                             "dag_complete", "provider_cleanup"))}
        summary.sort(key=lambda n: order.get(n["node_id"], 99))
        return dict(run_id=job.run_id, state=job.state, error=job.error, created_at=job.created_at,
                    finished_at=job.finished_at, request=request, config=config,
                    nodes=summary, total_usd=round(sum(n["usd"] for n in summary), 4),
                    unknown_cost_calls=sum(n["unknown_cost_calls"] for n in summary),
                    in_flight_calls=sum(n["in_flight_calls"] for n in summary),
                    components=sorted(job.sink.revisions))

    def result(self, run_id):
        job = self.jobs[run_id]
        return dict(run_id=run_id, state=job.state, **job.sink.revisions)

    def list(self):
        return [dict(run_id=j.run_id, state=j.state, created_at=j.created_at, youtube_id=j.request["youtube_id"],
                     sport=j.request["sport"], start_ms=j.request["start_ms"], end_ms=j.request["end_ms"],
                     label=j.request.get("label")) for j in sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)]
