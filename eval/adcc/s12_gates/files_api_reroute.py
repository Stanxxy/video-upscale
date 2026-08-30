"""§G.1 reroute lane — the 3 `pwGbW5GZgfc`-video windows (Gate-1 straddler
`w37`, Gate-2 forced-unsuccessful `occ0`/`occ11`) that 400 on every native
YouTube `file_uri` ingestion attempt regardless of offset/fps (confirmed via
6 direct SDK probes during the 2026-07-25 smoke test — see the engineer's
handoff report). Ruling (c), `GATES_1_2_DESIGN.md` §G.1: route ONLY these 3
windows through the Gemini Files API upload lane.

Pipeline per window (§G.1 points 2-3): yt-dlp section-download the window's
own `[start-2, end+2]` span from the gold YouTube URL -> normalize to
H.264/AAC MP4 via the harness's EXISTING `media_resolver.build_mp4_derivative`
passthrough-or-transcode (never a new transcode pipeline) -> upload via the
verbatim-copied `gemini_upload.py` -> `poll_until_active`. ONE upload per
window, reused across ALL of that window's calls (cross-arm pairing, §G.1
point 3) — the resulting Files API `.uri` is passed through the exact same
`youtube_url` parameter every other call already uses.

**TWO live bugs found during the reroute smoke test (2026-07-26), both fixed
here — NEITHER is in the design doc, and BOTH will also hit Gate 3
(`GATES_3_4_DESIGN.md` §H.2) when that runs, since its plan makes the exact
same "just pass the files/xxxx URI through analyze_chunk" assumption:**

1. **Missing `mime_type` on the Files API `FileData`.** This worktree's
   `service.pipelines.simplified_tags.SimplifiedTagsTimeAnalyzer.
   analyze_chunk` has NO `mime_type` parameter at all (added later, in
   `whole-video-analysis-wt-s12`'s S12 Phase-1b fork, whose own docstring
   says: "Production (Gemini Files API URIs) MUST pass a non-None value —
   Files API `file_data.mime_type` is documented 'Required' for non-YouTube
   sources."). Kept fixed here even though isolated testing (below) shows it
   is not what caused the observed 500s — it is a documented API
   requirement, not an optional nicety, and leaving it unset would be
   fixing the SYMPTOM (retry the call until root cause #2 is also found)
   while ignoring a real, documented contract violation.
2. **THE ACTUAL ROOT CAUSE of the observed 500s: `video_metadata.start_offset`/
   `end_offset` were sent as ABSOLUTE match-clock seconds (e.g. window
   `occ0`'s real `[28320.0, 28329.0]`) against an uploaded file that is only
   ~13 SECONDS LONG (already yt-dlp-cropped to the window's own span) — the
   uploaded file's OWN clock starts at 0:00, not at the original video's
   28320s. Confirmed via direct SDK isolation: the identical uploaded file,
   with `mime_type` set, `video_metadata=VideoMetadata(start_offset=
   '28320.000s', end_offset='28329.000s', fps=10)` -> `500 INTERNAL`
   (every time); the SAME file with FILE-RELATIVE offsets
   (`start_offset='2.000s', end_offset='11.000s'`, i.e. `absolute -
   sec_start`) -> succeeds cleanly. `ReroutedUpload.call_window_bounds()`
   below is the ONE place this conversion happens — every reroute call site
   MUST route through it, never pass a window's own absolute
   `window_start_s`/`window_end_s` directly to a Files API call.

`_call_axis_json`/`analyze_chunk` are therefore NOT fully agnostic to
`file_uri` kind, contrary to what an earlier draft of this module's
docstring assumed — the offset CLOCK the caller must reason in changes.
`FilesApiTimeAnalyzer` below is a LOCAL, DISCLOSED subclass (same
duplication discipline/justification as the verbatim `gemini_upload.py`
copy — this worktree's `simplified_tags.py` is production code, never
edited in place) fixing bug #1 (`mime_type`); bug #2 is fixed at the CALL
SITE (`run_gate1.py`/`run_gate2.py`/`smoke_reroute.py`, via
`call_window_bounds()`), not inside the analyzer, since the analyzer has no
way to know a given `youtube_url` string is actually a pre-cropped Files
API asset versus a full YouTube video. Used ONLY for the 3 rerouted
windows; every other window's calls still use the unmodified base
`SimplifiedTagsTimeAnalyzer` with absolute offsets against the full
YouTube video, unchanged.

Downloads/normalized derivatives are stored under `MEDIA_CACHE_DIR`
(``VLM_EVAL_HOME/media_cache/s12_gates_reroute`` — the SAME `VLM_EVAL_HOME`
convention `eval/adcc/predict/dashboard_app.py` already uses for its own
`DerivativeCache`), NEVER inside the repo. sha256 + the exact yt-dlp format
string are recorded per window in `reroute_manifest.json` (a NEW, separate
artifact — the already-frozen, already-sha256'd `gate1_subset_manifest.json`/
`gate2_subset_manifest.json` are never mutated by this module, matching the
same "never touches the original committed draw" discipline as §G.2's
replacement-draw procedure).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from eval.adcc.predict import media_resolver
from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.s12_gates import gemini_upload
from service.pipelines import gemini_retry
from service.pipelines.simplified_tags import (
    SimplifiedTagsTimeAnalyzer,
    _format_offset_seconds,
    _strip_markdown_fences,
    build_video_prompt,
    build_video_response_schema,
    extract_usage_metadata,
)

FILES_API_MIME_TYPE: str = "video/mp4"  # media_resolver.build_mp4_derivative's only output shape

logger = logging.getLogger("eval.adcc.s12_gates.files_api_reroute")


class FilesApiTimeAnalyzer(SimplifiedTagsTimeAnalyzer):
    """LOCAL, DISCLOSED override of ``analyze_chunk`` — see module docstring
    "Live bug found during the reroute smoke test" for the full writeup.
    Every line below is copied VERBATIM from
    ``SimplifiedTagsTimeAnalyzer.analyze_chunk`` (this worktree's version)
    with exactly ONE change: ``types.FileData(file_uri=..., mime_type=...)``
    now sets ``mime_type`` (via ``self._mime_type``, default
    ``FILES_API_MIME_TYPE``) instead of leaving it unset. Does not touch
    ``service/pipelines/simplified_tags.py`` — this is a subclass living
    entirely in ``eval/adcc/s12_gates/``, exactly the same "temporary,
    disclosed duplication, never a silent fork" discipline as the verbatim
    ``gemini_upload.py`` copy (§G.1 point 1)."""

    def __init__(self, *args, mime_type: str = FILES_API_MIME_TYPE, **kwargs):
        super().__init__(*args, **kwargs)
        self._mime_type = mime_type

    async def analyze_chunk(
        self, youtube_url: str, start_sec: float, end_sec: float, previous_context: Optional[str] = None,
        *, response_schema=None, fps: int = 1, media_resolution: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> str:
        effective_schema = response_schema if response_schema is not None else build_video_response_schema()
        prompt = prompt_text if prompt_text is not None else build_video_prompt(previous_context)
        video_part = types.Part(
            file_data=types.FileData(file_uri=youtube_url, mime_type=self._mime_type),  # <-- the one changed line
            video_metadata=types.VideoMetadata(
                start_offset=_format_offset_seconds(start_sec),
                end_offset=_format_offset_seconds(end_sec),
                fps=fps,
            ),
        )

        self.last_prompt_text = prompt
        self.last_raw_response_text = None
        self.last_usage_metadata = None

        try:
            logger.info(
                "Gemini simplified-tags-time-v1 (files_api): chunk [%.1f-%.1f]s, model %s, mime_type=%s",
                start_sec, end_sec, self.model_id, self._mime_type,
            )
            response = await gemini_retry.call_with_retry(
                lambda: self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=[types.Content(role="user", parts=[types.Part(text=prompt), video_part])],
                    config=self._build_generate_config(effective_schema, media_resolution),
                ),
                op_name="simplified-tags-time-v1-files-api",
                retry_config=self.retry_config,
            )
            text = response.text
            self.last_raw_response_text = text
            self.last_usage_metadata = extract_usage_metadata(response)
            logger.info(
                "Gemini simplified-tags-time-v1 (files_api): received response (%d chars)",
                len(text) if text else 0,
            )
            return _strip_markdown_fences(text or "")
        except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error, no fabrication
            logger.error("Gemini simplified-tags-time-v1 (files_api) call failed: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------- #
# The 3 rerouted windows (§G.1) — fixed, not a live "any window sharing this
# video" rule, matching the ruling's explicit enumeration.
# --------------------------------------------------------------------------- #
REROUTE_WINDOW_IDS: frozenset[str] = frozenset({
    "66b009481c62a1db7a53d504__w37",     # Gate 1 straddler, back_take, pwGbW5GZgfc t_s=30443
    "66afe5ab1c62a1db7a53b80b__occ0",    # Gate 2 forced-unsuccessful arm_lock, pwGbW5GZgfc t_s=28325
    "66a6ff021c62a1db7a5133c0__occ11",   # Gate 2 forced-unsuccessful leg_lock, pwGbW5GZgfc t_s=12775
})
REROUTE_BLOCKED_VIDEO_ID: str = "pwGbW5GZgfc"

YTDLP_BIN: str = "/opt/homebrew/bin/yt-dlp"
FFMPEG_BIN: str = "/opt/homebrew/bin/ffmpeg"  # informational — media_resolver invokes bare "ffmpeg"/"ffprobe" via PATH
YTDLP_FORMAT: str = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]"
SECTION_BUFFER_S: float = 2.0  # §G.1 point 2 — 2s past each edge so a keyframe-boundary cut never clips the window short

UPLOAD_POLL_INTERVAL_SEC: float = 5.0
UPLOAD_POLL_TIMEOUT_SEC: float = 600.0  # matches wt-s12's ServiceConfig.gemini_upload_poll_timeout_sec default

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../eval/adcc/s12_gates/files_api_reroute.py -> repo root
VLM_EVAL_HOME = Path(os.environ.get("VLM_EVAL_HOME", str(_REPO_ROOT.parent)))
MEDIA_CACHE_DIR = VLM_EVAL_HOME / "media_cache" / "s12_gates_reroute"
REROUTE_MANIFEST_PATH = gc.S12_GATES_DIR / "reroute_manifest.json"


@dataclass
class ReroutedUpload:
    window_id: str
    match_id: str
    file_uri: str
    file_name: str
    ytdlp_format: str
    ytdlp_section_spec: str
    local_source_path: str
    local_normalized_path: str
    source_sha256: str
    normalized_sha256: str
    download_latency_ms: float
    normalize_latency_ms: float
    upload_latency_ms: float
    active_wait_ms: float
    # §G.3 (live bug #2, see module docstring) — the uploaded file's OWN
    # clock starts at 0:00 (it is already a short, yt-dlp-cropped section of
    # the original video), NOT the original video's absolute match clock.
    # `sec_start_s` is the ORIGINAL video's absolute second the uploaded
    # file's 0:00 corresponds to (`max(0, window_start_s - SECTION_BUFFER_S)`
    # — see `download_section`). Callers MUST convert any Gemini
    # `video_metadata` offset from absolute match-clock seconds to
    # file-relative seconds via `absolute_s - sec_start_s` before calling
    # this window through the Files API path — `call_window_bounds()` below
    # does exactly that, once, so no call site re-derives it by hand.
    sec_start_s: float

    def call_window_bounds(self, window_start_s: float, window_end_s: float) -> tuple[float, float]:
        """Converts ABSOLUTE match-clock ``[window_start_s, window_end_s]``
        to FILE-RELATIVE offsets for a ``video_metadata`` call against this
        upload's ``file_uri``. Never negative (clamped to >=0 — a caller
        passing bounds outside what was actually downloaded is a real bug,
        not something to silently clamp past 0, but the lower bound is
        already guaranteed >=0 by construction: ``window_start_s >=
        sec_start_s`` always holds since ``sec_start_s`` is exactly
        ``window_start_s`` minus a POSITIVE buffer)."""
        return (
            round(window_start_s - self.sec_start_s, 3),
            round(window_end_s - self.sec_start_s, 3),
        )


async def _run_subprocess(cmd: list[str], *, op_name: str) -> tuple[bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"{op_name} failed (exit {proc.returncode}): {stderr.decode(errors='replace')[-2000:]}"
        )
    return stdout, stderr


async def download_section(
    video_url_watch: str, window_start_s: float, window_end_s: float, out_dir: Path, *, window_id: str,
) -> tuple[Path, str, float]:
    """yt-dlp section-download `[window_start_s-2, window_end_s+2]` (clamped
    to >=0 at the lower edge — never negative, defensively, though no
    selected window's start is within 2s of t=0). Returns
    ``(downloaded_file_path, section_spec_string, sec_start)`` — ``sec_start``
    is the ORIGINAL video's absolute second the downloaded file's OWN 0:00
    corresponds to (needed downstream to convert absolute match-clock
    offsets to file-relative ones, see ``ReroutedUpload.call_window_bounds``).
    Idempotent — a stale prior download for this ``window_id`` is removed
    first, so re-running a smoke/full pass never leaves a duplicate/
    mismatched file behind."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sec_start = max(0.0, window_start_s - SECTION_BUFFER_S)
    sec_end = window_end_s + SECTION_BUFFER_S
    section_spec = f"*{sec_start:.3f}-{sec_end:.3f}"
    out_template = str(out_dir / f"{window_id}.raw.%(ext)s")

    for stale in out_dir.glob(f"{window_id}.raw.*"):
        stale.unlink()

    cmd = [
        YTDLP_BIN, "--download-sections", section_spec, "-f", YTDLP_FORMAT,
        "-o", out_template, "--no-playlist", "--quiet", "--no-warnings", video_url_watch,
    ]
    await _run_subprocess(cmd, op_name=f"yt-dlp section-download ({window_id}, {section_spec})")

    matches = sorted(out_dir.glob(f"{window_id}.raw.*"))
    if not matches:
        raise RuntimeError(
            f"yt-dlp reported success but no output file found for {window_id} (template {out_template})"
        )
    return matches[0], section_spec, sec_start


async def prepare_files_api_upload(
    client: genai.Client, window: dict, *, cache_dir: Path = MEDIA_CACHE_DIR,
) -> ReroutedUpload:
    """Full pipeline for ONE window: download -> normalize -> upload ->
    poll ACTIVE. No fallback logic — every step raises loud on failure
    (yt-dlp non-zero exit, `media_resolver.MediaBuildError`,
    `gemini_upload`'s own FAILED/timeout raises); nothing here fabricates a
    success state."""
    window_id = window["window_id"]

    t0 = time.perf_counter()
    src_path, section_spec, sec_start = await download_section(
        window["video_url_watch"], window["window_start_s"], window["window_end_s"], cache_dir, window_id=window_id,
    )
    download_latency_ms = (time.perf_counter() - t0) * 1000.0

    norm_path = cache_dir / f"{window_id}.normalized.mp4"
    t0 = time.perf_counter()
    await media_resolver.build_mp4_derivative(src_path, norm_path)
    normalize_latency_ms = (time.perf_counter() - t0) * 1000.0

    source_sha256 = gc.sha256_file(src_path)
    normalized_sha256 = gc.sha256_file(norm_path)

    t0 = time.perf_counter()
    uploaded = await gemini_upload.upload_video_to_gemini(client, str(norm_path))
    upload_latency_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    active_file = await gemini_upload.poll_until_active(
        client, uploaded.name, poll_interval_sec=UPLOAD_POLL_INTERVAL_SEC, timeout_sec=UPLOAD_POLL_TIMEOUT_SEC,
    )
    active_wait_ms = (time.perf_counter() - t0) * 1000.0

    return ReroutedUpload(
        window_id=window_id, match_id=window["match_id"],
        file_uri=active_file.uri, file_name=active_file.name,
        ytdlp_format=YTDLP_FORMAT, ytdlp_section_spec=section_spec,
        local_source_path=str(src_path), local_normalized_path=str(norm_path),
        source_sha256=source_sha256, normalized_sha256=normalized_sha256,
        download_latency_ms=download_latency_ms, normalize_latency_ms=normalize_latency_ms,
        upload_latency_ms=upload_latency_ms, active_wait_ms=active_wait_ms,
        sec_start_s=sec_start,
    )


def build_files_api_time_analyzer(
    client: genai.Client, *, system_instruction: Optional[str],
    retry_config: "gemini_retry.GeminiRetryConfig", model_id: str = gc.MODEL_ID,
) -> FilesApiTimeAnalyzer:
    """``gate_common.build_time_analyzer``'s sibling for the reroute lane —
    same construction, but the ``FilesApiTimeAnalyzer`` subclass so calls
    against a Files API URI carry ``mime_type`` (see module docstring's
    "Live bug found" section). Used ONLY for the 3 rerouted windows.

    **[RE-RUN 2026-07-26, GATES_RERUN_DESIGN.md §2] fixed the SAME
    thinking="high" bug as ``gate_common.build_time_analyzer`` — found here
    independently during the re-run config change (the CEO's cited line
    list did not name this second call site, but leaving it at
    ``thinking=None`` would have silently left the 3 rerouted windows
    (w37/occ0/occ11) at NO thinking while every other window in the SAME
    Gate 1/2 batch ran at "high" — an uncontrolled, undisclosed confound
    inside a single re-run, exactly what this whole program exists to
    avoid).** Both this function and ``gate_common.build_time_analyzer``
    must always be kept in lockstep on ``thinking`` level going forward.
    """
    return FilesApiTimeAnalyzer(
        client, model_id=model_id,
        thinking_config=gc.thinking_config_for(model_id, "high"),
        system_instruction=system_instruction, retry_config=retry_config,
    )


class RerouteUploadCache:
    """One upload per window, reused across ALL of that window's calls
    (§G.1 point 3 — mandatory for cross-arm paired validity). Also persists
    a `reroute_manifest.json` record (sha256 + yt-dlp format string, per the
    CEO's explicit instruction) and one `gate_budget_ledger.json` upload
    event per window (§G.1 point 5 — the same upload_latency_ms/
    active_wait_ms numbers Gate 3 needs, GATES_3_4_DESIGN.md §H)."""

    def __init__(self, client: genai.Client, ledger: gc.CallLedger, *, gate: str):
        self.client = client
        self.ledger = ledger
        self.gate = gate
        self._cache: dict[str, ReroutedUpload] = {}
        self._manifest_records: list[dict] = (
            gc.read_json(REROUTE_MANIFEST_PATH) if REROUTE_MANIFEST_PATH.exists() else []
        )

    async def ensure(self, window: dict) -> ReroutedUpload:
        window_id = window["window_id"]
        if window_id in self._cache:
            return self._cache[window_id]

        # Bezos precondition 1 (2026-07-26): "reuse the existing cached upload
        # if still ACTIVE... verify state, re-ensure if expired" — check
        # reroute_manifest.json for a PRIOR upload of this window and confirm
        # it is still real/ACTIVE on Gemini's side (Files API TTL is ~48h;
        # a prior run may also have explicitly deleted it via cleanup()) before
        # trusting it. Never assumed valid from the manifest record alone.
        reused_upload = await self._try_reuse_persisted(window)
        if reused_upload is not None:
            self._cache[window_id] = reused_upload
            self.ledger.record_upload_event(
                gate=self.gate, window_id=window_id, match_id=window["match_id"],
                status="ok", error=None, upload_latency_ms=0.0, active_wait_ms=0.0, reused=True,
            )
            return reused_upload

        try:
            upload = await prepare_files_api_upload(self.client, window)
        except Exception as e:  # noqa: BLE001 — record the real failure, then re-raise (no fabricated URI)
            self.ledger.record_upload_event(
                gate=self.gate, window_id=window_id, match_id=window["match_id"],
                status="call_failed", error=str(e), upload_latency_ms=0.0, active_wait_ms=None,
            )
            self._flush_manifest()
            raise
        self._cache[window_id] = upload
        self.ledger.record_upload_event(
            gate=self.gate, window_id=window_id, match_id=window["match_id"],
            status="ok", error=None, upload_latency_ms=upload.upload_latency_ms,
            active_wait_ms=upload.active_wait_ms,
        )
        self._manifest_records = [r for r in self._manifest_records if r["window_id"] != window_id]
        self._manifest_records.append({**asdict(upload), "gate": self.gate})
        self._flush_manifest()
        return upload

    async def _try_reuse_persisted(self, window: dict) -> Optional[ReroutedUpload]:
        """Looks up ``reroute_manifest.json`` for a prior upload of this
        ``window_id`` and, if found, calls ``client.aio.files.get`` to
        confirm it is STILL in ``ACTIVE`` state on Gemini's side (never
        trusted from the manifest record alone — a prior run's ``cleanup()``
        may have deleted it, or the ~48h TTL may have expired it server-side
        since the record was written). Returns ``None`` (never raises) on
        any lookup failure/non-ACTIVE state — the caller falls through to a
        fresh upload, exactly the "re-ensure if expired" instruction."""
        window_id = window["window_id"]
        prior = next((r for r in self._manifest_records if r["window_id"] == window_id), None)
        if prior is None:
            return None
        try:
            f = await self.client.aio.files.get(name=prior["file_name"])
        except Exception as e:  # noqa: BLE001 — a 404/NotFound (deleted/expired) is expected, not a real error
            logger.info(
                "files_api_reroute: prior upload %s for %s not retrievable (%s) — will re-upload.",
                prior.get("file_name"), window_id, e,
            )
            return None
        if f.state != types.FileState.ACTIVE:
            logger.info(
                "files_api_reroute: prior upload %s for %s is state=%s (not ACTIVE) — will re-upload.",
                prior.get("file_name"), window_id, f.state,
            )
            return None
        logger.info("files_api_reroute: reusing still-ACTIVE prior upload %s for %s.", prior["file_name"], window_id)
        return ReroutedUpload(
            window_id=window_id, match_id=prior["match_id"],
            file_uri=f.uri, file_name=f.name,
            ytdlp_format=prior["ytdlp_format"], ytdlp_section_spec=prior["ytdlp_section_spec"],
            local_source_path=prior["local_source_path"], local_normalized_path=prior["local_normalized_path"],
            source_sha256=prior["source_sha256"], normalized_sha256=prior["normalized_sha256"],
            download_latency_ms=0.0, normalize_latency_ms=0.0,
            upload_latency_ms=0.0, active_wait_ms=0.0,
            sec_start_s=prior["sec_start_s"],
        )

    def _flush_manifest(self) -> None:
        gc.write_json(REROUTE_MANIFEST_PATH, self._manifest_records)

    async def cleanup(self) -> None:
        """Best-effort Files API delete for every uploaded file this cache
        instance created (mirrors `gemini_upload.delete_gemini_file`'s own
        best-effort, never-blocks discipline)."""
        for upload in self._cache.values():
            await gemini_upload.delete_gemini_file(self.client, upload.file_name)
