"""ADCC eval cost-control dashboard — FastAPI backend (Task 3).

Self-contained, ``qa_app.py``-pattern backend (no lifespan/Cassandra/S3
coupling): wraps the existing ``chunk-segment-tags`` capture engine
(``service/pipelines/executors.py``/``registry.py``, never forked) plus the
ADCC eval harness's own ``eval/adcc/predict/**`` modules (scope policy,
slicing, stitching, the smoke-test target selector, and this build's new
``global_cap_driver``/``cost_model``/``dashboard_scope``). Never imports
``service/routes/**`` — the server's two preserved shims there
(``service/routes/__init__.py`` emptied, ``service/routes/state.py`` reduced
to ``_config = ServiceConfig()``) exist for the ALREADY-DEPLOYED QA VLM Studio
app, not for this one; this module's only transitive touch of that package is
via ``service.pipelines.executors``/``registry`` importing
``service.routes.qa_vlm`` for shared helpers (unavoidable, pre-existing, and
exactly why those two shims must be preserved on every redeploy).

**The spine rule (Jobs): no single click anywhere can spend a dollar.**
  - ``GET /matches`` is PURE METADATA + ARITHMETIC (``match_catalog.build_match_catalog``)
    -- the searchable match-picker menu's data source: every match's human
    name, the length actually analyzed, and both the realistic (primary) and
    worst-case-ceiling (secondary) call/dollar estimates. Zero Gemini import
    ever executes a call from this path either; the menu only BUILDS the
    ``match_id``/``match_ids`` a scope request carries -- it never bypasses
    the ``/plan`` preview -> confirm -> cap gate below.
  - ``GET /plan`` is PURE ARITHMETIC — builds the real per-slice plan
    (``run_predictions.build_dry_run_plan``) and prices it
    (``cost_model.estimate_batch_cost_usd``). Zero Gemini import ever
    executes a call from this path.
  - ``POST /run`` requires ALL of: (a) ``confirm=True`` explicitly, (b) a
    ``plan_token`` that the server recomputes from the SUBMITTED scope+cap
    and must match exactly (stale-preview guard — changing scope/cap
    invalidates any earlier preview, enforced server-side, not just
    client-side UX), (c) a configured Gemini key, (d) no other run already in
    progress. Only then does it stream ``global_cap_driver.run_batch_with_global_cap``,
    which enforces the GLOBAL cross-slice/match call cap (Task 2) and can be
    aborted mid-flight by ``POST /stop``.
  - ``POST /stop`` flips the ACTIVE run's ``BatchRunState.stop_requested`` —
    guarantees zero NEW Gemini calls after it registers (at most the one
    call already in flight, if any, is allowed to complete; see
    ``global_cap_driver`` module docstring for the exact mechanism).

**Highlight Eval feature (frozen contract) — a NEW, separate namespace.**
``GET /highlight`` serves a SIBLING client page (``dashboard_client/
highlight_eval.html``, never a tab bolted onto ``cost_dashboard.html``).
``POST /highlight/plan``/``POST /highlight/run``/``POST /highlight/score``
follow the identical preview -> confirm -> cap -> stream -> score spine as
above, single-match scope only, backed by the NEW PEER module
``eval/adcc/predict/highlight_discovery.py`` (its OWN ``genai.Client``,
never ``executors.run_pipeline`` — see that module's docstring) and the NEW
scoring glue ``eval/adcc/score/highlight_scoring.py``/``highlight_baselines.
py`` (which reuse ``matcher.py``/``metrics.summarize`` 100% UNCHANGED).
Critically, ``POST /highlight/run`` SHARES the exact same
``app.state.active_run_state`` slot and the SAME ``POST /stop`` handler as
the tag-batch ``/run`` above — a highlight run and a tag-batch run can never
both be live, by construction (both check/set the identical slot), never a
second, parallel mutual-exclusion mechanism.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from eval.adcc.predict import highlight_discovery as hd
from eval.adcc.predict.cost_model import (
    MARGIN_MULTIPLIER,
    estimate_batch_calls_realistic,
    estimate_batch_cost_usd,
    estimate_highlight_call_cost_usd,
)
from eval.adcc.predict.dashboard_scope import (
    GoldCorpus,
    ScopeError,
    ScopeSpec,
    build_slices_by_match,
    compute_highlight_plan_token,
    compute_plan_token,
    load_gold_corpus,
    resolve_scope_targets,
)
from eval.adcc.predict.global_cap_driver import BatchRunState, run_batch_with_global_cap
from eval.adcc.predict.match_catalog import build_match_catalog
from eval.adcc.predict.media_resolver import (
    DerivativeCache,
    MediaBuildError,
    MediaResolutionError,
    MediaRootIndex,
    UnsupportedExtensionError,
)
from eval.adcc.predict.run_predictions import RunPredictionsError, build_dry_run_plan, load_video_durations_json
from eval.adcc.score import highlight_baselines as hb
from eval.adcc.score import highlight_scoring as hs
from service.config import ServiceConfig

logger = logging.getLogger("eval.adcc.predict.dashboard_app")

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../eval/adcc/predict/dashboard_app.py -> repo root
# On the server this is ~/vlm-eval (sibling of the deployed "code" checkout
# and "venv"); on a Mac dev checkout it resolves to the umbrella/parent dir,
# which simply won't have a .env/video_durations.json — harmless, the
# dashboard runs fine with an empty video-durations mapping (only single-
# match-video scopes need it; see RunPredictionsError below).
_DEFAULT_VLM_EVAL_HOME = _REPO_ROOT.parent
VLM_EVAL_HOME = Path(os.environ.get("VLM_EVAL_HOME", str(_DEFAULT_VLM_EVAL_HOME)))

_DEFAULT_GOLD_SNAPSHOT = _REPO_ROOT / "eval" / "adcc" / "gold_snapshot" / "normalized_gold.jsonl"
_DEFAULT_SPLIT_ASSIGNMENT = _REPO_ROOT / "eval" / "adcc" / "gold_snapshot" / "split_assignment.json"
_DEFAULT_VIDEO_DURATIONS = VLM_EVAL_HOME / "video_durations.json"
_DEFAULT_OUTPUT_DIR = VLM_EVAL_HOME / "dashboard_runs"
_DEFAULT_HTML_PATH = Path(__file__).resolve().parent / "dashboard_client" / "cost_dashboard.html"

# Highlight Eval feature (frozen contract) — the GT-provenance artifact
# ``build_highlight_gold.py`` writes, NEVER recomputed ad hoc here (frozen
# contract §B). New sibling client page (frozen contract §F — a SIBLING
# page, never a tab bolted onto ``cost_dashboard.html``).
_DEFAULT_HIGHLIGHT_GOLD = _REPO_ROOT / "eval" / "adcc" / "gold_snapshot" / "highlight_gold.jsonl"
_DEFAULT_HIGHLIGHT_GOLD_MANIFEST = _REPO_ROOT / "eval" / "adcc" / "gold_snapshot" / "highlight_gold_manifest.json"
_DEFAULT_HIGHLIGHT_HTML_PATH = Path(__file__).resolve().parent / "dashboard_client" / "highlight_eval.html"

# Item 2 media fallback — ADCC media root + lazy browser-playable cache.
# Paths never leave the server; the route accepts only opaque match_id.
_DEFAULT_ADCC_MEDIA_ROOT = Path(
    os.environ.get("BJJ_ADCC_MEDIA_ROOT", "/mnt/e/assets/data/bjj-vision/all|ADCC|all/videos")
)
_DEFAULT_DERIVATIVE_CACHE_DIR = VLM_EVAL_HOME / "media_cache"

# Path-free client messages for the media route (never echo resolver/build
# exception text — MediaBuildError may include source filenames / stderr).
_MEDIA_UNKNOWN_MSG = "unknown or unavailable match_id"
_MEDIA_UNSUPPORTED_MSG = "unsupported media type for this match"
_MEDIA_BUILD_FAILED_MSG = "Could not prepare a browser-playable copy for this match"

DEFAULT_PORT = 8321


# --------------------------------------------------------------------------- #
# Request/response models (API layer only — internal resolution uses
# dashboard_scope.ScopeSpec, a plain frozen dataclass).
# --------------------------------------------------------------------------- #
class ScopeRequest(BaseModel):
    mode: Literal["single", "list", "smoke", "split"]
    match_id: Optional[str] = None
    match_ids: Optional[list[str]] = None
    split: Optional[Literal["sandbox", "dev", "test"]] = None

    def to_spec(self) -> ScopeSpec:
        return ScopeSpec(
            mode=self.mode, match_id=self.match_id,
            match_ids=tuple(self.match_ids) if self.match_ids else None,
            split=self.split,
        )


class RunRequest(BaseModel):
    scope: ScopeRequest
    cap: int
    plan_token: str
    confirm: bool = False


class HighlightPlanRequest(BaseModel):
    """Highlight Eval feature (frozen contract §E/§F) — single-match scope
    only (section F's single-select match picker), no scope-mode dispatch:
    this is a coarse pass over ONE ADCC corpus video/match at a time.

    ``fps``/``frame_quality`` (an earlier increment's CEO-frozen controls)
    default to ``hd.DEFAULT_FPS``/``hd.DEFAULT_FRAME_QUALITY`` — reproducing
    exactly the pre-that-increment Gemini call shape when left unset.

    ``refine``/``refine_pad_before_s``/``refine_pad_after_s``/``refine_fps``
    (timestamp critique/refinement, this increment) default to ON with
    Rener's frozen pad envelope (12s before / 5s after) and 4fps. ``refine=
    False`` skips the entire per-highlight refinement pass — ``est_calls=1``
    and the persisted shape/cost stay BYTE-FOR-BYTE what they were before
    this feature existed (``reason`` is still requested/persisted regardless
    of this toggle, per frozen contract §A)."""

    match_id: str
    cap: int
    model: str = hd.DEFAULT_HIGHLIGHT_MODEL
    thinking_level: Optional[str] = None
    system_prompt: str = ""
    followup_prompt: str = ""
    fps: float = hd.DEFAULT_FPS
    frame_quality: str = hd.DEFAULT_FRAME_QUALITY
    refine: bool = True
    refine_pad_before_s: float = hd.DEFAULT_PAD_BEFORE_S
    refine_pad_after_s: float = hd.DEFAULT_PAD_AFTER_S
    refine_fps: float = hd.DEFAULT_REFINE_FPS


class HighlightRunRequest(HighlightPlanRequest):
    plan_token: str
    confirm: bool = False


class HighlightScoreRequest(BaseModel):
    match_id: str
    tau_t: float = 5.0


def _parse_scope_query(
    mode: str, match_id: Optional[str], match_ids: Optional[str], split: Optional[str],
) -> ScopeRequest:
    return ScopeRequest(
        mode=mode,  # type: ignore[arg-type]  # pydantic validates against the Literal
        match_id=match_id,
        match_ids=[m.strip() for m in match_ids.split(",") if m.strip()] if match_ids else None,
        split=split,  # type: ignore[arg-type]
    )


def create_app(
    *,
    gold_snapshot_path: Path = _DEFAULT_GOLD_SNAPSHOT,
    split_assignment_path: Optional[Path] = _DEFAULT_SPLIT_ASSIGNMENT,
    video_durations_path: Optional[Path] = _DEFAULT_VIDEO_DURATIONS,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    html_path: Path = _DEFAULT_HTML_PATH,
    highlight_gold_path: Path = _DEFAULT_HIGHLIGHT_GOLD,
    highlight_gold_manifest_path: Path = _DEFAULT_HIGHLIGHT_GOLD_MANIFEST,
    highlight_html_path: Path = _DEFAULT_HIGHLIGHT_HTML_PATH,
    media_root: Path = _DEFAULT_ADCC_MEDIA_ROOT,
    derivative_cache_dir: Path = _DEFAULT_DERIVATIVE_CACHE_DIR,
    service_config: Optional[ServiceConfig] = None,
) -> FastAPI:
    """Factory (dependency-injection-friendly, unlike a bare module-level
    singleton) so tests can point every path at an isolated ``tmp_path``
    fixture and never touch the real gold snapshot/.env. The module-level
    ``app`` at the bottom of this file is ``create_app()`` with the real
    on-disk defaults — what actually runs in production."""
    config = service_config or ServiceConfig()

    corpus: GoldCorpus = load_gold_corpus(gold_snapshot_path, split_assignment_path)
    video_durations: dict = (
        load_video_durations_json(video_durations_path)
        if video_durations_path is not None and video_durations_path.exists()
        else {}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    name_by_match_id: dict = {rec["match_id"]: rec["name"] for rec in corpus.gold_records}

    # Highlight Eval feature (frozen contract §B) — loads the PROVENANCED
    # artifact ``build_highlight_gold.py`` writes; NEVER recomputes
    # ``is_highlight`` here. Missing on a fresh test fixture (highlight
    # endpoints aren't exercised) degrades to an empty corpus, never a
    # create_app()-time crash — mirrors ``video_durations_path``'s own
    # "optional, absent is a real and expected state" discipline above.
    highlight_gold_by_match: dict = (
        hs.load_highlight_gold_by_match(highlight_gold_path) if highlight_gold_path.exists() else {}
    )
    highlight_gold_manifest: dict = (
        json.loads(highlight_gold_manifest_path.read_text()) if highlight_gold_manifest_path.exists() else {}
    )

    # Item 2 (frozen contract §7) — built once, eagerly: fails loud
    # (DuplicateStemError) at THIS point if the on-disk media root ever has
    # two files for one stem, never per-request.
    match_id_to_stem = {t.match_id: t.video_url_md5 for t in corpus.all_targets().values()}
    media_index = MediaRootIndex.build(media_root, match_id_to_stem)
    derivative_cache = DerivativeCache(derivative_cache_dir)

    app = FastAPI(title="ADCC Eval Cost-Control Dashboard", version="1.0.0")
    # Single active run at a time (Jobs's spec — no concurrency complexity).
    # Held on app.state so create_app() stays test-isolated (no module-level
    # mutable global shared across FastAPI app instances in the same process).
    app.state.active_run_state = None  # type: ignore[attr-defined]

    @app.get("/")
    async def index():
        if not html_path.exists():
            raise HTTPException(404, f"Dashboard client not found at {html_path}")
        return FileResponse(html_path, media_type="text/html")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "gemini_key_configured": bool(config.gemini_api_key),
            "match_count_in_corpus": len(corpus.gold_records),
            "video_durations_loaded": len(video_durations),
            "active_run": app.state.active_run_state is not None,
            "highlight_gold_match_count": len(highlight_gold_by_match),
        }

    @app.get("/highlight")
    async def highlight_index():
        """Serves the Highlight Eval SIBLING client page (frozen contract
        §F — a separate page, never a tab in ``cost_dashboard.html``)."""
        if not highlight_html_path.exists():
            raise HTTPException(404, f"Highlight Eval client not found at {highlight_html_path}")
        return FileResponse(highlight_html_path, media_type="text/html")

    @app.get("/matches")
    async def matches():
        """Zero-Gemini per-match catalog (name + length + dual call
        estimates) for the dashboard's searchable match-picker menu -- see
        ``match_catalog.build_match_catalog`` for the full contract. Pure
        metadata + arithmetic over the already-loaded ``corpus``/
        ``video_durations`` -- no live Gemini client anywhere in this call
        graph, safe to call at any time (menu population, not a run)."""
        catalog = build_match_catalog(corpus, video_durations)
        return {"match_count": len(catalog), "matches": catalog}

    @app.get("/plan")
    async def plan(
        mode: str, cap: int, match_id: Optional[str] = None, match_ids: Optional[str] = None,
        split: Optional[str] = None,
    ):
        """Dry-run preview. ZERO Gemini calls — pure arithmetic over the real
        per-slice plan (never a fabricated/approximate slice count)."""
        try:
            scope_req = _parse_scope_query(mode, match_id, match_ids, split)
        except ValidationError as e:
            # _parse_scope_query builds ScopeRequest manually from query
            # params (not FastAPI's own body-parsing path), so a bad `mode`
            # (e.g. the deliberately-unsupported "all") must be converted to
            # a 400 here explicitly — otherwise it propagates as an
            # unhandled 500.
            raise HTTPException(400, f"Invalid scope query parameters: {e}") from e
        scope_spec = scope_req.to_spec()

        try:
            targets = resolve_scope_targets(scope_spec, corpus)
        except ScopeError as e:
            raise HTTPException(400, str(e)) from e

        # build_dry_run_plan derives its OWN slice plan internally (verbatim
        # reuse — this endpoint never re-implements the slicing math); it's
        # the single source of truth for per-slice worst-case call counts.
        try:
            dry_run_plan = build_dry_run_plan(targets, video_durations)
        except RunPredictionsError as e:
            raise HTTPException(400, str(e)) from e

        scope_seconds = [
            sl["duration_s"] for m in dry_run_plan["per_match"] for sl in m["slices"]
        ]
        est_dollars_with_margin = estimate_batch_cost_usd(scope_seconds, apply_margin=True)
        est_dollars_realistic = estimate_batch_cost_usd(scope_seconds, apply_margin=False)
        # PRIMARY headline number (LeCun's realistic per-run formula, the
        # same one match_catalog.py sums for the menu's live aggregate and
        # cost_model.py prices in dollars above) -- additive field, kept
        # alongside the pre-existing "..._worst_case" field (now explicitly
        # the SECONDARY "worst-case ceiling / cap suggestion" number) so this
        # response stays backward-compatible.
        total_calls_realistic = estimate_batch_calls_realistic(scope_seconds)

        plan_token = compute_plan_token(scope_spec.canonical(), cap)

        return {
            "plan_token": plan_token,
            "scope": scope_spec.canonical(),
            "cap": cap,
            "pipeline_id": dry_run_plan["pipeline_id"],
            "match_count": dry_run_plan["match_count"],
            "over_cap_match_count": dry_run_plan["over_cap_match_count"],
            "total_slices": dry_run_plan["total_slices"],
            "total_calls_realistic": total_calls_realistic,
            "total_planned_gemini_calls_worst_case": dry_run_plan["total_planned_gemini_calls_worst_case"],
            "within_cap": dry_run_plan["total_planned_gemini_calls_worst_case"] <= cap,
            "est_dollars_with_margin": round(est_dollars_with_margin, 2),
            "est_dollars_realistic": round(est_dollars_realistic, 2),
            "per_match": dry_run_plan["per_match"],
        }

    @app.post("/run")
    async def run(body: RunRequest):
        if not config.gemini_api_key:
            raise HTTPException(400, "GEMINI_API_KEY is not configured server-side — cannot run.")
        if not body.confirm:
            raise HTTPException(400, "confirm=true is required — this endpoint spends real Gemini quota.")
        if body.cap <= 0:
            raise HTTPException(400, f"cap must be a positive integer, got {body.cap!r}.")
        if app.state.active_run_state is not None:
            raise HTTPException(409, "A run is already in progress. Stop it (POST /stop) before starting another.")

        scope_spec = body.scope.to_spec()
        expected_token = compute_plan_token(scope_spec.canonical(), body.cap)
        if expected_token != body.plan_token:
            raise HTTPException(
                400,
                "Stale or invalid preview: plan_token does not match the submitted scope+cap. "
                "Re-run GET /plan for the EXACT scope+cap you want to run, then confirm again.",
            )

        try:
            targets = resolve_scope_targets(scope_spec, corpus)
        except ScopeError as e:
            raise HTTPException(400, str(e)) from e

        try:
            slices_by_match = build_slices_by_match(targets, video_durations)
        except RunPredictionsError as e:
            raise HTTPException(400, str(e)) from e

        state = BatchRunState(cap=body.cap)
        app.state.active_run_state = state

        async def event_stream():
            try:
                async for event in run_batch_with_global_cap(
                    targets, slices_by_match,
                    gemini_api_key=config.gemini_api_key,
                    request_timeout_ms=config.gemini_request_timeout_ms,
                    state=state,
                ):
                    if event.get("type") == "batch_complete":
                        _persist_batch_results(output_dir, event.get("results", {}))
                    yield json.dumps(event) + "\n"
            except Exception as e:  # noqa: BLE001 — surface a fatal run-loop error as a final NDJSON line
                logger.error("dashboard /run: run loop crashed: %s", e, exc_info=True)
                yield json.dumps({"type": "error", "stage_id": "run", "message": str(e)}) + "\n"
            finally:
                app.state.active_run_state = None

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    # ----------------------------------------------------------------- #
    # Highlight Eval feature (frozen contract §E) — a NEW namespace, never
    # overloading /plan or /run. Single-match scope only (§F's single-select
    # match picker); SHARES ``app.state.active_run_state``/the ``/stop``
    # handler above with the tag-batch flow — a highlight run and a
    # tag-batch run are mutually exclusive by construction (both check/set
    # the SAME slot), never a second, parallel concurrency mechanism.
    # ----------------------------------------------------------------- #
    def _resolve_highlight_target_and_window(match_id: str):
        target = corpus.all_targets().get(match_id)
        if target is None:
            raise HTTPException(400, f"Unknown match_id {match_id!r} (not in the gold snapshot).")
        try:
            window = hd.resolve_highlight_window(target, video_durations)
        except hd.HighlightDiscoveryError as e:
            raise HTTPException(400, str(e)) from e
        return target, window

    def _validate_highlight_request(body) -> None:
        if body.cap <= 0:
            raise HTTPException(400, f"cap must be a positive integer, got {body.cap!r}.")
        if body.model not in hd.HIGHLIGHT_MODELS:
            raise HTTPException(400, f"Unsupported model {body.model!r}; must be one of {hd.HIGHLIGHT_MODELS}.")
        if body.thinking_level is not None and body.thinking_level not in hd.THINKING_LEVELS:
            raise HTTPException(
                400,
                f"thinking_level must be one of {hd.THINKING_LEVELS} or null, got {body.thinking_level!r}.",
            )
        if body.fps not in hd.FPS_OPTIONS:
            raise HTTPException(400, f"fps must be one of {hd.FPS_OPTIONS}, got {body.fps!r}.")
        if body.frame_quality not in hd.FRAME_QUALITY_OPTIONS:
            raise HTTPException(
                400, f"frame_quality must be one of {hd.FRAME_QUALITY_OPTIONS}, got {body.frame_quality!r}.",
            )
        if body.refine_fps not in hd.FPS_OPTIONS:
            raise HTTPException(400, f"refine_fps must be one of {hd.FPS_OPTIONS}, got {body.refine_fps!r}.")
        if body.refine_pad_before_s < 0:
            raise HTTPException(400, f"refine_pad_before_s must be >= 0, got {body.refine_pad_before_s!r}.")
        if body.refine_pad_after_s < 0:
            raise HTTPException(400, f"refine_pad_after_s must be >= 0, got {body.refine_pad_after_s!r}.")

    @app.post("/highlight/plan")
    async def highlight_plan(body: HighlightPlanRequest):
        """Dry-run preview. ZERO Gemini calls — pure arithmetic. Prices via
        the DEDICATED ``cost_model.estimate_highlight_call_cost_usd`` (LeCun's
        fps/media_resolution-aware decomposition) — NOT the tag-batch
        ``estimate_batch_cost_usd`` reuse (that formula's chunk multiplier
        never applied to this 1-call-per-match feature in the first place;
        the tag-batch ``/plan``/``/run`` pricing path above is untouched).

        ``refine=False`` returns EXACTLY today's (pre-refinement) single-call
        estimate. ``refine=True`` (frozen contract §E, Brooks — "cost-anchored
        range, not corpus-average"): the preview cannot know in advance how
        many highlights discovery will find, so it reports a RANGE anchored
        to the enforced cap ceiling, ``est_calls_low=1`` (discovery only, the
        worst case being zero highlights found) to ``est_calls_high=cap``
        (every remaining call slot spent on refinement), PLUS a separate,
        clearly-labeled INFORMATIONAL "typical" aside using the corpus's own
        real highlight density (``highlight_baselines.TARGET_SPANS_PER_MATCH``
        ~= 2.16/match) — never the priced ceiling."""
        _validate_highlight_request(body)
        target, window = _resolve_highlight_target_and_window(body.match_id)

        discovery_dollars_realistic = estimate_highlight_call_cost_usd(window.duration_s, body.fps, body.frame_quality)

        plan_token = compute_highlight_plan_token(
            body.match_id, body.cap, body.model, body.thinking_level, body.system_prompt, body.followup_prompt,
            body.fps, body.frame_quality, body.refine, body.refine_pad_before_s, body.refine_pad_after_s,
            body.refine_fps,
        )

        response = {
            "plan_token": plan_token,
            "match_id": body.match_id,
            "match_name": name_by_match_id.get(body.match_id, ""),
            "video_url": target.video_url,
            "is_multi_match": target.is_multi_match,
            "window": [window.start_s, window.end_s],
            "window_duration_s": window.duration_s,
            # Bezos fast-follow (non-blocking, evaluator-converged): a WARNING
            # only, surfaced in the priced preview so the founder sees it
            # before confirming — never a gate, never new slicing (CEO-ruled
            # out of scope; see highlight_discovery.is_long_window_risk).
            "long_window_risk": hd.is_long_window_risk(window.duration_s),
            "cap": body.cap,
            "model": body.model,
            "thinking_level": body.thinking_level,
            "fps": body.fps,
            "frame_quality": body.frame_quality,
            "refine": body.refine,
            "refine_pad_before_s": body.refine_pad_before_s,
            "refine_pad_after_s": body.refine_pad_after_s,
            "refine_fps": body.refine_fps,
            "within_cap": body.cap >= 1,
        }

        if not body.refine:
            # Byte-for-byte the pre-refinement estimate shape (frozen
            # contract §D: "OFF reproduces today BYTE-FOR-BYTE").
            response.update({
                "est_calls": 1,
                "est_dollars_with_margin": round(discovery_dollars_realistic * MARGIN_MULTIPLIER, 2),
                "est_dollars_realistic": round(discovery_dollars_realistic, 2),
                "message": "Refinement OFF: 1 discovery call only.",
            })
            return response

        # refine=True: cost-anchored RANGE (frozen contract §E), never a
        # corpus-average headline number.
        refine_window_duration_estimate = body.refine_pad_before_s + body.refine_pad_after_s
        refine_dollars_per_call = estimate_highlight_call_cost_usd(
            refine_window_duration_estimate, body.refine_fps, hd.REFINE_FRAME_QUALITY,
        )
        est_calls_low = 1
        est_calls_high = max(1, body.cap)
        est_dollars_low_realistic = discovery_dollars_realistic
        est_dollars_high_realistic = discovery_dollars_realistic + max(0, body.cap - 1) * refine_dollars_per_call

        typical_highlight_count = round(hb.TARGET_SPANS_PER_MATCH)
        est_calls_typical = 1 + typical_highlight_count
        est_dollars_typical_realistic = discovery_dollars_realistic + typical_highlight_count * refine_dollars_per_call

        response.update({
            "est_calls_low": est_calls_low,
            "est_calls_high": est_calls_high,
            "est_dollars_low_realistic": round(est_dollars_low_realistic, 2),
            "est_dollars_high_realistic": round(est_dollars_high_realistic, 2),
            "est_dollars_low_with_margin": round(est_dollars_low_realistic * MARGIN_MULTIPLIER, 2),
            "est_dollars_high_with_margin": round(est_dollars_high_realistic * MARGIN_MULTIPLIER, 2),
            "est_calls_typical": est_calls_typical,
            "est_dollars_typical_realistic": round(est_dollars_typical_realistic, 2),
            "message": (
                "Refinement ON: 1 discovery call + 1 call per highlight found, capped at your budget."
            ),
        })
        return response

    @app.post("/highlight/run")
    async def highlight_run(body: HighlightRunRequest):
        if not config.gemini_api_key:
            raise HTTPException(400, "GEMINI_API_KEY is not configured server-side — cannot run.")
        if not body.confirm:
            raise HTTPException(400, "confirm=true is required — this endpoint spends real Gemini quota.")
        _validate_highlight_request(body)
        if app.state.active_run_state is not None:
            raise HTTPException(409, "A run is already in progress. Stop it (POST /stop) before starting another.")

        expected_token = compute_highlight_plan_token(
            body.match_id, body.cap, body.model, body.thinking_level, body.system_prompt, body.followup_prompt,
            body.fps, body.frame_quality, body.refine, body.refine_pad_before_s, body.refine_pad_after_s,
            body.refine_fps,
        )
        if expected_token != body.plan_token:
            raise HTTPException(
                400,
                "Stale or invalid preview: plan_token does not match the submitted match_id/cap/model/"
                "thinking_level/fps/frame_quality/refine settings/prompts. Re-run POST /highlight/plan "
                "for the EXACT inputs you want to run, then confirm again.",
            )

        target, window = _resolve_highlight_target_and_window(body.match_id)

        state = BatchRunState(cap=body.cap)
        app.state.active_run_state = state

        async def event_stream():
            try:
                async for event in hd.run_highlight_batch_with_global_cap(
                    [(target, window)],
                    gemini_api_key=config.gemini_api_key,
                    request_timeout_ms=config.gemini_request_timeout_ms,
                    model=body.model, thinking_level=body.thinking_level,
                    system_prompt=body.system_prompt, followup_prompt=body.followup_prompt,
                    fps=body.fps, frame_quality=body.frame_quality,
                    refine=body.refine, refine_pad_before_s=body.refine_pad_before_s,
                    refine_pad_after_s=body.refine_pad_after_s, refine_fps=body.refine_fps,
                    state=state,
                ):
                    # Persist whenever a match's result is available —
                    # ``batch_complete`` (the normal end) AND ``cap_reached``/
                    # ``run_stopped`` (which now ALSO carry ``results`` for
                    # any match that finished before the cap/stop fired —
                    # frozen contract §C: "emit the PARTIAL result", never
                    # silently drop it because the run didn't reach the end).
                    if "results" in event:
                        _persist_highlight_results(
                            output_dir, event.get("results", {}), model=body.model,
                            thinking_level=body.thinking_level, system_prompt=body.system_prompt,
                            followup_prompt=body.followup_prompt, window=window,
                            fps=body.fps, frame_quality=body.frame_quality,
                            refine=body.refine, refine_pad_before_s=body.refine_pad_before_s,
                            refine_pad_after_s=body.refine_pad_after_s, refine_fps=body.refine_fps,
                        )
                    yield json.dumps(event) + "\n"
            except Exception as e:  # noqa: BLE001 — surface a fatal run-loop error as a final NDJSON line
                logger.error("dashboard /highlight/run: run loop crashed: %s", e, exc_info=True)
                yield json.dumps({"type": "error", "stage_id": "highlight_run", "message": str(e)}) + "\n"
            finally:
                app.state.active_run_state = None

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.post("/highlight/score")
    async def highlight_score(body: HighlightScoreRequest):
        """Scores the LAST persisted ``POST /highlight/run`` result for
        ``match_id`` against the provenanced highlight-GT artifact, plus the
        two MANDATORY baselines (frozen contract §C) — every path through
        ``matcher.match_all``/``metrics.summarize`` is UNCHANGED, only
        ``pred_by_match`` differs (real Gemini output vs. each baseline
        generator)."""
        highlights_path = output_dir / "highlights" / f"{body.match_id}.json"
        if not highlights_path.exists():
            raise HTTPException(
                404,
                f"No persisted highlight-discovery result for match_id {body.match_id!r} — "
                "run POST /highlight/run first.",
            )
        persisted = json.loads(highlights_path.read_text())

        if body.match_id not in highlight_gold_by_match:
            raise HTTPException(400, f"Unknown match_id {body.match_id!r} (not in the highlight-gold artifact).")
        gold_by_match = {body.match_id: highlight_gold_by_match[body.match_id]}
        # Frozen contract §G (LeCun): score the EFFECTIVE timestamp --
        # refined_* when refined=true, else original_* -- never the raw
        # enriched dict shape directly (matcher.py only understands flat
        # {start_s, end_s}). Order-preserving, so render_timeline_marks's
        # pred_marks[i] still cross-references enriched_highlights[i].
        enriched_highlights = persisted["highlights"]
        # Defense-in-depth (requirement §2): a rejected highlight is never
        # scoreable and never reaches the timeline -- filter BEFORE calling
        # effective_pred_events so the returned enriched_highlights stays
        # index-aligned with timeline.pred_marks (both derived from the SAME
        # filtered list). hs.effective_pred_events would also independently
        # skip a rejected entry if one somehow survived this filter -- this
        # filter's job is keeping the TWO lists' indices in lockstep, not
        # re-implementing the scoring-side rejection check.
        scoreable_highlights = [h for h in enriched_highlights if not h.get("rejected")]
        pred_by_match_raw = {body.match_id: hs.effective_pred_events(scoreable_highlights)}

        real = hs.score_highlights(gold_by_match, pred_by_match_raw, body.tau_t)
        timeline = hs.render_timeline_marks(
            body.match_id,
            real["gold_by_match"][body.match_id], real["pred_by_match"][body.match_id],
            real["outcomes"][body.match_id],
        )

        window_start_s, window_end_s = persisted["window"]
        match_windows = {body.match_id: (window_start_s, window_end_s)}
        baseline_whole = hs.score_highlights(gold_by_match, hb.whole_match_baseline(match_windows), body.tau_t)
        baseline_uniform = hs.score_highlights(gold_by_match, hb.uniform_span_baseline(match_windows), body.tau_t)

        return {
            "match_id": body.match_id,
            "tau_t": body.tau_t,
            "predicate_version": highlight_gold_manifest.get("predicate_version"),
            "include_takedowns": highlight_gold_manifest.get("include_takedowns"),
            "real": real["summary"],
            "timeline": timeline,
            # Scoreable enriched per-highlight dicts (frozen contract §F),
            # order-preserving with timeline.pred_marks -- the client
            # cross-references pred_marks[i] <-> enriched_highlights[i] by
            # index to show reason/refined/original-vs-corrected (§H), no
            # separate lookup needed. Rejected rows are filtered above.
            "enriched_highlights": scoreable_highlights,
            "baseline_whole_match": baseline_whole["summary"],
            "baseline_uniform_span": baseline_uniform["summary"],
            "window": persisted["window"],
            "model": persisted.get("model"),
            "thinking_level": persisted.get("thinking_level"),
            "fps": persisted.get("fps"),
            "frame_quality": persisted.get("frame_quality"),
            "refine": persisted.get("refine"),
            "refine_pad_before_s": persisted.get("refine_pad_before_s"),
            "refine_pad_after_s": persisted.get("refine_pad_after_s"),
            "refine_fps": persisted.get("refine_fps"),
            "system_prompt": persisted.get("system_prompt"),
            "followup_prompt": persisted.get("followup_prompt"),
        }

    @app.get("/highlight/media/{match_id}")
    async def highlight_media(match_id: str):
        """Item 2 (AC2-4..AC2-8) — the ONLY route that ever reads bytes from
        the ADCC media root. Accepts ONLY an opaque match_id; never a path,
        filename, or URL. Range requests are handled entirely by Starlette's
        native FileResponse support (206/416, Accept-Ranges: bytes) — this
        handler adds no Range logic of its own.

        Client-facing error bodies are path-free generics; real ffmpeg/ffprobe
        detail is logged server-side only (never echoed via ``str(exc)``).
        """
        try:
            resolved = media_index.resolve(match_id)
        except UnsupportedExtensionError as e:
            raise HTTPException(415, _MEDIA_UNSUPPORTED_MSG) from e
        except MediaResolutionError as e:
            raise HTTPException(404, _MEDIA_UNKNOWN_MSG) from e
        try:
            cached_path = await derivative_cache.ensure_derivative(resolved)
        except MediaBuildError as e:
            logger.error(
                "highlight_media: derivative build failed for match_id=%s: %s",
                match_id,
                e,
            )
            raise HTTPException(500, _MEDIA_BUILD_FAILED_MSG) from e
        return FileResponse(cached_path, media_type="video/mp4")

    @app.post("/stop")
    async def stop():
        state = app.state.active_run_state
        if state is None:
            raise HTTPException(400, "No active run to stop.")
        state.stop_requested = True
        state.stop_reason = "operator requested stop"
        return {"status": "stopping", "calls_spent_so_far": state.calls_spent}

    return app


def _persist_batch_results(output_dir: Path, results: dict) -> None:
    for match_id, result in results.items():
        out_path = output_dir / f"{match_id}.json"
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _persist_highlight_results(
    output_dir: Path, results: dict, *, model: str, thinking_level, system_prompt: str,
    followup_prompt: str, window, fps: float, frame_quality: str,
    refine: bool = True, refine_pad_before_s: float = None, refine_pad_after_s: float = None,
    refine_fps: float = None,
) -> None:
    """Persists a completed (or PARTIAL, cap/stop-truncated) ``/highlight/
    run``'s output to ``<output_dir>/highlights/<match_id>.json`` —
    ``/highlight/score`` reads THIS file (never re-runs Gemini, never
    re-derives the window) so scoring is repeatable against whatever the
    last real run actually produced. ``fps``/``frame_quality``/``refine``/
    the ``refine_*`` knobs (LeCun: "scores only comparable at identical
    settings") are provenance fields — a legacy file written before a given
    increment simply lacks the newer keys, and ``/highlight/score`` reads
    them back with ``.get(...)`` (``None`` -> rendered as "unknown/legacy"
    by the client), never a crash on an older persisted result.

    ``result["highlights"]`` is, as of the timestamp critique/refinement
    increment, the ENRICHED per-highlight dict shape (frozen contract §F):
    ``{original_start_s, original_end_s, refined_start_s, refined_end_s,
    refined, reason, action_confirmed, refinement_failed}`` — this function
    persists it verbatim, it does not itself know or care whether refinement
    ran; that decision already happened in ``run_highlight_batch_with_
    global_cap``.

    Also persists ``raw_response_text`` and ``raw_highlights`` (the RAW,
    UNMODIFIED clip-relative ``{start_s,end_s,reason}`` Gemini actually
    returned, before ``window.start_s`` normalization) plus
    ``out_of_bounds_count``/``out_of_bounds_spans`` (``discover_match_
    highlights``'s detection — never dropped/clamped here, that policy call
    is LeCun's) — makes the NEXT real run definitively diagnosable
    (clip-relative drift vs. a harness normalization bug) with zero extra
    Gemini spend."""
    highlights_dir = output_dir / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)
    for match_id, result in results.items():
        out_path = highlights_dir / f"{match_id}.json"
        payload = {
            "match_id": match_id,
            "highlights": result["highlights"],
            "raw_highlights": result.get("raw_highlights"),
            "out_of_bounds_count": result.get("out_of_bounds_count"),
            "out_of_bounds_spans": result.get("out_of_bounds_spans"),
            "raw_response_text": result.get("raw_response_text"),
            "model": model,
            "thinking_level": thinking_level,
            "fps": fps,
            "frame_quality": frame_quality,
            "refine": refine,
            "refine_pad_before_s": refine_pad_before_s,
            "refine_pad_after_s": refine_pad_after_s,
            "refine_fps": refine_fps,
            "system_prompt": system_prompt,
            "followup_prompt": followup_prompt,
            "window": [window.start_s, window.end_s],
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# Load `.env` from VLM_EVAL_HOME (server-side ONLY — the key must never reach
# the browser or a log line). Harmless no-op on a Mac dev checkout where that
# file doesn't exist.
load_dotenv(VLM_EVAL_HOME / ".env")

app = create_app()
