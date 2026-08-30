"""Shared constants/utilities for the Gates 1 & 2 driver
(``GATES_1_2_DESIGN.md``, Yann LeCun, 2026-07-25 — the binding spec; every
constant/derivation below is traceable to a specific section of that doc).

Scripts in this module ONLY touch ``eval/adcc/s12_gates/`` outputs. They
read (never modify) ``eval/adcc/gold_snapshot/``, ``eval/adcc/score/``,
``eval/adcc/data_prep/apply_gracie_mapping.py``, ``eval/adcc/predict/
youtube_url.py``, and the production ``service/pipelines/{simplified_tags,
highlight_axes,executors,gemini_retry}.py`` (read-only reuse of
``SimplifiedTagsTimeAnalyzer``/``_call_axis_json``/the axis prompt+schema
builders — never a reimplementation of those, per the design doc's own
"Summary for the engineer").

No fallback logic: every function here either returns a real value derived
from the checked-in gold artifact / a real Gemini call, or raises. Corpus
drift (the frozen frame counts in ``GATES_1_2_DESIGN.md`` §A.2/§A.3 no
longer matching the live gold snapshot) is a hard error, not a silently
re-derived subset — the whole point of a FROZEN manifest is that its
membership is reproducible from the seed alone, and a silent re-derivation
on drift would defeat that.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from eval.adcc.predict.youtube_url import normalize_youtube_watch_url
from service.config import ServiceConfig
from service.pipelines import gemini_retry, highlight_axes, simplified_tags
from service.pipelines.executors import _call_axis_json
from service.routes.qa_vlm import thinking_config_for

logger = logging.getLogger("eval.adcc.s12_gates.gate_common")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
S12_GATES_DIR = Path(__file__).resolve().parent
GOLD_SNAPSHOT_DIR = S12_GATES_DIR.parent / "gold_snapshot"
GOLD_PATH = GOLD_SNAPSHOT_DIR / "highlight_gold.jsonl"

GATE1_MANIFEST_PATH = S12_GATES_DIR / "gate1_subset_manifest.json"
GATE2_MANIFEST_PATH = S12_GATES_DIR / "gate2_subset_manifest.json"
GATE1_ARM_S_PATH = S12_GATES_DIR / "gate1_predictions_arm_s.json"
GATE1_ARM_T_PATH = S12_GATES_DIR / "gate1_predictions_arm_t.json"
GATE2_PREDICTIONS_PATH = S12_GATES_DIR / "gate2_predictions.json"
GATE1_SCORING_PATH = S12_GATES_DIR / "gate1_scoring.json"
GATE2_SCORING_PATH = S12_GATES_DIR / "gate2_scoring.json"
GATE_BUDGET_LEDGER_PATH = S12_GATES_DIR / "gate_budget_ledger.json"
GATE1_REPORT_PATH = S12_GATES_DIR / "GATE1_REPORT.md"
GATE2_REPORT_PATH = S12_GATES_DIR / "GATE2_REPORT.md"
GATE5_ADDENDUM_PATH = S12_GATES_DIR / "GATE5_ADDENDUM.md"

# --------------------------------------------------------------------------- #
# [RE-RUN 2026-07-26, GATES_RERUN_DESIGN.md] Artifact versioning — the
# qualified-model (gemini-3.5-flash, thinking=high) re-run MUST NOT
# overwrite the flash-lite v1 canonicals (they are their own data point,
# per VERDICTS.md's SUPERSEDED-BY-SETTINGS banner, never a deleted one).
# Every driver/scorer that participates in the re-run routes its
# predictions/scoring/report I/O through `versioned_path()` instead of the
# bare canonical constants above. The subset MANIFESTS (GATE1_MANIFEST_PATH/
# GATE2_MANIFEST_PATH) and the LEDGER (GATE_BUDGET_LEDGER_PATH) are
# deliberately NEVER versioned: manifests are re-frozen NEVER (design §1,
# reused byte-for-byte), and the ledger is one single, append-only,
# cumulative audit log across every run this whole task has ever made —
# each entry already self-identifies its own `model_id`/`thinking_level`,
# so splitting it would only make cross-run auditing harder, not safer.
# --------------------------------------------------------------------------- #
RERUN_LABEL: str = "gemini35flash_highthink"


def versioned_path(canonical_path: Path, label: str = RERUN_LABEL) -> Path:
    """Returns ``canonical_path`` with ``.{label}`` inserted before the
    final suffix — e.g. ``gate1_predictions_arm_s.json`` ->
    ``gate1_predictions_arm_s.gemini35flash_highthink.json``. NEVER used
    for `GATE1_MANIFEST_PATH`/`GATE2_MANIFEST_PATH`/`GATE_BUDGET_LEDGER_PATH`
    (see module-level note above)."""
    return canonical_path.with_name(f"{canonical_path.stem}.{label}{canonical_path.suffix}")


# --------------------------------------------------------------------------- #
# Settings tuple (design doc §D — frozen, identical across every call site
# unless the table explicitly varies it).
#
# **[RE-RUN 2026-07-26, GATES_RERUN_DESIGN.md §1/§2] MODEL CHANGE — founder
# ruling: `gemini-3.1-flash-lite` is NON-QUALIFIED for multimodal work on
# this project.** Minimum qualified model: `gemini-3.5-flash` @ max
# thinking. This is a config-only change (confirmed by engineer recon,
# design §3): `gemini-3.5-flash` is already in `qa_vlm.EVENT_MODELS`'s
# allowlist and is this codebase's own default-of-record for highlight
# discovery elsewhere — no new integration surface. The frozen subsets
# (manifests, Gate 3's `w17->w1` replacement, Gate 4's reference crops) are
# explicitly NOT re-derived (design §1 — they are model-independent
# functions of the gold corpus + a fixed seed). Every re-run artifact this
# model change produces MUST carry a rerun label and MUST NOT overwrite the
# flash-lite v1 canonicals (design's output contract) — v1 stands as its
# own data point, per `VERDICTS.md`'s SUPERSEDED-BY-SETTINGS banner, not a
# deleted one.
# --------------------------------------------------------------------------- #
MODEL_ID: str = "gemini-3.5-flash"
FPS: int = 10
PREROLL_S: float = 5.0
POSTROLL_S: float = 4.0
TAU_T: float = 5.0
SEED: int = 20260726  # date-derived, matches metrics.DEFAULT_BOOTSTRAP_SEED's convention (§A.2)

# --------------------------------------------------------------------------- #
# §A.2 Step 2 — the 8 forced Gate-5 straddlers, verbatim from GATE5_REPORT.md
# ("Straddling events — production anchoring" W=60 table). (match_id,
# action_class, t_s).
# --------------------------------------------------------------------------- #
GATE5_STRADDLERS: list[tuple[str, str, int]] = [
    ("6470211a3ec8dca9d1a32710", "submission_arm_lock", 727),
    ("650d20c745120c8433e9cfe0", "guard_pass", 774),
    ("650d20c745120c8433e9cff3", "back_take", 761),
    ("650d20c745120c8433e9d020", "guard_pass", 760),
    ("65de1567a2434704339dbc8c", "sweep", 729),
    ("65de222ea2434704339dbe9c", "back_take", 2235),
    ("65edf42aa2434704339f7e93", "back_take", 754),
    ("66b009481c62a1db7a53d504", "back_take", 30443),
]

# §A.2 Step 1 — frame composition AFTER removing the 8 straddlers, and the
# largest-remainder quota derived from it. Both frozen constants below are
# ASSERTED against the live gold snapshot at load time (never silently
# re-derived on drift — see build_gate1_subset).
GATE1_EXPECTED_FRAME_COUNTS: dict[str, int] = {
    "guard_pass": 193, "back_take": 133, "sweep": 76,
    "submission_choke": 67, "submission_leg_lock": 52, "submission_arm_lock": 23,
}
GATE1_QUOTAS: dict[str, int] = {
    "guard_pass": 11, "back_take": 7, "sweep": 4,
    "submission_choke": 4, "submission_leg_lock": 3, "submission_arm_lock": 1,
}
assert sum(GATE1_QUOTAS.values()) == 30
assert sum(GATE1_EXPECTED_FRAME_COUNTS.values()) == 544

# §A.3 — Gate 2 forced "unsuccessful" occlusion events, verbatim from the
# design doc's own corpus scan (re-verified against the live snapshot in
# build_gate2_subset).
GATE2_FORCED_UNSUCCESSFUL: dict[str, list[tuple[str, int]]] = {
    "submission_arm_lock": [("66afe5ab1c62a1db7a53b80b", 28325)],
    "submission_leg_lock": [
        ("65e3a69da2434704339e4460", 333),
        ("65f648c51dcf397e6900a827", 241),
        ("6643b593532ae8f232692d70", 2647),
        ("66a6ff021c62a1db7a5133c0", 12775),
        ("66d4e7091c7649ed30cbef2b", 164),
    ],
}
GATE2_QUOTA_PER_CLASS: int = 8
GATE2_CLASS_ORDER: tuple[str, str] = ("submission_arm_lock", "submission_leg_lock")  # §A.3: arm_lock sampled first

# §D.1 halt-and-escalate thresholds. ">3 of 38" for Gate 1 is unambiguous
# (10% of 38 = 3.8 -> trigger at 4+). Gate 2's stated ">1 of 16" (10% of 16 =
# 1.6 -> would literally trigger at 2+) directly conflicts with the SAME
# sentence's own clarifying restatement "i.e. any Gate-2 failure at all
# triggers this" (trigger at 1+). This is a genuine internal inconsistency
# in the design doc (flagged in the engineer's handoff report, not silently
# resolved) — the more conservative reading (halt sooner on a safety
# threshold, and the more specific/most-recently-stated clarification wins)
# is used here: ANY Gate-2 call_failed clip halts.
GATE1_SUBSET_N: int = 38
GATE2_SUBSET_N: int = 16
GATE1_FAILURE_HALT_THRESHOLD: int = 3  # trigger when failed_window_count > 3
GATE2_FAILURE_HALT_THRESHOLD: int = 0  # trigger when failed_clip_count > 0 (i.e. any failure)


class GateHaltError(RuntimeError):
    """Raised when §D.1's halt-and-escalate threshold is tripped on a FULL
    (non-smoke) run. Never caught/suppressed by the driver — this is meant
    to stop the script and surface to the human operator."""


# --------------------------------------------------------------------------- #
# Gold loading
# --------------------------------------------------------------------------- #
def load_hard_track_events(gold_path: Path = GOLD_PATH) -> list[dict]:
    """Flattens ``highlight_gold.jsonl`` (one row per MATCH, with a nested
    ``events`` list) to one dict per EVENT, restricted to ``track=="hard"``
    (design §A.1). Each event dict carries the match-level ``match_id``/
    ``video_url``/``video_url_md5`` folded in."""
    events: list[dict] = []
    with open(gold_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for e in row["events"]:
                if e.get("track") != "hard":
                    continue
                ev = dict(e)
                ev["match_id"] = row["match_id"]
                ev["video_url"] = row["video_url"]
                ev["video_url_md5"] = row.get("video_url_md5")
                events.append(ev)
    return events


def _event_key(e: dict) -> tuple[str, int]:
    """``(match_id, t_s)`` — the stable sort/identity key used throughout
    §A.2/§A.3 for deterministic ordering and straddler/forced-event lookup."""
    return (e["match_id"], int(e["t_s"]))


# --------------------------------------------------------------------------- #
# §A.4 window cut — IDENTICAL for every arm/condition of a given gold event.
# --------------------------------------------------------------------------- #
def compute_window(
    t_s: float, *, preroll_s: float = PREROLL_S, postroll_s: float = POSTROLL_S,
    video_duration_s: Optional[float] = None,
) -> tuple[float, float, Optional[str]]:
    """Returns ``(window_start_s, window_end_s, clamp_warning)``. Millisecond-
    quantized (matches ``executors.highlight_analyze_node``'s own rounding
    discipline). ``clamp_warning`` is ``None`` unless a real
    ``video_duration_s`` lookup was available AND the un-clamped window_end
    would exceed it — in which case the window is clamped and the warning is
    non-None (never silently clamped, per §A.4's own "log a warning — never
    silently skip")."""
    window_start = max(0.0, t_s - preroll_s)
    window_end = t_s + postroll_s
    warning: Optional[str] = None
    if video_duration_s is not None and window_end > video_duration_s:
        warning = (
            f"window_end {window_end:.3f}s exceeds known video_duration_s {video_duration_s:.3f}s "
            f"(t_s={t_s}); clamping window_end to video_duration_s."
        )
        logger.warning("compute_window: %s", warning)
        window_end = video_duration_s
    return round(window_start, 3), round(window_end, 3), warning


_VIDEO_DURATIONS_CACHE: Optional[dict[str, float]] = None


def load_video_durations() -> dict[str, float]:
    """Best-effort cached-duration lookup (design §A.4: "the driver should
    still defensively clamp ... if a duration lookup is available via
    eval/adcc/predict/generate_video_durations.py's cached durations").

    That cache (``~/vlm-eval/video_durations.json``, keyed by
    ``video_url_md5``) is generated SERVER-SIDE on bjj-server, where the
    on-disk ADCC video corpus lives — this Mac worktree has no local copy.
    Returns ``{}`` (never fabricated) when absent; ``compute_window``'s clamp
    is then a no-op for every window, which the design doc explicitly
    allows ("no clamp needed... none of the selected t_s are within 4s of a
    video's known end").
    """
    global _VIDEO_DURATIONS_CACHE
    if _VIDEO_DURATIONS_CACHE is not None:
        return _VIDEO_DURATIONS_CACHE
    candidate = Path.home() / "vlm-eval" / "video_durations.json"
    if candidate.exists():
        _VIDEO_DURATIONS_CACHE = json.loads(candidate.read_text())
        logger.info("load_video_durations: loaded %d entries from %s", len(_VIDEO_DURATIONS_CACHE), candidate)
    else:
        logger.warning(
            "load_video_durations: no local cache at %s — window-end clamp is a no-op for this run "
            "(design doc §A.4 explicitly allows this for the current corpus).", candidate,
        )
        _VIDEO_DURATIONS_CACHE = {}
    return _VIDEO_DURATIONS_CACHE


# --------------------------------------------------------------------------- #
# sha256 helpers (manifest provenance, F.1)
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json_obj(obj) -> str:
    """Canonical (sorted-keys, no whitespace-sensitivity) sha256 of a JSON-
    serializable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# §A.2 Gate 1 subset construction
# --------------------------------------------------------------------------- #
def largest_remainder_quotas(frame_counts: dict[str, int], total_quota: int) -> dict[str, int]:
    """Standard largest-remainder proportional allocation (§A.2 Step 1's
    literal derivation method: ``count/total*quota``, floored, remainder
    slots assigned to the largest fractional remainders). Used here as a
    SELF-CHECK against the frozen ``GATE1_QUOTAS`` table, not as the live
    source of truth (the design doc froze the table; this function proves
    the table is reproducible from the frame counts, it does not replace
    the table)."""
    total_frame = sum(frame_counts.values())
    exact = {cls: n * total_quota / total_frame for cls, n in frame_counts.items()}
    floors = {cls: int(v) for cls, v in exact.items()}
    remainder = total_quota - sum(floors.values())
    order = sorted(frame_counts, key=lambda c: (exact[c] - floors[c]), reverse=True)
    for cls in order[:remainder]:
        floors[cls] += 1
    return floors


def build_gate1_subset(events: list[dict]) -> list[dict]:
    """§A.2 — 30 stratified + 8 forced straddlers = 38 windows, in the
    manifest's fixed enumeration order (30 stratified, strata visited in
    ascending alphabetical ``action_class`` order, THEN the 8 straddlers in
    the design doc's table order). Every returned dict is a COPY of the
    matching gold event dict plus a ``gate5_straddler`` bool. Raises on any
    corpus drift from the design doc's frozen frame-count/straddler tables —
    never silently re-derives a different subset."""
    straddler_keys = {(mid, t) for mid, _cls, t in GATE5_STRADDLERS}
    frame = [e for e in events if _event_key(e) not in straddler_keys]

    strata: dict[str, list[dict]] = {}
    for e in frame:
        strata.setdefault(e["action_class"], []).append(e)
    for cls in strata:
        strata[cls].sort(key=_event_key)

    live_frame_counts = {cls: len(strata.get(cls, [])) for cls in GATE1_EXPECTED_FRAME_COUNTS}
    if live_frame_counts != GATE1_EXPECTED_FRAME_COUNTS:
        raise ValueError(
            f"gate1 subset build: live post-straddler-removal frame counts {live_frame_counts} "
            f"disagree with GATES_1_2_DESIGN.md §A.2's frozen table {GATE1_EXPECTED_FRAME_COUNTS} "
            "— gold corpus has drifted since the design was frozen; the seeded draw sequence is no "
            "longer reproducible against the documented derivation. Refusing to silently re-derive."
        )
    derived_quotas = largest_remainder_quotas(GATE1_EXPECTED_FRAME_COUNTS, 30)
    if derived_quotas != GATE1_QUOTAS:
        raise ValueError(
            f"gate1 subset build: largest-remainder derivation {derived_quotas} disagrees with the "
            f"frozen GATE1_QUOTAS table {GATE1_QUOTAS} — design doc self-check failed."
        )

    # ONE freshly-seeded Random(20260726) instance for this ENTIRE selection
    # step ("Gate-1 stratified core" — Bezos condition 5's naming), consumed
    # sequentially across all 6 strata in ascending action_class order. This
    # is a single named step (unlike Gate 2's two independent per-class
    # steps below) — matches both the design doc's original §A.2 text and
    # condition 5's explicit step list.
    rng = random.Random(SEED)
    selected: list[dict] = []
    for cls in sorted(GATE1_QUOTAS):  # ascending alphabetical action_class order (§A.2)
        k = GATE1_QUOTAS[cls]
        pool = strata[cls]
        if len(pool) < k:
            raise ValueError(f"gate1 subset build: stratum {cls!r} has only {len(pool)} candidates, need {k}")
        drawn = rng.sample(pool, k)
        for e in drawn:
            row = dict(e)
            row["gate5_straddler"] = False
            selected.append(row)

    straddler_lookup = {_event_key(e): e for e in events}
    for mid, cls, t in GATE5_STRADDLERS:
        e = straddler_lookup.get((mid, t))
        if e is None:
            raise ValueError(f"gate1 subset build: straddler {mid}/{t} not found in the live hard-track gold")
        if e["action_class"] != cls:
            raise ValueError(
                f"gate1 subset build: straddler {mid}/{t} action_class drifted "
                f"(design doc says {cls!r}, live gold says {e['action_class']!r})"
            )
        row = dict(e)
        row["gate5_straddler"] = True
        selected.append(row)

    if len(selected) != GATE1_SUBSET_N:
        raise ValueError(f"gate1 subset build: expected {GATE1_SUBSET_N} windows, got {len(selected)}")
    return selected


def build_gate2_subset(events: list[dict]) -> list[dict]:
    """§A.3 — 16 occlusion clips (8 arm_lock + 8 leg_lock).

    **RNG procedure — Bezos condition 5 (founder-review pass-with-conditions,
    2026-07-25) SUPERSEDES the design doc's original §A.3 text here.** The
    design doc's literal wording ("both using fresh .sample() calls on the
    ONE seeded Random instance") was overridden before any manifest was
    frozen: each class pool gets its OWN independent,
    freshly-seeded ``random.Random(20260726)`` instance (arm_lock's sample
    draw does not consume/advance leg_lock's RNG state, and vice versa) —
    three named selection steps total across both gates (Gate-1 stratified
    core; Gate-2 arm_lock pool; Gate-2 leg_lock pool), each with its own
    documented draw order. This changes leg_lock's realized 3-of-47 sample
    relative to the design doc's original one-shared-instance wording (which
    was never executed — no manifest existed before this correction);
    arm_lock's 7-of-23 sample is unaffected since it is the first consumption
    in both procedures. Raises on corpus drift from the frozen forced-event
    table."""
    lookup = {_event_key(e): e for e in events}
    selected: list[dict] = []
    for cls in GATE2_CLASS_ORDER:
        rng = random.Random(SEED)  # fresh instance PER CLASS POOL (Bezos condition 5)
        forced_keys = GATE2_FORCED_UNSUCCESSFUL[cls]
        forced_events: list[dict] = []
        for mid, t in forced_keys:
            e = lookup.get((mid, t))
            if e is None:
                raise ValueError(f"gate2 subset build: forced event {mid}/{t} not found in live hard-track gold")
            if e["action_class"] != cls or e["outcome"] != "unsuccessful":
                raise ValueError(
                    f"gate2 subset build: forced event {mid}/{t} drifted — expected "
                    f"action_class={cls!r}/outcome='unsuccessful', got "
                    f"action_class={e['action_class']!r}/outcome={e['outcome']!r}"
                )
            forced_events.append(e)

        successful_pool = sorted(
            (e for e in events if e["action_class"] == cls and e["outcome"] == "successful"),
            key=_event_key,
        )
        need = GATE2_QUOTA_PER_CLASS - len(forced_events)
        if need < 0 or len(successful_pool) < need:
            raise ValueError(
                f"gate2 subset build: class {cls!r} needs {need} successful draws from a pool of "
                f"{len(successful_pool)} — quota/forced-count mismatch"
            )
        drawn = rng.sample(successful_pool, need) if need > 0 else []
        for e in forced_events + drawn:
            selected.append(dict(e))

    if len(selected) != GATE2_SUBSET_N:
        raise ValueError(f"gate2 subset build: expected {GATE2_SUBSET_N} clips, got {len(selected)}")
    return selected


# --------------------------------------------------------------------------- #
# §D.1/§G.2 replacement-window drawing (amended 2026-07-26 — the original
# "next item in the seeded sequence" phrasing was not implementable:
# random.Random(seed).sample(pool, k) and .sample(pool, k+1) do not share a
# common prefix, so there is no well-defined "next" item, and re-deriving via
# a full-pool permutation would silently disturb the ALREADY-FROZEN,
# already-sha256'd primary draw in gate1_subset_manifest.json/
# gate2_subset_manifest.json. The implementable rule (§G.2, LeCun ruling
# 2026-07-26): a replacement is a SEPARATE, freshly-seeded sub-draw over the
# stratum's remaining (not-yet-selected) pool, seed SEED_REPLACEMENT =
# 20260727 (SEED + 1, same date-derived convention). NEVER touches the
# original committed draw — callers pass `already_used_keys` (primary draw +
# any replacements already consumed this run) and this module only reads
# `events`/gold snapshot, never gate1_subset_manifest.json/
# gate2_subset_manifest.json themselves.
# --------------------------------------------------------------------------- #
SEED_REPLACEMENT: int = 20260727


def draw_replacement(remaining_pool: list[dict]) -> dict:
    """ONE replacement draw against the CURRENT remaining pool, via a
    FRESHLY re-instantiated ``random.Random(SEED_REPLACEMENT)`` every call
    (never a shared/advancing instance across multiple replacements — that
    is what makes each individual replacement draw well-defined and
    reproducible regardless of how many replacements a run ends up needing,
    per §G.2: "each time re-instantiating Random(20260727) fresh against
    the then-current (shrinking) remaining pool"). Raises (no fallback) if
    the pool is exhausted."""
    if not remaining_pool:
        raise ValueError(
            "gate_common.draw_replacement: remaining pool is empty — no candidates left to redraw from."
        )
    rng = random.Random(SEED_REPLACEMENT)
    return rng.sample(remaining_pool, 1)[0]


def draw_gate1_stratum_replacement(
    events: list[dict], action_class: str, already_used_keys: set[tuple[str, int]],
) -> dict:
    """Gate 1 §A.2 stratum replacement for a `call_failed`-excluded window:
    remaining = that `action_class`'s hard-track pool minus
    `already_used_keys` (the primary 30-draw + any replacements already
    consumed earlier in the SAME run), sorted `(match_id, t_s)` ascending —
    the same deterministic ordering discipline as the primary draw. NOT
    valid for a `gate5_straddler` window's failure — the 8 straddlers are a
    fixed enumeration, not a stratified sample (§G.1: dropping to n=7 is
    the accepted fallback there, not a redraw)."""
    pool = sorted(
        (e for e in events if e["action_class"] == action_class and _event_key(e) not in already_used_keys),
        key=_event_key,
    )
    return draw_replacement(pool)


def draw_gate2_class_replacement(
    events: list[dict], action_class: str, already_used_keys: set[tuple[str, int]],
) -> tuple[dict, str]:
    """Gate 2 §A.3 forced-slot replacement: try the class's remaining
    `unsuccessful` sub-pool FIRST (preserving the oversampling intent that
    is the entire premise of Gate 2 — §A.3), falling back to the
    `successful` sub-pool ONLY if the `unsuccessful` sub-pool is exhausted.
    Returns `(event, pool_name)` so the caller can record which pool a
    given replacement came from in the manifest (§D.1's explicit
    requirement), never silently blending the two."""
    unsuccessful_pool = sorted(
        (e for e in events if e["action_class"] == action_class and e["outcome"] == "unsuccessful"
         and _event_key(e) not in already_used_keys),
        key=_event_key,
    )
    if unsuccessful_pool:
        return draw_replacement(unsuccessful_pool), "unsuccessful"
    successful_pool = sorted(
        (e for e in events if e["action_class"] == action_class and e["outcome"] == "successful"
         and _event_key(e) not in already_used_keys),
        key=_event_key,
    )
    return draw_replacement(successful_pool), "successful"


# --------------------------------------------------------------------------- #
# Manifest row -> per-window record shared fields (video url normalization,
# window cut) — used by both freeze_subsets.py (to build the manifest) and
# the run_gate*.py drivers (which re-derive window bounds identically from
# the frozen manifest row rather than trusting a possibly-stale persisted
# value, belt-and-suspenders against a manifest/driver drift bug).
# --------------------------------------------------------------------------- #
def resolve_window_for_event(event: dict, video_durations: dict[str, float]) -> tuple[str, float, float, Optional[str]]:
    """Returns ``(video_url_watch, window_start_s, window_end_s, clamp_warning)``
    for one gold event. ``video_url`` MUST be run through
    ``normalize_youtube_watch_url`` before any Gemini call (§A.4 — the exact
    live bug that module's docstring documents)."""
    video_url_watch = normalize_youtube_watch_url(event["video_url"])
    duration = video_durations.get(event.get("video_url_md5"))
    window_start_s, window_end_s, warning = compute_window(float(event["t_s"]), video_duration_s=duration)
    return video_url_watch, window_start_s, window_end_s, warning


# --------------------------------------------------------------------------- #
# Gemini client / analyzer construction (real key, real retry policy —
# read-only reuse of production plumbing, per the design doc's "Summary for
# the engineer": "Read SimplifiedTagsTimeAnalyzer and highlight_axes.py/
# _call_axis_json directly ... rather than reimplementing the schemas/
# prompts").
# --------------------------------------------------------------------------- #
def load_service_config() -> ServiceConfig:
    """``ServiceConfig`` already dual-accepts ``BJJ_GEMINI_API_KEY`` (prefixed)
    and plain ``GEMINI_API_KEY`` from ``.env``/the process env (see
    ``service/config.py``'s ``_alias_gemini_api_key``) — this worktree's
    ``.env`` sets ``BJJ_GEMINI_API_KEY`` only, which the prefixed field reads
    directly. No key-loading logic is reimplemented here; this is a thin,
    fail-loud wrapper that raises if the resolved key is empty (never sends
    an unauthenticated request to find that out)."""
    cfg = ServiceConfig()
    if not cfg.gemini_api_key:
        raise RuntimeError(
            "gate_common.load_service_config: ServiceConfig resolved an EMPTY gemini_api_key. "
            "Set BJJ_GEMINI_API_KEY (or GEMINI_API_KEY) in .env at the repo root, or in the "
            "process environment, before running any --execute gate driver."
        )
    return cfg


def build_gemini_client(cfg: ServiceConfig) -> genai.Client:
    return genai.Client(
        api_key=cfg.gemini_api_key,
        http_options=types.HttpOptions(timeout=cfg.gemini_request_timeout_ms),
    )


def build_retry_config(cfg: ServiceConfig) -> gemini_retry.GeminiRetryConfig:
    return gemini_retry.GeminiRetryConfig.from_service_config(cfg)


def build_time_analyzer(
    client: genai.Client, *, system_instruction: Optional[str],
    retry_config: gemini_retry.GeminiRetryConfig, model_id: str = MODEL_ID,
) -> "simplified_tags.SimplifiedTagsTimeAnalyzer":
    """One ``SimplifiedTagsTimeAnalyzer`` per axis/arm (never shared across
    axes with different system instructions — mirrors
    ``executors.highlight_analyze_node``'s own per-axis analyzer
    instantiation, built once outside the per-window loop).

    **[RE-RUN 2026-07-26, GATES_RERUN_DESIGN.md §2] ``thinking`` is now
    ``"high"`` (max thinking) for every call site this module drives** — the
    qualified-model re-run requires BOTH arms at max thinking (a fair paired
    A/B needs the same model+thinking on both sides; see design §2's
    explicit confound-avoidance ruling). ``thinking_config_for(model_id,
    "high")`` maps through the existing, already-tested
    ``ThinkingQualityMixin`` plumbing to ``types.ThinkingConfig(
    thinking_level="high")`` — no new code needed."""
    return simplified_tags.SimplifiedTagsTimeAnalyzer(
        client, model_id=model_id,
        thinking_config=thinking_config_for(model_id, "high"),
        system_instruction=system_instruction,
        retry_config=retry_config,
    )


def assert_thinking_high_in_generate_config(model_id: str = MODEL_ID) -> None:
    """[RE-RUN 2026-07-26, CEO instruction] Fail-loud verification that
    ``thinking_level="high"`` actually lands in the REAL
    ``types.GenerateContentConfig`` Gemini would receive — not just that
    ``thinking_config_for``'s mapping function returns the right value in
    isolation, but that it survives the full path through
    ``SimplifiedTagsTimeAnalyzer._build_generate_config``. Builds a
    throwaway, client-less analyzer (``_build_generate_config`` never
    touches ``self.client``) purely to inspect the constructed config.
    Called from every driver's dry-run path (so it's visible before any
    real spend) AND at the top of every real ``--execute`` run (so a
    silent regression can never reach a real call unnoticed). Raises
    ``AssertionError`` immediately on any mismatch — never proceeds on a
    misconfigured thinking level."""
    dummy = simplified_tags.SimplifiedTagsTimeAnalyzer(
        client=None, model_id=model_id, thinking_config=thinking_config_for(model_id, "high"),
        system_instruction="verification probe — never sent to Gemini", retry_config=gemini_retry.GeminiRetryConfig(),
    )
    cfg = dummy._build_generate_config(simplified_tags.build_video_response_schema())
    assert cfg.thinking_config is not None, (
        f"assert_thinking_high_in_generate_config: thinking_config is None for model={model_id!r} — "
        "the qualified-model re-run REQUIRES thinking=high on every call."
    )
    actual_level = cfg.thinking_config.thinking_level
    # The SDK returns a real types.ThinkingLevel enum (e.g. <ThinkingLevel.HIGH: 'HIGH'>),
    # not a bare lowercase string — compare against the enum member, never a naive string
    # equality that would falsely fail on a correctly-configured value (live bug caught
    # while writing this exact assertion: comparing == "high" against the enum failed even
    # though the level WAS correctly HIGH — fixed to compare the actual enum type).
    assert actual_level == types.ThinkingLevel.HIGH, (
        f"assert_thinking_high_in_generate_config: expected thinking_level=ThinkingLevel.HIGH for "
        f"model={model_id!r}, got {actual_level!r} instead."
    )
    print(f"[VERIFIED] model={model_id} -> GenerateContentConfig.thinking_config="
          f"ThinkingConfig(thinking_level={actual_level!r}) — confirmed end-to-end, not just the mapping function.",
          file=__import__("sys").stderr)


# --------------------------------------------------------------------------- #
# Calls ledger (F.1's gate_budget_ledger.json, enriched per the orchestrator
# handoff's explicit ask: "Log every Gemini call (model id, window, arm,
# latency, http status)" — a strict superset of F.1's own
# "op, window_id, arm, tokens, $, ok/call_failed" schema, not a second file).
# --------------------------------------------------------------------------- #
import re as _re

_HTTP_STATUS_RE = _re.compile(r"(\d{3}) [A-Z_]+\.")


def extract_http_status(error_message: Optional[str]) -> Optional[int]:
    """Best-effort HTTP status parse from a genai ``APIError``'s ``str()``
    (format confirmed by reading ``google.genai.errors.APIError.__init__``:
    ``f'{self.code} {self.status}. {self.details}'``, e.g. ``"400
    INVALID_ARGUMENT. {...}"``). ``_call_axis_json`` PREFIXES this with its
    own context (``"PASS-2 Gemini call failed: 400 INVALID_ARGUMENT. ..."``),
    so this searches ANYWHERE in the string, not just an anchored prefix —
    live bug fix, confirmed against a real 400 during the Gate 2 smoke test
    (the original ``^(\\d{3})\\s`` anchor never matched the prefixed message
    and always returned ``None``). Returns ``None`` for non-HTTP errors
    (timeouts, non-JSON response, connection errors) or when
    ``error_message`` is ``None`` — never fabricated, and NOT re-derived by
    re-raising/catching the original exception a second time (``
    _call_axis_json`` already swallows it into a string; this is a
    documented best-effort parse of that string, not a second retry
    layer)."""
    if not error_message:
        return None
    m = _HTTP_STATUS_RE.search(error_message)
    return int(m.group(1)) if m else None


@dataclass
class LedgerEntry:
    gate: str                       # "gate1" | "gate2"
    window_id: str
    match_id: str
    arm: str                        # "arm_s" | "arm_t" | "gate2"
    call_name: str                  # "arm_s" | "position" | "technique" | "validator" | "validator_low" |
                                     # "validator_high" | "files_api_upload" (§G.1 point 5 — one per rerouted window)
    model_id: str
    status: str                     # "ok" | "call_failed"
    error: Optional[str]
    http_status: Optional[int]
    latency_ms: float
    usage_metadata: Optional[dict]
    cost_usd: Optional[float]
    timestamp_utc: str
    ingestion_path: str = "youtube_uri"  # "youtube_uri" | "files_api" — §G.1 point 4, tag every call, never blend silently
    active_wait_ms: Optional[float] = None  # only set on call_name="files_api_upload" rows (§G.1 point 5)
    thinking_level: Optional[str] = None  # [RE-RUN] "high" (qualified model) | None (v1 flash-lite, off) — full settings-tuple audit trail
    provider_route: Optional[str] = None  # [PHASE B, §6.4] "direct_anthropic" | "openrouter" | None (Gemini-native calls — no route ambiguity) — recorded on EVERY frontier-lane call per the CEO's explicit "record provider_route on every call" instruction, auditable either way regardless of which route wins the §6.4 equivalence check

    def to_dict(self) -> dict:
        return {
            "gate": self.gate, "window_id": self.window_id, "match_id": self.match_id,
            "arm": self.arm, "call_name": self.call_name, "model_id": self.model_id,
            "status": self.status, "error": self.error, "http_status": self.http_status,
            "latency_ms": round(self.latency_ms, 1), "usage_metadata": self.usage_metadata,
            "cost_usd": self.cost_usd, "timestamp_utc": self.timestamp_utc,
            "ingestion_path": self.ingestion_path, "thinking_level": self.thinking_level,
            "active_wait_ms": None if self.active_wait_ms is None else round(self.active_wait_ms, 1),
            "provider_route": self.provider_route,
        }


class CallLedger:
    """Append-only in-memory ledger, flushed to ``gate_budget_ledger.json``
    (a single JSON array — read-modify-write, appending to whatever prior
    entries exist on disk, since Gate 1 and Gate 2 drivers both write to the
    SAME ledger file per F.1's file layout)."""

    def __init__(self, path: Path = GATE_BUDGET_LEDGER_PATH):
        self.path = path
        self.entries: list[dict] = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry.to_dict())

    def record_upload_event(
        self, *, gate: str, window_id: str, match_id: str, status: str, error: Optional[str],
        upload_latency_ms: float, active_wait_ms: Optional[float], reused: bool = False,
    ) -> None:
        """§G.1 point 5 — one ledger row per REROUTED window recording
        ``upload_latency_ms`` (the ``upload_video_to_gemini`` call's own
        return latency) and ``active_wait_ms`` (``poll_until_active``'s
        total elapsed PROCESSING->ACTIVE wait) — the exact numbers Gate 3
        also needs (GATES_3_4_DESIGN.md §H), captured once here, reused
        there. ``call_name="files_api_upload"`` (fresh download+normalize+
        upload+poll) vs ``"files_api_upload_reused"`` (a still-ACTIVE prior
        upload was found and reused — Bezos precondition 1, 2026-07-26:
        "reuse the existing cached upload if still ACTIVE... verify state,
        re-ensure if expired") distinguishes the two paths in the ledger
        without a schema change."""
        from datetime import datetime, timezone

        self.record(LedgerEntry(
            gate=gate, window_id=window_id, match_id=match_id, arm="reroute",
            call_name="files_api_upload_reused" if reused else "files_api_upload",
            model_id="n/a", status=status, error=error, http_status=None,
            latency_ms=upload_latency_ms, usage_metadata=None, cost_usd=None,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ingestion_path="files_api", active_wait_ms=active_wait_ms,
        ))

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2, sort_keys=False))

    def total_cost_usd(self) -> float:
        return sum(e["cost_usd"] for e in self.entries if e.get("cost_usd") is not None)

    def total_calls(self) -> int:
        return len(self.entries)


# --------------------------------------------------------------------------- #
# The one real-call wrapper every driver uses. Timing + ledger recording
# wrap _call_axis_json (never reimplemented) so instrumentation is uniform
# across arm_s/position/technique/validator/validator_low/validator_high.
# --------------------------------------------------------------------------- #
async def timed_axis_call(
    *, ledger: CallLedger, gate: str, window_id: str, match_id: str, arm: str, call_name: str,
    analyzer: "simplified_tags.SimplifiedTagsTimeAnalyzer", youtube_url: str,
    window_start: float, window_end: float, response_schema, prompt_text: str,
    fps: int, media_resolution: Optional[str], model_id: str = MODEL_ID,
    ingestion_path: str = "youtube_uri",
) -> tuple[Optional[dict], Optional[str], dict]:
    """Calls ``executors._call_axis_json`` (read-only reuse), times it,
    records one ``LedgerEntry``, and returns ``(data, error, call_record)``
    where ``call_record`` is the F.2 per-call schema fragment (``call_name``,
    ``status``, ``error``, ``raw_response_text``, ``parsed``, ``usage_metadata``,
    ``latency_ms``, ``ingestion_path``).

    ``youtube_url`` is, despite its name, whatever URI value the caller wants
    ``types.FileData(file_uri=...)`` to carry — a YouTube ``watch?v=`` URL
    (``ingestion_path="youtube_uri"``, the default) OR a Gemini Files API
    uploaded file's ``.uri`` (``ingestion_path="files_api"``, §G.1's reroute
    lane). ``SimplifiedTagsTimeAnalyzer.analyze_chunk``/``_call_axis_json``
    are agnostic to which — no code path below this function branches on
    ``ingestion_path`` at all, it is pure tagging/disclosure (§G.1 point 4:
    "tag and disclose, never blend silently")."""
    from datetime import datetime, timezone

    t0 = time.perf_counter()
    data, err = await _call_axis_json(
        analyzer, youtube_url, window_start, window_end,
        response_schema=response_schema, prompt_text=prompt_text,
        fps=fps, media_resolution=media_resolution,
        # [RE-RUN 2026-07-26] thinking="high" — inert for the Gemini path (this
        # kwarg only feeds _call_axis_json's is_claude_model branch; the REAL
        # thinking_level for Gemini calls comes from the analyzer's own
        # thinking_config, set in build_time_analyzer/FilesApiTimeAnalyzer
        # above), kept consistent here per the design's explicit instruction
        # and for forward-compat if this path is ever reused for a Claude call.
        ctx=None, model_id=model_id, system_instruction=None, thinking="high",
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    usage = getattr(analyzer, "last_usage_metadata", None)
    status = "ok" if err is None else "call_failed"
    http_status = extract_http_status(err)

    cost_usd: Optional[float] = None
    if usage and usage.get("total_token_count") is not None:
        # $0.25/1M input + $0.60/1M output — informational only (usage_metadata
        # does not split input/output cleanly for this SDK path; use total
        # tokens at the (higher, output) blended rate as a conservative
        # upper-bound real-token-derived estimate, NEVER the pre-call cost
        # model — see design §D: "sourced from each call's usage_metadata").
        cost_usd = usage["total_token_count"] * (0.60 / 1_000_000)

    # [RE-RUN] Read the REAL thinking level off the analyzer's own thinking_config
    # (set at construction time by build_time_analyzer/FilesApiTimeAnalyzer) —
    # never a second, independently-passed value that could drift from what was
    # actually sent. None if no thinking_config was set at all (budget=0/off).
    thinking_cfg = getattr(analyzer, "thinking_config", None)
    thinking_level_raw = getattr(thinking_cfg, "thinking_level", None)
    thinking_level = (
        thinking_level_raw.name.lower() if hasattr(thinking_level_raw, "name") else thinking_level_raw
    )

    ledger.record(LedgerEntry(
        gate=gate, window_id=window_id, match_id=match_id, arm=arm, call_name=call_name,
        model_id=model_id, status=status, error=err, http_status=http_status,
        latency_ms=latency_ms, usage_metadata=usage, cost_usd=cost_usd,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), ingestion_path=ingestion_path,
        thinking_level=thinking_level,
    ))

    call_record = {
        "call_name": call_name, "status": status, "error": err,
        "raw_response_text": getattr(analyzer, "last_raw_response_text", None),
        "parsed": data, "usage_metadata": usage, "latency_ms": round(latency_ms, 1),
        "ingestion_path": ingestion_path,
    }
    return data, err, call_record


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False))


def read_json(path: Path):
    return json.loads(path.read_text())
