"""Scope resolution for the ADCC eval cost-control dashboard (Task 3).

Exactly FOUR scope modes — no "all", per Jobs's non-negotiable AC (the scope
selector must never offer a way to select the entire 258-match corpus in one
click):
  - ``single``: exactly one ``match_id``.
  - ``list``: an explicit, hand-picked ``match_ids`` list.
  - ``smoke``: the fixed 3-video go/no-go set
    (``smoke_test.select_smoke_targets`` — 1 single-match video, 1 small
    multi-match VOD, 1 worst-case 25-match VOD).
  - ``split``: one whole NAMED split (``sandbox``/``dev``/``test``) — the
    split's own corpus, never "every split at once".

Reuses ``run_predictions.build_match_targets``/``plan_match_slices`` and
``smoke_test.select_smoke_targets`` verbatim; never re-derives match-target
or slicing logic here — this module is ONLY the scope-mode -> target-list
dispatch, plus the plan-token staleness guard.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eval.adcc.predict.run_predictions import MatchTarget, build_match_targets, plan_match_slices
from eval.adcc.predict.scope_policy import Slice
from eval.adcc.predict.smoke_test import select_smoke_targets

__all__ = [
    "VALID_MODES",
    "VALID_SPLITS",
    "ScopeError",
    "ScopeSpec",
    "GoldCorpus",
    "load_gold_corpus",
    "resolve_scope_targets",
    "build_slices_by_match",
    "compute_plan_token",
    "compute_highlight_plan_token",
]

VALID_MODES = ("single", "list", "smoke", "split")
VALID_SPLITS = ("sandbox", "dev", "test")


class ScopeError(ValueError):
    """Raised for a malformed/unresolvable scope request. The route layer
    converts this to HTTP 400 — never a silent fallback to some other
    scope."""


@dataclass(frozen=True)
class ScopeSpec:
    mode: str
    match_id: Optional[str] = None
    match_ids: Optional[tuple] = None
    split: Optional[str] = None

    def canonical(self) -> dict:
        """Deterministic, order-independent dict — the ONLY thing
        ``compute_plan_token`` hashes. Two ``ScopeSpec``s that mean the same
        scope (e.g. ``match_ids=("b","a")`` vs ``("a","b")``) must hash
        identically."""
        return {
            "mode": self.mode,
            "match_id": self.match_id,
            "match_ids": sorted(self.match_ids) if self.match_ids else None,
            "split": self.split,
        }


@dataclass(frozen=True)
class GoldCorpus:
    gold_records: list
    split_assignment: Optional[dict]

    def all_targets(self) -> dict:
        """Built fresh from the FULL corpus every call — never cached across
        a mutation, and never a pre-filtered subset (the smoke mode's
        "worst-case 25-match VOD" criterion needs to see the whole corpus to
        evaluate correctly)."""
        return build_match_targets(self.gold_records)


def load_gold_corpus(gold_snapshot_path: Path, split_assignment_path: Optional[Path] = None) -> GoldCorpus:
    gold_records = [json.loads(line) for line in gold_snapshot_path.read_text().splitlines() if line.strip()]
    split_assignment = None
    if split_assignment_path is not None and split_assignment_path.exists():
        split_assignment = json.loads(split_assignment_path.read_text())
    return GoldCorpus(gold_records=gold_records, split_assignment=split_assignment)


def _split_match_ids(split_assignment: dict, split: str) -> set:
    wanted: set = set()
    for entry in split_assignment["assignment"].values():
        if entry["split"] == split:
            wanted.update(entry["match_ids"])
    return wanted


def resolve_scope_targets(scope: ScopeSpec, corpus: GoldCorpus) -> list:
    """Resolves ``scope`` into an ORDERED list of ``MatchTarget``. Raises
    ``ScopeError`` for any malformed/unknown request — never silently
    substitutes a default scope or drops an unresolvable match_id."""
    if scope.mode not in VALID_MODES:
        raise ScopeError(f"Unknown scope mode {scope.mode!r}; must be one of {VALID_MODES}.")

    all_targets = corpus.all_targets()

    if scope.mode == "single":
        if not scope.match_id:
            raise ScopeError("scope.mode='single' requires match_id.")
        target = all_targets.get(scope.match_id)
        if target is None:
            raise ScopeError(f"Unknown match_id {scope.match_id!r} (not in the gold snapshot).")
        return [target]

    if scope.mode == "list":
        if not scope.match_ids:
            raise ScopeError("scope.mode='list' requires a non-empty match_ids list.")
        missing = [mid for mid in scope.match_ids if mid not in all_targets]
        if missing:
            raise ScopeError(f"Unknown match_id(s) {missing!r} (not in the gold snapshot).")
        return [all_targets[mid] for mid in scope.match_ids]

    if scope.mode == "smoke":
        return select_smoke_targets(all_targets)

    # scope.mode == "split"
    if scope.split not in VALID_SPLITS:
        raise ScopeError(f"scope.mode='split' requires split in {VALID_SPLITS}, got {scope.split!r}.")
    if corpus.split_assignment is None:
        raise ScopeError("No split_assignment loaded — cannot resolve scope.mode='split'.")
    wanted_ids = _split_match_ids(corpus.split_assignment, scope.split)
    return [all_targets[mid] for mid in all_targets if mid in wanted_ids]


def build_slices_by_match(targets: list, video_durations: dict) -> dict:
    """One ``plan_match_slices`` call per target. Raises
    ``RunPredictionsError`` (never a silent 0-duration guess — propagated
    from ``run_predictions.plan_match_slices``) if a single-match target's
    video duration is missing from ``video_durations``."""
    return {t.match_id: plan_match_slices(t, video_durations) for t in targets}


def compute_plan_token(scope_canonical: dict, cap: int) -> str:
    """Deterministic short hash of (scope, cap) — the dashboard's stale-
    preview guard. ``/plan`` returns this; ``/run`` recomputes it itself from
    the SUBMITTED scope+cap (never trusts a client-echoed token blindly) and
    401/400s if it doesn't match what the client claims it previewed — the
    server-side enforcement of "changing scope/cap invalidates the
    preview"."""
    payload = json.dumps({"scope": scope_canonical, "cap": cap}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_highlight_plan_token(
    match_id: str,
    cap: int,
    model: str,
    thinking_level: Optional[str],
    system_prompt: str,
    followup_prompt: str,
    fps: float,
    frame_quality: str,
    refine: bool = True,
    refine_pad_before_s: float = 12.0,
    refine_pad_after_s: float = 5.0,
    refine_fps: float = 4.0,
) -> str:
    """The Highlight Eval feature's OWN plan-token (frozen contract §E): the
    single-match ``/highlight/plan`` -> confirm -> ``/highlight/run`` gate has
    TEN staleness inputs beyond scope+cap that ``compute_plan_token`` above
    doesn't cover — ``model``/``thinking_level`` (which Gemini call actually
    runs), the founder's own ``system_prompt``/``followup_prompt`` free text
    (what the model is asked to find), ``fps``/``frame_quality`` (how
    densely/at what resolution the video is sampled), and (this increment's
    timestamp critique/refinement feature) ``refine``/``refine_pad_before_s``/
    ``refine_pad_after_s``/``refine_fps`` (whether/how a second per-highlight
    pass re-checks each discovered timestamp — a real cost AND accuracy
    lever, not cosmetic). Changing ANY of the ten invalidates the preview,
    same server-side-recomputed discipline as ``compute_plan_token``:
    ``/highlight/run`` recomputes this from the SUBMITTED request body and
    400s on a mismatch, never trusting a client-echoed token.

    ``fps``/``frame_quality``/``refine``/``refine_pad_before_s``/
    ``refine_pad_after_s``/``refine_fps`` are hashed IN THE CLEAR alongside
    ``model``/``thinking_level`` (not folded into ``prompt_hash`` — they are
    call-shape knobs, not free text, and belong with the other structured
    fields for the same reason ``model`` isn't hashed as if it were prose).

    The prompt text itself is hashed first (never embedded verbatim in the
    token payload) purely to keep the outer hash's input small and
    order-of-magnitude-independent of prompt length — the SHA-256 of the
    prompt pair is exactly as sensitive to a single-character prompt edit as
    the raw text would be."""
    prompt_hash = hashlib.sha256((system_prompt + "\x00" + followup_prompt).encode("utf-8")).hexdigest()
    payload = json.dumps(
        {
            "match_id": match_id, "cap": cap, "prompt_hash": prompt_hash, "model": model,
            "thinking_level": thinking_level, "fps": fps, "frame_quality": frame_quality,
            "refine": refine, "refine_pad_before_s": refine_pad_before_s,
            "refine_pad_after_s": refine_pad_after_s, "refine_fps": refine_fps,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
