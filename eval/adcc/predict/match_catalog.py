"""Zero-Gemini per-match catalog for the dashboard's searchable match-picker
menu (the founder can't memorize hex ``match_id``s -- this is the menu's data
source).

Deliberately its own module, not folded into ``scope_policy.py`` (scoped
narrowly to window-policy math) or ``dashboard_scope.py`` (scoped narrowly to
scope-mode -> target-list dispatch + the plan-token staleness guard): this is
a THIRD, catalog-building concern -- name lookup + "length actually analyzed"
+ dual (realistic/worst-case) call estimates for ALL matches in the corpus at
once, for DISPLAY only. It never feeds a real run directly; the menu's
checked ``match_id``s still go through the existing ``mode="list"``/``"single"``
``/plan`` -> confirm gate in ``dashboard_app.py``, unchanged.

Reuses, verbatim, the SAME building blocks the real dry-run plan uses so the
numbers shown here can never drift from what ``/plan``/a real run would
compute:
  - ``scope_policy.single_match_scope``/``multi_match_scope`` for the scope
    WINDOW (the length actually analyzed -- for a multi-match VOD this is the
    gold-derived event span, NOT the sum of over-cap slice durations, which
    would double-count the ``SLICE_OVERLAP_S`` seam overlap).
  - ``scope_policy.slice_scope`` for the ordered per-run slice list (only the
    over-cap slicing needs multiple runs; each run gets its OWN realistic +
    worst-case call estimate, then both are summed for the match).
  - ``cost_model.estimate_batch_calls_realistic``/``estimate_batch_cost_usd``
    for LeCun's realistic (PRIMARY) call/dollar estimate.
  - ``run_predictions.estimate_slice_calls`` (``executors.estimate_run_plan``
    under the hood) for the pathological worst-case (SECONDARY, cap-suggestion
    only) call estimate -- the exact same function ``run_predictions.build_dry_run_plan``
    uses for ``/plan``.
"""
from __future__ import annotations

from typing import Optional

from eval.adcc.predict.cost_model import estimate_batch_calls_realistic, estimate_batch_cost_usd
from eval.adcc.predict.dashboard_scope import GoldCorpus
from eval.adcc.predict.run_predictions import PIPELINE_ID, MatchTarget, estimate_slice_calls
from eval.adcc.predict.scope_policy import ScopeWindow, multi_match_scope, single_match_scope, slice_scope
from service.pipelines import registry
from service.pipelines.models import PipelineDef

__all__ = ["build_match_catalog"]


def _resolve_window(target: MatchTarget, video_durations: dict) -> Optional[ScopeWindow]:
    """Returns ``None`` -- never raises, never guesses a duration -- when a
    single-match video's ffprobe duration isn't in ``video_durations``. The
    catalog entry flags this via ``has_duration=False`` instead of failing
    the WHOLE ``/matches`` response for one bad/missing entry (this is a
    display-only catalog, not the run-gating path, where a missing duration
    correctly stays a hard ``RunPredictionsError``)."""
    if target.is_multi_match:
        return multi_match_scope(target.match_id, target.gold_t_s_values)
    if target.video_url_md5 not in video_durations:
        return None
    return single_match_scope(target.match_id, video_durations[target.video_url_md5])


def _format_mmss(total_seconds: float) -> str:
    total = int(round(total_seconds))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}:{seconds:02d}"


def _catalog_entry(
    target: MatchTarget, name: str, video_durations: dict, pipeline_def: PipelineDef,
) -> dict:
    entry = {
        "match_id": target.match_id,
        "name": name,
        "video_url": target.video_url,
        "video_md5": target.video_url_md5,
        "is_multi_match": target.is_multi_match,
    }

    window = _resolve_window(target, video_durations)
    if window is None:
        entry.update(
            has_duration=False,
            length_seconds=None,
            length_mmss=None,
            est_calls_realistic=None,
            est_calls_worst_case=None,
            est_dollars_realistic=None,
        )
        return entry

    # The SAME slice list a real run would use -- so an over-cap match's
    # realistic/worst-case call counts reflect every run it actually takes,
    # while length_seconds stays the raw window (not slice-sum-with-overlap).
    slices = slice_scope(window)
    scope_seconds = [sl.duration_s for sl in slices]
    est_calls_worst_case = sum(
        estimate_slice_calls(pipeline_def, sl)["planned_gemini_calls"] for sl in slices
    )

    entry.update(
        has_duration=True,
        length_seconds=window.duration_s,
        length_mmss=_format_mmss(window.duration_s),
        est_calls_realistic=estimate_batch_calls_realistic(scope_seconds),
        est_calls_worst_case=est_calls_worst_case,
        est_dollars_realistic=round(estimate_batch_cost_usd(scope_seconds, apply_margin=False), 2),
    )
    return entry


def build_match_catalog(corpus: GoldCorpus, video_durations: dict) -> list[dict]:
    """All matches in ``corpus``, each with a human name + length + dual
    (realistic/worst-case) call estimates. ZERO Gemini calls anywhere in this
    call graph -- ``estimate_slice_calls``/``estimate_run_plan`` and
    ``cost_model``'s functions are pure arithmetic over already-known
    durations, no live client, no network. Sorted by ``name`` (the athlete
    matchup) so the menu is browsable, not a raw hex-id dump.

    ``corpus.gold_records`` is the source of the human ``name`` -- read
    directly (``rec["name"]``, no ``.get`` fallback): every real gold record
    carries this field (asserted by ``build_gold_snapshot.py``), so a record
    missing it is a real gold-snapshot bug that must surface loudly, not a
    silently-defaulted hex id."""
    pipeline_def = registry.get_default(PIPELINE_ID)
    name_by_match_id = {rec["match_id"]: rec["name"] for rec in corpus.gold_records}
    all_targets = corpus.all_targets()

    catalog = [
        _catalog_entry(target, name_by_match_id[match_id], video_durations, pipeline_def)
        for match_id, target in all_targets.items()
    ]
    catalog.sort(key=lambda entry: entry["name"])
    return catalog
