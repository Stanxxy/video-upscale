"""Windowed strike census (workflow ``mma-windowed-strikes``): pure functions only.

Strikes are dense and sub-second, so they are not discovered by the whole-scope
technique scan and reviewed one at a time. The scope is tiled into short overlapping
windows; each window gets one census call (and optionally one audit call that keeps,
drops, retimes or adds strikes). This module tiles the scope and merges the answers
into MMA technique analyses. A strike is never moved or invented here: strikes in
not-live footage are rejected, and only cross-window duplicates in the overlap are
merged.
"""

from bjj_analysis_core.mma_outputs import (
    MmaTechniqueAnalysisV1,
    StrikeAuditV1,
    StrikeBatchV1,
    StrikeCensusV1,
)
from shared_lib.models.analysis_contracts import TimeRangeMs

# A merged strike event spans its contact moment +-250 ms; outcome evidence is the
# contact +-50 ms. Same-fighter, same-class strikes this close from two sources (two
# windows of an overlap, or the census and its audit's additions) are one strike seen
# twice; within one source they are a real fast combination.
EVENT_HALF_MS = 250
CONTACT_HALF_MS = 50
DUPLICATE_MS = 400


def strike_windows(run, config):
    """Tile the selected scope into overlapping windows, in order."""
    scope = run.spec.scope
    size, overlap = config.strike_window_ms, config.strike_overlap_ms
    windows, start = [], scope.start_ms
    while True:
        end = min(start + size, scope.end_ms)
        windows.append(TimeRangeMs(start_ms=start, end_ms=end))
        if end == scope.end_ms:
            return tuple(windows)
        start = end - overlap


def _not_live(positions, at_ms):
    return any(
        p.observation == "inactive" and p.bounds.start_ms <= at_ms < p.bounds.end_ms
        for p in positions.positions or ()
    )


def _analysis(run, strike, at_ms):
    scope = run.spec.scope
    bounds = TimeRangeMs(
        start_ms=max(scope.start_ms, at_ms - EVENT_HALF_MS),
        end_ms=min(scope.end_ms, at_ms + EVENT_HALF_MS),
    )
    known = strike.outcome != "unclear"
    contact = TimeRangeMs(
        start_ms=max(bounds.start_ms, at_ms - CONTACT_HALF_MS),
        end_ms=min(bounds.end_ms, at_ms + CONTACT_HALF_MS),
    )
    return MmaTechniqueAnalysisV1(
        bounds=bounds,
        action_classes=(strike.action.value,),
        actor_player_id=strike.actor_player_id,
        actor_sentinel=strike.actor_sentinel,
        outcome=strike.outcome,
        notes=strike.evidence,
        evidence=strike.evidence,
        outcome_evidence=strike.evidence,
        outcome_bounds=contact if known else None,
        strike_target=strike.strike_target.value,
        significant=strike.significant,
        knockdown=strike.knockdown,
    )


def merge_strikes(run, config, censuses, audits, positions):
    """One census per window (and one audit per window when audited) -> final strikes."""
    windows = strike_windows(run, config)
    censuses = tuple(StrikeCensusV1.model_validate(c) for c in censuses)
    if [c.window_index for c in censuses] != list(range(len(windows))):
        raise ValueError("exactly one census is required for every strike window")
    if audits is not None:
        audits = tuple(StrikeAuditV1.model_validate(a) for a in audits)
        if [a.window_index for a in audits] != list(range(len(windows))):
            raise ValueError("exactly one audit is required for every strike window")
    found, dropped, added = [], 0, 0
    for index, census in enumerate(censuses):
        if audits is None:
            found += [(s.at_ms, (index, "census"), s) for s in census.strikes]
            continue
        audit = audits[index]
        if len(audit.decisions) != len(census.strikes):
            raise ValueError("audit must decide every census strike")
        for strike, decision in zip(census.strikes, audit.decisions, strict=True):
            if decision.verdict == "drop":
                dropped += 1
            else:
                at = decision.at_ms if decision.verdict == "retime" else strike.at_ms
                found.append((at, (index, "census"), strike))
        found += [(s.at_ms, (index, "added"), s) for s in audit.added]
        added += len(audit.added)
    for at, (index, _), _ in found:
        if not windows[index].start_ms <= at <= windows[index].end_ms:
            raise ValueError("strike time falls outside its window")
    live = [f for f in found if not _not_live(positions, f[0])]
    rejected = len(found) - len(live)
    live.sort(key=lambda f: (f[0], f[1]))
    kept, merged = [], 0
    for at, source, strike in live:
        actor = strike.actor_player_id or strike.actor_sentinel
        twin = any(
            other_source != source
            and abs(other_at - at) <= DUPLICATE_MS
            and (other.actor_player_id or other.actor_sentinel) == actor
            and other.action == strike.action
            for other_at, other_source, other in kept
        )
        if twin:
            merged += 1
        else:
            kept.append((at, source, strike))
    return StrikeBatchV1(
        strikes=tuple(_analysis(run, strike, at) for at, _, strike in kept),
        census_count=sum(len(c.strikes) for c in censuses),
        dropped_by_audit=dropped,
        added_by_audit=added,
        rejected_not_live=rejected,
        merged_duplicates=merged,
    )
