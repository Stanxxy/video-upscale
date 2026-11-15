"""Ground re-scan (workflow mma-windowed-strikes-ground): pure functions only.

The whole-scope position scan samples 1 fps over minutes. On the ground it can lose
which fighter is on top, or call ground control a clinch. Each ground phase of the scan
is re-scanned in a short window at a higher frame rate, starting a few seconds before the
phase (both fighters are usually easier to tell apart there). The model names the top
fighter; the ordered role pair is derived here. The re-scan replaces the scan's states
inside each phase; the scan stands everywhere else.
"""

from bjj_analysis_core.mma_outputs import GroundRescanV1
from shared_lib.models.analysis_contracts import TimeRangeMs

# position -> (top role, bottom role); every other position is neutral/neutral.
TOP_BOTTOM = {
    "closed_guard": ("guard_opponent", "guard_player"),
    "open_guard": ("guard_opponent", "guard_player"),
    "half_guard": ("guard_opponent", "guard_player"),
    "side_control": ("pinning_player", "pinned_player"),
    "mount": ("pinning_player", "pinned_player"),
    "back_control": ("back_controller", "back_defender"),
    "standing_back_control": ("back_controller", "back_defender"),
    "turtle": ("turtle_opponent", "turtled_player"),
    "wall_seated": ("wall_pinner", "wall_seated_player"),
}
# Scan positions whose spans are re-scanned (scramble: who comes out on top).
PHASE_POSITIONS = frozenset(TOP_BOTTOM) | {"scramble"}
MERGE_GAP_MS = 6000  # phases this close are one phase
AFTER_MS = 2000  # the window runs a little past the phase
MIN_PHASE_MS = 2000


def _intervals(run, scan):
    scope = run.spec.scope
    starts = [(scope.start_ms, scan.initial_state)] + [(t.at_ms, t.state_after) for t in scan.transitions]
    ends = [t for t, _ in starts[1:]] + [scope.end_ms]
    return [(s, e, state) for (s, state), e in zip(starts, ends, strict=True)]


def ground_phases(run, config, scan):
    """(core, window) for every ground phase of the scan, in order."""
    scope = run.spec.scope
    spans = [
        [s, e]
        for s, e, state in _intervals(run, scan)
        if state.observation == "observed" and state.position is not None
        and state.position.value in PHASE_POSITIONS
    ]
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= MERGE_GAP_MS:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    phases = []
    for s, e in merged:
        while s < e:
            core_end = min(e, s + config.ground_window_ms)
            if core_end - s >= MIN_PHASE_MS:
                core = TimeRangeMs(start_ms=s, end_ms=core_end)
                window = TimeRangeMs(
                    start_ms=max(scope.start_ms, s - config.ground_lookback_ms),
                    end_ms=min(scope.end_ms, core_end + AFTER_MS),
                )
                phases.append((core, window))
            s = core_end
    return tuple(phases)


def ground_state(pack, run, state):
    """A re-scan state as a position state of the run's pack (role pair derived)."""
    position = state.position.value
    if position in TOP_BOTTOM:
        ids = [a.player_id for a in run.spec.athletes]
        if state.top_player_id not in ids:
            raise ValueError("a ground position names its top fighter")
        top, bottom = TOP_BOTTOM[position]
        roles = (top, bottom) if state.top_player_id == ids[0] else (bottom, top)
    elif state.top_player_id is not None:
        raise ValueError("a neutral position has no top fighter")
    else:
        roles = ("neutral", "neutral")
    return pack.position_scan.model_fields["initial_state"].annotation.model_validate(
        dict(
            observation="observed",
            position=position,
            roles=roles,
            notes=state.geometry or "",
            evidence=state.evidence,
            uncertainty_note=None,
            interruption_note=None,
            interruption_kind=None,
            cage_contact=None,
        )
    )


def apply_ground_rescans(pack, run, config, scan, rescans):
    """The scan with every ground phase replaced by its re-scan (one per phase, in order)."""
    phases = ground_phases(run, config, scan)
    rescans = tuple(GroundRescanV1.model_validate(r) for r in rescans)
    if [r.phase_index for r in rescans] != list(range(len(phases))):
        raise ValueError("exactly one ground re-scan is required for every ground phase")
    scope = run.spec.scope
    original = _intervals(run, scan)

    def state_at(t):
        return next(state for s, e, state in original if s <= t < e)

    def in_core(t):
        return any(core.start_ms <= t < core.end_ms for core, _ in phases)

    boundaries = [(s, state) for s, _, state in original if not in_core(s)]
    for (core, _), rescan in zip(phases, rescans, strict=True):
        placed = []
        for g in rescan.states:
            at = max(g.at_ms, core.start_ms)
            if at >= core.end_ms:
                continue
            if placed and placed[-1][0] == at:
                placed.pop()  # a later state at the core start supersedes an earlier one
            placed.append((at, ground_state(pack, run, g)))
        if not placed or placed[0][0] != core.start_ms:
            raise ValueError("a ground re-scan must cover its phase from the start")
        boundaries += placed
        if core.end_ms < scope.end_ms and not in_core(core.end_ms):
            boundaries.append((core.end_ms, state_at(core.end_ms)))
    boundaries.sort(key=lambda b: b[0])
    timeline = []
    for at, state in boundaries:
        if timeline and timeline[-1][0] == at:
            timeline[-1] = (at, state)  # the scan's own boundary at a core end
        elif not timeline or timeline[-1][1] != state:
            timeline.append((at, state))
    if timeline[0][0] != scope.start_ms:
        raise ValueError("the merged timeline must start at the scope start")
    return pack.position_scan(
        initial_state=timeline[0][1],
        transitions=tuple(dict(at_ms=at, state_after=state) for at, state in timeline[1:]),
    )


def rescanned_transitions(config, phases, merged):
    """Indexes of merged-scan transitions a position check must leave alone.

    The re-scan is the check inside ground phases: a position critique there (or one
    whose retime could reach into a phase) would re-time the high-fps boundaries from
    a coarser look, and could cross them.
    """
    reach = [
        (core.start_ms - config.position_critique_forward_ms, core.end_ms + config.position_critique_back_ms)
        for core, _ in phases
    ]
    return tuple(i for i, t in enumerate(merged.transitions) if any(a <= t.at_ms <= b for a, b in reach))
