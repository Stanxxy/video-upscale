"""Bounded parallel execution: same steps, same requests, same results as serial."""

import asyncio
import json
from datetime import timedelta

import pytest
from bjj_analysis_core.execution import NotDispatched, RecoveryRequired
from conftest import state, technique
from journal_fixture import DiskJournal
from test_executor import FixtureProvider, Sink, execute_pipeline

# Ordered, non-overlapping scan events inside the fixture scope [1000, 11000).
EVENTS = ((1000, 2200), (3000, 4200), (5200, 6400), (7400, 8600), (9400, 10600))
TRANSITIONS = (
    (2600, "closed_guard", ("guard_player", "guard_opponent")),
    (4800, "side_control", ("pinning_player", "pinned_player")),
    (8900, "mount", ("pinning_player", "pinned_player")),
)


class FanoutProvider(FixtureProvider):
    """Several transitions and technique events, with latency and in-flight tracking."""

    def __init__(self, clock, sink, *, delays=None):
        super().__init__(clock, sink)
        self.delays = dict(delays or {})
        self.items = []  # (node, item id) in dispatch order
        self.active = []  # node ids of calls currently in flight
        self.max_in_flight = 0
        self.max_by_node = {}
        # (node, item) -> (gate, exception): raise once `gate()` is true instead of replying
        self.failures = {}
        self.watch = None  # (node, count): set `reached` once that many are in flight
        self.reached = asyncio.Event()

    async def generate(self, run, request, execution_id, call_id):
        node = request.node_id
        self.calls.append((node, call_id))
        self.items.append((node, item_of(request)))
        self.active.append(node)
        self.max_in_flight = max(self.max_in_flight, len(self.active))
        self.max_by_node[node] = max(
            self.max_by_node.get(node, 0), self.active.count(node)
        )
        if self.watch and self.active.count(self.watch[0]) >= self.watch[1]:
            self.reached.set()
        try:
            failure = self.failures.get((node, item_of(request)))
            if failure is not None:
                while not failure[0]():
                    await asyncio.sleep(0.001)
                raise failure[1]
            await asyncio.sleep(self.delays.get(node, 0.0))
            return self.reply(
                run, request, execution_id, call_id, json.dumps(reply_for(request))
            )
        finally:
            self.active.remove(node)


def item_of(request):
    context = json.loads(request.initial_prompt)["stage_context"]
    if request.node_id == "technique_critique":
        return context["candidate"]["candidate_id"]
    if request.node_id in ("technique_analyze", "technique_validate"):
        return context["candidate_id"]
    return None


def reply_for(request):
    context = json.loads(request.initial_prompt)["stage_context"]
    node = request.node_id
    if node == "position_scan":
        return dict(
            initial_state=state(),
            transitions=[
                dict(at_ms=at, state_after=state(position, roles))
                for at, position, roles in TRANSITIONS
            ],
        )
    if node == "position_critique":
        proposed = context["proposed"]
        return dict(
            transition_index=context["transition_index"],
            confirmed=True,
            corrected_at_ms=proposed["at_ms"],
            state_after=proposed["state_after"],
            reason="transition visible",
        )
    if node == "technique_scan":
        return dict(
            events=[
                dict(
                    bounds=dict(start_ms=start, end_ms=end),
                    description=f"attempt at {start}",
                    reasoning="grip visible",
                )
                for start, end in EVENTS
            ]
        )
    if node == "technique_critique":
        candidate = context["candidate"]
        return dict(
            candidate_id=candidate["candidate_id"],
            confirmed=True,
            bounds=candidate["bounds"],
            end_status="outcome_visible",
            reason="outcome visible",
        )
    if node == "technique_analyze":
        bounds = context["bounds"]
        return technique(
            bounds=bounds,
            outcome_bounds=dict(start_ms=bounds["end_ms"] - 100, end_ms=bounds["end_ms"]),
        )
    return dict(
        candidate_id=context["candidate_id"],
        verdict="agree",
        intentional_technique_visible=True,
        technique=context["proposed"],
        reason="outcome visible",
    )


class FrozenClock:
    """One instant for every lane, so revisions (and the advisory positions and
    candidate identities derived from them) are byte-comparable across schedules."""

    def __init__(self, run):
        self.at = run.registered_at + timedelta(seconds=1)

    def __call__(self):
        return self.at


@pytest.fixture
def lanes(run, tmp_path):
    made = []

    def lane(name, **provider_options):
        journal, clock, sink = DiskJournal(tmp_path / f"{name}.sqlite"), FrozenClock(run), Sink()
        made.append(journal)
        return journal, clock, sink, FanoutProvider(clock, sink, **provider_options)

    yield lane
    for journal in made:
        journal.close()


async def execute(run, configuration, lane, **kwargs):
    journal, clock, sink, provider = lane
    return await execute_pipeline(
        run,
        configuration,
        journal=journal,
        provider=provider,
        sink=sink,
        clock=clock,
        **kwargs,
    )


def identity(journal):
    """Every durable fact about each step head (frozen clock: timestamps included)."""
    return {
        record.step_id: record.model_dump_json(exclude={"usage": {"raw_usage"}})
        for record in journal.records()
    }


def analyze_items_in_work_order(provider):
    return [item for node, item in provider.items if node == "technique_analyze"]


@pytest.mark.asyncio
async def test_parallel_run_has_identical_steps_requests_and_results(run, configuration, lanes):
    serial, parallel = lanes("serial"), lanes("parallel")
    one = await execute(run, configuration, serial)
    six = await execute(run, configuration, parallel, max_parallel_calls=6)

    assert identity(serial[0]) == identity(parallel[0])
    assert sorted(map(str, serial[3].calls)) == sorted(map(str, parallel[3].calls))
    assert one.positions.positions == six.positions.positions
    assert one.techniques.techniques == six.techniques.techniques
    assert serial[3].max_in_flight == 1
    # The fixture genuinely fans out: several items on every fan-out node.
    nodes = [node for node, _ in parallel[3].calls]
    assert nodes.count("position_critique") == len(TRANSITIONS)
    assert nodes.count("technique_critique") >= len(EVENTS)
    assert nodes.count("technique_validate") >= len(EVENTS)


@pytest.mark.asyncio
async def test_in_flight_calls_never_exceed_bound_and_position_chain_stays_serial(
    run, configuration, lanes
):
    delays = dict.fromkeys(
        ("technique_scan", "technique_critique", "technique_analyze", "technique_validate"),
        0.02,
    )
    lane = lanes("bounded", delays=delays)
    await execute(run, configuration, lane, max_parallel_calls=3)
    provider = lane[3]

    assert provider.max_in_flight == 3
    for node in ("technique_critique", "technique_analyze", "technique_validate"):
        assert provider.max_by_node[node] == 3
    assert provider.max_by_node["position_critique"] == 1
    # The slow technique scan ran beside the position chain, not after publication.
    nodes = [node for node, _ in provider.calls]
    assert nodes.index("technique_scan") < nodes.index("position_critique")


@pytest.mark.asyncio
async def test_failure_drains_in_flight_steps_then_resume_completes_identically(
    run, configuration, lanes
):
    reference = lanes("reference")
    expected = await execute(run, configuration, reference)
    work = analyze_items_in_work_order(reference[3])  # serial = work order
    assert len(work) >= 4

    lane = lanes("failing", delays={"technique_analyze": 0.5})
    journal, _, _, provider = lane
    # Fails only once items 0, 1 and 2 are all in flight (peers still sleeping).
    full = lambda: provider.active.count("technique_analyze") == 3  # noqa: E731
    provider.failures[("technique_analyze", work[1])] = (full, NotDispatched("refused"))
    with pytest.raises(NotDispatched):
        await execute(run, configuration, lane, max_parallel_calls=3)

    heads = journal.records()
    assert not [r for r in heads if r.phase == "running"]  # every started step settled
    analyze = {
        r.progress.item_id: r.phase for r in heads if r.progress.node_id == "technique_analyze"
    }
    # Items 0 and 2 were in flight with the failure and finished; nothing new started.
    assert analyze == {work[0]: "succeeded", work[1]: "failed", work[2]: "succeeded"}
    assert not [r for r in heads if r.progress.node_id == "technique_validate"]

    with pytest.raises(RecoveryRequired):  # failed work still needs authorized retry
        await execute(run, configuration, lane, max_parallel_calls=3)
    del provider.failures[("technique_analyze", work[1])]
    before = len(provider.calls)
    result = await execute(run, configuration, lane, max_parallel_calls=3, retry_failed=True)
    repeated = [node for node, _ in provider.calls[before:]]
    # Completed work is reused; only the failed and never-started items are called.
    assert repeated.count("technique_analyze") == len(work) - 2
    assert repeated.count("technique_validate") == len(work)
    assert set(repeated) == {"technique_analyze", "technique_validate"}
    assert result.positions.positions == expected.positions.positions
    assert result.techniques.techniques == expected.techniques.techniques


@pytest.mark.asyncio
async def test_earliest_failure_in_input_order_is_raised(run, configuration, lanes):
    reference = lanes("order-reference")
    await execute(run, configuration, reference)
    work = analyze_items_in_work_order(reference[3])

    lane = lanes("two-failures", delays={"technique_analyze": 0.5})
    provider = lane[3]
    full = lambda: provider.active.count("technique_analyze") == 3  # noqa: E731
    later = NotDispatched("item 2 fails first")
    earlier = NotDispatched("item 1 fails second")
    provider.failures[("technique_analyze", work[2])] = (full, later)
    # Item 1 fails only after item 2 already has.
    provider.failures[("technique_analyze", work[1])] = (
        lambda: provider.active.count("technique_analyze") == 2,
        earlier,
    )
    with pytest.raises(NotDispatched) as raised:
        await execute(run, configuration, lane, max_parallel_calls=3)
    assert raised.value is earlier


@pytest.mark.asyncio
async def test_technique_scan_failure_still_publishes_positions_and_stops_techniques(
    run, configuration, lanes
):
    lane = lanes("scan-fails", delays={"position_critique": 0.02})
    journal, _, sink, provider = lane
    provider.failures[("technique_scan", None)] = (lambda: True, NotDispatched("refused"))
    with pytest.raises(NotDispatched):
        await execute(run, configuration, lane, max_parallel_calls=3)

    # The position branch is one item: it runs to publication, as when technique
    # work fails serially; no technique step after the scan ever starts.
    assert "positions" in sink.outputs and "techniques" not in sink.outputs
    nodes = [node for node, _ in provider.calls]
    assert nodes.count("position_critique") == len(TRANSITIONS)
    assert set(nodes) == {"position_scan", "position_critique", "technique_scan"}
    assert not [r for r in journal.records() if r.phase == "running"]


@pytest.mark.asyncio
async def test_cancellation_settles_every_in_flight_call_as_cancelled(
    run, configuration, lanes
):
    lane = lanes("cancel", delays={"technique_analyze": 30.0})
    journal, _, _, provider = lane
    provider.watch = ("technique_analyze", 3)
    task = asyncio.create_task(execute(run, configuration, lane, max_parallel_calls=3))
    await asyncio.wait_for(provider.reached.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    heads = journal.records()
    assert not [r for r in heads if r.phase == "running"]
    cancelled = [r for r in heads if r.phase == "cancelled"]
    assert len(cancelled) == 3
    assert {r.progress.node_id for r in cancelled} == {"technique_analyze"}
    assert all(r.usage.cost_nano_usd is None for r in cancelled)  # spend stays unknown
    assert not provider.active


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [0, 17, True, 2.0, None])
async def test_parallelism_bound_is_explicit_and_small(run, configuration, lanes, bound):
    with pytest.raises(ValueError, match="max_parallel_calls"):
        await execute(run, configuration, lanes(f"bad{bound}"), max_parallel_calls=bound)
