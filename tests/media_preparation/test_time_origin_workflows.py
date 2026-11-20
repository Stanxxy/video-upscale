"""Every production workflow, end to end, on a file whose video starts after its container.

The real core executor drives the real S3RunMedia + GeminiProvider; only the Gemini HTTP
endpoint is a fixture. That endpoint plays the provider's clock: it receives the prompt
and window as sent, checks them independently of ``analysis_time_map`` (every ``*_ms``
integer must be exactly the source value minus the measured offset), answers on its
own re-based clock, and the published result must equal the same run on a zero-origin
file. A node whose times were not shifted at the Gemini boundary fails here.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from bjj_analysis_core import MMA_GROUND_WORKFLOW_ID, MMA_STRIKES_WORKFLOW_ID, WORKFLOW_ID
from bjj_analysis_core.executor import execute_pipeline
from bjj_analysis_core.requests import canonical_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis_core" / "tests"))
from journal_fixture import DiskJournal  # noqa: E402
from test_grounds import SCAN as GROUND_SCAN  # noqa: E402
from test_grounds import ground_config  # noqa: E402
from test_windowed_strikes import Sink, config_bytes, strike, strikes_run  # noqa: E402

from tests.media_preparation.test_gemini import Wire, uploaded  # noqa: E402
from tests.media_preparation.test_media import altered  # noqa: E402
from tests.media_preparation.test_time_origin import probe, run as ffmpeg, video_offset  # noqa: E402

LIVE = dict(observation="observed", position="striking_range", roles=["neutral", "neutral"], notes="",
            evidence="both standing", uncertainty_note=None, interruption_note=None, interruption_kind=None,
            cage_contact=None)  # fmt: skip
REPLAY = dict(LIVE, observation="inactive", position=None, roles=["not_applicable", "not_applicable"],
              evidence="slow-motion replay graphic", interruption_note="broadcast replay",
              interruption_kind="replay")  # fmt: skip
BJJ_STATE = dict(observation="observed", position="back_control", roles=["back_controller", "back_defender"],
                 notes="", evidence="visible torso and leg relationship", uncertainty_note=None,
                 interruption_note=None)  # fmt: skip


def technique(bounds, **changes):
    value = dict(bounds=bounds, action_classes=["submission_choke"], actor_player_id="a", actor_sentinel=None,
                 outcome="successful", notes="rear naked choke", evidence="visible intentional choke attempt",
                 outcome_evidence="defender taps", outcome_bounds=dict(start_ms=7400, end_ms=7500))  # fmt: skip
    value.update(changes)
    return value


def answer(workflow, node, context):
    """The scripted provider, on the Analyzer source clock (scope 1000-11000)."""
    if node == "position_scan":
        if workflow == MMA_GROUND_WORKFLOW_ID:
            return {k: v for k, v in GROUND_SCAN.items() if k != "schema_version"}
        if workflow == MMA_STRIKES_WORKFLOW_ID:
            return dict(initial_state=LIVE, transitions=[dict(at_ms=8000, state_after=REPLAY)])
        return dict(initial_state=dict(BJJ_STATE, position="standing", roles=["neutral", "neutral"]),
                    transitions=[dict(at_ms=4000, state_after=BJJ_STATE)])  # fmt: skip
    if node == "position_critique":
        return dict(transition_index=context["transition_index"], confirmed=True,
                    corrected_at_ms=context["proposed"]["at_ms"] + 300,
                    state_after=context["proposed"]["state_after"], reason="looks later")  # fmt: skip
    if node == "technique_scan":
        if workflow == WORKFLOW_ID:
            return dict(events=[dict(bounds=dict(start_ms=2000, end_ms=5000), description="A applies a rear choke",
                                     reasoning="grip and defense visible")])  # fmt: skip
        return dict(events=[])
    if node == "technique_critique":
        return dict(candidate_id=context["candidate"]["candidate_id"], confirmed=True,
                    bounds=dict(start_ms=2000, end_ms=8000), end_status="outcome_visible",
                    reason="tap and release visible after rough end")  # fmt: skip
    if node == "technique_analyze":
        if workflow == WORKFLOW_ID:
            return technique(context["bounds"])
        end = context["bounds"]["end_ms"]  # MMA: a ground-phase transition candidate
        return technique(context["bounds"], action_classes=["transition"], strike_target=None, significant=None,
                         knockdown=False, outcome_bounds=dict(start_ms=end - 100, end_ms=end))  # fmt: skip
    if node == "technique_validate":
        return dict(candidate_id=context["candidate_id"], verdict="agree", intentional_technique_visible=True,
                    technique=context["proposed"], reason="tap visible")  # fmt: skip
    if node == "ground_rescan":
        return dict(phase_index=context["phase_index"], states=[dict(
            at_ms=context["window"]["start_ms"], position="side_control", top_player_id="a",
            evidence="a on top", geometry=None)])  # fmt: skip
    if node == "strike_census":
        start = context["window"]["start_ms"]
        return dict(window_index=context["window_index"],
                    strikes=[strike(start + 600, knockdown=context["window_index"] == 0),
                             strike(start + 1500, actor="a")])  # fmt: skip
    if node == "knockdown_check":
        return dict(strike_index=context["strike_index"], knocked_down=False,
                    reason="opponent only steps back; both feet stay planted")  # fmt: skip
    if node == "strike_audit":
        return dict(window_index=context["window_index"], added=[],
                    decisions=[dict(census_index=i, verdict="keep", at_ms=None, reason="visible")
                               for i, _ in enumerate(context["census"]["strikes"])])  # fmt: skip
    raise AssertionError(node)


def shifted(value, by):
    """Independent of analysis_time_map: every integer under a ``*_ms`` key moves by ``by``."""
    if isinstance(value, dict):
        return {k: (v + by if k.endswith("_ms") and type(v) is int else shifted(v, by)) for k, v in value.items()}
    if isinstance(value, list):
        return [shifted(v, by) for v in value]
    return value


def times(value, found=None):
    found = [] if found is None else found
    if isinstance(value, dict):
        for k, v in value.items():
            if k.endswith("_ms") and type(v) is int:
                found.append(v)
            times(v, found)
    elif isinstance(value, list):
        for v in value:
            times(v, found)
    return found


class RecordingProvider:
    """The real GeminiProvider, with the core's source-clock request kept for comparison."""

    def __init__(self, provider):
        self.provider, self.name, self.sent = provider, provider.name, {}

    async def generate(self, run, request, execution_id, call_id):
        self.sent[request.initial_prompt] = request
        return await self.provider.generate(run, request, execution_id, call_id)


def projection(result):
    return (
        [(p.bounds.start_ms, p.bounds.end_ms, p.position, p.observation) for p in result.positions.positions],
        [(t.bounds.start_ms, t.bounds.end_ms, t.actor_player_id, tuple(t.action_classes))
         for t in result.techniques.techniques],
    )  # fmt: skip


@pytest.fixture(scope="module")
def raw20(tmp_path_factory):
    path = tmp_path_factory.mktemp("origin20") / "raw.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10", "-f", "lavfi", "-i", "sine=frequency=440",
           "-t", "20", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path))  # fmt: skip
    return path


async def execute(workflow, offset_s, raw20, tmp_path, gemini, resolver, inputs, s3, media_bytes):
    """One whole analysis; returns (projection, offset_ms, source-clock requests by node)."""
    from tests.media_preparation.test_gemini import provider_case  # noqa: PLC0415

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "source.mp4"
    video_offset(raw20, path, offset_s)
    container, video, duration_ms = probe(path)
    run0, _, put = inputs
    _, png, jpeg = media_bytes
    if workflow == WORKFLOW_ID:
        config = inputs[1]
        run = run0
    else:
        config = config_bytes() if workflow == MMA_STRIKES_WORKFLOW_ID else ground_config(
            position_critique=None, technique_critique=None, technique_validate=None,
            strike_audit=None, knockdown_check=None)  # fmt: skip
        run = strikes_run(altered(run0, effective_config=put(config, "config.json")), config, workflow)
    run = altered(
        run,
        source=put(path.read_bytes(), "source.mp4"),
        source_duration_ms=duration_ms,
        scope=dict(start_ms=1000, end_ms=11000),
        athletes=(
            dict(player_id="a", display_name="a", reference_crop=put(png, "a.png")),
            dict(player_id="b", display_name="b", reference_crop=put(jpeg, "b.jpg")),
        ),
    )
    async with provider_case(gemini, resolver, (run, config, put), s3) as (provider, wire, media):
        assert media.time_origin.offset_ms == int((video - container) * 1000)
        offset = media.time_origin.offset_ms
        await uploaded(provider)
        recorder, nodes = RecordingProvider(provider), []

        def play_gemini():
            """The provider end: sees only re-based times, answers on that clock."""
            body = json.loads(wire.calls[-1][2])
            prompt = body["contents"][0]["parts"][-1]["text"]
            window = body["contents"][0]["parts"][-2]["videoMetadata"]
            request = recorder.sent[canonical_json(shifted(json.loads(prompt), offset))]
            observed = json.loads(prompt)["observation_window"]
            assert (observed["start_ms"], observed["end_ms"]) == (
                request.input_window.start_ms - offset, request.input_window.end_ms - offset)
            assert window["startOffset"] == f"{observed['start_ms'] // 1000}.{observed['start_ms'] % 1000:03d}s"
            assert window["endOffset"] == f"{observed['end_ms'] // 1000}.{observed['end_ms'] % 1000:03d}s"
            context = json.loads(request.initial_prompt)["stage_context"]
            nodes.append(request.node_id)
            reply = dict(schema_version=1, **answer(workflow, request.node_id, context))
            text = json.dumps(shifted(reply, -offset))
            wire.result["candidates"][0]["content"]["parts"][0]["text"] = text

        wire.after_generate = play_gemini
        wire.result["modelVersion"] = "fixture-model-r1"
        journal = DiskJournal(tmp_path / "journal.sqlite")
        try:
            result = await execute_pipeline(
                media.run, media.config_bytes, journal=journal, provider=recorder, sink=Sink(),
                guard=lambda: resolver.guard(media.run), clock=lambda: datetime.now(timezone.utc),
                max_parallel_calls=1)  # fmt: skip
        finally:
            journal.close()
        return projection(result), offset, nodes, recorder.sent


WORKFLOW_NODES = {
    WORKFLOW_ID: {"position_scan", "position_critique", "technique_scan", "technique_critique",
                  "technique_analyze", "technique_validate"},
    MMA_STRIKES_WORKFLOW_ID: {"position_scan", "position_critique", "technique_scan", "strike_census",
                              "strike_audit", "knockdown_check"},
    MMA_GROUND_WORKFLOW_ID: {"position_scan", "ground_rescan", "technique_scan", "technique_analyze",
                             "strike_census"},
}  # fmt: skip


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", WORKFLOW_NODES)
@pytest.mark.parametrize("offset_s", [0.041, 0.5])
async def test_every_workflow_node_maps_times_at_the_provider_boundary(
    workflow, offset_s, raw20, tmp_path, gemini, resolver, inputs, s3, media_bytes
):
    mapped, offset, nodes, sent = await execute(
        workflow, offset_s, raw20, tmp_path / "offset", gemini, resolver, inputs, s3, media_bytes
    )
    plain, zero, _, _ = await execute(
        workflow, 0, raw20, tmp_path / "plain", gemini, resolver, inputs, s3, media_bytes
    )
    assert offset == int(offset_s * 1000) and zero == 0
    assert set(nodes) == WORKFLOW_NODES[workflow], "the scripted run must reach every node of the workflow"
    assert all(times(json.loads(request.initial_prompt)) for request in sent.values())
    assert mapped == plain
    assert mapped[0] and (mapped[1] or workflow == MMA_GROUND_WORKFLOW_ID)


def test_every_response_schema_time_field_is_mapped():
    """A time-bearing property added to any node schema must be added to the map."""
    from bjj_analysis_core.packs import PACKS
    from bjj_analysis_core.requests import prompt_bundle

    from service.analysis_time_map import TIME_FIELDS

    def keys(value, found):
        if isinstance(value, dict):
            for k, v in value.items():
                if k.endswith("_ms"):
                    found.add(k)
                keys(v, found)
        elif isinstance(value, list):
            for v in value:
                keys(v, found)
        return found

    for name, pack in PACKS.items():
        bundle = prompt_bundle(pack)
        for node, schema in bundle["schemas"].items():
            assert keys(schema, set()) <= TIME_FIELDS, (name, node)
