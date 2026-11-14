"""VLM Studio production-core host (service.qa_core): journal, config, report, crops."""
import asyncio
import json
import subprocess
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from bjj_analysis_core import CORE_VERSION, MMA_WORKFLOW_ID
from bjj_analysis_core.executor import execute_pipeline
from bjj_analysis_core.packs import MMA
from bjj_analysis_core.execution import ProviderReplyV1
from bjj_analysis_core.requests import prompt_bundle_sha256
from shared_lib.models.analysis_contracts import CallUsageV1
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

from service import qa_core


def _run(config_bytes):
    import hashlib

    now = datetime(2026, 11, 14, tzinfo=timezone.utc)
    ref = dict(bucket="x-mma-vlm-core", key="k", sha256="a" * 64)
    return RegisteredAIRunV1.model_validate(dict(
        spec=dict(
            run_id=uuid4(), analysis_id=uuid4(), base_edit_version=0, scope_id=uuid4(),
            owner=dict(kind="account", principal_id="vlm-studio"), video_id="1ACTKcuQZyQ", source=ref,
            source_duration_ms=60_000, scope=dict(start_ms=0, end_ms=30_000),
            athletes=(dict(player_id="fighter-a", display_name="Orange", reference_crop=ref),
                      dict(player_id="fighter-b", display_name="Black", reference_crop=ref)),
            workflow_id=MMA_WORKFLOW_ID, taxonomy_id="mma.v1", core_version=CORE_VERSION,
            effective_config={**ref, "sha256": hashlib.sha256(config_bytes).hexdigest()},
            admission_id="vlm-studio", usage_policy_version="vlm-studio-v1", admitted_at=now,
        ),
        baseline=ref, baseline_revision_id=uuid4(), baseline_revision_sha256="b" * 64,
        registered_at=now + timedelta(seconds=1),
    ))


class EmptyFightProvider:
    """One live striking span, no transitions, no attacks: every fan-out is explicitly empty."""

    name = "fixture"

    def __init__(self, clock):
        self.clock, self.calls = clock, []

    async def generate(self, run, request, execution_id, call_id):
        self.calls.append(request.node_id)
        if request.node_id == "position_scan":
            data = dict(initial_state=dict(observation="observed", position="striking_range", roles=("neutral", "neutral"),
                                           notes="", evidence="both standing at range", uncertainty_note=None,
                                           interruption_note=None, interruption_kind=None, cage_contact=None),
                        transitions=[])
        else:
            data = dict(events=[])
        usage = CallUsageV1(run_id=run.spec.run_id, execution_id=execution_id, call_id=call_id, recorded_at=self.clock(),
                            kind="model", provider=self.name, model=request.config.model, usage_status="complete",
                            input_tokens=1_000_000, output_tokens=100_000, thinking_tokens=100_000, cost_status="unknown")
        return ProviderReplyV1(text=json.dumps(data), usage=usage)


class Sink:
    def __init__(self):
        self.outputs = {}

    async def publish(self, run, component, revision):
        self.outputs[component] = revision


class StubS3:
    def head_bucket(self, Bucket):
        return {}


def test_mma_config_is_the_production_preset_and_overrides_only_named_fields():
    config = qa_core.build_config("mma", {})
    assert config.prompt_bundle_sha256 == prompt_bundle_sha256(MMA)
    assert (config.technique_scan.fps, config.position_scan.fps, config.technique_analyze.fps) == (5, 1, 10)
    changed = qa_core.build_config("mma", {"technique_scan": qa_core.NodeOverride(fps=10, media_resolution="medium")})
    assert (changed.technique_scan.fps, changed.technique_scan.media_resolution, changed.technique_scan.model) == (10, "medium", "gemini-3.8-flash")
    assert changed.position_scan == config.position_scan
    with pytest.raises(ValueError, match="unknown node"):
        qa_core.build_config("mma", {"strike_census": qa_core.NodeOverride(fps=10)})


def test_estimate_counts_scan_frames_and_prices_only_listed_models():
    config = qa_core.build_config("mma", {})
    scans = qa_core.estimate(config, 0, 60_000)
    assert scans["position_scan"]["frames"] == 60 and scans["technique_scan"]["frames"] == 300
    assert scans["technique_scan"]["input_tokens"] == 300 * 70
    unknown = qa_core.build_config("mma", {"technique_scan": qa_core.NodeOverride(model="some-other-model")})
    assert qa_core.estimate(unknown, 0, 60_000)["technique_scan"]["usd"] is None


def test_journal_is_compare_and_swap_with_full_history(tmp_path):
    config_bytes = qa_core.build_config("mma", {}).model_dump_json().encode()
    run = _run(config_bytes)
    clock = lambda: run.registered_at + timedelta(seconds=5)
    journal = qa_core.SqliteJournal(tmp_path / "j.sqlite", run)
    provider = EmptyFightProvider(clock)
    sink = Sink()

    async def guard():
        return None

    result = asyncio.run(execute_pipeline(run, config_bytes, journal=journal, provider=provider, sink=sink, guard=guard, clock=clock))
    assert provider.calls == ["position_scan", "technique_scan"]
    assert [p.position for p in result.positions.positions] == ["striking_range"] and result.techniques.techniques == ()
    history = journal.history()
    head = history[-1]
    # A stale expected revision is refused and leaves the head unchanged.
    assert asyncio.run(journal.compare_and_swap(head.revision + 5, head)) is False
    assert asyncio.run(journal.read(head.step_id)) == head
    # Resume over the same journal replays from checkpoints: no new provider call.
    again = EmptyFightProvider(clock)
    asyncio.run(execute_pipeline(run, config_bytes, journal=journal, provider=again, sink=Sink(), guard=guard, clock=clock))
    assert again.calls == []
    journal.close()


def test_status_reports_progress_tokens_and_published_rate_cost(tmp_path):
    settings = qa_core.Settings(root=tmp_path, media_dir=tmp_path / "media", bucket="x-mma-vlm-core", api_key="k")
    core = qa_core.CoreRuns(settings, client=StubS3())
    config = qa_core.build_config("mma", {})
    config_bytes = config.model_dump_json().encode()
    run = _run(config_bytes)
    folder = tmp_path / "runs" / str(run.spec.run_id)
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(config_bytes)
    request = dict(youtube_id="1ACTKcuQZyQ", sport="mma", start_ms=0, end_ms=30_000, label=None, max_parallel_calls=4, overrides={},
                   fighters=[dict(name="Orange", frame_ms=0, box=[0, 0, 0.5, 1]), dict(name="Black", frame_ms=0, box=[0.5, 0, 0.5, 1])])
    core.jobs[str(run.spec.run_id)] = qa_core.Job(run_id=str(run.spec.run_id), folder=folder, request=request, state="succeeded")
    clock = lambda: run.registered_at + timedelta(seconds=5)
    journal = qa_core.SqliteJournal(folder / "journal.sqlite", run)

    async def guard():
        return None

    asyncio.run(execute_pipeline(run, config_bytes, journal=journal, provider=EmptyFightProvider(clock), sink=Sink(), guard=guard, clock=clock))
    journal.close()
    status = core.status(str(run.spec.run_id))
    nodes = {n["node_id"]: n for n in status["nodes"]}
    assert nodes["position_scan"]["calls"] == 1 and nodes["position_scan"]["succeeded"] == 1
    # 1M input at $0.75 + (100k output + 100k thinking) at $3.75 per million (promotional Standard rate).
    assert nodes["technique_scan"]["usd"] == pytest.approx(0.75 + 0.75)
    assert status["total_usd"] == pytest.approx(3.0) and status["unknown_cost_calls"] == 0
    assert nodes["positions_normalize"]["calls"] == 0


def test_crop_is_one_png_frame_inside_the_box_and_rejects_boxes_outside_it(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10", "-t", "2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True, timeout=30)
    png = qa_core.crop_png(video, 1000, (0.25, 0.5, 0.5, 0.5))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json", "-"],
                           input=png, capture_output=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (160, 120)
    assert qa_core.probe_duration_ms(video) == 2000
    with pytest.raises(ValueError, match="inside the frame"):
        qa_core.crop_png(video, 1000, (0.8, 0, 0.5, 1))


def test_media_path_accepts_only_youtube_ids(tmp_path):
    settings = qa_core.Settings(root=tmp_path, media_dir=tmp_path, bucket="x-mma-vlm-core", api_key="k")
    assert qa_core.media_path(settings, "1ACTKcuQZyQ", download=False) == tmp_path / "1ACTKcuQZyQ.mp4"
    with pytest.raises(ValueError):
        qa_core.media_path(settings, "../../etc/pa", download=False)
