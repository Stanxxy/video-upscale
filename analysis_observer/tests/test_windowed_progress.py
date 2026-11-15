"""Progress of the windowed strike workflow from a real DAG journal (no Keyspaces)."""

import asyncio
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from bjj_analysis_core import CORE_VERSION, WORKFLOW_ID
from bjj_analysis_core.executor import execute_pipeline
from bjj_analysis_observer.progress import read_progress
from shared_lib.models.analysis_contracts import AnalysisRunSpecV1
from shared_lib.models.analysis_progress import WINDOWED_NODE_ORDER, DagObservationV1
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis_core" / "tests"))
from journal_fixture import DiskJournal  # noqa: E402
from test_windowed_strikes import Provider, Sink, config_bytes, strikes_run  # noqa: E402


class Journal(DiskJournal):
    """The checkpoint journal plus the run's admitted configuration, as the reader has."""

    def __init__(self, path, run, config):
        super().__init__(path)
        self.run, self._config = run, config

    async def config_bytes(self):
        return self._config


def windowed_run(config):
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    ref = dict(bucket="fixture-private", key="input", sha256="a" * 64)
    spec = AnalysisRunSpecV1(
        run_id=uuid4(), analysis_id=uuid4(), base_edit_version=2, scope_id=uuid4(),
        owner=dict(kind="account", principal_id="owner"), video_id="video", source=ref,
        source_duration_ms=20000, scope=dict(start_ms=1000, end_ms=11000),
        athletes=tuple(dict(player_id=p, display_name=p, reference_crop=ref) for p in ("a", "b")),
        workflow_id=WORKFLOW_ID, core_version=CORE_VERSION,
        effective_config={**ref, "sha256": hashlib.sha256(config).hexdigest()},
        admission_id="admission", usage_policy_version="fixture-v1", admitted_at=now,
    )
    base = RegisteredAIRunV1(spec=spec, baseline=ref, baseline_revision_id=uuid4(),
                             baseline_revision_sha256="b" * 64, registered_at=now + timedelta(seconds=1))
    return strikes_run(base, config)


async def guard():
    return None


def execute(run, config, journal, provider):
    at = [run.registered_at]

    def clock():
        at[0] += timedelta(milliseconds=1)
        return at[0]

    provider.clock = clock
    return execute_pipeline(run, config, journal=journal, provider=provider, sink=Sink(), guard=guard,
                            clock=clock, max_parallel_calls=4)


def public(view):
    """The backend's public projection: the shared contract must accept it."""
    return DagObservationV1(
        observed_at=view.observed_at,
        nodes=[dict(node_id=n.node_id, recorded_status=n.recorded_status, completed_items=n.completed_items,
                    total_items=n.total_items,
                    executions=[dict(p.model_dump(), error_code=None if p.error_code is None else "EXECUTION_FAILED")
                                for p in n.checkpoints])
               for n in view.nodes],
    )


def test_unstarted_windowed_run_lists_the_strike_steps_with_their_window_count(tmp_path):
    config = config_bytes()
    run = windowed_run(config)
    journal = Journal(tmp_path / "j.sqlite", run, config)
    view = asyncio.run(read_progress(journal, guard=guard))
    rows = {n.node_id: n for n in view.nodes}
    assert tuple(rows) == WINDOWED_NODE_ORDER
    assert all(n.recorded_status == "not_started" for n in view.nodes)
    public(view)
    journal.close()


def test_completed_windowed_run_reports_census_audit_merge_and_knockdown_check(tmp_path):
    config = config_bytes()
    run = windowed_run(config)
    journal = Journal(tmp_path / "j.sqlite", run, config)
    asyncio.run(execute(run, config, journal, Provider(None)))
    view = asyncio.run(read_progress(journal, guard=guard))
    rows = {n.node_id: n for n in view.nodes}
    # Scope 1000-11000 in 5 s windows with 1 s overlap -> 3 windows; one flagged knockdown.
    for node, count in (("strike_census", 3), ("strike_audit", 3), ("knockdown_check", 1),
                        ("strikes_merge", 1), ("knockdowns_apply", 1)):
        assert (rows[node].recorded_status, rows[node].completed_items, rows[node].total_items) == (
            "succeeded", count, count), node
    assert rows["techniques_publish"].recorded_status == "succeeded"  # dag_complete is the host's
    public(view)
    journal.close()


class FailingAudit(Provider):
    async def generate(self, run, request, execution_id, call_id):
        if request.node_id == "strike_audit":
            raise RuntimeError("fixture provider outage")
        return await super().generate(run, request, execution_id, call_id)


def test_a_failed_audit_shows_where_the_strike_branch_stopped(tmp_path):
    config = config_bytes()
    run = windowed_run(config)
    journal = Journal(tmp_path / "j.sqlite", run, config)
    with pytest.raises(Exception):
        asyncio.run(execute(run, config, journal, FailingAudit(None)))
    view = asyncio.run(read_progress(journal, guard=guard))
    rows = {n.node_id: n for n in view.nodes}
    assert (rows["strike_census"].recorded_status, rows["strike_census"].completed_items) == ("succeeded", 3)
    assert rows["strike_audit"].recorded_status == "failed"
    assert rows["strike_audit"].total_items == 3 and rows["strike_audit"].completed_items == 0
    # The merge never ran, so the knockdown checks have no topology yet.
    assert rows["strikes_merge"].recorded_status == "not_started"
    assert (rows["knockdown_check"].recorded_status, rows["knockdown_check"].total_items) == ("not_started", None)
    public(view)
    journal.close()


def test_a_run_without_audit_or_knockdown_check_reports_them_as_empty(tmp_path):
    config = config_bytes(audit=False, knockdown_check=None)
    run = windowed_run(config)
    journal = Journal(tmp_path / "j.sqlite", run, config)
    asyncio.run(execute(run, config, journal, Provider(None)))
    rows = {n.node_id: n for n in asyncio.run(read_progress(journal, guard=guard)).nodes}
    assert rows["strike_census"].recorded_status == "succeeded"
    for node in ("strike_audit", "knockdown_check", "knockdowns_apply"):
        assert rows[node].recorded_status == "not_started", node
    journal.close()
