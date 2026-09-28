import hashlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from bjj_analysis_core import CORE_VERSION, WORKFLOW_ID
from bjj_analysis_core.outputs import CandidateConfigV1
from bjj_analysis_core.requests import CoreConfigV1, NodeConfigV1, prompt_bundle_sha256
from shared_lib.models.analysis_contracts import AnalysisRunSpecV1
from shared_lib.models.analysis_run_record import RegisteredAIRunV1


@pytest.fixture
def configuration():
    node = NodeConfigV1(model="fixture-model", media_resolution="high", fps=10)
    config = CoreConfigV1(
        prompt_bundle_sha256=prompt_bundle_sha256(),
        position_scan=node.model_copy(update={"fps": 1}),
        position_critique=node,
        technique_scan=node.model_copy(update={"fps": 1}),
        technique_critique=node,
        technique_analyze=node,
        technique_validate=node,
        candidates=CandidateConfigV1(),
    )
    return config.model_dump_json().encode()


@pytest.fixture
def run(configuration):
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    ref = dict(bucket="fixture-private", key="input", sha256="a" * 64)
    spec = AnalysisRunSpecV1(
        run_id=uuid4(),
        analysis_id=uuid4(),
        base_edit_version=2,
        scope_id=uuid4(),
        owner=dict(kind="account", principal_id="owner"),
        video_id="video",
        source=ref,
        source_duration_ms=20000,
        scope=dict(start_ms=1000, end_ms=11000),
        athletes=tuple(
            dict(player_id=p, display_name=p, reference_crop=ref) for p in ("a", "b")
        ),
        workflow_id=WORKFLOW_ID,
        core_version=CORE_VERSION,
        effective_config={**ref, "sha256": hashlib.sha256(configuration).hexdigest()},
        admission_id="admission",
        usage_policy_version="fixture-v1",
        admitted_at=now,
    )
    return RegisteredAIRunV1(
        spec=spec,
        baseline=ref,
        baseline_revision_id=uuid4(),
        baseline_revision_sha256="b" * 64,
        registered_at=now + timedelta(seconds=1),
    )


@pytest.fixture
def completed_at(run):
    return run.registered_at + timedelta(seconds=20)


def state(position="standing", roles=("neutral", "neutral")):
    return dict(
        observation="observed",
        position=position,
        roles=roles,
        notes="",
        evidence="visible torso and leg relationship",
        uncertainty_note=None,
        interruption_note=None,
    )


def technique(**changes):
    value = dict(
        schema_version=1,
        bounds=dict(start_ms=2000, end_ms=6000),
        action_classes=["submission_choke"],
        actor_player_id="a",
        actor_sentinel=None,
        outcome="successful",
        technique_shortlist=None,
        technique_guess="rear naked choke",
        notes="",
        evidence="visible intentional choke attempt",
        outcome_evidence="defender taps",
        outcome_bounds=dict(start_ms=5500, end_ms=5900),
    )
    value.update(changes)
    return value


def candidate(**changes):
    value = dict(
        candidate_id="scan-1",
        description="visible intentional choke attempt",
        observation_window=dict(start_ms=1000, end_ms=11000),
        proposed=technique(),
    )
    value.update(changes)
    return value


def decision(**changes):
    value = dict(
        schema_version=1,
        candidate_id="scan-1",
        verdict="agree",
        intentional_technique_visible=True,
        technique=technique(),
        reason="video confirms tap",
    )
    value.update(changes)
    return value
