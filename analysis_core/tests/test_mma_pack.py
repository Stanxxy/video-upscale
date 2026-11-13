"""x-mma: the MMA pack runs the same DAG with mma.v1 vocabulary; BJJ is unchanged."""

import hashlib
import json

import pytest
from bjj_analysis_core import MMA_WORKFLOW_ID, WORKFLOW_ID
from bjj_analysis_core.normalization import normalize_positions, normalize_techniques
from bjj_analysis_core.packs import BJJ, MMA, pack_for
from bjj_analysis_core.requests import (
    CoreConfigV1,
    load_config,
    pack_for_bundle,
    parse_response,
    prepare_call,
    prompt_bundle,
    prompt_bundle_sha256,
)
from conftest import candidate, decision, state, technique
from pydantic import ValidationError
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

# The value pinned in config/analysis/gemini38.json for every admitted BJJ run.
BJJ_BUNDLE = "ce925cb824762900dc29544f3f5892afd9f4f9d8636a6a875c3539f6653db634"


def mma_run(run, configuration, *, taxonomy_id="mma.v1", workflow_id=MMA_WORKFLOW_ID):
    config = CoreConfigV1.model_validate_json(configuration)
    config_bytes = config.model_copy(
        update={"prompt_bundle_sha256": prompt_bundle_sha256(MMA)}
    ).model_dump_json().encode()
    spec = run.spec.model_copy(
        update={
            "workflow_id": workflow_id,
            "taxonomy_id": taxonomy_id,
            "effective_config": run.spec.effective_config.model_copy(
                update={"sha256": hashlib.sha256(config_bytes).hexdigest()}
            ),
        }
    )
    return RegisteredAIRunV1.model_validate({**run.model_dump(), "spec": spec.model_dump()}), config_bytes


def mma_state(position="striking_range", roles=("neutral", "neutral"), **changes):
    value = {**state(position, roles), "interruption_kind": None, "cage_contact": False}
    value.update(changes)
    return value


def strike(**changes):
    value = {
        **technique(),
        "action_classes": ["strike_punch"],
        "technique_shortlist": "hook",
        "technique_guess": "left hook",
        "evidence": "visible left hook to the jaw",
        "outcome_evidence": "head snaps back on impact",
        "strike_target": "head",
        "significant": True,
        "knockdown": False,
    }
    value.update(changes)
    return value


def test_bjj_bundle_is_byte_identical_to_the_admitted_preset():
    assert prompt_bundle_sha256() == prompt_bundle_sha256(BJJ) == BJJ_BUNDLE
    assert "strike_classes" not in prompt_bundle()


def test_mma_bundle_is_its_own_and_names_its_pack():
    mma = prompt_bundle(MMA)
    assert prompt_bundle_sha256(MMA) != BJJ_BUNDLE
    assert pack_for_bundle(prompt_bundle_sha256(MMA)) is MMA
    assert pack_for_bundle(BJJ_BUNDLE) is BJJ
    assert "clinch" in mma["positions"]["positions"] and "standing" not in mma["positions"]["positions"]
    assert "strike_punch" in mma["action_classes"] and "guard_pull" not in mma["action_classes"]
    assert all("BJJ" not in text and "Jiu-Jitsu" not in text for text in mma["prompts"].values())
    with pytest.raises(ValueError):
        pack_for_bundle("0" * 64)


def test_runs_select_their_pack_and_reject_a_taxonomy_mismatch(run, configuration):
    assert pack_for(run) is BJJ
    mma, config_bytes = mma_run(run, configuration)
    assert pack_for(mma) is MMA
    load_config(mma, config_bytes)
    with pytest.raises(ValueError):
        load_config(mma, configuration)  # BJJ bundle on the MMA workflow
    mismatched, config_bytes = mma_run(run, configuration, taxonomy_id="bjj.v1")
    with pytest.raises(ValueError):
        load_config(mismatched, config_bytes)


def test_mma_prompt_carries_mma_vocabulary_and_strike_fields(run, configuration):
    mma, config_bytes = mma_run(run, configuration)
    call = prepare_call(mma, config_bytes, "technique_scan", input_window=mma.spec.scope, context={})
    initial = json.loads(call.initial_prompt)
    assert initial["strike_targets"] == ["head", "body", "leg"]
    assert "strike_kick" in initial["strike_classes"]
    assert call.system_prompt == MMA.prompt("technique_scan")
    bjj_call = prepare_call(run, configuration, "technique_scan", input_window=run.spec.scope, context={})
    assert "strike_targets" not in json.loads(bjj_call.initial_prompt)


def _analyze_call(mma, config_bytes):
    target = dict(
        candidate_id="scan-1",
        description="left hook",
        bounds=dict(start_ms=2000, end_ms=6000),
        end_status="outcome_visible",
        temporal_reason="hook lands",
    )
    return prepare_call(mma, config_bytes, "technique_analyze", input_window=target["bounds"], context=target)


def test_parse_rejects_incomplete_or_misplaced_strike_fields(run, configuration):
    mma, config_bytes = mma_run(run, configuration)
    call = _analyze_call(mma, config_bytes)
    assert parse_response(call, json.dumps(strike())).strike_target.value == "head"
    for bad in (
        strike(strike_target=None),
        strike(significant=None),
        strike(action_classes=["takedown_attempt"], technique_shortlist="double_leg"),
        {**technique(), "strike_target": None, "significant": None, "knockdown": True},
    ):
        with pytest.raises(ValidationError):
            parse_response(call, json.dumps(bad))


def test_parse_requires_interruption_kind_exactly_for_inactive_states(run, configuration):
    mma, config_bytes = mma_run(run, configuration)
    call = prepare_call(mma, config_bytes, "position_scan", input_window=mma.spec.scope, context={})
    inactive = mma_state(
        observation="inactive", position=None, roles=("not_applicable", "not_applicable"),
        evidence="corners in", interruption_note="end of round 1", interruption_kind="round_break",
        cage_contact=None,
    )
    ok = dict(initial_state=mma_state(), transitions=[dict(at_ms=5000, state_after=inactive)])
    assert parse_response(call, json.dumps(ok)).transitions[0].state_after.interruption_kind.value == "round_break"
    for bad in (
        {**inactive, "interruption_kind": None},
        {**inactive, "cage_contact": True},
        mma_state(interruption_kind="replay"),
    ):
        with pytest.raises(ValidationError):
            parse_response(call, json.dumps(dict(initial_state=mma_state(), transitions=[dict(at_ms=5000, state_after=bad)])))


def test_mma_positions_normalize_into_mma_items(run, configuration, completed_at):
    mma, _ = mma_run(run, configuration)
    clinch = mma_state("clinch", ("cage_pinner", "cage_pinned"), cage_contact=True)
    replay = mma_state(
        observation="inactive", position=None, roles=("not_applicable", "not_applicable"),
        interruption_note="broadcast replay of the takedown", interruption_kind="replay",
        cage_contact=None,
    )
    scan = dict(
        initial_state=mma_state(),
        transitions=[dict(at_ms=4000, state_after=clinch), dict(at_ms=8000, state_after=replay)],
    )
    critiques = [
        dict(transition_index=i, confirmed=True, corrected_at_ms=t,
             state_after=scan["transitions"][i]["state_after"], reason="visible")
        for i, t in enumerate((4000, 8000))
    ]
    result = normalize_positions(mma, scan, critiques, completed_at=completed_at)
    assert result.taxonomy_id == "mma.v1"
    assert [p.position and p.position.value for p in result.positions] == ["striking_range", "clinch", None]
    assert result.positions[1].cage_contact is True
    assert result.positions[2].interruption_kind.value == "replay"
    assert all(p.taxonomy_id == "mma.v1" for p in result.positions)


def test_mma_strikes_normalize_into_mma_techniques(run, configuration, completed_at):
    mma, _ = mma_run(run, configuration)
    result = normalize_techniques(
        mma, [candidate(proposed=strike())], [decision(technique=strike())], completed_at=completed_at
    )
    (event,) = result.techniques
    assert (event.taxonomy_id, event.strike_target.value, event.significant) == ("mma.v1", "head", True)


def test_bjj_normalization_output_has_no_sport_fields(run, completed_at):
    result = normalize_techniques(run, [candidate()], [decision()], completed_at=completed_at)
    dumped = result.model_dump(mode="json")
    assert "taxonomy_id" not in dumped and "strike_target" not in dumped["techniques"][0]
    assert run.spec.workflow_id == WORKFLOW_ID
