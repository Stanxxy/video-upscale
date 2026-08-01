"""R4 production analysis-settings contract and admission tests."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from service import routes as routes_mod
from service.analysis_settings import (
    AnalysisSettingsValidationError,
    apply_effective_analysis_config,
    get_analysis_settings_capabilities,
    resolve_analysis_settings,
)
from service.models import AdmittedTrackRequest, EffectiveAnalysisConfig, TrackRequest
from service.pipelines.registry import get_default
from service.routes.recovery import recover_interrupted_job


@pytest.fixture()
def scheduled_jobs(monkeypatch):
    scheduled: list[tuple[str, AdmittedTrackRequest]] = []

    def _schedule(job_id: str, request: AdmittedTrackRequest) -> None:
        scheduled.append((job_id, request))

    async def _cleanup() -> None:
        return None

    monkeypatch.setattr(routes_mod, "_schedule_job", _schedule)
    monkeypatch.setattr(routes_mod, "_cleanup_orphaned_tasks", _cleanup)
    return scheduled


def test_capability_manifest_is_the_approved_production_contract():
    manifest = get_analysis_settings_capabilities()

    assert manifest.schema_version == 1
    assert manifest.qualified_models == ("gemini-3.5-flash",)
    assert manifest.media_resolutions == ("low", "medium", "high")
    assert manifest.analysis_fps == (1, 10)
    assert manifest.thinking_levels == ("off", "low", "medium", "high")
    assert manifest.recommended.model == "gemini-3.5-flash"
    assert manifest.recommended.media_resolution == "low"
    assert manifest.recommended.analysis_fps == 10
    assert manifest.recommended.thinking == "high"
    assert manifest.mapping.analysis_fps == ("taxonomy", "actor")


def test_omitted_overrides_resolve_to_complete_immutable_stage_snapshot():
    admitted = resolve_analysis_settings(TrackRequest(bucket="b", key="match.mp4"))

    assert isinstance(admitted, AdmittedTrackRequest)
    assert admitted.requested_analysis_settings == {}
    assert admitted.capability_schema_version == 1
    effective = admitted.effective_analysis_config
    for stage_name in ("scan", "critique", "taxonomy", "actor"):
        stage = getattr(effective, stage_name)
        assert stage.model == "gemini-3.5-flash"
        assert stage.media_resolution == "low"
        assert stage.thinking == "high"
    assert effective.scan.fps is None
    assert effective.critique.fps is None
    assert effective.taxonomy.fps == 10
    assert effective.actor.fps == 10

    with pytest.raises(Exception):
        effective.taxonomy.fps = 1


def test_effective_snapshot_rejects_taxonomy_actor_fps_divergence():
    payload = resolve_analysis_settings(
        TrackRequest(bucket="b", key="match.mp4"),
    ).effective_analysis_config.model_dump()
    payload["actor"]["fps"] = 1

    with pytest.raises(
        ValidationError,
        match="taxonomy and actor analysis_fps must be identical",
    ):
        EffectiveAnalysisConfig.model_validate(payload)


def test_one_override_is_retained_raw_and_changes_only_its_mapped_axis():
    admitted = resolve_analysis_settings(
        TrackRequest(
            bucket="b",
            key="match.mp4",
            analysis_media_resolution="medium",
        ),
    )

    assert admitted.requested_analysis_settings == {
        "analysis_media_resolution": "medium",
    }
    effective = admitted.effective_analysis_config
    assert {getattr(effective, stage).media_resolution for stage in (
        "scan", "critique", "taxonomy", "actor"
    )} == {"medium"}
    assert effective.taxonomy.fps == 10
    assert effective.actor.fps == 10


@pytest.mark.parametrize(
    ("field", "value", "allowed"),
    [
        ("analysis_fps", 15, "1, 10"),
        ("analysis_model", "gemini-3.1-flash-lite", "gemini-3.5-flash"),
        ("analysis_thinking", "maximum", "off, low, medium, high"),
    ],
)
def test_off_manifest_values_fail_closed_with_allowed_values(field, value, allowed):
    request = TrackRequest.model_construct(bucket="b", key="match.mp4", **{field: value})

    with pytest.raises(AnalysisSettingsValidationError) as exc:
        resolve_analysis_settings(request)

    assert field in str(exc.value)
    assert allowed in str(exc.value)


def test_effective_snapshot_maps_to_a_deep_copied_pipeline():
    original = get_default("highlight-scan-critique-analyze")
    admitted = resolve_analysis_settings(
        TrackRequest(bucket="b", key="match.mp4", analysis_fps=1),
    )

    configured = apply_effective_analysis_config(
        original,
        admitted.effective_analysis_config,
    )
    stages = {stage.id: stage.config for stage in configured.stages}
    assert stages["highlight_scan"]["model"] == "gemini-3.5-flash"
    assert stages["highlight_scan"]["media_resolution"] == "low"
    assert stages["highlight_scan"]["thinking"] == "high"
    assert "fps" not in stages["highlight_scan"]
    assert stages["highlight_critique"]["model"] == "gemini-3.5-flash"
    assert "fps" not in stages["highlight_critique"]
    assert stages["highlight_analyze"]["fps"] == 1
    assert stages["highlight_analyze"]["model"] == "gemini-3.5-flash"
    assert stages["highlight_analyze"]["actor"]["model"] == "gemini-3.5-flash"
    assert get_default("highlight-scan-critique-analyze") == original


@pytest.mark.asyncio
async def test_track_admission_persists_and_schedules_the_resolved_request(
    service_client,
    service_components,
    scheduled_jobs,
):
    _, _, jobs_store = service_components

    response = await service_client.post(
        "/track",
        json={"bucket": "b", "key": "match.mp4", "analysis_fps": 1},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    persisted = json.loads(await jobs_store.get_request(job_id))
    assert persisted["requested_analysis_settings"] == {"analysis_fps": 1}
    assert persisted["effective_analysis_config"]["taxonomy"]["fps"] == 1
    assert persisted["effective_analysis_config"]["actor"]["fps"] == 1
    assert isinstance(scheduled_jobs[0][1], AdmittedTrackRequest)


@pytest.mark.asyncio
async def test_invalid_setting_is_rejected_before_any_job_state_is_created(
    service_client,
    service_components,
    scheduled_jobs,
):
    _, job_store, jobs_store = service_components

    response = await service_client.post(
        "/track",
        json={"bucket": "b", "key": "match.mp4", "analysis_fps": 15},
    )

    assert response.status_code == 400
    assert "1, 10" in response.json()["detail"]
    assert jobs_store._lifecycles == {}
    assert jobs_store._requests == {}
    assert job_store._jobs == {}
    assert scheduled_jobs == []


@pytest.mark.asyncio
async def test_capability_endpoint_serves_the_same_manifest(service_client):
    response = await service_client.get("/analysis-settings/capabilities")

    assert response.status_code == 200
    assert response.json() == get_analysis_settings_capabilities().model_dump(mode="json")


@pytest.mark.asyncio
async def test_highlight_recovery_uses_persisted_effective_snapshot_after_default_change(
    monkeypatch,
    service_components,
    scheduled_jobs,
):
    from service import analysis_settings as settings_module

    _, _, jobs_store = service_components
    admitted = resolve_analysis_settings(TrackRequest(bucket="b", key="match.mp4"))
    await jobs_store.save_request("job-1", admitted.model_dump_json())
    changed_recommendation = settings_module._CAPABILITIES.recommended.model_copy(
        update={"media_resolution": "high", "analysis_fps": 1, "thinking": "low"},
    )
    monkeypatch.setattr(
        settings_module,
        "_CAPABILITIES",
        settings_module._CAPABILITIES.model_copy(
            update={"recommended": changed_recommendation},
        ),
    )

    await recover_interrupted_job(
        {"job_id": "job-1", "pipeline_kind": "highlight_v2"},
    )

    recovered = scheduled_jobs[0][1]
    assert recovered.effective_analysis_config.taxonomy.media_resolution == "low"
    assert recovered.effective_analysis_config.taxonomy.fps == 10
    assert recovered.effective_analysis_config.taxonomy.thinking == "high"


@pytest.mark.asyncio
async def test_interrupted_pre_r4_request_fails_visibly(
    service_components,
    scheduled_jobs,
):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle(
        "job-pre-r4",
        "video-1",
        "user-1",
        pipeline_kind="highlight_v2",
    )
    await jobs_store.save_request(
        "job-pre-r4",
        TrackRequest(bucket="b", key="match.mp4").model_dump_json(),
    )

    with pytest.raises(RuntimeError, match="valid effective analysis settings snapshot"):
        await recover_interrupted_job(
            {"job_id": "job-pre-r4", "pipeline_kind": "highlight_v2"},
        )

    lifecycle = jobs_store._lifecycles["job-pre-r4"]
    assert lifecycle["job_state"] == "FAILED"
    assert "valid effective analysis settings snapshot" in lifecycle["error_message"]
    assert scheduled_jobs == []
