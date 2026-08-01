"""Engine-owned production analysis-settings manifest and admission resolver."""
from __future__ import annotations

from service.models import (
    AdmittedTrackRequest,
    AnalysisSettingsCapabilities,
    AnalysisSettingsMapping,
    AnalysisSettingsRecommendation,
    EffectiveAnalysisConfig,
    EffectiveAnalysisStageConfig,
    TrackRequest,
)
from service.pipelines.models import PipelineDef


class AnalysisSettingsValidationError(ValueError):
    """A requested production setting is outside the qualified manifest."""


_CAPABILITIES = AnalysisSettingsCapabilities(
    schema_version=1,
    qualified_models=("gemini-3.5-flash",),
    media_resolutions=("low", "medium", "high"),
    analysis_fps=(1, 10),
    thinking_levels=("off", "low", "medium", "high"),
    recommended=AnalysisSettingsRecommendation(
        model="gemini-3.5-flash",
        media_resolution="low",
        analysis_fps=10,
        thinking="high",
    ),
    mapping=AnalysisSettingsMapping(
        model=("scan", "critique", "taxonomy", "actor"),
        media_resolution=("scan", "critique", "taxonomy", "actor"),
        thinking=("scan", "critique", "taxonomy", "actor"),
        analysis_fps=("taxonomy", "actor"),
    ),
)


def get_analysis_settings_capabilities() -> AnalysisSettingsCapabilities:
    return _CAPABILITIES


def _validate(field: str, value: object, allowed: tuple[object, ...]) -> None:
    if value not in allowed:
        rendered = ", ".join(str(item) for item in allowed)
        raise AnalysisSettingsValidationError(
            f"Unsupported {field}={value!r}; allowed values: {rendered}",
        )


def resolve_analysis_settings(request: TrackRequest) -> AdmittedTrackRequest:
    """Validate overrides and freeze the exact per-stage configuration."""
    manifest = get_analysis_settings_capabilities()
    override_values = {
        "analysis_model": request.analysis_model,
        "analysis_media_resolution": request.analysis_media_resolution,
        "analysis_fps": request.analysis_fps,
        "analysis_thinking": request.analysis_thinking,
    }
    allowed = {
        "analysis_model": manifest.qualified_models,
        "analysis_media_resolution": manifest.media_resolutions,
        "analysis_fps": manifest.analysis_fps,
        "analysis_thinking": manifest.thinking_levels,
    }
    requested = {
        field: value for field, value in override_values.items() if value is not None
    }
    for field, value in requested.items():
        _validate(field, value, allowed[field])

    recommended = manifest.recommended
    model = request.analysis_model or recommended.model
    media_resolution = request.analysis_media_resolution or recommended.media_resolution
    fps = request.analysis_fps if request.analysis_fps is not None else recommended.analysis_fps
    thinking = request.analysis_thinking or recommended.thinking
    shared = {
        "model": model,
        "media_resolution": media_resolution,
        "thinking": thinking,
    }
    effective = EffectiveAnalysisConfig(
        scan=EffectiveAnalysisStageConfig(**shared),
        critique=EffectiveAnalysisStageConfig(**shared),
        taxonomy=EffectiveAnalysisStageConfig(**shared, fps=fps),
        actor=EffectiveAnalysisStageConfig(**shared, fps=fps),
    )
    return AdmittedTrackRequest(
        **request.model_dump(),
        capability_schema_version=manifest.schema_version,
        requested_analysis_settings=requested,
        effective_analysis_config=effective,
    )


def apply_effective_analysis_config(
    pipeline: PipelineDef,
    effective: EffectiveAnalysisConfig,
) -> PipelineDef:
    """Apply a stored snapshot to a deep copy of the production pipeline."""
    configured = pipeline.model_copy(deep=True)
    by_id = {stage.id: stage for stage in configured.stages}

    scan = effective.scan
    by_id["highlight_scan"].config.update(
        model=scan.model,
        media_resolution=scan.media_resolution,
        thinking=scan.thinking,
    )
    critique = effective.critique
    by_id["highlight_critique"].config.update(
        model=critique.model,
        media_resolution=critique.media_resolution,
        thinking=critique.thinking,
    )
    taxonomy = effective.taxonomy
    actor = effective.actor
    analyze_config = by_id["highlight_analyze"].config
    analyze_config.update(
        model=taxonomy.model,
        media_resolution=taxonomy.media_resolution,
        thinking=taxonomy.thinking,
        fps=taxonomy.fps,
    )
    analyze_config["actor"] = {
        **analyze_config["actor"],
        "model": actor.model,
        "media_resolution": actor.media_resolution,
        "thinking": actor.thinking,
    }
    return configured
