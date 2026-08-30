"""Gemini 3.6/3.7 selection and request-compatibility contracts for VLM Studio."""
from __future__ import annotations

from pathlib import Path

import pytest

from service.pipelines.gemini_model_compat import generation_sampling_kwargs
from service.pipelines.registry import get_default, validate_pipeline_def
from service.routes.qa_vlm import DETECT_MODELS, EVENT_MODELS, IMAGE_MODELS, thinking_config_for


NEW_FLASH_MODELS = ("gemini-3.7-flash", "gemini-3.6-flash")


@pytest.mark.parametrize("model_id", NEW_FLASH_MODELS)
def test_new_flash_models_are_allowed_for_text_video_and_detect_not_image_generation(model_id):
    assert model_id in EVENT_MODELS
    assert model_id in DETECT_MODELS
    assert model_id not in IMAGE_MODELS


@pytest.mark.parametrize("model_id", NEW_FLASH_MODELS)
def test_new_flash_models_validate_in_every_pipeline_model_slot(model_id):
    """Every model-bearing DAG node and nested highlight axis accepts the model."""
    for pipeline_id in (
        "single-shot",
        "vision-mimic",
        "vision-mimic-multi",
        "vision-mimic-simplified",
        "chunk-segment-tags",
        "highlight-scan-analyze",
        "highlight-scan-critique-analyze",
        "position-timeline-techniques",
    ):
        pipeline = get_default(pipeline_id)
        for stage in pipeline.stages:
            if "model" in stage.config:
                stage.config["model"] = model_id
            if stage.type == "highlight_analyze":
                for axis_name in ("position", "technique", "validator"):
                    stage.config[axis_name]["model"] = model_id
        validate_pipeline_def(pipeline)


def test_new_flash_models_use_supported_level_based_thinking_for_off():
    assert thinking_config_for("gemini-3.6-flash", "off").thinking_level == "MINIMAL"
    assert thinking_config_for("gemini-3.7-flash", "off").thinking_level == "LOW"


@pytest.mark.parametrize("model_id", NEW_FLASH_MODELS)
def test_new_flash_models_keep_explicit_thinking_levels(model_id):
    assert thinking_config_for(model_id, "high").thinking_level == "HIGH"


@pytest.mark.parametrize("model_id", NEW_FLASH_MODELS)
def test_new_flash_models_omit_unsupported_sampling_parameters(model_id):
    assert generation_sampling_kwargs(model_id, temperature=0.2) == {}


def test_existing_models_keep_their_existing_sampling_parameter_contract():
    assert generation_sampling_kwargs("gemini-3.5-flash", temperature=0.2) == {"temperature": 0.2}


def test_html_exposes_new_flash_models_to_tasks_and_all_pipeline_node_selectors():
    html = (Path(__file__).parents[1] / "qa_client" / "vlm_studio.html").read_text()
    for model_id in NEW_FLASH_MODELS:
        assert html.count(f"'{model_id}'") >= 2  # Event and Detect task selectors.
    assert "const NODE_MODELS = TASK_CONFIG.event.models" in html
    assert "model: { widget: 'select', options: NODE_MODELS" in html


def test_position_timeline_html_defaults_every_gemini_node_to_high_thinking():
    html = (Path(__file__).parents[1] / "qa_client" / "vlm_studio.html").read_text()
    for stage_id in (
        "position_scan",
        "position_critique",
        "highlight_scan",
        "highlight_critique",
        "technique_analyze",
    ):
        stage_start = html.index(
            f"{{ id: '{stage_id}'", html.index("const POSITION_TIMELINE_DEFAULT")
        )
        stage_end = html.index("} },", stage_start)
        assert "thinking: 'high'" in html[stage_start:stage_end], stage_id
