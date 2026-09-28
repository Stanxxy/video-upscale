"""Explicit new-run preset; never rewrites an admitted configuration artifact.

Provider qualification remains a separate host input, backed by live evidence.
"""

from bjj_analysis_core.outputs import CandidateConfigV1
from bjj_analysis_core.requests import CoreConfigV1, NodeConfigV1, prompt_bundle_sha256


def gemini38_preset() -> CoreConfigV1:
    node = NodeConfigV1(
        model="gemini-3.8-flash", thinking="high", media_resolution="low", fps=10
    )
    return CoreConfigV1(
        prompt_bundle_sha256=prompt_bundle_sha256(),
        position_scan=node.model_copy(update={"fps": 1}),
        position_critique=node,
        technique_scan=node.model_copy(update={"fps": 1}),
        technique_critique=node,
        technique_analyze=node,
        technique_validate=node,
        candidates=CandidateConfigV1(),
    )
