"""Explicit new-run preset; never rewrites an admitted configuration artifact.

Provider qualification remains a separate host input, backed by live evidence.
"""

from bjj_analysis_core.outputs import CandidateConfigV1
from bjj_analysis_core.packs import MMA
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


def gemini38_mma_preset() -> CoreConfigV1:
    """The BJJ preset on the MMA pack; the technique scan samples 5 fps.

    Single strikes last a few hundred milliseconds, so a 1 fps scan cannot
    propose them as separate candidates. The x-mma strike experiment measures
    1/5/10 fps; retune this value from its result.
    """
    bjj = gemini38_preset()
    return bjj.model_copy(
        update={
            "prompt_bundle_sha256": prompt_bundle_sha256(MMA),
            "technique_scan": bjj.technique_scan.model_copy(update={"fps": 5}),
        }
    )
