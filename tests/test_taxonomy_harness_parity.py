"""shared_lib <-> bjj-dataset-harness taxonomy parity gate (T2, revised
2026-07-12 per CEO ruling on Evaluator pass 1 MEDIUM-1).

``service/taxonomy_mapper.py`` no longer hand-maintains a vendored copy of
the axis vocabulary — it imports directly from
``shared_lib.models.simplified_taxonomy`` (a real runtime dependency), so
there is nothing left to drift *within this repo* and nothing left here to
parity-test against shared_lib.

What CAN still drift independently is ``bjj-dataset-harness``'s own
``src/harness/taxonomy.py`` — per D2 (plan
``working_log/plans/2026-07-12-simplified-4axis-taxonomy-adoption.md``):
"harness ``taxonomy.py`` stays authoritative for probe axes and is
regenerated from the same manifest" — but that regeneration is NOT yet
wired (T0/T1 only shipped the shared_lib side). Until it is, this is a real,
open two-copy problem, and this test is the fail-loud (NEVER SKIP) guard
against it — same idiom as the (now-removed) vendored-manifest parity test
and the s8 ``bjjTaxonomy.parity.test.ts`` precedent: umbrella-relative
default path (not ``cwd()``, so this resolves correctly from a ``-wt-*``
sibling worktree), env-overridable, missing source file is a hard failure
with a clear message.

The harness module is read as TEXT and parsed via ``ast`` (never imported —
``bjj-dataset-harness`` is a separate repo/package, not a declared
dependency of this engine; a sys.path hack to import it would violate the
"never use sys.path hacks" project convention). ``shared_lib``'s module,
conversely, IS a real dependency here and is imported directly.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from shared_lib.models import simplified_taxonomy as sl_taxonomy

# Repo-root-relative default (not cwd()) so this resolves correctly whether
# pytest is invoked from the main checkout or a `-wt-*` sibling worktree:
# this test file lives at <repo>/tests/, so one level up is <repo>/, and the
# harness lives at a sibling of <repo> under the shared umbrella
# (bjj-proj/{whole-video-analysis*,bjj-dataset-harness}).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HARNESS_TAXONOMY_PATH = (
    REPO_ROOT.parent / "bjj-dataset-harness" / "src" / "harness" / "taxonomy.py"
)
HARNESS_TAXONOMY_PATH = Path(
    os.environ.get("BJJ_HARNESS_TAXONOMY", str(DEFAULT_HARNESS_TAXONOMY_PATH))
)


def _load_harness_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"[taxonomy-harness-parity] harness taxonomy.py not found at \"{path}\". "
            "Set BJJ_HARNESS_TAXONOMY to the correct path, or restore the file — "
            "this gate must fail loudly, never skip."
        )
    return path.read_text(encoding="utf-8")


def _extract_module_level_literals(source: str) -> Dict[str, Any]:
    """Parse top-level `NAME = <literal>` assignments via ast (never import
    — bjj-dataset-harness is not a dependency of this engine)."""
    tree = ast.parse(source)
    values: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue  # non-literal assignment (e.g. a tuple of names) — skip
    return values


def test_missing_harness_source_raises_not_skips():
    bogus_path = Path(str(HARNESS_TAXONOMY_PATH) + ".__does_not_exist__")
    with pytest.raises(FileNotFoundError, match="harness taxonomy.py not found"):
        _load_harness_source(bogus_path)


def test_harness_source_resolves_at_default_or_overridden_path():
    assert HARNESS_TAXONOMY_PATH.exists(), (
        f"bjj-dataset-harness taxonomy.py not found at {HARNESS_TAXONOMY_PATH}. "
        "Set BJJ_HARNESS_TAXONOMY if bjj-dataset-harness lives at a non-default "
        "path relative to this umbrella checkout."
    )


def _harness_values() -> Dict[str, Any]:
    return _extract_module_level_literals(_load_harness_source(HARNESS_TAXONOMY_PATH))


def test_axis1_position_matches_harness():
    harness = _harness_values()
    assert list(sl_taxonomy.AXIS1_POSITION) == harness["AXIS1_POSITION"]


def test_axis3_action_matches_harness():
    harness = _harness_values()
    assert list(sl_taxonomy.AXIS3_ACTION) == harness["AXIS3_ACTION"]


def test_axis4_outcome_matches_harness():
    harness = _harness_values()
    assert list(sl_taxonomy.AXIS4_OUTCOME) == harness["AXIS4_OUTCOME"]


def test_axis2_actor_sentinels_matches_harness():
    harness = _harness_values()
    assert list(sl_taxonomy.AXIS2_ACTOR_SENTINELS) == harness["AXIS2_ACTOR_SENTINELS"]


def test_hard_pair_actions_matches_harness():
    harness = _harness_values()
    assert dict(sl_taxonomy.HARD_PAIR_ACTIONS) == harness["HARD_PAIR_ACTIONS"]


def test_taxonomy_mapper_re_exports_match_shared_lib_directly():
    """Guard against a future regression back toward hand-copying: confirm
    service/taxonomy_mapper.py's constants really are shared_lib's objects
    (or equal values), not a second hand-maintained list."""
    from service import taxonomy_mapper as tm

    assert tm.AXIS1_POSITION == sl_taxonomy.AXIS1_POSITION
    assert tm.AXIS3_ACTION == sl_taxonomy.AXIS3_ACTION
    assert tm.AXIS4_OUTCOME == sl_taxonomy.AXIS4_OUTCOME
    assert tm.AXIS2_ACTOR_SENTINELS == sl_taxonomy.AXIS2_ACTOR_SENTINELS
    assert tm.HARD_PAIR_ACTIONS == sl_taxonomy.HARD_PAIR_ACTIONS
    assert tm.TECHNIQUE_SHORTLISTS == sl_taxonomy.TECHNIQUE_SHORTLISTS
