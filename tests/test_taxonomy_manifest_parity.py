"""Taxonomy manifest parity gate (T2, item 1) — s8 precedent.

``service/taxonomy_manifest.json`` is a vendored, byte-identical copy of
``shared_lib``'s ``taxonomy_manifest.json`` (see ``service/taxonomy_mapper.py``
module docstring for why it is vendored rather than read from the installed
package). This test is the fail-loud (NEVER SKIP) drift guard, following the
exact idiom of ``bjj-vision-frontend``'s
``src/lib/__tests__/bjjTaxonomy.parity.test.ts`` (s8, AC-11): a repo-root-
relative default path (not ``cwd()``, so this resolves correctly whether
pytest is invoked from the main checkout or a ``-wt-*`` sibling worktree),
env-overridable, and a missing source file is a hard failure with a clear
message — a skipped parity gate is a lie.

NOTE (s9/T2, 2026-07-12): the DEFAULT path below points at
``bjj-vision-backend/shared_lib`` (the eventual merged/production location,
matching the s8 default-path convention). As of this writing, shared_lib's
simplified-4-axis-taxonomy work (T1) lives on the UNMERGED branch
``feature/simplified-4axis-taxonomy`` in the ``shared_lib-wt-s9`` worktree,
not yet on shared_lib's ``develop``. Until that branch merges, run this
suite with::

    BJJ_SHARED_LIB_TAXONOMY_MANIFEST=/Users/stanliu/bjj-proj/bjj-vision-backend/shared_lib-wt-s9/src/shared_lib/models/taxonomy_manifest.json \
        pytest tests/test_taxonomy_manifest_parity.py -v

Once T1 merges to shared_lib ``develop``/``main``, drop the env override —
the default path will resolve correctly on its own.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Repo-root-relative default (not cwd()) so this resolves correctly whether
# pytest is invoked from the main checkout or a `-wt-*` sibling worktree:
# this test file lives at <repo>/tests/, so one level up is <repo>/, and the
# backend lives at a sibling of <repo> under the shared umbrella
# (bjj-proj/{whole-video-analysis*,bjj-vision-backend}).
REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED_MANIFEST_PATH = REPO_ROOT / "service" / "taxonomy_manifest.json"
DEFAULT_SHARED_LIB_MANIFEST_PATH = (
    REPO_ROOT.parent / "bjj-vision-backend" / "shared_lib"
    / "src" / "shared_lib" / "models" / "taxonomy_manifest.json"
)
SHARED_LIB_MANIFEST_PATH = Path(
    os.environ.get("BJJ_SHARED_LIB_TAXONOMY_MANIFEST", str(DEFAULT_SHARED_LIB_MANIFEST_PATH))
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"[taxonomy-manifest-parity] source not found at \"{path}\". "
            "Set BJJ_SHARED_LIB_TAXONOMY_MANIFEST to the correct path, or restore "
            "the file — this gate must fail loudly, never skip."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_vendored_manifest_exists():
    assert VENDORED_MANIFEST_PATH.exists(), (
        f"vendored manifest missing at {VENDORED_MANIFEST_PATH}"
    )


def test_missing_shared_lib_source_raises_not_skips():
    bogus_path = Path(str(SHARED_LIB_MANIFEST_PATH) + ".__does_not_exist__")
    with pytest.raises(FileNotFoundError, match="source not found"):
        _load_json(bogus_path)


def test_shared_lib_manifest_resolves_at_default_or_overridden_path():
    """This is the actual fail-loud gate: if this fails, either
    BJJ_SHARED_LIB_TAXONOMY_MANIFEST is unset/wrong, or shared_lib's T1
    branch (feature/simplified-4axis-taxonomy) genuinely hasn't merged yet —
    see the module docstring for how to run this locally in that case."""
    assert SHARED_LIB_MANIFEST_PATH.exists(), (
        f"shared_lib taxonomy_manifest.json not found at {SHARED_LIB_MANIFEST_PATH}. "
        "See this test file's module docstring for the BJJ_SHARED_LIB_TAXONOMY_MANIFEST "
        "override needed while shared_lib T1 is unmerged."
    )


def test_vendored_manifest_is_byte_identical_to_shared_lib_source():
    if not SHARED_LIB_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"[taxonomy-manifest-parity] source not found at \"{SHARED_LIB_MANIFEST_PATH}\". "
            "Set BJJ_SHARED_LIB_TAXONOMY_MANIFEST to the correct path, or restore "
            "the file — this gate must fail loudly, never skip."
        )
    vendored_bytes = VENDORED_MANIFEST_PATH.read_bytes()
    shared_lib_bytes = SHARED_LIB_MANIFEST_PATH.read_bytes()
    assert vendored_bytes == shared_lib_bytes, (
        "service/taxonomy_manifest.json has DRIFTED from shared_lib's "
        "taxonomy_manifest.json — re-copy the vendored file byte-for-byte."
    )


def test_vendored_manifest_is_json_equal_to_shared_lib_source():
    """Belt-and-suspenders JSON-structural check alongside the byte check
    above — catches drift even if someone re-serializes with different
    whitespace/ordering (which would legitimately still be "equal" data but
    should still be flagged, since the byte-identical check is the actual
    contract; this second check documents the failure mode more clearly)."""
    vendored = _load_json(VENDORED_MANIFEST_PATH)
    shared_lib = _load_json(SHARED_LIB_MANIFEST_PATH)
    assert vendored == shared_lib


def test_vendored_axes_match_taxonomy_mapper_module():
    """taxonomy_mapper.py must actually be DERIVED from the vendored file
    (not a second hand-copy) — cross-check its loaded constants."""
    from service import taxonomy_mapper as tm

    vendored = _load_json(VENDORED_MANIFEST_PATH)
    assert tm.AXIS1_POSITION == vendored["axis1_position"]
    assert tm.AXIS3_ACTION == vendored["axis3_action"]
    assert tm.AXIS4_OUTCOME == vendored["axis4_outcome"]
    assert tm.TECHNIQUE_SHORTLISTS == vendored["technique_shortlists"]
