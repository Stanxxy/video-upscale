"""Default service startup must stay independent from the legacy GPU profile."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_app_registers_gemini_routes_without_legacy_gpu_imports():
    """Catch a default-startup import of a package moved to the optional profile."""
    script = """
import importlib.abc
import sys


class BlockLegacyGpuImports(importlib.abc.MetaPathFinder):
    roots = {
        'torch', 'torchvision', 'sam2', 'spandrel', 'ultralytics',
        'supervision', 'rtmlib', 'scipy', 'huggingface_hub', 'tqdm',
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in self.roots:
            raise ModuleNotFoundError(f'blocked legacy GPU import: {fullname}')
        return None


sys.meta_path.insert(0, BlockLegacyGpuImports())
from service.app import app
from service.routes import scheduling

paths = set(app.openapi()['paths'])
assert {'/health', '/analysis-settings/capabilities', '/track'} <= paths
assert '/debug/memory' not in paths
assert callable(scheduling.run_highlight_job)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
