"""Installed-wheel checks: no engine imports or writer capability in read hosts."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from bjj_analysis_observer.checkpoints import KeyspacesCheckpointReader
from bjj_analysis_observer.observer import AuthorizedRunObserver


def test_installed_package_imports_without_engine_or_source_path(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
import bjj_analysis_observer.checkpoints as cp
import bjj_analysis_observer.lifecycle
import bjj_analysis_observer.observer
import bjj_analysis_observer.results
assert 'site-packages' in cp.__file__, cp.__file__
assert not any(n == 'service' or n.startswith('service.') for n in sys.modules)
assert not any(n.startswith(('tracking_pipeline', 'google.genai')) for n in sys.modules)
assert not hasattr(cp.KeyspacesCheckpointReader, 'compare_and_swap')
assert not hasattr(cp.KeyspacesCheckpointReader, '_put_bytes')
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_installed_reader_sources_do_not_import_engine_or_write_storage():
    import bjj_analysis_observer.checkpoints as module

    root = Path(module.__file__).parent
    disallowed = {
        "put_object",
        "put_item",
        "update_item",
        "delete_item",
        "delete_object",
        "transact_write_items",
        "publish",
        "compare_and_swap",
        "acquire",
        "dispatch",
    }
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("service"), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in disallowed, (path, node.func.attr)


@pytest.mark.parametrize(
    "query,conditional",
    [
        ("UPDATE example SET x=1", False),
        ("INSERT INTO example VALUES (1)", False),
        ("DELETE FROM example", False),
        ("BEGIN BATCH", False),
        ("SELECT * FROM example", True),
    ],
)
def test_reader_refuses_write_and_conditional_queries_without_touching_a_client(
    query, conditional
):
    reader = object.__new__(KeyspacesCheckpointReader)
    with pytest.raises(ValueError, match="only accepts read queries"):
        reader._execute_sync(query, [], conditional=conditional)


def test_host_cannot_construct_observer_without_resource_authority():
    with pytest.raises(ValueError, match="authorization"):
        AuthorizedRunObserver(
            None,
            None,
            None,
            keyspace="x",
            bucket="bucket",
            authorize=None,
            lease_seconds=60,
        )
