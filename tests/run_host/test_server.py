"""Entrypoint configuration and real host startup fence; no provider/network calls."""

import os
import subprocess
import sys

import pytest

from service.dag_server import settings_from_env


@pytest.fixture
def environment(tmp_path):
    certificate = tmp_path / "certificate"
    certificate.touch()  # settings checks paths; actual listener validates TLS bytes
    values = dict(
        BJJ_DAG_ENGINE_ENABLED="1",
        BJJ_DAG_ENGINE_OWNERS="account,session",
        BJJ_DAG_QUALIFIED_MODELS_JSON='{"fixture":["fixture-r1"]}',
        BJJ_DAG_INTERNAL_HOST="127.0.0.1",
        BJJ_DAG_INTERNAL_PORT="9101",
        BJJ_DAG_INTERNAL_CERTIFICATE=str(certificate),
        BJJ_DAG_INTERNAL_PRIVATE_KEY=str(certificate),
        BJJ_DAG_MAX_SOURCE_BYTES="10000000",
        BJJ_DAG_KEYSPACE="fixture",
        BJJ_DAG_GEMINI_API_KEY="fixture",
        BJJ_DAG_ENQUEUE_AUDIENCE="engine",
        BJJ_DAG_AWS_REGION="us-east-1",
        BJJ_DAG_CASSANDRA_HOST="cassandra.internal",
        BJJ_DAG_CASSANDRA_PORT="9142",
        BJJ_DAG_CASSANDRA_USERNAME="fixture",
        BJJ_DAG_CASSANDRA_PASSWORD="fixture",
        BJJ_DAG_CASSANDRA_CA_FILE=str(certificate),
    )
    for index, owner in enumerate(("ACCOUNT", "SESSION")):
        prefix = f"BJJ_DAG_{owner}_"
        for key, value in dict(
            ANNOTATION_TABLE_PREFIX=owner.lower(),
            PROPOSAL_TABLE_PREFIX=owner.lower(),
            PUBLICATION_TABLE=owner.lower() + "_publication",
            ARTIFACT_BUCKET="fixture-artifacts",
            SOURCE_BUCKETS_JSON='["fixture-source"]',
            SCRATCH_ROOT=str(tmp_path),
            RESULT_TOPIC_ARN="arn:aws:sns:us-east-1:000000000000:" + owner.lower(),
            ACCESS_ORIGIN="https://" + owner.lower() + ".internal",
            ACCESS_AUDIENCE=owner.lower(),
            ACCESS_KEY_ID=owner.lower() + "-access",
            ENQUEUE_KEY_ID=owner.lower() + "-enqueue",
            ACCESS_SECRET=str(index * 2) * 32,
            ENQUEUE_SECRET=str(index * 2 + 1) * 32,
        ).items():
            values[prefix + key] = value
    return values


def test_explicit_settings_keep_owner_lanes_and_tls(environment):
    settings = settings_from_env(environment)
    assert set(settings["lanes"]) == {"account", "session"}
    assert not settings["listener_settings"].local_test
    assert settings["qualified_models"] == {"fixture": frozenset({"fixture-r1"})}
    for owner, lane in settings["lanes"].items():
        assert lane["access_key"].owner_kind == lane["enqueue_key"].owner_kind == owner


def test_session_only_engine_needs_no_account_lane(environment):
    environment["BJJ_DAG_ENGINE_OWNERS"] = "session"
    for key in [key for key in environment if key.startswith("BJJ_DAG_ACCOUNT_")]:
        del environment[key]
    settings = settings_from_env(environment)
    assert set(settings["lanes"]) == {"session"}
    lane = settings["lanes"]["session"]
    assert lane["access_key"].owner_kind == lane["enqueue_key"].owner_kind == "session"


@pytest.mark.parametrize(
    "owners", ["", "guest", "session,session", "session,", "account;session"]
)
def test_engine_owners_must_be_explicit_known_lanes(environment, owners):
    environment["BJJ_DAG_ENGINE_OWNERS"] = owners
    with pytest.raises(ValueError):
        settings_from_env(environment)


@pytest.mark.parametrize(
    "name,value",
    [
        ("BJJ_DAG_ENGINE_ENABLED", "0"),
        ("BJJ_DAG_QUALIFIED_MODELS_JSON", "{}"),
        ("BJJ_DAG_QUALIFIED_MODELS_JSON", '{"fixture":["r1"],"fixture":["r2"]}'),
        ("BJJ_DAG_INTERNAL_HOST", "0.0.0.0"),
        ("BJJ_DAG_INTERNAL_CERTIFICATE", ""),
        ("BJJ_DAG_MAX_SOURCE_BYTES", "0"),
        ("BJJ_DAG_SESSION_SOURCE_BUCKETS_JSON", '"bucket"'),
        ("BJJ_DAG_SESSION_ENQUEUE_SECRET", "0" * 32),
        ("BJJ_DAG_KEYSPACE", "bad;keyspace"),
        ("BJJ_DAG_GEMINI_API_KEY", ""),
    ],
)
def test_invalid_settings_fail_before_client_construction(environment, name, value):
    environment[name] = value
    with pytest.raises(ValueError):
        settings_from_env(environment)


def test_module_rejects_unconfigured_activation_before_legacy_import():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BJJ_DAG_")
    }
    result = subprocess.run(
        [sys.executable, "-m", "service.dag_server"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "BJJ_DAG_ENGINE_ENABLED" in result.stderr
    assert "service/app.py" not in result.stderr


@pytest.mark.asyncio
async def test_actual_app_rejects_unsupervised_dag_before_legacy_io(monkeypatch):
    from service.app import create_app

    monkeypatch.setenv("BJJ_DAG_ENGINE_ENABLED", "1")
    app = create_app()
    with pytest.raises(RuntimeError, match="DAG engine requires"):
        async with app.router.lifespan_context(app):
            pytest.fail("unsupervised activation must not enter legacy startup")
