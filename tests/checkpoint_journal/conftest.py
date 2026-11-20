"""Disposable real Cassandra + LocalStack S3, explicit loopback endpoints only."""

import hashlib
import importlib.util
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import pytest
from bjj_analysis_core import CORE_VERSION, WORKFLOW_ID
from bjj_analysis_core.outputs import CandidateConfigV1
from bjj_analysis_core.requests import CoreConfigV1, NodeConfigV1, prompt_bundle_sha256
from botocore.config import Config
from cassandra.cluster import Cluster
from shared_lib.models.analysis_contracts import AnalysisRunSpecV1
from shared_lib.models.analysis_run_record import RegisteredAIRunV1


@pytest.fixture(scope="module")
def adapter():
    path = (
        Path(__file__).resolve().parents[2] / "service/analysis_checkpoint_journal.py"
    )
    spec = importlib.util.spec_from_file_location("s39_checkpoint_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def resources(adapter):
    host = os.environ.get("CHECKPOINT_CASSANDRA_HOST")
    endpoint = os.environ.get("ANNOTATION_TEST_ENDPOINT")
    if (
        host not in {"127.0.0.1", "localhost", "::1"}
        or not endpoint
        or urlparse(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}
    ):
        raise pytest.UsageError("explicit local Cassandra/S3 endpoints required")
    cluster = Cluster([host], port=9042, protocol_version=4)
    session = cluster.connect()
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 0}),
    )
    keyspace, bucket = "s39_cp_" + uuid4().hex, "s39-cp-" + uuid4().hex
    created_keyspace = created_bucket = False
    try:
        session.execute(
            f"CREATE KEYSPACE {keyspace} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
        created_keyspace = True
        session.execute(adapter.checkpoint_table_ddl(keyspace))
        s3.create_bucket(Bucket=bucket)
        created_bucket = True
        yield session, s3, keyspace, bucket
    finally:
        if created_bucket:
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                for obj in page.get("Contents", []):
                    s3.delete_object(Bucket=bucket, Key=obj["Key"])
            s3.delete_bucket(Bucket=bucket)
        if created_keyspace:
            session.execute(f"DROP KEYSPACE {keyspace}")
        cluster.shutdown()
        s3.close()


@pytest.fixture
def configuration():
    node = NodeConfigV1(model="fixture-model", fps=10, media_resolution="high")
    return (
        CoreConfigV1(
            prompt_bundle_sha256=prompt_bundle_sha256(),
            position_scan=node,
            position_critique=node,
            technique_scan=node,
            technique_critique=node,
            technique_analyze=node,
            technique_validate=node,
            candidates=CandidateConfigV1(),
        )
        .model_dump_json()
        .encode()
    )


@pytest.fixture
def run(configuration, resources):
    _, _, _, bucket = resources
    now = datetime.now(timezone.utc) - timedelta(seconds=1)
    ref = dict(bucket=bucket, key="fixture", sha256="a" * 64)
    return RegisteredAIRunV1(
        spec=AnalysisRunSpecV1(
            run_id=uuid4(),
            analysis_id=uuid4(),
            base_edit_version=0,
            scope_id=uuid4(),
            owner=dict(kind="account", principal_id="owner"),
            video_id="video",
            source=ref,
            source_duration_ms=20000,
            scope=dict(start_ms=1000, end_ms=11000),
            athletes=tuple(
                dict(player_id=p, display_name=p, reference_crop=ref)
                for p in ("a", "b")
            ),
            workflow_id=WORKFLOW_ID,
            core_version=CORE_VERSION,
            effective_config={
                **ref,
                "sha256": hashlib.sha256(configuration).hexdigest(),
            },
            admission_id="admission",
            usage_policy_version="fixture-v1",
            admitted_at=now,
        ),
        baseline=ref,
        baseline_revision_id=uuid4(),
        baseline_revision_sha256="b" * 64,
        registered_at=now,
    )


async def authorized():
    """Explicit local fixture guard; production host must check live authority."""


@pytest.fixture
def journal(adapter, resources, run):
    session, s3, keyspace, bucket = resources
    return adapter.KeyspacesCheckpointJournal(
        run, session, s3, bucket=bucket, keyspace=keyspace, guard=authorized
    )
