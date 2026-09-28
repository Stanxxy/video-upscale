"""Real local S3 and FFmpeg fixtures; no provider requests or cloud discovery."""

import hashlib
import importlib.util
import os
import subprocess
import sys
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
from shared_lib.models.analysis_run_record import RegisteredAIRunV1


@pytest.fixture(scope="module")
def adapter():
    path = Path(__file__).resolve().parents[2] / "service/analysis_media.py"
    spec = importlib.util.spec_from_file_location("s39_analysis_media", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gemini():
    path = Path(__file__).resolve().parents[2] / "service/analysis_gemini.py"
    spec = importlib.util.spec_from_file_location("s39_analysis_gemini", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def s3():
    endpoint = os.environ.get("ANNOTATION_TEST_ENDPOINT")
    if not endpoint or urlparse(endpoint).hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise pytest.UsageError("explicit loopback S3 endpoint required")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 0}),
    )
    bucket = "s39-media-" + uuid4().hex
    client.create_bucket(Bucket=bucket)
    try:
        yield client, bucket
    finally:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                client.delete_object(Bucket=bucket, Key=obj["Key"])
        client.delete_bucket(Bucket=bucket)
        client.close()


def ffmpeg(path, *arguments):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", *arguments, str(path)],
        capture_output=True,
        check=True,
        timeout=30,
    )


@pytest.fixture(scope="module")
def media_bytes(tmp_path_factory):
    root = tmp_path_factory.mktemp("generated-media")
    video, png, jpeg = root / "video.mp4", root / "a.png", root / "b.jpg"
    ffmpeg(
        video,
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x48:rate=10",
        "-t",
        "3",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
    )
    for path, color in ((png, "red"), (jpeg, "blue")):
        ffmpeg(
            path,
            "-f",
            "lavfi",
            "-i",
            f"color={color}:size=32x24",
            "-frames:v",
            "1",
            "-threads",
            "1",
        )
    return video.read_bytes(), png.read_bytes(), jpeg.read_bytes()


@pytest.fixture
def inputs(s3, media_bytes):
    client, bucket = s3
    prefix = uuid4().hex

    def put(data, name):
        key = f"{prefix}/literal %20 #?/{name}"
        client.put_object(
            Bucket=bucket, Key=key, Body=data, ContentType="application/octet-stream"
        )
        return dict(bucket=bucket, key=key, sha256=hashlib.sha256(data).hexdigest())

    node = NodeConfigV1(model="fixture-model", fps=10, media_resolution="high")
    config = (
        CoreConfigV1(
            prompt_bundle_sha256=prompt_bundle_sha256(),
            candidates=CandidateConfigV1(),
            **{
                name: node
                for name in (
                    "position_scan",
                    "position_critique",
                    "technique_scan",
                    "technique_critique",
                    "technique_analyze",
                    "technique_validate",
                )
            },
        )
        .model_dump_json()
        .encode()
    )
    video, png, jpeg = media_bytes
    now = datetime.now(timezone.utc) - timedelta(seconds=1)
    source = put(video, "source.mp4")
    run = RegisteredAIRunV1.model_validate(
        dict(
            spec=dict(
                run_id=uuid4(),
                analysis_id=uuid4(),
                base_edit_version=0,
                scope_id=uuid4(),
                owner=dict(kind="account", principal_id="owner"),
                video_id="video",
                source=source,
                source_duration_ms=3000,
                scope=dict(start_ms=500, end_ms=2500),
                athletes=(
                    dict(
                        player_id="z",
                        display_name="Red player",
                        reference_crop=put(png, "z.jpg"),
                    ),
                    dict(
                        player_id="a",
                        display_name="Blue player",
                        reference_crop=put(jpeg, "a.png"),
                    ),
                ),
                workflow_id=WORKFLOW_ID,
                core_version=CORE_VERSION,
                effective_config=put(config, "config.json"),
                admission_id="admission",
                usage_policy_version="fixture-v1",
                admitted_at=now,
            ),
            baseline=source,
            baseline_revision_id=uuid4(),
            baseline_revision_sha256="b" * 64,
            registered_at=now,
        )
    )
    return run, config, put


@pytest.fixture
def resolver(adapter, s3, tmp_path):
    client, bucket = s3

    async def guard(run):
        """Local fixture only; host must check live authority for this run."""

    return adapter.S3RunMedia(
        client,
        allowed_buckets={bucket},
        scratch_root=tmp_path,
        max_source_bytes=1024 * 1024,
        guard=guard,
    )
