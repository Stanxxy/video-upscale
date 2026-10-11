import asyncio
import json
import os
import subprocess
import threading
from decimal import ROUND_FLOOR, Decimal

import pytest
from shared_lib.models.analysis_run_record import RegisteredAIRunV1


def altered(run, **spec):
    data = run.model_dump()
    data["spec"].update(spec)
    return RegisteredAIRunV1.model_validate(data)


@pytest.mark.asyncio
async def test_exact_media_identity_order_clock_permissions_and_cleanup(
    resolver, inputs, media_bytes
):
    run, config, _ = inputs
    async with resolver.open(run) as media:
        assert media.run == run and media.config_bytes == config
        assert media.source_duration_ms == 3000
        assert (
            media.run.spec.scope.start_ms == 500
        )  # full source, no silent trim/rebase
        assert media.source_size == len(media_bytes[0])
        assert media.source_format == "mov,mp4,m4a,3gp,3g2,mj2"
        assert [c.player_id for c in media.crops] == ["z", "a"]
        assert [c.display_name for c in media.crops] == ["Red player", "Blue player"]
        assert [c.mime_type for c in media.crops] == ["image/png", "image/jpeg"]
        assert [c.data for c in media.crops] == list(media_bytes[1:])
        assert all((c.width, c.height) == (32, 24) for c in media.crops)
        assert media.source_path.stat().st_mode & 0o777 == 0o400
        assert media.source_path.parent.stat().st_mode & 0o777 == 0o700
        with media.open_source() as source:
            assert source.tell() == 0 and source.read() == media_bytes[0]
    assert not media.source_path.exists()
    assert list(resolver.root.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        with media.open_source():
            pytest.fail("expired media must not reopen")


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["source", "effective_config", "crop"])
async def test_digest_mismatch_never_yields_media(resolver, inputs, which, adapter):
    run, _, _ = inputs
    data = run.model_dump()
    ref = (
        data["spec"][which]
        if which != "crop"
        else data["spec"]["athletes"][0]["reference_crop"]
    )
    ref["sha256"] = "f" * 64
    with pytest.raises(adapter.MediaIntegrityError, match="digest"):
        async with resolver.open(data):
            pytest.fail("bad admitted digest")
    assert list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
async def test_invalid_admitted_config_stops_before_video_get(resolver, inputs):
    run, _, put = inputs
    calls, get = [], resolver.client.get_object

    def observed(**kw):
        calls.append(kw["Key"])
        return get(**kw)

    resolver.client = type("Observed", (), {"get_object": staticmethod(observed)})()
    ref = put(b'{"schema_version":1}', "bad-config")
    with pytest.raises(ValueError):
        async with resolver.open(altered(run, effective_config=ref)):
            pytest.fail("invalid config")
    assert calls == [ref["key"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["bucket", "authority"])
async def test_untrusted_access_has_no_storage_calls(resolver, inputs, boundary):
    run, _, _ = inputs

    def no_io(**kw):
        pytest.fail("unauthorized I/O")

    resolver.client = type("Denied", (), {"get_object": staticmethod(no_io)})()
    if boundary == "bucket":
        run = altered(
            run, source={**run.spec.source.model_dump(), "bucket": "foreign-bucket"}
        )
    else:

        async def denied(given):
            assert given == run
            raise PermissionError("revoked")

        resolver.guard = denied
    with pytest.raises((ValueError, PermissionError)):
        async with resolver.open(run):
            pytest.fail("access denied")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit", ["max_source", "max_crop", "max_config", "max_pixels"]
)
async def test_resource_limits_fail_and_clean(resolver, inputs, adapter, limit):
    setattr(resolver, limit, 1)
    with pytest.raises(adapter.MediaIntegrityError):
        async with resolver.open(inputs[0]):
            pytest.fail("resource limit")
    assert list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
async def test_video_duration_must_match_registration(resolver, inputs, adapter):
    run, _, _ = inputs
    with pytest.raises(adapter.MediaIntegrityError, match="duration"):
        async with resolver.open(altered(run, source_duration_ms=3001)):
            pytest.fail("duration changed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", [b"not an image", b"#EXTM3U\nhttp://127.0.0.1:9999/secret\n"]
)
async def test_reference_is_decoded_not_trusted_from_name(resolver, inputs, payload):
    run, _, put = inputs
    athletes = [a.model_dump() for a in run.spec.athletes]
    athletes[0]["reference_crop"] = put(payload, "bad.png")
    with pytest.raises(subprocess.CalledProcessError):
        async with resolver.open(altered(run, athletes=athletes)):
            pytest.fail("invalid crop")
    assert list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
async def test_concatenated_images_are_not_one_reference(
    resolver, inputs, media_bytes, adapter
):
    run, _, put = inputs
    athletes = [a.model_dump() for a in run.spec.athletes]
    athletes[0]["reference_crop"] = put(media_bytes[1] * 2, "two.png")
    with pytest.raises(adapter.MediaIntegrityError, match="exactly one"):
        async with resolver.open(altered(run, athletes=athletes)):
            pytest.fail("ambiguous multi-image reference")


@pytest.mark.asyncio
async def test_source_rechecked_on_same_descriptor(resolver, inputs, adapter):
    async with resolver.open(inputs[0]) as media:
        os.chmod(media.source_path, 0o600)
        with media.source_path.open("r+b") as target:
            target.write(b"corrupted")
        with pytest.raises(adapter.MediaIntegrityError, match="changed"):
            with media.open_source():
                pytest.fail("modified local cache")


@pytest.mark.asyncio
async def test_blocking_cancellation_keeps_scratch_until_worker_exits(resolver, inputs):
    started, release = threading.Event(), threading.Event()
    download = resolver._download
    paths = []

    def blocked(ref, path, limit):
        paths.append(path)
        started.set()
        assert release.wait(5)
        assert path.parent.exists()
        return download(ref, path, limit)

    resolver._download = blocked

    async def consume():
        async with resolver.open(inputs[0]):
            pytest.fail("cancelled acquisition yielded")

    task = asyncio.create_task(consume())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # repeated cancellation must not remove active worker files
        await asyncio.sleep(0)
        assert paths[0].parent.exists() and not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(paths) == 1 and list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
async def test_revocation_after_video_download_prevents_crop_reads(resolver, inputs):
    run, _, _ = inputs
    gets, get = [], resolver.client.get_object

    def observe(**kw):
        gets.append(kw["Key"])
        return get(**kw)

    async def guard(given):
        if run.spec.source.key in gets:
            raise PermissionError("revoked during download")

    resolver.client = type("Observed", (), {"get_object": staticmethod(observe)})()
    resolver.guard = guard
    with pytest.raises(PermissionError):
        async with resolver.open(run):
            pytest.fail("revoked handoff")
    assert gets == [run.spec.effective_config.key, run.spec.source.key]
    assert list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
async def test_declared_length_lie_closes_body_and_cleans(resolver, inputs, adapter):
    get, bodies = resolver.client.get_object, []

    def wrong_length(**kw):
        response = get(**kw)
        bodies.append(response["Body"])
        response["ContentLength"] = 1
        return response

    resolver.client = type(
        "WrongLength", (), {"get_object": staticmethod(wrong_length)}
    )()
    with pytest.raises(adapter.MediaIntegrityError, match="declared length"):
        async with resolver.open(inputs[0]):
            pytest.fail("length mismatch")
    assert all(body._raw_stream.closed for body in bodies)
    assert list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
async def test_storage_outage_is_not_an_empty_reference(resolver, inputs):
    def unavailable(**kw):
        raise TimeoutError("S3 unavailable")

    resolver.client = type(
        "Unavailable", (), {"get_object": staticmethod(unavailable)}
    )()
    with pytest.raises(TimeoutError):
        async with resolver.open(inputs[0]):
            pytest.fail("outage cannot yield media")
    assert list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
async def test_consumer_failure_also_cleans(resolver, inputs):
    with pytest.raises(RuntimeError, match="consumer"):
        async with resolver.open(inputs[0]):
            raise RuntimeError("consumer failed")
    assert list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["offset", "audio", "multiple_video"])
async def test_unqualified_video_clock_or_streams_rejected(
    resolver, inputs, adapter, tmp_path, kind
):
    run, _, put = inputs
    path = tmp_path / "alternative.mp4"
    base = ["ffmpeg", "-v", "error", "-nostdin"]
    if kind == "audio":
        args = ["-f", "lavfi", "-i", "sine=frequency=440", "-t", "3", "-c:a", "aac"]
    else:
        args = [
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
        ]
        args += (
            ["-output_ts_offset", "5"]
            if kind == "offset"
            else ["-map", "0:v", "-map", "0:v"]
        )
    subprocess.run(
        base + args + [str(path)], capture_output=True, timeout=30, check=True
    )
    metadata = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        timeout=30,
        check=True,
    )
    duration = Decimal(json.loads(metadata.stdout)["format"]["duration"])
    duration_ms = int((duration * 1000).to_integral_value(rounding=ROUND_FLOOR))
    source = put(path.read_bytes(), kind + ".mp4")
    with pytest.raises(
        adapter.MediaIntegrityError,
        match="time origin" if kind == "offset" else "one unambiguous",
    ):
        async with resolver.open(
            altered(run, source=source, source_duration_ms=duration_ms)
        ):
            pytest.fail("unsupported video must not be silently normalized")
    # This test owns a generated input in the same scratch parent.
    assert list(resolver.root.iterdir()) == [path]


@pytest.mark.asyncio
async def test_final_authority_guard_prevents_media_handoff(resolver, inputs):
    crop, revoked = resolver._crop, False

    def revoke_after_last_crop(path, athlete):
        nonlocal revoked
        result = crop(path, athlete)
        if athlete.player_id == "a":
            revoked = True
        return result

    async def guard(run):
        if revoked:
            raise PermissionError("revoked before yield")

    resolver._crop, resolver.guard = revoke_after_last_crop, guard
    with pytest.raises(PermissionError):
        async with resolver.open(inputs[0]):
            pytest.fail("revoked media handoff")
    assert list(resolver.root.iterdir()) == []


@pytest.mark.asyncio
async def test_probe_timeout_cleans_scratch(resolver, inputs, monkeypatch, adapter):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 30)

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        async with resolver.open(inputs[0]):
            pytest.fail("probe timeout must propagate")
    assert list(resolver.root.iterdir()) == []
