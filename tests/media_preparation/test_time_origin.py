"""Non-zero video/container time origins map onto the Analyzer source timeline.

Source ms = presentation time - container start_time (the preview convention).
Gemini re-bases the uploaded file so its first video frame is 0 (measured live
2026-11-09), so provider ms = presentation time - video stream start_time.
"""

import json
import os
import subprocess
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import httpx
import pytest
from bjj_analysis_core.requests import parse_response

from tests.media_preparation.test_gemini import (
    provider_case,
    request_for,
    uploaded,
)
from tests.media_preparation.test_media import altered
from uuid import uuid4

REAL_YT = Path("/Volumes/SanDisk256/mls")
FPS = 25


def run(*args):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", *args],
        capture_output=True,
        check=True,
        timeout=60,
    )


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    path = tmp_path_factory.mktemp("origin") / "raw.mp4"
    run(
        "-f", "lavfi", "-i", f"testsrc2=size=64x48:rate={FPS}",
        "-f", "lavfi", "-i", "sine=frequency=440", "-t", "3",
        "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(path),
    )  # fmt: skip
    return path


def video_offset(raw, out, seconds):
    run("-itsoffset", str(seconds), "-i", str(raw), "-i", str(raw),
        "-map", "0:v", "-map", "1:a", "-c", "copy", str(out))  # fmt: skip


def container_offset(raw, out, seconds):
    run("-output_ts_offset", str(seconds), "-i", str(raw), "-c", "copy", str(out))


def real_clip(name):
    def make(raw, out, _):
        source = REAL_YT / name
        if not source.exists():
            pytest.skip("real yt-dlp download not on this machine")
        run("-i", str(source), "-t", "3", "-c", "copy", str(out))

    return make


CASES = {
    "video_0041_audio_0": (video_offset, 0.041, "mp4"),
    "video_1s_audio_0": (video_offset, 1.0, "mp4"),
    "video_0033_audio_0": (video_offset, 0.033, "mp4"),
    "container_and_video_5s": (container_offset, 5, "mp4"),
    "mkv_container_2s": (container_offset, 2, "mkv"),
    "mpegts_video_after_audio": (video_offset, 0.5, "ts"),
    "real_yt_0041": (real_clip("CFE7E1D.mp4"), 0, "mp4"),
    "real_yt_0033": (real_clip("42F55F2-720p.mp4"), 0, "mp4"),
}


def probe(path):
    data = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=start_time,duration:stream=codec_type,start_time",
             "-of", "json", str(path)],
            capture_output=True, check=True, timeout=30,
        ).stdout
    )  # fmt: skip
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    return (
        Decimal(data["format"]["start_time"]),
        Decimal(video["start_time"]),
        int((Decimal(data["format"]["duration"]) * 1000).to_integral_value(ROUND_FLOOR)),
    )


def frame_pts(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, check=True, timeout=30, text=True,
    ).stdout  # fmt: skip
    return sorted(
        Decimal(line.split(",")[0]) for line in out.split() if line[:1].isdigit()
    )


def build(raw, tmp_path, case):
    make, seconds, suffix = CASES[case]
    path = tmp_path / f"{case}.{suffix}"
    make(raw, path, seconds)
    return path


def with_source(inputs, put, path):
    run_, config, _ = inputs
    container, video, duration_ms = probe(path)
    source = put(path.read_bytes(), path.name)
    # scope [0.5 s, 2.5 s) of the source timeline; the file always outlasts it
    return altered(run_, source=source, source_duration_ms=duration_ms), container, video


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES)
async def test_events_land_on_source_ms_whatever_the_origin(
    case, raw, gemini, resolver, inputs, s3, tmp_path
):
    path = build(raw, tmp_path, case)
    run_, container, video = with_source(inputs, inputs[2], path)
    offset_ms = int(((video - container) * 1000).to_integral_value(ROUND_FLOOR))
    pts = frame_pts(path)
    # Frames nearest 0.9 s and 2.2 s on the source timeline: truth comes from the file.
    source_ms = lambda t: int((t - container) * 1000)  # noqa: E731
    first = min(pts, key=lambda t: abs(source_ms(t) - 900))
    last = min(pts, key=lambda t: abs(source_ms(t) - 2200))
    provider_ms = lambda t: int((t - video) * 1000)  # noqa: E731
    async with provider_case(gemini, resolver, (run_, *inputs[1:]), s3) as (
        provider, wire, media,
    ):
        assert media.time_origin.offset_ms == offset_ms
        assert media.time_origin.container_start_time and media.time_origin.video_start_time
        await uploaded(provider)
        request = request_for(media, "technique_scan")
        wire.result["candidates"][0]["content"]["parts"][0]["text"] = json.dumps(
            dict(
                schema_version=1,
                events=[dict(
                    bounds=dict(start_ms=provider_ms(first), end_ms=provider_ms(last)),
                    description="a", reasoning="b",
                )],
            )
        )  # fmt: skip
        reply = await provider.generate(media.run, request, uuid4(), uuid4())
        sent = json.loads(wire.generates()[0][2])["contents"][0]["parts"][-2]
        scope = media.run.spec.scope
        assert sent["videoMetadata"]["startOffset"] == f"{max(0, scope.start_ms - offset_ms) / 1000:.3f}s"
        assert sent["videoMetadata"]["endOffset"] == f"{(scope.end_ms - offset_ms) / 1000:.3f}s"
        prompt = json.loads(json.loads(wire.generates()[0][2])["contents"][0]["parts"][-1]["text"])
        assert prompt["observation_window"]["end_ms"] == scope.end_ms - offset_ms
        result = parse_response(request, reply.text)
        bounds = result.events[0].bounds
        for got, want in ((bounds.start_ms, source_ms(first)), (bounds.end_ms, source_ms(last))):
            assert abs(got - want) <= 1 <= 1000 / FPS, (case, got, want)


def reject_probe(resolver, monkeypatch, streams, container):
    probe_value = dict(
        streams=streams, format=dict(format_name="mov,mp4", duration="3.0", **container)
    )
    monkeypatch.setattr(resolver, "_probe", lambda path, formats: probe_value)


VIDEO = dict(codec_type="video", width=64, height=48)


@pytest.mark.parametrize(
    "streams, container, reason",
    [
        ([dict(VIDEO)], dict(start_time="0.0"), "video stream start_time is unreadable"),
        ([dict(VIDEO, start_time="N/A")], dict(start_time="0.0"), "video stream start_time is unreadable"),
        ([dict(VIDEO, start_time="0.0")], dict(), "container start_time is unreadable"),
        ([dict(VIDEO, start_time="-0.5")], dict(start_time="0.0"), "starts before the container time origin"),
        ([dict(VIDEO, start_time="3.5")], dict(start_time="0.0"), "starts after the container duration"),
        ([dict(VIDEO, start_time="0"), dict(VIDEO, start_time="0")], dict(start_time="0"), "more than one video stream"),
        ([dict(VIDEO, start_time="0", disposition=dict(attached_pic=1))], dict(start_time="0"), "only an attached picture"),
    ],
)  # fmt: skip
def test_untimable_sources_reject_with_a_specific_reason(
    resolver, adapter, monkeypatch, streams, container, reason
):
    from types import SimpleNamespace

    reject_probe(resolver, monkeypatch, streams, container)
    run_ = SimpleNamespace(spec=SimpleNamespace(source_duration_ms=3000))
    with pytest.raises(adapter.MediaIntegrityError, match=reason):
        resolver._video("ignored", run_)


def test_negative_container_origin_is_mapped_not_rejected(resolver, monkeypatch):
    from types import SimpleNamespace

    reject_probe(
        resolver, monkeypatch, [dict(VIDEO, start_time="-0.460")], dict(start_time="-0.500")
    )
    run_ = SimpleNamespace(spec=SimpleNamespace(source_duration_ms=3000))
    _, _, origin = resolver._video("ignored", run_)
    assert origin.offset_ms == 40 and origin.container_start_time == "-0.500"


@pytest.mark.asyncio
async def test_cover_art_is_rejected_end_to_end(raw, resolver, inputs, tmp_path):
    art, path = tmp_path / "art.png", tmp_path / "cover.mp4"
    run("-f", "lavfi", "-i", "color=red:size=32x32", "-frames:v", "1", str(art))
    run("-i", str(raw), "-i", str(art), "-map", "0", "-map", "1", "-c", "copy",
        "-disposition:v:1", "attached_pic", str(path))  # fmt: skip
    run_, container, video = with_source(inputs, inputs[2], path)
    with pytest.raises(Exception, match="more than one video stream"):
        async with resolver.open(run_):
            pytest.fail("cover art must not be normalized")
