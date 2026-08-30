"""eval/adcc/tests/test_media_resolver.py"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from eval.adcc.predict import media_resolver as mr


pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH in this environment",
)


def _make_real_test_clip(path: Path, *, codec: str = "libx264", duration_s: float = 1.0) -> None:
    """Generates a tiny REAL synthetic video via ffmpeg's testsrc filter --
    no network, no fixture asset to maintain, deterministic. Used to
    exercise the REAL ffmpeg/ffprobe subprocess path end-to-end rather than
    mocking the transcode itself (Evaluator Loop Protocol: reject shallow
    mocks for the mechanism actually under test)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration_s}:size=64x64:rate=5",
            "-c:v", codec, "-pix_fmt", "yuv420p" if codec == "libx264" else "yuv420p",
            str(path),
        ],
        capture_output=True, check=True, timeout=30,
    )


class TestMediaRootIndexBuild:
    def test_unknown_match_id_raises_unknown_media_error(self, tmp_path):
        idx = mr.MediaRootIndex.build(tmp_path, match_id_to_stem={})
        with pytest.raises(mr.UnknownMediaError):
            idx.resolve("ghost_match")

    def test_known_match_id_with_no_on_disk_file_raises_unknown_media_error(self, tmp_path):
        idx = mr.MediaRootIndex.build(tmp_path, match_id_to_stem={"m1": "stem1"})
        with pytest.raises(mr.UnknownMediaError):
            idx.resolve("m1")

    def test_unknown_media_error_message_is_mode_indistinguishable(self, tmp_path):
        """Unknown / missing / outside-root must share one path-free message."""
        outside = tmp_path.parent / f"outside-oracle-{tmp_path.name}"
        outside.mkdir(exist_ok=True)
        escaped = outside / "secret.mp4"
        escaped.write_bytes(b"x")
        (tmp_path / "stem1.mp4").write_bytes(b"ok")
        idx = mr.MediaRootIndex.build(tmp_path, match_id_to_stem={"m1": "stem1", "m2": "missing"})
        with pytest.raises(mr.UnknownMediaError) as unknown:
            idx.resolve("ghost")
        with pytest.raises(mr.UnknownMediaError) as missing:
            idx.resolve("m2")
        # Poison index to exercise handoff re-realpath escape path.
        idx.stem_to_source_path["stem1"] = escaped
        with pytest.raises(mr.UnknownMediaError) as outside_root:
            idx.resolve("m1")
        msgs = {str(unknown.value), str(missing.value), str(outside_root.value)}
        assert len(msgs) == 1
        only = next(iter(msgs))
        assert only == "unknown or unavailable match_id"
        assert "stem" not in only
        assert "/" not in only
        assert "secret" not in only

    def test_resolves_a_real_on_disk_file_by_stem(self, tmp_path):
        (tmp_path / "stem1.mp4").write_bytes(b"fake video bytes")
        idx = mr.MediaRootIndex.build(tmp_path, match_id_to_stem={"m1": "stem1"})
        resolved = idx.resolve("m1")
        assert resolved.match_id == "m1"
        assert resolved.stem == "stem1"
        assert resolved.source_path == (tmp_path / "stem1.mp4").resolve()
        assert resolved.source_extension == ".mp4"

    def test_unsupported_extension_raises_415_class_error(self, tmp_path):
        (tmp_path / "stem1.txt").write_bytes(b"not a video")
        idx = mr.MediaRootIndex.build(tmp_path, match_id_to_stem={"m1": "stem1"})
        with pytest.raises(mr.UnsupportedExtensionError):
            idx.resolve("m1")

    def test_duplicate_on_disk_stem_fails_loud_at_build_time(self, tmp_path):
        (tmp_path / "stem1.mp4").write_bytes(b"a")
        (tmp_path / "stem1.mkv").write_bytes(b"b")
        with pytest.raises(mr.DuplicateStemError):
            mr.MediaRootIndex.build(tmp_path, match_id_to_stem={"m1": "stem1"})

    def test_multiple_match_ids_sharing_one_stem_is_not_a_duplicate_error(self, tmp_path):
        # Multi-match VODs legitimately share ONE video file across MANY
        # match_ids -- this is expected, correct behavior (build_match_
        # targets.is_multi_match), NOT the ambiguity DuplicateStemError
        # guards against (which is about >1 ON-DISK FILE for one stem).
        (tmp_path / "shared_stem.mp4").write_bytes(b"a")
        idx = mr.MediaRootIndex.build(
            tmp_path, match_id_to_stem={"m1": "shared_stem", "m2": "shared_stem"},
        )
        r1 = idx.resolve("m1")
        r2 = idx.resolve("m2")
        assert r1.source_path == r2.source_path

    def test_symlink_escaping_the_media_root_is_never_indexed(self, tmp_path):
        outside_dir = tmp_path.parent / f"outside-{tmp_path.name}"
        outside_dir.mkdir(exist_ok=True)
        secret_file = outside_dir / "secret.mp4"
        secret_file.write_bytes(b"should never be servable")
        media_root = tmp_path / "media_root"
        media_root.mkdir()
        (media_root / "stem1.mp4").symlink_to(secret_file)
        idx = mr.MediaRootIndex.build(media_root, match_id_to_stem={"m1": "stem1"})
        with pytest.raises(mr.UnknownMediaError):
            idx.resolve("m1")  # never resolvable -- the escaping symlink was skipped at index time

    def test_symlink_staying_inside_the_media_root_resolves_normally(self, tmp_path):
        media_root = tmp_path / "media_root"
        media_root.mkdir()
        real_file = media_root / "real.mp4"
        real_file.write_bytes(b"real bytes")
        (media_root / "stem1.mp4").symlink_to(real_file)
        idx = mr.MediaRootIndex.build(media_root, match_id_to_stem={"m1": "stem1"})
        resolved = idx.resolve("m1")
        assert resolved.source_path == real_file.resolve()

    def test_missing_media_root_directory_builds_an_empty_index_not_a_crash(self, tmp_path):
        idx = mr.MediaRootIndex.build(tmp_path / "does-not-exist", match_id_to_stem={"m1": "stem1"})
        with pytest.raises(mr.UnknownMediaError):
            idx.resolve("m1")

    def test_known_video_extensions_reuses_generate_video_durations_constant(self):
        from eval.adcc.predict.generate_video_durations import VIDEO_EXTENSIONS
        assert mr.KNOWN_VIDEO_EXTENSIONS is VIDEO_EXTENSIONS


class TestProbeIsH264:
    @pytestmark_ffmpeg
    @pytest.mark.asyncio
    async def test_h264_source_is_detected_as_h264(self, tmp_path):
        clip = tmp_path / "h264.mp4"
        _make_real_test_clip(clip, codec="libx264")
        assert await mr.probe_is_h264(clip) is True

    @pytestmark_ffmpeg
    @pytest.mark.asyncio
    async def test_non_h264_source_is_detected_as_not_h264(self, tmp_path):
        clip = tmp_path / "mpeg4.mp4"
        _make_real_test_clip(clip, codec="mpeg4")
        assert await mr.probe_is_h264(clip) is False


class TestBuildMp4Derivative:
    @pytestmark_ffmpeg
    @pytest.mark.asyncio
    async def test_builds_a_real_playable_derivative_from_an_h264_source(self, tmp_path):
        source = tmp_path / "source.mp4"
        _make_real_test_clip(source, codec="libx264")
        out = tmp_path / "derivative.mp4"
        await mr.build_mp4_derivative(source, out)
        assert out.exists() and out.stat().st_size > 0
        assert await mr.probe_is_h264(out) is True

    @pytestmark_ffmpeg
    @pytest.mark.asyncio
    async def test_builds_a_real_playable_derivative_from_a_non_h264_source(self, tmp_path):
        source = tmp_path / "source.mp4"
        _make_real_test_clip(source, codec="mpeg4")
        out = tmp_path / "derivative.mp4"
        await mr.build_mp4_derivative(source, out)
        assert out.exists() and out.stat().st_size > 0
        assert await mr.probe_is_h264(out) is True  # always re-encoded to h264 in the derivative

    @pytest.mark.asyncio
    async def test_ffmpeg_failure_raises_media_build_error_with_real_stderr(self, tmp_path, monkeypatch):
        async def _fake_exec(*args, **kwargs):
            class _FakeProc:
                returncode = 1
                async def communicate(self):
                    return b"", b"ffmpeg: real error text naming the actual problem"
            return _FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
        with pytest.raises(mr.MediaBuildError, match="real error text naming the actual problem"):
            await mr.build_mp4_derivative(tmp_path / "nonexistent.mp4", tmp_path / "out.mp4")


class TestDerivativeCache:
    @pytestmark_ffmpeg
    @pytest.mark.asyncio
    async def test_first_call_builds_second_call_reuses_the_cache_no_rebuild(self, tmp_path):
        source = tmp_path / "source.mp4"
        _make_real_test_clip(source, codec="libx264")
        resolved = mr.ResolvedMedia(match_id="m1", stem="stem1", source_path=source, source_extension=".mp4")
        cache = mr.DerivativeCache(cache_dir=tmp_path / "cache")

        path1 = await cache.ensure_derivative(resolved)
        mtime1 = path1.stat().st_mtime_ns
        path2 = await cache.ensure_derivative(resolved)
        mtime2 = path2.stat().st_mtime_ns
        assert path1 == path2
        assert mtime1 == mtime2  # never rebuilt on the second call

    @pytest.mark.asyncio
    async def test_concurrent_requests_for_the_same_stem_trigger_exactly_one_build(self, tmp_path, monkeypatch):
        build_calls = {"n": 0}

        async def _fake_build(source_path, out_path):
            build_calls["n"] += 1
            await asyncio.sleep(0.05)  # simulate real transcode latency
            out_path.write_bytes(b"fake derivative bytes")

        monkeypatch.setattr(mr, "build_mp4_derivative", _fake_build)
        resolved = mr.ResolvedMedia(match_id="m1", stem="stem1", source_path=tmp_path / "src.mp4", source_extension=".mp4")
        cache = mr.DerivativeCache(cache_dir=tmp_path / "cache")

        results = await asyncio.gather(
            cache.ensure_derivative(resolved), cache.ensure_derivative(resolved), cache.ensure_derivative(resolved),
        )
        assert build_calls["n"] == 1  # concurrency-safe single generation per stem (AC2-6)
        assert all(p == results[0] for p in results)

    @pytest.mark.asyncio
    async def test_a_build_failure_never_leaves_a_partial_file_at_the_cached_path(self, tmp_path, monkeypatch):
        async def _fake_build(source_path, out_path):
            out_path.write_bytes(b"partial garbage")
            raise mr.MediaBuildError("simulated ffmpeg crash mid-write")

        monkeypatch.setattr(mr, "build_mp4_derivative", _fake_build)
        resolved = mr.ResolvedMedia(match_id="m1", stem="stem1", source_path=tmp_path / "src.mp4", source_extension=".mp4")
        cache = mr.DerivativeCache(cache_dir=tmp_path / "cache")
        with pytest.raises(mr.MediaBuildError):
            await cache.ensure_derivative(resolved)
        assert not cache.cached_path_for("stem1").exists()  # never a corrupt/partial file at the FINAL path
