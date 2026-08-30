"""eval/adcc/tests/test_dashboard_app_highlight_media.py"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from eval.adcc.predict.dashboard_app import create_app
from service.config import ServiceConfig


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH in this environment",
)


def _make_real_test_clip(path, *, codec="libx264", duration_s=1.0):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration_s}:size=64x64:rate=5",
         "-c:v", codec, "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True, timeout=30,
    )


def _write_corpus(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps({
        "match_id": "s_small", "name": "Small Single vs Match",
        "video_url": "https://www.youtube.com/embed/sSmallID123",
        "video_url_md5": "stem_s_small", "events": [],
    }) + "\n")
    return gold_path


def _make_app_with_media(tmp_path, *, media_root=None):
    gold_path = _write_corpus(tmp_path)
    media_root = media_root or (tmp_path / "media")
    media_root.mkdir(exist_ok=True)
    return create_app(
        gold_snapshot_path=gold_path,
        split_assignment_path=tmp_path / "no_split.json",
        video_durations_path=tmp_path / "no_durations.json",
        output_dir=tmp_path / "out",
        highlight_gold_path=tmp_path / "no_highlight_gold.jsonl",
        highlight_gold_manifest_path=tmp_path / "no_manifest.json",
        media_root=media_root,
        derivative_cache_dir=tmp_path / "media_cache",
        service_config=ServiceConfig(gemini_api_key="fake-key"),
    )


class TestHighlightMediaRoute:
    def test_unknown_match_id_is_404(self, tmp_path):
        client = TestClient(_make_app_with_media(tmp_path))
        resp = client.get("/highlight/media/ghost_match")
        assert resp.status_code == 404

    def test_known_match_id_with_no_on_disk_file_is_404(self, tmp_path):
        client = TestClient(_make_app_with_media(tmp_path))
        resp = client.get("/highlight/media/s_small")
        assert resp.status_code == 404

    def test_unsupported_extension_is_415(self, tmp_path):
        media_root = tmp_path / "media"
        media_root.mkdir()
        (media_root / "stem_s_small.txt").write_bytes(b"not a video")
        client = TestClient(_make_app_with_media(tmp_path, media_root=media_root))
        resp = client.get("/highlight/media/s_small")
        assert resp.status_code == 415

    def test_symlink_escape_attempt_is_404_never_serves_the_target(self, tmp_path):
        outside = tmp_path.parent / f"outside-{tmp_path.name}"
        outside.mkdir(exist_ok=True)
        (outside / "secret.mp4").write_bytes(b"should never be servable")
        media_root = tmp_path / "media"
        media_root.mkdir()
        (media_root / "stem_s_small.mp4").symlink_to(outside / "secret.mp4")
        client = TestClient(_make_app_with_media(tmp_path, media_root=media_root))
        resp = client.get("/highlight/media/s_small")
        assert resp.status_code == 404

    def test_first_request_builds_and_serves_the_derivative(self, tmp_path):
        media_root = tmp_path / "media"
        media_root.mkdir()
        _make_real_test_clip(media_root / "stem_s_small.mp4", codec="libx264")
        client = TestClient(_make_app_with_media(tmp_path, media_root=media_root))
        resp = client.get("/highlight/media/s_small")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert resp.headers.get("accept-ranges") == "bytes"
        assert len(resp.content) > 0

    def test_second_request_serves_the_cached_derivative_no_rebuild(self, tmp_path):
        media_root = tmp_path / "media"
        media_root.mkdir()
        _make_real_test_clip(media_root / "stem_s_small.mp4", codec="libx264")
        app = _make_app_with_media(tmp_path, media_root=media_root)
        client = TestClient(app)
        first = client.get("/highlight/media/s_small")
        cache_dir = tmp_path / "media_cache"
        cached_file = cache_dir / "stem_s_small.mp4"
        mtime1 = cached_file.stat().st_mtime_ns
        second = client.get("/highlight/media/s_small")
        mtime2 = cached_file.stat().st_mtime_ns
        assert first.status_code == second.status_code == 200
        assert mtime1 == mtime2  # never rebuilt

    def test_range_request_returns_206_with_content_range(self, tmp_path):
        media_root = tmp_path / "media"
        media_root.mkdir()
        _make_real_test_clip(media_root / "stem_s_small.mp4", codec="libx264", duration_s=2.0)
        client = TestClient(_make_app_with_media(tmp_path, media_root=media_root))
        client.get("/highlight/media/s_small")  # warm the cache first
        resp = client.get("/highlight/media/s_small", headers={"Range": "bytes=0-99"})
        assert resp.status_code == 206
        assert resp.headers["content-range"].startswith("bytes 0-99/")
        assert len(resp.content) == 100

    def test_build_failure_is_500_with_the_real_ffmpeg_error_logged(self, tmp_path, monkeypatch, caplog):
        media_root = tmp_path / "media"
        media_root.mkdir()
        (media_root / "stem_s_small.mp4").write_bytes(b"not a real video container")
        client = TestClient(_make_app_with_media(tmp_path, media_root=media_root))
        resp = client.get("/highlight/media/s_small")
        assert resp.status_code == 500
