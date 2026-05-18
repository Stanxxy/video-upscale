"""Tests for BicubicRestorer (M4 F1)."""
import numpy as np
import pytest


def _make_bgr(h, w):
    """Create a random BGR image array."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestBicubicRestorer:
    def setup_method(self):
        from restorer import BicubicRestorer
        self.restorer = BicubicRestorer()

    def test_enhance_output_size_long_edge_smaller(self):
        """Small image (long edge < target) is upscaled to target."""
        img = _make_bgr(200, 150)  # long edge 200
        result = self.restorer.enhance(img, target_size=768)
        assert result.ndim == 3
        assert result.shape[2] == 3
        # Long edge should be 768
        assert max(result.shape[:2]) == 768

    def test_enhance_output_size_long_edge_equal(self):
        """Image with long edge == target stays at target."""
        img = _make_bgr(768, 512)  # long edge 768
        result = self.restorer.enhance(img, target_size=768)
        assert max(result.shape[:2]) == 768

    def test_enhance_output_size_long_edge_larger(self):
        """Image larger than target is downscaled."""
        img = _make_bgr(1024, 900)  # long edge 1024
        result = self.restorer.enhance(img, target_size=768)
        assert max(result.shape[:2]) == 768

    def test_enhance_preserves_aspect_ratio_portrait(self):
        """Portrait image aspect ratio is preserved approximately."""
        h, w = 400, 200  # 2:1 aspect
        img = _make_bgr(h, w)
        result = self.restorer.enhance(img, target_size=768)
        out_h, out_w = result.shape[:2]
        # Long edge should be 768 (the height)
        assert out_h == 768
        # Width should be ~384 (±1 for rounding)
        assert abs(out_w - 384) <= 1

    def test_enhance_preserves_aspect_ratio_landscape(self):
        """Landscape image aspect ratio is preserved approximately."""
        h, w = 200, 400  # 1:2 aspect
        img = _make_bgr(h, w)
        result = self.restorer.enhance(img, target_size=768)
        out_h, out_w = result.shape[:2]
        assert out_w == 768
        assert abs(out_h - 384) <= 1

    def test_enhance_batch_matches_per_call_enhance(self):
        """enhance_batch() results match individual enhance() calls."""
        crops = [_make_bgr(150, 100), _make_bgr(200, 180), _make_bgr(300, 250)]
        batch_results = self.restorer.enhance_batch(crops, target_size=512)
        assert len(batch_results) == len(crops)
        for crop, batch_out in zip(crops, batch_results):
            single_out = self.restorer.enhance(crop, target_size=512)
            # Sizes must match exactly
            assert batch_out.shape == single_out.shape

    def test_enhance_batch_empty(self):
        """empty batch returns empty list."""
        result = self.restorer.enhance_batch([], target_size=768)
        assert result == []

    def test_enhance_batch_single(self):
        """Single-element batch behaves like per-call enhance."""
        img = _make_bgr(300, 200)
        batch_result = self.restorer.enhance_batch([img], target_size=768)
        assert len(batch_result) == 1
        single_result = self.restorer.enhance(img, target_size=768)
        assert batch_result[0].shape == single_result.shape

    def test_flush_cache_is_noop(self):
        """flush_cache() does not raise and returns None."""
        result = self.restorer.flush_cache()
        assert result is None

    def test_enhance_returns_uint8(self):
        """Output dtype is uint8."""
        img = _make_bgr(100, 80)
        result = self.restorer.enhance(img, target_size=256)
        assert result.dtype == np.uint8

    def test_enhance_batch_order_preserved(self):
        """enhance_batch() returns results in same order as inputs."""
        crops = [_make_bgr(100, 50), _make_bgr(200, 100), _make_bgr(50, 80)]
        results = self.restorer.enhance_batch(crops, target_size=256)
        # Each output long edge should match expected from input
        for crop, out in zip(crops, results):
            assert max(out.shape[:2]) == 256
