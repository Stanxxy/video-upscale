"""Tests for ProcessingMode enum and TrackRequest integration (M4 api)."""
import pytest
from pydantic import ValidationError


class TestProcessingModeEnum:
    def test_fast_member_exists(self):
        from service.models import ProcessingMode
        assert ProcessingMode.FAST == "fast"
        assert ProcessingMode.FAST.value == "fast"

    def test_standard_member_exists(self):
        from service.models import ProcessingMode
        assert ProcessingMode.STANDARD == "standard"
        assert ProcessingMode.STANDARD.value == "standard"

    def test_str_subclass(self):
        """ProcessingMode is a str subclass so it serializes cleanly."""
        from service.models import ProcessingMode
        assert isinstance(ProcessingMode.FAST, str)
        assert isinstance(ProcessingMode.STANDARD, str)

    def test_comparison_with_string(self):
        from service.models import ProcessingMode
        assert ProcessingMode.FAST == "fast"
        assert ProcessingMode.STANDARD == "standard"

    def test_only_two_members(self):
        from service.models import ProcessingMode
        members = list(ProcessingMode)
        assert len(members) == 2
        values = {m.value for m in members}
        assert values == {"fast", "standard"}


class TestTrackRequestProcessingMode:
    def _make_minimal_request(self, **kwargs):
        """Build a minimal TrackRequest with required fields."""
        from service.models import TrackRequest
        return TrackRequest(bucket="test-bucket", key="test/video.mp4", **kwargs)

    def test_default_is_standard(self):
        req = self._make_minimal_request()
        from service.models import ProcessingMode
        assert req.processing_mode == ProcessingMode.STANDARD

    def test_accepts_fast_string(self):
        req = self._make_minimal_request(processing_mode="fast")
        from service.models import ProcessingMode
        assert req.processing_mode == ProcessingMode.FAST

    def test_accepts_standard_string(self):
        req = self._make_minimal_request(processing_mode="standard")
        from service.models import ProcessingMode
        assert req.processing_mode == ProcessingMode.STANDARD

    def test_accepts_fast_enum(self):
        from service.models import ProcessingMode
        req = self._make_minimal_request(processing_mode=ProcessingMode.FAST)
        assert req.processing_mode == ProcessingMode.FAST

    def test_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            self._make_minimal_request(processing_mode="turbo")

    def test_processing_mode_serializes_to_string(self):
        req = self._make_minimal_request(processing_mode="fast")
        dumped = req.model_dump()
        assert dumped["processing_mode"] == "fast"

    def test_processing_mode_in_json_schema(self):
        from service.models import TrackRequest
        schema = TrackRequest.model_json_schema()
        # processing_mode field should appear in the schema
        props = schema.get("properties", {})
        assert "processing_mode" in props
