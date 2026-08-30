"""``eval/adcc/predict/dashboard_app.py``'s Highlight Eval endpoints
(``/highlight``, ``/highlight/plan``, ``/highlight/run``, ``/highlight/score``
— frozen contract §E). Every test builds an ISOLATED app via
``create_app(...)`` pointed at ``tmp_path`` fixtures. ZERO real Gemini calls:
``/highlight/plan`` is pure arithmetic; ``/highlight/run``'s mocked-Gemini
test monkeypatches ``highlight_discovery.build_gemini_client`` (this
feature's OWN client constructor, never ``RunContext.gemini_client``).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from eval.adcc.predict import highlight_discovery as hd
from eval.adcc.predict.dashboard_app import create_app
from service.config import ServiceConfig


def _hl(start_s, end_s, reason="submission — armbar — arm fully extended and secured"):
    """A discovery-schema highlight item WITH the (frozen contract §A)
    required 'reason' field — every synthetic Gemini discovery JSON payload
    in this file must include it or discover_match_highlights correctly
    rejects the response."""
    return {"start_s": start_s, "end_s": end_s, "reason": reason}


def _corpus_records():
    # video_url is a real-SHAPED (fake) YouTube /embed/<11-char-id> URL --
    # live-bug fix regression: discover_match_highlights now rejects
    # unparseable video URLs (normalize_youtube_watch_url), mirroring the
    # real ADCC gold snapshot's own embed-URL field (INS-084).
    return [
        {"match_id": "s_small", "name": "Small Single vs Match", "video_url": "https://www.youtube.com/embed/sSmallID123",
         "video_url_md5": "stem_s_small", "events": [
             {"t_s": 10, "action_class": "submission_choke", "outcome": "successful"},
             {"t_s": 400, "action_class": "escape", "outcome": "successful"},
         ]},
        # >720s (LONG_WINDOW_RISK_THRESHOLD_S) — the Bezos fast-follow's
        # long_window_risk=True case.
        {"match_id": "s_long", "name": "Long Single vs Match", "video_url": "https://www.youtube.com/embed/sLongID1234",
         "video_url_md5": "stem_s_long", "events": [{"t_s": 10, "action_class": "sweep", "outcome": "successful"}]},
    ]


def _write_corpus(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    with open(gold_path, "w") as f:
        for rec in _corpus_records():
            f.write(json.dumps(rec) + "\n")
    durations_path = tmp_path / "durations.json"
    durations_path.write_text(json.dumps({"stem_s_small": 600.0, "stem_s_long": 1000.0}))
    return gold_path, durations_path


def _write_highlight_gold(tmp_path):
    gold_path = tmp_path / "highlight_gold.jsonl"
    gold_path.write_text(json.dumps({
        "match_id": "s_small", "video_url": "https://x/s_small", "video_url_md5": "stem_s_small",
        "name": "Small Single vs Match", "events": [{"t_s": 10, "action_class": "submission_choke", "outcome": "successful"}],
    }) + "\n")
    manifest_path = tmp_path / "highlight_gold_manifest.json"
    manifest_path.write_text(json.dumps({"predicate_version": "highlight_predicate_v1", "include_takedowns": False}))
    return gold_path, manifest_path


def _make_app(tmp_path, gemini_api_key: str = "fake-key"):
    gold_path, durations_path = _write_corpus(tmp_path)
    highlight_gold_path, highlight_gold_manifest_path = _write_highlight_gold(tmp_path)
    return create_app(
        gold_snapshot_path=gold_path,
        split_assignment_path=tmp_path / "no_split.json",
        video_durations_path=durations_path,
        output_dir=tmp_path / "out",
        highlight_gold_path=highlight_gold_path,
        highlight_gold_manifest_path=highlight_gold_manifest_path,
        service_config=ServiceConfig(gemini_api_key=gemini_api_key),
    )


def _fake_client_returning(highlights_json_text: str):
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=highlights_json_text, usage_metadata=None)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    return fake_client


# --------------------------------------------------------------------------- #
# GET /highlight
# --------------------------------------------------------------------------- #
def test_highlight_index_404s_when_client_html_missing(tmp_path):
    gold_path, durations_path = _write_corpus(tmp_path)
    highlight_gold_path, highlight_gold_manifest_path = _write_highlight_gold(tmp_path)
    app = create_app(
        gold_snapshot_path=gold_path,
        split_assignment_path=tmp_path / "no_split.json",
        video_durations_path=durations_path,
        output_dir=tmp_path / "out",
        highlight_gold_path=highlight_gold_path,
        highlight_gold_manifest_path=highlight_gold_manifest_path,
        highlight_html_path=tmp_path / "does-not-exist.html",
        service_config=ServiceConfig(gemini_api_key="fake-key"),
    )
    client = TestClient(app)
    resp = client.get("/highlight")
    assert resp.status_code == 404


def test_highlight_index_serves_the_real_client_html():
    from eval.adcc.predict.dashboard_app import app
    client = TestClient(app)
    resp = client.get("/highlight")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# --------------------------------------------------------------------------- #
# POST /highlight/plan — zero Gemini calls, pure arithmetic
# --------------------------------------------------------------------------- #
def test_highlight_plan_returns_estimate_with_zero_gemini_calls(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("POST /highlight/plan must NEVER touch a Gemini client")

    monkeypatch.setattr(hd, "build_gemini_client", _boom)
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "refine": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["match_id"] == "s_small"
    assert body["window"] == [0.0, 600.0]
    assert body["est_calls"] == 1
    assert "plan_token" in body and len(body["plan_token"]) == 16


def test_highlight_plan_defaults_fps_and_frame_quality_when_omitted(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5})
    body = resp.json()
    assert body["fps"] == hd.DEFAULT_FPS
    assert body["frame_quality"] == hd.DEFAULT_FRAME_QUALITY


def test_highlight_plan_prices_via_the_dedicated_highlight_cost_formula(tmp_path):
    from eval.adcc.predict.cost_model import estimate_highlight_call_cost_usd
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/highlight/plan",
        json={"match_id": "s_small", "cap": 5, "fps": 2.0, "frame_quality": "high", "refine": False},
    )
    body = resp.json()
    expected = estimate_highlight_call_cost_usd(600.0, 2.0, "high")
    assert body["est_dollars_realistic"] == pytest.approx(round(expected, 2))


def test_highlight_plan_unknown_match_id_is_400(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "ghost", "cap": 5})
    assert resp.status_code == 400


def test_highlight_plan_rejects_out_of_family_model(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "model": "gemini-2.0-flash"})
    assert resp.status_code == 400


def test_highlight_plan_rejects_out_of_family_fps(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "fps": 3.0})
    assert resp.status_code == 400


def test_highlight_plan_rejects_out_of_family_frame_quality(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "frame_quality": "ultra"})
    assert resp.status_code == 400


def test_highlight_plan_rejects_out_of_family_refine_fps(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "refine_fps": 3.0})
    assert resp.status_code == 400


def test_highlight_plan_rejects_negative_refine_pad_before(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "refine_pad_before_s": -1.0})
    assert resp.status_code == 400


def test_highlight_plan_rejects_negative_refine_pad_after(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "refine_pad_after_s": -1.0})
    assert resp.status_code == 400


class TestHighlightPlanRefinementPreview:
    """Frozen contract §E (Brooks — cost-anchored range, not corpus-average)."""

    def test_refine_true_returns_a_cost_anchored_range_not_a_single_number(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5})  # refine defaults True
        assert resp.status_code == 200
        body = resp.json()
        assert body["refine"] is True
        assert body["est_calls_low"] == 1
        assert body["est_calls_high"] == 5  # anchored to the enforced cap ceiling, not a corpus average
        assert body["est_dollars_high_realistic"] >= body["est_dollars_low_realistic"]
        assert "est_calls" not in body  # the refine=False single-number shape must NOT leak into refine=True

    def test_refine_true_includes_an_informational_typical_aside(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 50})
        body = resp.json()
        # typical ~= 1 discovery + round(2.16) highlights = 3, per the corpus's
        # own real highlight density -- informational only, never the priced ceiling.
        assert body["est_calls_typical"] == 3
        assert body["est_calls_typical"] < body["est_calls_high"]

    def test_refine_false_returns_exactly_the_pre_refinement_single_call_shape(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "refine": False})
        body = resp.json()
        assert body["refine"] is False
        assert body["est_calls"] == 1
        assert "est_calls_low" not in body and "est_calls_high" not in body
        assert "est_calls_typical" not in body


class TestLongWindowRisk:
    """Bezos fast-follow (non-blocking, evaluator-converged): a WARNING
    surfaced in the priced preview, never a gate."""

    def test_short_window_reports_no_risk(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        body = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5}).json()
        assert body["window_duration_s"] == pytest.approx(600.0)
        assert body["long_window_risk"] is False

    def test_long_window_reports_risk(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        body = client.post("/highlight/plan", json={"match_id": "s_long", "cap": 5}).json()
        assert body["window_duration_s"] == pytest.approx(1000.0)
        assert body["long_window_risk"] is True

    def test_risk_flag_never_blocks_the_preview(self, tmp_path):
        # A WARNING only -- still 200, still a usable plan_token, per the
        # "do NOT add slicing / do NOT gate" instruction.
        client = TestClient(_make_app(tmp_path))
        resp = client.post("/highlight/plan", json={"match_id": "s_long", "cap": 5})
        assert resp.status_code == 200
        assert resp.json()["plan_token"]


class TestTokenInvalidation:
    """The frozen contract's core AC: changing model/thinking_level/prompt/
    cap/scope invalidates the token (server-recomputed, never client-side
    only)."""

    def _base_payload(self):
        return {
            "match_id": "s_small", "cap": 5, "model": hd.DEFAULT_HIGHLIGHT_MODEL,
            "thinking_level": "off", "system_prompt": "sys", "followup_prompt": "find highlights",
            "fps": hd.DEFAULT_FPS, "frame_quality": hd.DEFAULT_FRAME_QUALITY,
        }

    def test_token_stable_for_identical_inputs(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        r1 = client.post("/highlight/plan", json=self._base_payload()).json()
        r2 = client.post("/highlight/plan", json=self._base_payload()).json()
        assert r1["plan_token"] == r2["plan_token"]

    def test_token_changes_when_model_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, model="gemini-3.1-flash-lite")
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_thinking_level_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, thinking_level="high")
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_system_prompt_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, system_prompt="a completely different system prompt")
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_followup_prompt_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, followup_prompt="a completely different followup prompt")
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_cap_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, cap=99)
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_match_id_changes(self, tmp_path):
        # a second match_id doesn't exist in this fixture's corpus, so use
        # the plan_token computation directly via two different match_ids
        # against the SAME single-match corpus by comparing to a manually
        # computed token for a hypothetical different match_id.
        from eval.adcc.predict.dashboard_scope import compute_highlight_plan_token
        base = self._base_payload()
        t1 = compute_highlight_plan_token(
            base["match_id"], base["cap"], base["model"], base["thinking_level"],
            base["system_prompt"], base["followup_prompt"], base["fps"], base["frame_quality"],
        )
        t2 = compute_highlight_plan_token(
            "different_match_id", base["cap"], base["model"], base["thinking_level"],
            base["system_prompt"], base["followup_prompt"], base["fps"], base["frame_quality"],
        )
        assert t1 != t2

    def test_token_changes_when_fps_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, fps=2.0)
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_frame_quality_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, frame_quality="high")
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_stays_constant_when_nothing_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        r2 = client.post("/highlight/plan", json=dict(base)).json()
        assert r1["plan_token"] == r2["plan_token"]

    def test_token_changes_when_refine_toggle_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, refine=False)
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_refine_pad_before_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, refine_pad_before_s=20.0)
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_refine_pad_after_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, refine_pad_after_s=10.0)
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]

    def test_token_changes_when_refine_fps_changes(self, tmp_path):
        client = TestClient(_make_app(tmp_path))
        base = self._base_payload()
        r1 = client.post("/highlight/plan", json=base).json()
        changed = dict(base, refine_fps=2.0)
        r2 = client.post("/highlight/plan", json=changed).json()
        assert r1["plan_token"] != r2["plan_token"]


# --------------------------------------------------------------------------- #
# POST /highlight/run gating
# --------------------------------------------------------------------------- #
def test_highlight_run_without_confirm_is_rejected(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "plan_token": "irrelevant", "confirm": False,
    })
    assert resp.status_code == 400
    assert "confirm" in resp.json()["detail"].lower()


def test_highlight_run_without_gemini_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("BJJ_GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    client = TestClient(_make_app(tmp_path, gemini_api_key=""))
    plan_resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5})
    token = plan_resp.json()["plan_token"]
    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 400
    assert "GEMINI_API_KEY" in resp.json()["detail"]


def test_highlight_run_with_stale_token_after_prompt_change_is_rejected(tmp_path):
    client = TestClient(_make_app(tmp_path))
    plan_resp = client.post("/highlight/plan", json={
        "match_id": "s_small", "cap": 5, "system_prompt": "original",
    })
    stale_token = plan_resp.json()["plan_token"]
    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "system_prompt": "CHANGED", "plan_token": stale_token, "confirm": True,
    })
    assert resp.status_code == 400
    assert "stale" in resp.json()["detail"].lower()


def test_highlight_run_fresh_token_is_accepted_and_persists_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hd, "build_gemini_client",
        lambda *a, **k: _fake_client_returning(json.dumps([_hl(5.0, 5.0)])),
    )
    client = TestClient(_make_app(tmp_path))
    plan_resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "refine": False})
    token = plan_resp.json()["plan_token"]

    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "refine": False, "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert lines[-1]["type"] == "batch_complete"
    assert (tmp_path / "out" / "highlights" / "s_small.json").exists()


def test_highlight_run_persists_fps_and_frame_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hd, "build_gemini_client",
        lambda *a, **k: _fake_client_returning(json.dumps([_hl(5.0, 5.0)])),
    )
    client = TestClient(_make_app(tmp_path))
    plan_resp = client.post(
        "/highlight/plan",
        json={"match_id": "s_small", "cap": 5, "fps": 2.0, "frame_quality": "high", "refine": False},
    )
    token = plan_resp.json()["plan_token"]
    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "fps": 2.0, "frame_quality": "high", "refine": False,
        "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 200
    persisted = json.loads((tmp_path / "out" / "highlights" / "s_small.json").read_text())
    assert persisted["fps"] == 2.0
    assert persisted["frame_quality"] == "high"


def test_highlight_run_persists_raw_response_text_raw_highlights_and_out_of_bounds(tmp_path, monkeypatch):
    """Live investigation (2026-07-18, LONG-video out-of-range timestamps):
    persist BOTH raw_response_text and the raw clip-relative highlights
    (before window.start_s normalization), plus out_of_bounds detection —
    makes the NEXT real run definitively diagnosable with zero extra spend."""
    raw_text = json.dumps([_hl(590.0, 610.0)])  # s_small's window is [0.0, 600.0] -> 610 is past the end
    monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: _fake_client_returning(raw_text))
    client = TestClient(_make_app(tmp_path))
    plan_resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5, "refine": False})
    token = plan_resp.json()["plan_token"]
    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "refine": False, "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 200
    persisted = json.loads((tmp_path / "out" / "highlights" / "s_small.json").read_text())
    assert persisted["raw_response_text"] == raw_text
    assert persisted["raw_highlights"] == [_hl(590.0, 610.0)]
    assert persisted["out_of_bounds_count"] == 1
    assert persisted["out_of_bounds_spans"] == [{"start_s": 590.0, "end_s": 610.0}]
    # This span [590,610] in window [0,600] fails AC1-3 containment
    # (610 > 600) -- it is now REJECTED at the batch-driver level, never
    # dropped/clamped (the raw candidate is preserved, diagnostics-only).
    assert persisted["highlights"] == [{
        "original_start_s": 590.0, "original_end_s": 610.0,
        "refined_start_s": None, "refined_end_s": None, "refined": False,
        "reason": _hl(590.0, 610.0)["reason"], "action_confirmed": None, "refinement_failed": False,
        "rejected": True, "rejection_reason": "out_of_range",
    }]


def test_highlight_run_persists_the_enriched_dict_shape_when_refinement_accepts(tmp_path, monkeypatch):
    """Frozen contract §F: each persisted highlight is ONE enriched dict
    (never parallel arrays), with refined_start_s/end_s actually populated
    when refinement accepted a correction — an end-to-end test through the
    REAL orchestration path (discovery + refine), not just the driver-level
    unit tests in test_highlight_discovery.py."""
    call_count = {"n": 0}
    responses = [
        json.dumps([_hl(10.0, 20.0)]),  # discovery: one highlight
        json.dumps({"action_confirmed": True, "start_s": 15.0, "end_s": 18.0}),  # refine: confirmed + interior
    ]

    async def _fake_generate(*, model, contents, config):
        idx = call_count["n"]
        call_count["n"] += 1
        return SimpleNamespace(text=responses[idx], usage_metadata=None)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)

    client = TestClient(_make_app(tmp_path))
    plan_resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5})  # refine defaults True
    token = plan_resp.json()["plan_token"]
    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 200
    assert call_count["n"] == 2  # discovery + one refine call

    persisted = json.loads((tmp_path / "out" / "highlights" / "s_small.json").read_text())
    assert persisted["refine"] is True
    highlight = persisted["highlights"][0]
    # ONE enriched dict, all frozen contract §F keys present, no parallel arrays.
    assert set(highlight.keys()) == {
        "original_start_s", "original_end_s", "refined_start_s", "refined_end_s",
        "refined", "reason", "action_confirmed", "refinement_failed",
        "rejected", "rejection_reason",
    }
    assert highlight["rejected"] is False
    assert highlight["rejection_reason"] is None
    assert highlight["original_start_s"] == 10.0 and highlight["original_end_s"] == 20.0
    assert highlight["refined"] is True
    assert highlight["refinement_failed"] is False
    assert highlight["action_confirmed"] is True
    # padded_start = max(0, 10-12) = 0 -> 0+15=15, 0+18=18
    assert highlight["refined_start_s"] == pytest.approx(15.0)
    assert highlight["refined_end_s"] == pytest.approx(18.0)
    assert highlight["reason"]  # requested + persisted regardless of refine toggle


def test_highlight_run_persists_partial_result_when_cap_fires_mid_refinement(tmp_path, monkeypatch):
    """Bezos gap-closure: the prior test only proves the persist-on-
    cap_reached/run_stopped fix via a generator-level unit test
    (test_highlight_discovery.py's
    test_cap_fires_mid_refinement_emits_partial_with_unrefined_remainder).
    This drives the REAL FastAPI ``/highlight/run`` endpoint end-to-end and
    asserts the PARTIAL file actually lands on disk when the cap fires
    mid-refinement (frozen contract §C, the crux): cap=2 -> discovery(1) +
    highlight #0's refine call(2) exhausts the cap; highlight #1's
    refinement is never attempted -- NEITHER highlight is dropped from the
    persisted file, and the second is tagged unrefined, never fabricated."""
    call_count = {"n": 0}
    responses = [
        json.dumps([_hl(10.0, 20.0), _hl(100.0, 110.0)]),  # discovery: TWO highlights
        json.dumps({"action_confirmed": True, "start_s": 15.0, "end_s": 18.0}),  # highlight #0's refine: accepted
    ]

    async def _fake_generate(*, model, contents, config):
        idx = call_count["n"]
        call_count["n"] += 1
        return SimpleNamespace(text=responses[idx], usage_metadata=None)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(hd, "build_gemini_client", lambda *a, **k: fake_client)

    client = TestClient(_make_app(tmp_path))
    plan_resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 2})  # refine defaults True
    token = plan_resp.json()["plan_token"]
    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 2, "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 200
    assert call_count["n"] == 2  # discovery + exactly ONE refine call -- cap enforced, no overspend

    lines = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert lines[-1]["type"] == "cap_reached"
    assert lines[-1]["calls_spent"] == 2

    highlights_path = tmp_path / "out" / "highlights" / "s_small.json"
    assert highlights_path.exists(), "the PARTIAL result must be persisted even though the run never reached batch_complete"
    persisted = json.loads(highlights_path.read_text())
    assert len(persisted["highlights"]) == 2  # NEITHER highlight dropped

    h0, h1 = persisted["highlights"]
    assert h0["original_start_s"] == 10.0 and h0["original_end_s"] == 20.0
    assert h0["refined"] is True
    assert h0["refined_start_s"] == pytest.approx(15.0)
    assert h0["refined_end_s"] == pytest.approx(18.0)

    assert h1["original_start_s"] == 100.0 and h1["original_end_s"] == 110.0
    assert h1["refined"] is False
    assert h1["refinement_failed"] is False  # "never attempted" != "failed"
    assert h1["refined_start_s"] is None
    assert h1["refined_end_s"] is None


# --------------------------------------------------------------------------- #
# Mutual exclusion (frozen contract §E) — shares app.state.active_run_state
# with the tag-batch /run.
# --------------------------------------------------------------------------- #
def test_highlight_run_409s_while_a_tag_batch_run_is_active(tmp_path):
    app = _make_app(tmp_path)
    app.state.active_run_state = object()  # simulate an in-flight /run
    client = TestClient(app)
    plan_resp = client.post("/highlight/plan", json={"match_id": "s_small", "cap": 5})
    token = plan_resp.json()["plan_token"]
    resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 409


def test_stop_endpoint_stops_a_highlight_run_state(tmp_path):
    app = _make_app(tmp_path)
    from eval.adcc.predict.global_cap_driver import BatchRunState
    state = BatchRunState(cap=5)
    app.state.active_run_state = state
    client = TestClient(app)
    resp = client.post("/stop")
    assert resp.status_code == 200
    assert state.stop_requested is True


# --------------------------------------------------------------------------- #
# POST /highlight/score
# --------------------------------------------------------------------------- #
def test_highlight_score_without_a_run_is_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post("/highlight/score", json={"match_id": "s_small", "tau_t": 5.0})
    assert resp.status_code == 404


def test_highlight_score_reports_real_plus_two_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hd, "build_gemini_client",
        lambda *a, **k: _fake_client_returning(json.dumps([_hl(10.0, 10.0)])),
    )
    client = TestClient(_make_app(tmp_path))
    plan_resp = client.post(
        "/highlight/plan",
        json={"match_id": "s_small", "cap": 5, "fps": 2.0, "frame_quality": "medium", "refine": False},
    )
    token = plan_resp.json()["plan_token"]
    run_resp = client.post("/highlight/run", json={
        "match_id": "s_small", "cap": 5, "fps": 2.0, "frame_quality": "medium", "refine": False,
        "plan_token": token, "confirm": True,
    })
    assert run_resp.status_code == 200

    score_resp = client.post("/highlight/score", json={"match_id": "s_small", "tau_t": 5.0})
    assert score_resp.status_code == 200
    body = score_resp.json()
    assert "real" in body and "baseline_whole_match" in body and "baseline_uniform_span" in body
    assert body["real"]["totals"]["tp"] == 1  # the fake pred at t=10 matches the GT submission_choke at t_s=10
    assert body["predicate_version"] == "highlight_predicate_v1"
    assert body["include_takedowns"] is False
    assert "gold_marks" in body["timeline"] and "pred_marks" in body["timeline"]
    # LeCun's provenance requirement: scores only comparable at identical
    # settings -- the settings the run actually used must echo back.
    assert body["fps"] == 2.0
    assert body["frame_quality"] == "medium"


def test_highlight_score_on_a_legacy_persisted_file_reports_null_fps_quality_not_a_crash(tmp_path):
    """A file persisted before this increment lacks fps/frame_quality keys
    entirely -- /highlight/score must degrade to null (the client renders
    "unknown/legacy"), never a 500."""
    client = TestClient(_make_app(tmp_path))
    out_dir = tmp_path / "out" / "highlights"
    out_dir.mkdir(parents=True)
    (out_dir / "s_small.json").write_text(json.dumps({
        "match_id": "s_small", "highlights": [{"start_s": 10.0, "end_s": 10.0}], "window": [0.0, 600.0],
        "model": "gemini-3.5-flash", "thinking_level": "off", "system_prompt": "", "followup_prompt": "",
    }))
    resp = client.post("/highlight/score", json={"match_id": "s_small", "tau_t": 5.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["fps"] is None
    assert body["frame_quality"] is None


def test_highlight_score_filters_rejected_highlights_out_of_the_scoreable_response(tmp_path):
    """Requirement §2: a rejected highlight must never appear in
    enriched_highlights/timeline.pred_marks -- and the two lists must stay
    index-aligned (both derived from the SAME filtered set)."""
    client = TestClient(_make_app(tmp_path))
    out_dir = tmp_path / "out" / "highlights"
    out_dir.mkdir(parents=True)
    (out_dir / "s_small.json").write_text(json.dumps({
        "match_id": "s_small",
        "highlights": [
            {"original_start_s": 10.0, "original_end_s": 10.0, "refined_start_s": None,
             "refined_end_s": None, "refined": False, "reason": "x", "action_confirmed": None,
             "refinement_failed": False, "rejected": False, "rejection_reason": None},
            {"original_start_s": 590.0, "original_end_s": 610.0, "refined_start_s": None,
             "refined_end_s": None, "refined": False, "reason": "y", "action_confirmed": None,
             "refinement_failed": False, "rejected": True, "rejection_reason": "out_of_range"},
        ],
        "window": [0.0, 600.0], "model": "gemini-3.5-flash", "thinking_level": "off",
        "system_prompt": "", "followup_prompt": "",
    }))
    resp = client.post("/highlight/score", json={"match_id": "s_small", "tau_t": 5.0})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["enriched_highlights"]) == 1  # the rejected one is dropped
    assert body["enriched_highlights"][0]["rejected"] is False
    assert len(body["timeline"]["pred_marks"]) == 1
    assert body["timeline"]["pred_marks"][0]["index"] == 0


def test_highlight_score_unknown_match_id_is_400(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hd, "build_gemini_client", lambda *a, **k: _fake_client_returning("[]"),
    )
    client = TestClient(_make_app(tmp_path))
    # Fabricate a persisted highlight result for an unknown match_id directly
    # (bypassing /highlight/run, which would 400 first) to isolate this check.
    out_dir = tmp_path / "out" / "highlights"
    out_dir.mkdir(parents=True)
    (out_dir / "ghost_match.json").write_text(json.dumps({
        "match_id": "ghost_match", "highlights": [], "window": [0.0, 100.0],
        "model": "x", "thinking_level": None, "system_prompt": "", "followup_prompt": "",
    }))
    resp = client.post("/highlight/score", json={"match_id": "ghost_match", "tau_t": 5.0})
    assert resp.status_code == 400
