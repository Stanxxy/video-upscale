"""``eval/adcc/predict/dashboard_app.py`` — the ADCC eval cost-control
dashboard FastAPI backend (Task 3). Every test builds an ISOLATED app via
``create_app(...)`` pointed at tmp_path fixtures — never the real gold
snapshot/.env. ZERO real Gemini calls anywhere: ``/plan`` is pure arithmetic
(no monkeypatch needed to prove it never calls Gemini — it never imports a
live client at all in this test's fixture) and ``/run``'s mocked-Gemini test
monkeypatches ``RunContext.gemini_client`` exactly like ``test_pipelines_executors.py``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from eval.adcc.predict.dashboard_app import create_app
from service.config import ServiceConfig
from service.pipelines.executors import RunContext


def _corpus_records():
    """Mirrors eval/adcc/tests/test_smoke_test.py's fixture shape exactly —
    one single-match video, two distinct multi-match VODs (so scope
    mode='smoke' resolves), plus enough shape for mode='single'/'list'/'split'.
    Every record carries a "name" field (real gold records always do — see
    match_catalog.py) so /matches can resolve a human name for each.

    video_url is a real-SHAPED (fake) YouTube /embed/<11-char-id> URL --
    live-bug fix regression: global_cap_driver now normalizes
    target.video_url (youtube_url.normalize_youtube_watch_url) before
    building RunContext, mirroring the real ADCC gold snapshot's own
    embed-URL field (INS-084)."""
    records = [
        {"match_id": "s_small", "name": "Small Single vs Match", "video_url": "https://www.youtube.com/embed/sSmallID123",
         "video_url_md5": "stem_s_small", "events": [{"t_s": 10}]},
    ]
    records += [
        {"match_id": f"multi_small_{i}", "name": f"Multi Small {i} vs Sibling",
         "video_url": "https://www.youtube.com/embed/multiSmall1", "video_url_md5": "stem_multi_small",
         "events": [{"t_s": 100 * i}]}
        for i in range(2)
    ]
    records += [
        {"match_id": f"multi_big_{i}", "name": f"Multi Big {i} vs Sibling",
         "video_url": "https://www.youtube.com/embed/multiBigID1", "video_url_md5": "stem_multi_big",
         "events": [{"t_s": 1000 * i}]}
        for i in range(6)
    ]
    return records


def _write_corpus(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    with open(gold_path, "w") as f:
        for rec in _corpus_records():
            f.write(json.dumps(rec) + "\n")

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({
        "assignment": {
            "stem_s_small": {"split": "dev", "match_ids": ["s_small"], "event_count": 1},
            "stem_multi_small": {"split": "test", "match_ids": ["multi_small_0", "multi_small_1"], "event_count": 2},
            "stem_multi_big": {
                "split": "sandbox",
                "match_ids": [f"multi_big_{i}" for i in range(6)],
                "event_count": 6,
            },
        },
        "ratios": {}, "seed": 0, "video_counts": {},
    }))

    durations_path = tmp_path / "durations.json"
    durations_path.write_text(json.dumps({"stem_s_small": 300.0}))

    return gold_path, split_path, durations_path


def _make_app(tmp_path, gemini_api_key: str = ""):
    gold_path, split_path, durations_path = _write_corpus(tmp_path)
    return create_app(
        gold_snapshot_path=gold_path,
        split_assignment_path=split_path,
        video_durations_path=durations_path,
        output_dir=tmp_path / "out",
        service_config=ServiceConfig(gemini_api_key=gemini_api_key),
    )


# --------------------------------------------------------------------------- #
# GET / — serves the single-file dashboard client
# --------------------------------------------------------------------------- #
def test_index_serves_the_dashboard_html(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "ADCC Eval" in resp.text
    assert "no single click" in resp.text.lower() or "no click" in resp.text.lower()


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_reports_key_configured_and_corpus_size(tmp_path):
    client = TestClient(_make_app(tmp_path, gemini_api_key="fake-key"))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["gemini_key_configured"] is True
    assert body["match_count_in_corpus"] == 9  # 1 single + 2 + 6 multi
    assert body["active_run"] is False


# --------------------------------------------------------------------------- #
# /matches — the searchable match-picker menu's data source. ZERO Gemini
# calls, pure metadata + arithmetic (match_catalog.build_match_catalog).
# --------------------------------------------------------------------------- #
def test_matches_returns_every_match_in_the_corpus_with_zero_gemini_calls(tmp_path, monkeypatch):
    def _boom(self):
        raise AssertionError("GET /matches must NEVER touch a Gemini client")

    monkeypatch.setattr(RunContext, "gemini_client", _boom)
    client = TestClient(_make_app(tmp_path))

    resp = client.get("/matches")
    assert resp.status_code == 200
    body = resp.json()
    assert body["match_count"] == 9  # 1 single + 2 + 6 multi, mirrors /health's match_count_in_corpus
    assert len(body["matches"]) == 9
    match_ids = {m["match_id"] for m in body["matches"]}
    assert match_ids == {"s_small", "multi_small_0", "multi_small_1",
                          "multi_big_0", "multi_big_1", "multi_big_2",
                          "multi_big_3", "multi_big_4", "multi_big_5"}


def test_matches_row_has_name_length_and_dual_call_estimates(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.get("/matches")
    body = resp.json()
    entry = next(m for m in body["matches"] if m["match_id"] == "s_small")
    assert entry["name"] == "Small Single vs Match"
    assert entry["video_md5"] == "stem_s_small"
    assert entry["is_multi_match"] is False
    assert entry["has_duration"] is True
    assert entry["length_seconds"] == pytest.approx(300.0)  # from _write_corpus's durations.json
    assert entry["length_mmss"] == "5:00"
    assert entry["est_calls_realistic"] > 0
    assert entry["est_calls_worst_case"] >= entry["est_calls_realistic"]
    assert entry["est_dollars_realistic"] > 0


def test_matches_sorted_by_name_for_browsability(tmp_path):
    client = TestClient(_make_app(tmp_path))
    body = client.get("/matches").json()
    names = [m["name"] for m in body["matches"]]
    assert names == sorted(names)


def test_matches_flags_missing_duration_instead_of_guessing(tmp_path):
    """multi_small_* / multi_big_* are multi-match (gold-derived window, no
    ffprobe duration needed) but s_small (single-match) has NO duration in
    this empty-durations app — must be flagged, never silently dropped or
    given a fabricated length."""
    gold_path, split_path, _ = _write_corpus(tmp_path)
    empty_durations = tmp_path / "empty_durations.json"
    empty_durations.write_text("{}")
    app = create_app(
        gold_snapshot_path=gold_path, split_assignment_path=split_path,
        video_durations_path=empty_durations, output_dir=tmp_path / "out",
        service_config=ServiceConfig(gemini_api_key="fake-key"),
    )
    client = TestClient(app)
    body = client.get("/matches").json()
    assert body["match_count"] == 9  # still every match present
    entry = next(m for m in body["matches"] if m["match_id"] == "s_small")
    assert entry["has_duration"] is False
    assert entry["length_seconds"] is None
    assert entry["est_calls_realistic"] is None
    # multi-match entries are unaffected (their window never needed a
    # ffprobe duration in the first place).
    multi_entry = next(m for m in body["matches"] if m["match_id"] == "multi_small_0")
    assert multi_entry["has_duration"] is True


# --------------------------------------------------------------------------- #
# /plan — ZERO Gemini calls, pure arithmetic
# --------------------------------------------------------------------------- #
def test_plan_single_match_returns_estimate_with_zero_gemini_calls(tmp_path, monkeypatch):
    def _boom(self):
        raise AssertionError("GET /plan must NEVER touch a Gemini client")

    monkeypatch.setattr(RunContext, "gemini_client", _boom)
    client = TestClient(_make_app(tmp_path))

    resp = client.get("/plan", params={"mode": "single", "match_id": "s_small", "cap": 100})
    assert resp.status_code == 200
    body = resp.json()
    assert body["match_count"] == 1
    assert body["total_slices"] == 1
    assert body["total_planned_gemini_calls_worst_case"] > 0
    assert body["total_calls_realistic"] > 0
    # The realistic (primary) estimate must never exceed the pathological
    # worst-case ceiling (secondary, cap-suggestion-only) — the ceiling is
    # derived from the theoretical MINIMUM chunk size, a stricter bound.
    assert body["total_calls_realistic"] <= body["total_planned_gemini_calls_worst_case"]
    assert body["est_dollars_with_margin"] > 0
    assert body["est_dollars_with_margin"] > body["est_dollars_realistic"]
    assert body["within_cap"] is True
    assert "plan_token" in body and len(body["plan_token"]) == 16


def test_plan_smoke_scope_resolves_the_fixed_3_target_set(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.get("/plan", params={"mode": "smoke", "cap": 500})
    assert resp.status_code == 200
    body = resp.json()
    assert body["match_count"] == 3
    match_ids = {m["match_id"] for m in body["per_match"]}
    assert "s_small" in match_ids
    assert any(mid.startswith("multi_small_") for mid in match_ids)
    assert any(mid.startswith("multi_big_") for mid in match_ids)


def test_plan_split_scope(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.get("/plan", params={"mode": "split", "split": "sandbox", "cap": 500})
    assert resp.status_code == 200
    assert resp.json()["match_count"] == 6  # the 6 multi_big_* matches


def test_plan_list_scope(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.get(
        "/plan", params={"mode": "list", "match_ids": "multi_small_0,multi_small_1", "cap": 500},
    )
    assert resp.status_code == 200
    assert resp.json()["match_count"] == 2


def test_plan_mode_all_is_rejected_as_400_not_500(tmp_path):
    """Regression: 'all' is deliberately NOT a supported scope mode (Jobs's
    AC — the selector must never offer the whole corpus in one click).
    GET /plan builds its ScopeRequest manually from query params (not
    FastAPI's own body-parsing path), so this must be explicitly caught and
    turned into a 400, never an unhandled 500."""
    client = TestClient(_make_app(tmp_path))
    resp = client.get("/plan", params={"mode": "all", "cap": 100})
    assert resp.status_code == 400


def test_plan_unknown_match_id_is_400_not_silently_dropped(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.get("/plan", params={"mode": "single", "match_id": "does-not-exist", "cap": 100})
    assert resp.status_code == 400


def test_plan_missing_video_duration_for_single_match_is_400(tmp_path):
    """s_small's duration is deliberately omitted from durations.json here —
    the endpoint must surface RunPredictionsError as 400, never guess a
    duration."""
    gold_path, split_path, _ = _write_corpus(tmp_path)
    empty_durations = tmp_path / "empty_durations.json"
    empty_durations.write_text("{}")
    app = create_app(
        gold_snapshot_path=gold_path, split_assignment_path=split_path,
        video_durations_path=empty_durations, output_dir=tmp_path / "out",
        service_config=ServiceConfig(gemini_api_key="fake-key"),
    )
    client = TestClient(app)
    resp = client.get("/plan", params={"mode": "single", "match_id": "s_small", "cap": 100})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# /run gating — confirm, stale plan_token, missing key, no double-run
# --------------------------------------------------------------------------- #
def test_run_without_confirm_is_rejected(tmp_path):
    client = TestClient(_make_app(tmp_path, gemini_api_key="fake-key"))
    resp = client.post("/run", json={
        "scope": {"mode": "single", "match_id": "s_small"}, "cap": 100,
        "plan_token": "irrelevant", "confirm": False,
    })
    assert resp.status_code == 400
    assert "confirm" in resp.json()["detail"].lower()


def test_run_without_gemini_key_is_rejected(tmp_path, monkeypatch):
    # ServiceConfig's `_alias_gemini_api_key` validator falls back to the
    # process env / a `.env` in cwd when the explicit field is empty (by
    # design, for founder convenience) — neutralize BOTH so this test proves
    # a genuinely unconfigured key is rejected, not an accidental real one
    # picked up from the repo's own dev `.env`.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("BJJ_GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    client = TestClient(_make_app(tmp_path, gemini_api_key=""))
    plan_resp = client.get("/plan", params={"mode": "single", "match_id": "s_small", "cap": 100})
    token = plan_resp.json()["plan_token"]
    resp = client.post("/run", json={
        "scope": {"mode": "single", "match_id": "s_small"}, "cap": 100,
        "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 400
    assert "GEMINI_API_KEY" in resp.json()["detail"]


def test_run_with_stale_plan_token_after_cap_change_is_rejected(tmp_path):
    """THE stale-preview guarantee: preview scope=single/s_small, cap=100 ->
    submit with cap=999 (never re-previewed) -> must be rejected, never
    silently run against the new cap."""
    client = TestClient(_make_app(tmp_path, gemini_api_key="fake-key"))
    plan_resp = client.get("/plan", params={"mode": "single", "match_id": "s_small", "cap": 100})
    stale_token = plan_resp.json()["plan_token"]

    resp = client.post("/run", json={
        "scope": {"mode": "single", "match_id": "s_small"}, "cap": 999,  # changed! token no longer matches
        "plan_token": stale_token, "confirm": True,
    })
    assert resp.status_code == 400
    assert "stale" in resp.json()["detail"].lower()


def test_run_with_stale_plan_token_after_scope_change_is_rejected(tmp_path):
    client = TestClient(_make_app(tmp_path, gemini_api_key="fake-key"))
    plan_resp = client.get("/plan", params={"mode": "single", "match_id": "s_small", "cap": 100})
    stale_token = plan_resp.json()["plan_token"]

    resp = client.post("/run", json={
        "scope": {"mode": "single", "match_id": "multi_small_0"},  # changed scope, same cap
        "cap": 100, "plan_token": stale_token, "confirm": True,
    })
    assert resp.status_code == 400
    assert "stale" in resp.json()["detail"].lower()


def test_run_fresh_plan_token_for_exact_scope_and_cap_is_accepted(tmp_path, monkeypatch):
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"chunks": [{"start_s": 0, "end_s": 20, "reason": "broadcast"}]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    client = TestClient(_make_app(tmp_path, gemini_api_key="fake-key"))
    plan_resp = client.get("/plan", params={"mode": "single", "match_id": "s_small", "cap": 100})
    token = plan_resp.json()["plan_token"]

    resp = client.post("/run", json={
        "scope": {"mode": "single", "match_id": "s_small"}, "cap": 100,
        "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert lines[-1]["type"] == "batch_complete"


# --------------------------------------------------------------------------- #
# /stop
# --------------------------------------------------------------------------- #
def test_stop_with_no_active_run_is_400(tmp_path):
    client = TestClient(_make_app(tmp_path, gemini_api_key="fake-key"))
    resp = client.post("/stop")
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# THE cost guarantee, end-to-end through the real FastAPI route (not just the
# driver in isolation — proves the wiring, not just the algorithm).
# --------------------------------------------------------------------------- #
def test_run_end_to_end_enforces_global_cap_and_persists_no_results_on_abort(tmp_path, monkeypatch):
    call_log = []

    async def _fake_generate(*, model, contents, config):
        call_log.append(1)
        if len(call_log) == 1:
            return SimpleNamespace(text=json.dumps({"chunks": [
                {"start_s": 0, "end_s": 20, "reason": "match_action"},
                {"start_s": 20, "end_s": 40, "reason": "match_action"},
                {"start_s": 40, "end_s": 60, "reason": "match_action"},
                {"start_s": 60, "end_s": 80, "reason": "match_action"},
            ]}))
        return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    client = TestClient(_make_app(tmp_path, gemini_api_key="fake-key"))
    plan_resp = client.get("/plan", params={"mode": "single", "match_id": "s_small", "cap": 3})
    token = plan_resp.json()["plan_token"]

    resp = client.post("/run", json={
        "scope": {"mode": "single", "match_id": "s_small"}, "cap": 3,
        "plan_token": token, "confirm": True,
    })
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert len(call_log) == 3
    assert lines[-1]["type"] == "cap_reached"
    # Aborted run never persisted a match's stitched result to disk.
    assert not list((tmp_path / "out").glob("*.json"))

    # A second /run attempt (fresh preview) is accepted — the aborted run's
    # slot cleared, no lingering "run already in progress" 409.
    plan_resp2 = client.get("/plan", params={"mode": "single", "match_id": "s_small", "cap": 3})
    token2 = plan_resp2.json()["plan_token"]
    resp2 = client.post("/run", json={
        "scope": {"mode": "single", "match_id": "s_small"}, "cap": 3,
        "plan_token": token2, "confirm": True,
    })
    assert resp2.status_code == 200
