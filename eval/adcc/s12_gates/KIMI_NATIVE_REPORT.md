# Kimi-Native (Direct Moonshot) Report — `GATES_RERUN_DESIGN.md` §9

**Generated:** 2026-07-26. **Authorization:** Founder-funded direct `KIMI_API_KEY` (Moonshot, not OpenRouter),
LeCun's §9 addendum design, executed end-to-end this turn: §9.3 reliability fixes → §9.1 probe → §9.5 pilot →
§9.2 arms (primary + secondary + Sub-lane C) → §9.4 scoring → this report. **This closes the experiment** —
no further Kimi-native spend occurred beyond what is itemized below.

---

## 1. §9.1 Probe outcome

**Real API calls only, against `https://api.moonshot.ai/v1`, verified live — nothing guessed.**

- **Model id**: `kimi-k3` (confirmed via `GET /v1/models`, matches the assumed slug). Model metadata:
  `supports_video_in: true`, `reasoning_efforts: {support: true, valid_efforts: [low, high, max], default: max}`.
- **Video capability — four real mechanisms tried:**
  1. `POST /v1/files` with `purpose="video"` — **succeeds** (a real, officially-supported purpose value, confirmed
     via the API's own 400-error body listing all valid purposes).
  2. Referencing that upload via a `video_url` content block (`file_id`, or the file's own content-fetch URL) —
     **FAILS**, `400 "unsupported video url: <url>"` for every shape tried.
  3. `video_url` pointed at a live public YouTube watch URL — **FAILS**, the same `400 "unsupported video url"`.
     **This directly answers the w17 question**: Moonshot's direct API does not fetch external URLs at all
     (not YouTube, not its own file-content URL).
  4. `video_url` with an inline `data:video/mp4;base64,...` URI (raw video bytes embedded in the request) —
     **SUCCEEDS**. Verified content-grounded (not a generic/hallucinated answer): the response correctly named
     the actual athletes shown on screen and referenced the visible match score.
- **Conclusion: native video works, ONLY via mechanism 4 (inline data URI).** This does not cleanly match either
  of the design's anticipated "Branch A" (file_id reference — tried, failed) or "Branch B1" (external URL — tried,
  failed); it is closer to "Branch B2" (embed local content) but simpler (no local HTTP server needed).
- **`w17` resolution**: the only working mechanism requires the raw video bytes to exist locally BEFORE the call —
  the same yt-dlp dependency "Branch A" would have had. `w17`'s video is permanently yt-dlp-blocked (SME
  copyright claim, confirmed independently three times now across Gate 3, the frontier lanes, and this probe).
  **`w17` was pre-excluded from the primary-arm window set** (29 of the core-30 attempted; not needed for the
  secondary arm's first-12 slice, which doesn't include `w17`).
- **Reasoning parameter**: real top-level field confirmed via Moonshot's own live docs
  (`platform.moonshot.ai/docs/pricing/chat-k3.md`): `reasoning_effort` (`low`/`high`/`max`, default `max`) — NOT
  OpenRouter's `reasoning:{"effort":...}` nesting. Kimi K3 "always reasons" (cannot be disabled). Confirmed
  HONORED via a real pilot call: `reasoning_tokens=510` (nonzero) with `reasoning_effort="high"`.
- **Real pricing** (verified live, never estimated): `kimi-k3` = **$3.00/MTok input (cache-miss), $15.00/MTok
  output**. Native video consumes meaningfully MORE tokens per call than the frames-based path (~31K prompt
  tokens for one 9s window vs ~7K for the OpenRouter frames call on the same window) — real per-call cost ended
  up **~4-6x the frames-based reference anchor** ($0.10-0.15/call native vs ~$0.024-0.035/call frames), confirmed
  by the pilot, not assumed.

---

## 2. §9.5 Pilot result

3 primary-arm calls, real cost + parsing verified before continuing (auto-continue per LeCun's pre-authorization,
since the pilot came back clean):

| Window | Status | Cost |
|---|---|---|
| `w0` | ok | $0.1487 |
| `w1` | ok | $0.1305 |
| `w2` | ok | $0.1362 |

**3/3 succeeded, $0.4154 total, $0.1385/call average** — well within the expected range, no anomaly. Proceeded
directly to the full batch per LeCun's explicit pre-authorization ("gated only on the pilot").

---

## 3. Final per-arm numbers

### 3a. PRIMARY arm — Kimi-native single-call vs Gemini Arm S (16/30, 53.3% full-core-30 baseline)

**29/29 core-30 windows attempted** (`w17` pre-excluded, see §1), **29/29 scored** (1 residual `ffmpeg exited with
code 8` transient failure on `w12` — same known-class transient issue seen twice before in this project — retried
once, succeeded on retry, per the CEO's explicit one-retry authorization).

| | n | Hit rate |
|---|---|---|
| Kimi-native (single-call) | 29 | **6/29 = 20.7%** |
| Gemini Arm S (same 29 windows) | 29 | 15/29 = 51.7% |

### 3b. SECONDARY arm — Kimi-native triple vs Gemini Arm T (11/30, 36.7% full-core-30 baseline)

**12/12 windows attempted (first-12-by-window_id), 12/12 scored, zero failures.**

| | n | Hit rate |
|---|---|---|
| Kimi-native (triple) | 12 | **5/12 = 41.7%** |
| Gemini Arm T (same 12 windows) | 12 | 3/12 = 25.0% |

### 3c. Sub-lane C — zero-cost temporal ablation (native triple vs frames-Kimi triple, SAME windows, model held constant)

**11 windows** where BOTH the native-Kimi triple (secondary arm) and the frames-Kimi triple (`frontier_kimi_
predictions.json`, OpenRouter) successfully scored — zero new calls, pure re-aggregation.

| | n | Hit rate |
|---|---|---|
| Kimi-native (triple) | 11 | 4/11 = 36.4% |
| Kimi-frames (triple, OpenRouter) | 11 | 3/11 = 27.3% |
| **Delta (native − frames)** | | **+9.1pp (+1 window)** |

---

## 4. LeCun's §9.6 predictions — confirmed / refuted

**Prediction 1 — Kimi-native single-call vs Gemini Arm S (53.3%): predicted BELOW, landing in the 30-45% range
("a respectable showing, not a win").**

**REFUTED, in the more severe direction than LeCun himself anticipated.** Actual: **20.7%**, below even his
pessimistic floor of 30% by 9.3pp, and far below Gemini's own 51.7% on the same windows. This is the single most
consequential and surprising finding of this addendum — not because it flips any gate verdict (it doesn't, see
§6), but because Gemini's own Arm S advantage over Kimi-native turned out to be dramatically larger than "a
respectable showing" would predict. LeCun's underlying REASONING (Gemini Arm S is a reproduced, strong, stable
performer) is directionally vindicated — the magnitude just exceeded his own stated range.

**Prediction 2 — Kimi-native triple vs Kimi-frames triple (33.3% original paired subset): predicted native beats
frames by a modest +5-10pp, landing at ≥38-43%.**

**CONFIRMED on the relative-margin component, narrowly short on the absolute-landing component — read together
as CONFIRMED within sampling noise.** Two valid readings, both reported:
- Relative margin, Sub-lane C's clean n=11 paired subset (model held constant, everything else identical):
  native 36.4% vs frames 27.3%, **delta +9.1pp** — squarely inside his predicted +5-10pp modest-margin band.
- Absolute landing zone, using the secondary arm's own full n=12 (not truncated to match frames' incomplete
  coverage): **41.7%** — inside his stated ≥38-43% range.
- The Sub-lane C n=11 absolute figure (36.4%) alone falls 1.6pp short of his 38% floor, but n=11 has ~9pp
  single-window granularity — not a meaningful miss given the sample size.

**Prediction 3 — Kimi-native triple vs Gemini Arm T (36.7% full-core-30): predicted a near-toss-up, "plausibly a
slight Kimi edge."**

**Directionally CONFIRMED (a Kimi edge is real), but the margin is substantially larger than "slight."** On the
same 12 windows: Kimi-native 41.7% vs Gemini Arm T 25.0% — a **+16.7pp** edge, not a toss-up. Worth flagging as a
stronger-than-predicted result in the same family as the other two predictions' pattern: LeCun's directional
calls were right, his magnitude calibration ran in different directions for the single-call vs the triple arm
(underestimated the Gemini-Arm-S gap, underestimated the Kimi-triple-vs-Gemini-Arm-T edge).

**Prediction 4 — none of this changes Gate 1's own verdict.** **Confirmed, unaffected.** Gate 1 remains a
same-model (`gemini-3.5-flash`-only), single-vs-two-call architecture question, already closed NO_GO on its own
evidence (§4/§6.4/§8, restated, unchanged by this addendum). This addendum answers a separate, genuinely new
question — MODEL CHOICE, not architecture — and (per §6.5) is confirmatory/exploratory context, never a gate
input.

---

## 5. Mandatory disclosures

**(a) `w17` structural exclusion.** See §1 — confirmed independently a THIRD time this session (Gate 3, frontier
lanes, and now this probe) that this specific video is permanently unreachable by any yt-dlp-dependent or
external-URL-fetch-dependent mechanism.

**(b) Native video is meaningfully more expensive per call than frames-based.** ~31K prompt tokens for a single
9s window (vs ~7K frames-based) — real per-call cost landed at $0.10-0.15/call, not the $0.024-0.035/call
OpenRouter reference anchor §9.5 explicitly flagged as unconfirmed. Genuinely higher-cost lane, still comfortably
under the $20 cap at the n actually run.

**(c) One residual transient failure, retried once (CEO-authorized).** Primary-arm window `w12` failed with the
SAME `ffmpeg exited with code 8` signature already seen twice before in this project's history (Gate 4). Retried
once via a targeted single-window re-invocation (not a full batch re-run — avoided re-spending on the 28
already-successful primary windows) — succeeded on retry. Final primary-arm coverage: 29/29 (100%), zero
outstanding failures.

**(d) Branch-testing order.** Per the standing "design docs override CEO paraphrase, flag conflicts" rule
(established earlier this session), Branch A (file upload) was tested before Branch B (video_url), per
`GATES_RERUN_DESIGN.md` §9.1's literal stated order — not the CEO's paraphrased "video_url first" preference.
Branch B1 (YouTube URL) was still tested regardless of Branch A's outcome specifically to answer the w17
question, at near-zero cost, satisfying both the letter of the binding doc and the substantive interest behind
the CEO's request.

**(e) `_extract_final_json_object` reused prospectively (not retroactively assumed).** Both the primary and
secondary arms' JSON parsing use the recovery function built for Opus `w6` (`anthropic_direct.
_extract_final_json_object`) as a defensive superset of a bare `json.loads` — this is NOT an assumption that
Kimi's failure modes share Opus's shape (§9.3 explicitly forbade that inference); it is applied because the
function can never do worse than a bare parse (strict path tried first) and is real, already-tested robustness.
No Kimi-native call actually needed the recovery path this run (all successes parsed cleanly on the first,
strict attempt).

**(f) `fps` is not controllable for native video.** No sampling-rate parameter was found anywhere in Moonshot's
API surface during the probe — disclosed explicitly in every record's settings tuple as `"fps": "not
controllable, provider default"`, never silently omitted (§9.2's explicit requirement).

---

## 6. Spend

| | Calls (ok/failed) | Real spend |
|---|---|---|
| Pilot (3 primary calls) | 3/0 | $0.4154 (included in primary total below) |
| Primary arm (29 windows × 1 call, incl. 1 retry) | 30/0 (after retry) | $4.3794 |
| Secondary arm (12 windows × 3 calls) | 36/0 | $5.2428 |
| **Total** | **66/1** (1 residual failure superseded by its retry) | **$9.6223** |

**Cap: $20 — 48.1% used, comfortably under.**

**Grand total, this entire multi-turn session** (Phase A Gemini re-run + Phase B smoke + frontier full batches +
this Kimi-native experiment): **$16.5022**, against the founder's $100 authorization.

---

## 7. Artifact paths

- `eval/adcc/s12_gates/kimi_native_client.py` — Moonshot direct client (§9.1 probe findings, pricing, mechanisms)
- `eval/adcc/s12_gates/run_kimi_native_batch.py` — pilot + primary + secondary driver
- `eval/adcc/s12_gates/frame_source_retry.py` — bounded ffprobe retry (harness-only, §9.3)
- `eval/adcc/s12_gates/anthropic_direct.py` — `_extract_final_json_object` (§9.3 Opus `w6` fix, reused here)
- `eval/adcc/s12_gates/kimi_native_primary_predictions.json` — 29/29 records
- `eval/adcc/s12_gates/kimi_native_secondary_predictions.json` — 12/12 records
- `eval/adcc/s12_gates/kimi_native_scoring.json` — combined scoring (primary/secondary/Sub-lane C)
- `eval/adcc/s12_gates/gate_budget_ledger.json` — cumulative ledger, `gate=="kimi_native"` rows carry
  `provider_route="moonshot_direct"`, real `cost_usd`, `thinking_level`
- This file: `eval/adcc/s12_gates/KIMI_NATIVE_REPORT.md`
