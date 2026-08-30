# Frontier Lane Report — Kimi K3 (OpenRouter) vs Opus 4.8 (direct Anthropic) vs qualified `gemini-3.5-flash`

**Generated:** 2026-07-26. **Authorization:** Bezos frontier-checkpoint "AUTHORIZE-WITH-PRECONDITIONS" (2026-07-26) —
route ruling (Opus=`direct_anthropic`, Kimi=`openrouter`) CONFIRMED, two code preconditions landed, both full
batches run this turn (Kimi retried once per explicit authorization; Opus run once, no retry authorized/attempted).
**LeCun falsifiable prediction under test** (`GATES_RERUN_DESIGN.md` §8): frame-bound frontier models
(Opus via frames, Kimi via frames — both confirmed frames-only, §6.2) underperform native-video
`gemini-3.5-flash`'s own Arm T, "by a visible (not razor-thin) margin."

**This is the final frontier-phase report for this turn's authorized scope** (both lanes' full batches +
one authorized Kimi retry). No further batch spend occurred beyond what is itemized below.

---

## 1. Headline numbers

Both lanes ran an Arm-T-style position→technique→validator triple (`run_gate1._reconcile_validator`,
byte-faithful reuse) on the SAME frozen Gate 1 core-30 windows. Real per-call cost was measured directly
(never estimated) — OpenRouter's own `usage.cost` for Kimi, and a local usage-surfacing wrapper
(`anthropic_direct.py`, precondition 1) reading `response.usage` against Anthropic's live-verified Opus 4.8
rate card ($5/MTok input, $25/MTok output — thinking billed as output tokens) for Opus.

| Lane | Windows attempted | Windows scored (lane succeeded) | Calls (ok/failed) | Real spend | Cap |
|---|---|---|---|---|---|
| Kimi K3 (OpenRouter) | 18/30 | 15/30 | 49 ok / 6 failed = 55 | **$1.9083** | $40 (25pp used) |
| Opus 4.8 (direct Anthropic) | 8/30 | 5/30 | 17 ok / 3 failed = 20 | **$0.8933** | $40 (2.2pp used) |
| **Combined frontier lane** | — | — | 66 ok / 9 failed = 75 | **$2.8016** | — |

Both lanes are **far under** their $40 caps — the cap-check logic never engaged; both lanes stopped due to the
**halt-and-escalate** rule (>2 of N windows failed in a single invocation), not budget exhaustion.

### 1a. Gemini `gemini-3.5-flash` core-30 baseline (context, full 30 windows — from `gate1_scoring.gemini35flash_highthink.json`)

- Arm S: **hit_S = 16/30 (53.3%)**
- Arm T: **hit_T = 11/30 (36.7%)**

### 1b. Strict paired comparison — Gemini scored on the EXACT SAME windows each frontier lane itself successfully scored

This is the scientifically valid comparison: for each lane, restrict BOTH the lane's own hit count and Gemini's
hit count to the windows where **that lane** produced a valid final answer (i.e., Gemini's denominator changes
per lane, matching whatever subset the lane could actually be scored on).

| | n (paired) | Lane hit rate | Gemini Arm T hit rate (same n) | Gemini Arm S hit rate (same n) |
|---|---|---|---|---|
| **Kimi** | 15 | **5/15 = 33.3%** | 3/15 = 20.0% | 7/15 = 46.7% |
| **Opus** | 5 | **0/5 = 0.0%** | 1/5 = 20.0% | 1/5 = 20.0% |

### 1c. LeCun prediction — explicit CONFIRM / REFUTE

- **Opus vs Gemini Arm T**: Opus (0/5) is directionally **below** Gemini Arm T (1/5) on the paired subset —
  **directionally consistent with LeCun's prediction**, but **NOT CONFIRMED at his stated bar** ("a visible,
  not razor-thin margin"): n=5 is a single-digit sample (one Gemini hit total), and the absolute gap is exactly
  one window. This is too statistically underpowered to claim a visible margin either way. **Verdict: INCONCLUSIVE — directionally consistent, not confirmed.**
- **Kimi vs Gemini Arm T**: Kimi (5/15 = 33.3%) is **above** Gemini Arm T (3/15 = 20.0%) on the paired subset —
  this runs **opposite** to the general "frame-bound underperforms native-video" logic LeCun's prediction rests
  on. **Verdict: REFUTED IN THIS SAMPLE** for Kimi specifically, with strong caveats (n=15, only 60% of the core-30
  actually attempted, 50% actually scored — see §3). LeCun's own design doc (§6.3) already flagged that any
  Kimi-vs-Gemini comparison is confounded by ~4.5x lower effective temporal sampling (≤20 frames over 9s ≈ 2.2fps
  vs native fps=10) — this result should be read as "even with that handicap, Kimi did not underperform on the
  windows it managed to complete," not as a clean model-capability finding.
- **Neither lane's Arm-T-style result beats Gemini's own Arm S** (53.3% baseline) on the paired subsets either
  (Kimi 33.3% < 46.7%, Opus 0% < 20%) — consistent with the qualified-model's single-call architecture still
  being the strongest baseline in this corpus regardless of which model runs the two/three-call flow.
- **Neither result flips Gate 1's own verdict** — per `GATES_RERUN_DESIGN.md` §6.4 point 3, the frontier lane was
  never able to substitute for Gate 1's own missing Arm-S equivalent and does not change Gate 1's NO_GO.

---

## 2. Mandatory disclosures (Bezos precondition 3 + self-disclosed items)

**(a) Route-check window hardness (Bezos precondition 3a).** The original Opus route-equivalence smoke
(`smoke_frontier_lanes.py`) used window `w0` and its code comment called it "well-understood" — that was
**factually wrong** and has been corrected in-code. Cross-checked against `gate1_predictions_arm_t.
gemini35flash_highthink.json`: the QUALIFIED `gemini-3.5-flash` model's own Arm T also got `w0` wrong
(predicted position "mount" vs gold "turtle"; validator agreed with the wrong label). `w0` is one of Gate 1's
known-hard back_take-from-turtle cases, not a curated easy clip.

**(b) Pre-fix payload confound (Bezos precondition 3b).** The route-check's two calls (direct-Anthropic vs
OpenRouter) predate the precondition-2 framing/ordering fix landed in `openrouter_client.py` this turn — the
smoke's enum-disagreement (`sweep` vs `guard_pass`) is **not** clean independent route-equivalence evidence,
since the two payloads differed by more than just the provider endpoint at that time. Only the
`reasoning_tokens=0` finding (OpenRouter did not confirm thinking was honored for Opus specifically) carries
the actual routing conclusion (Opus=`direct_anthropic`).

**(c) ~2-call spend leak from the killed parallel-race attempt (self-disclosed).** An early attempt to run both
lanes as parallel background processes was killed within ~6 seconds of starting, before either process reached
its first flush (verified: zero ledger corruption, zero predictions written). However, each process's first
`position` call may have already been in-flight over the network when `kill` (SIGTERM) was sent, meaning up to
~2 API calls (~$0.04–0.08 combined, by comparison to this run's own measured per-call costs) may have been
billed by the providers without ever being recorded in this project's ledger. This is a small, permanent,
undocumented leak — disclosed here rather than silently absorbed.

**(d) Inferred halt threshold (self-disclosed).** `FRONTIER_FAILURE_HALT_THRESHOLD = 2` (of 30 windows, per
lane, per invocation) is **not** an explicit number from `GATES_RERUN_DESIGN.md` §6 — it is an engineer-inferred,
conservative, proportional analog of Gate 1's own explicit ">3 of 38" rule, flagged in `run_frontier_batch.py`'s
own docstring at write time (same treatment Gate 3/4's inferred thresholds received). It is threshold-PER-
INVOCATION, not cumulative across resume passes — a real, disclosed design choice that let Kimi's retry pass
independently re-trip the same rule on a fresh count.

**(e) `max_tokens` truncation and OpenRouter 429 fixes (self-disclosed, landed before any full-batch spend).**
Two real failures were found and fixed during the FIRST live windows of this turn's batch (see `openrouter_client.
py`'s module docstring "[FIX 2026-07-26]" note for full detail): (1) `max_tokens=4096` was insufficient once
`reasoning.effort="high"` is requested (reasoning tokens count against the same budget as the final answer for
OpenAI-compatible completions) — raised to 16384; (2) a genuine transient `429` from Kimi/Moonshot upstream — a
bounded retry (`_TRANSIENT_RETRY_MAX_ATTEMPTS=4`, honoring `Retry-After`) was added. The pre-fix partial run was
fully discarded (ledger rows removed, predictions files deleted) before the clean, post-fix run began — no mixed-
configuration data is included in the numbers above.

**(f) Kimi's real coverage is 18/30 (60%) core windows, scored on 15/30 (50%) — not the full core-30
(self-disclosed).** Two retry-authorized passes both hit the per-invocation halt-and-escalate rule. Per explicit
instruction, no third pass was attempted. The three residual Kimi failures are three DIFFERENT failure classes
(see §3) — this is disclosed as a real, partial-coverage result, not silently extrapolated to "Kimi's true
core-30 hit rate."

**(g) Opus's real coverage is 8/30 (27%) core windows, scored on 5/30 (17%) (self-disclosed).** One pass only
(no retry authorized for Opus). Halted after 3 failures (2 known-class transient `ffprobe`, 1 new JSON-parse
failure — see §3).

**(h) `w17` (video `FO4jG04aGqc`) is structurally, permanently unavailable — confirmed independently a second
time this turn (self-disclosed, matches Gate 3's own prior §G.2 finding).** `[youtube] FO4jG04aGqc: Video
unavailable. It was blocked due to the claimed content by SME.` This is an EXTERNAL YouTube-side block on public
yt-dlp access specifically — Gemini's own Arm T succeeded on this exact window (Google's internal video-fetch
path for the Gemini API is evidently unaffected by whatever public-access block yt-dlp hits). Both frontier
lanes require yt-dlp-downloaded frames (no native-video ingestion exists for either), so `w17` can **never** be
scored by any frames-based lane regardless of provider — this is excluded from both lanes' scoring on a
principled, external-unavailability basis, not counted as a lane-capability failure, and not treated as
retriable (retrying an SME-blocked video is pointless).

---

## 3. Full per-window failure ledger (9 failed windows total across both lanes)

| Lane | Window | Failure class | Retriable? |
|---|---|---|---|
| Kimi | `w0` (pass 1 only — succeeded on retry) | transient `ffprobe fps probe failed (rc=1)` | Yes — succeeded on retry |
| Kimi | `w6` (pass 1 only — succeeded on retry) | transient `ffprobe fps probe failed (rc=1)` | Yes — succeeded on retry |
| Kimi | `w8` (both passes) | JSON-parse error (`Expecting value: line 157 column 1`) — genuine, unresolved robustness gap, NOT the same as the already-fixed `max_tokens`-truncation class (this one did not report `finish_reason="length"`) | Unknown — not yet root-caused |
| Kimi | `w14` (pass 2) | transient `ffprobe fps probe failed (rc=1)` | Presumed yes (same class as `w0`/`w6`), not retried further per instruction |
| Kimi | `w17` (pass 2) | **permanent** — SME YouTube takedown, confirmed independently (see §2h) | No |
| Opus | `w5` | transient `ffprobe fps probe failed (rc=1)` | Presumed yes, not retried (no retry authorized) |
| Opus | `w6` | `anthropic_direct: Claude non-JSON response: Extra data: line 6 column 1` — Claude emitted valid JSON followed by unexpected trailing content not caught by the `finish_reason="max_tokens"` guard added this turn; genuine, unresolved robustness gap | Unknown — not yet root-caused |
| Opus | `w7` | transient `ffprobe fps probe failed (rc=1)` | Presumed yes, not retried (no retry authorized) |

**Pattern**: the `ffprobe fps probe failed (rc=1)` transient class now accounts for **5 of 9** failures across
both lanes — a real, recurring (not purely one-off) transient rate (~12–15% of individual calls hitting this
specific yt-dlp/ffprobe step), consistent with, but higher-frequency than, this project's two prior documented
transient yt-dlp/ffmpeg incidents. This is flagged as a follow-up worth root-causing (possibly a bounded retry
at the `probe_native_fps` call site in the eval harness's OWN wrapper code — never in the production
`service/pipelines/frame_source.py` file itself) if further frontier-lane batches are authorized. The remaining
2 failures (`w8` Kimi, `w6` Opus) are two DIFFERENT, still-unresolved JSON-parsing robustness gaps — genuinely
new findings from this turn, not yet fixed.

---

## 4. Artifact paths

- `eval/adcc/s12_gates/anthropic_direct.py` — Opus usage-surfacing wrapper (precondition 1)
- `eval/adcc/s12_gates/openrouter_client.py` — OpenRouter client, framing fix (precondition 2), max_tokens/429 fixes
- `eval/adcc/s12_gates/run_frontier_batch.py` — two-lane batch driver
- `eval/adcc/s12_gates/frontier_kimi_predictions.json` — Kimi lane records (18/30 attempted)
- `eval/adcc/s12_gates/frontier_opus_predictions.json` — Opus lane records (8/30 attempted)
- `eval/adcc/s12_gates/frontier_scoring.json` — combined scoring (both lanes, includes the broader
  attempted-windows Gemini comparison in addition to this report's strict-paired numbers)
- `eval/adcc/s12_gates/gate_budget_ledger.json` — cumulative ledger, `gate=="frontier_batch"` rows carry
  `provider_route`, `thinking_level`, real `cost_usd`
- `eval/adcc/s12_gates/smoke_frontier_lanes.py` — corrected comment (§2a)
- This file: `eval/adcc/s12_gates/FRONTIER_LANE_REPORT.md`
