# Gates 1 & 2 — Concrete Experimental Design

Author: Yann LeCun (`/scientist`) | Date: 2026-07-25 | Status: DESIGN FROZEN — ready for engineer implementation, zero Gemini calls spent writing this.

**Formal PASS/NO-GO verdicts on the real 216-call batch live in `VERDICTS.md`, not here — this file is the pre-registered design only.** Both gates returned NO-GO; see `VERDICTS.md` §1/§2 for the numbers and scientific interpretation.

Grounded against:
- `team-learnings/handoff/active/task-lecun-5-cutover-gates.md` (the 5-gate spec)
- `eval/adcc/gold_snapshot/highlight_gold.jsonl` + `highlight_gold_manifest.json` (258 matches, 557 events, `highlight_predicate_v1`)
- `service/pipelines/highlight_axes.py`, `service/pipelines/models.py` (`HighlightAnalyzeConfig`/`PositionAxisConfig`/`TechniqueAxisConfig`/`ValidatorAxisConfig`), `service/pipelines/executors.py` (`highlight_analyze_node`, `_call_axis_json`)
- `service/pipelines/simplified_tags.py` (`SimplifiedTagsTimeAnalyzer`, `simplified-tags-time-v1`)
- `eval/adcc/score/matcher.py`, `eval/adcc/score/metrics.py`, `eval/adcc/score/highlight_baselines.py`
- `eval/adcc/gracie_mapping/v1.yaml` (`match_compat.family_by_raw_sequence_type`, `non_scoreable_prediction_classes`)
- `eval/adcc/predict/youtube_url.py`, `eval/adcc/predict/cost_model.py`, `eval/adcc/predict/run_predictions.py` (dry-run/`--execute` discipline)
- `eval/adcc/s12_gates/GATE5_REPORT.md` + `gate5_results.json` (already-run, zero-Gemini boundary count)

**Critical fact this design is built around**: `eval/adcc/gracie_mapping/v1.yaml`'s `match_compat.family_by_raw_sequence_type["SUBMISSION"] = [submission_choke, submission_arm_lock, submission_leg_lock]` means the standard Hungarian matcher (`matcher.py`) treats any submission-family prediction as **class-compatible** with any submission-family gold event. A predicted `submission_leg_lock` against a gold `submission_arm_lock` scores as a **TP** under the harness's own compat rule. This is fine (even correct) for Gate 1's headline metric — it is the same convention the historical 63% number would have used — but it makes the standard matcher **structurally blind to exactly the armbar↔kneebar confusion Gate 2 exists to measure**. Gate 2 therefore does NOT use `matcher.py` at all (see §C).

---

## A. SUBSET FREEZE

### A.1 Universe

`eval/adcc/gold_snapshot/highlight_gold.jsonl`, flattened to one row per event, **restricted to `track == "hard"`** (552 of 557 events). The excluded 5 are all `action_class == "submission_unclear"` (`track == "soft"`) — by definition ambiguous ground truth (the label itself says "unclear"); scoring an A/B against a gold label that admits it doesn't know the true class would inject noise into a paired comparison designed to detect a real accuracy delta on a small n. Excluded from both gates, not just Gate 1.

Verified corpus composition (`track=="hard"`, n=552):

| action_class | count |
|---|---|
| guard_pass | 195 |
| back_take | 137 |
| sweep | 77 |
| submission_choke | 67 |
| submission_leg_lock | 52 |
| submission_arm_lock | 24 |

### A.2 Gate 1 subset — n=38 (30 stratified + 8 forced Gate-5 straddlers)

**Why n=38, not n=19**: the n=19 number was never a committed artifact and had no stated derivation. Cost of re-running at n=19 vs n=38 is trivial (4 calls/window either way — see §D), but statistical power on a paired sign test roughly doubles going 19→38, and n=38 lets the subset ALSO double as Gate 5's missing accuracy-delta measurement (§E.3) at zero extra Gemini spend, which n=19 alone could not reliably do (expected straddler count in a random 19-draw ≈ 0.3). n=38 is the smallest number that does both jobs.

**Step 1 — stratified core (n=30).** Sampling frame = the 544 hard-track events remaining after removing the 8 W=60 production-anchored straddlers listed in `GATE5_REPORT.md` §"Straddling events — production anchoring, per W" → W=60 table (excluded from the frame so they never collide with the Step-2 forced set):

| action_class | frame count (552 − straddlers) | quota (of 30) |
|---|---|---|
| guard_pass | 193 | 11 |
| back_take | 133 | 7 |
| sweep | 76 | 4 |
| submission_choke | 67 | 4 |
| submission_leg_lock | 52 | 3 |
| submission_arm_lock | 23 | 1 |
| **total** | **544** | **30** |

Quota derivation: proportional allocation (`count/544*30`), floored, remainder 3 slots assigned by largest fractional remainder (`submission_leg_lock` +1, `submission_choke` +1, `guard_pass` +1) — standard largest-remainder stratification, fully reproducible from the table above.

Per-stratum draw: sort each stratum's candidate events by `(match_id, t_s)` ascending (stable, deterministic ordering independent of file order), then `random.Random(20260726).sample(stratum_list, k)` — iterate strata in ascending alphabetical `action_class` order so the RNG consumes draws in a fixed sequence regardless of dict-ordering. Seed `20260726` = today's date, matching the repo's existing date-derived seed convention (`metrics.DEFAULT_BOOTSTRAP_SEED = 20260712`).

**Step 2 — forced straddler set (n=8, fixed).** The exact 8 events in `GATE5_REPORT.md`'s W=60 "production anchoring" table (verbatim, by `match_id`+`t_s`):

| match_id | action_class | t_s |
|---|---|---|
| 6470211a3ec8dca9d1a32710 | submission_arm_lock | 727 |
| 650d20c745120c8433e9cfe0 | guard_pass | 774 |
| 650d20c745120c8433e9cff3 | back_take | 761 |
| 650d20c745120c8433e9d020 | guard_pass | 760 |
| 65de1567a2434704339dbc8c | sweep | 729 |
| 65de222ea2434704339dbe9c | back_take | 2235 |
| 65edf42aa2434704339f7e93 | back_take | 754 |
| 66b009481c62a1db7a53d504 | back_take | 30443 |

These are appended unconditionally — not sampled — because Gate 5 needs them specifically (§E.3), not a representative sample of them.

**Total n=38.** Tag every row in the manifest with `"gate5_straddler": true/false` so downstream scoring can partition by it without re-deriving membership.

### A.3 Gate 2 subset — occlusion, m=16 (8 arm_lock + 8 leg_lock)

Universe: the 24 `submission_arm_lock` + 52 `submission_leg_lock` hard-track events (76 total) — deliberately **not** deduplicated against the Gate 1 subset (a handful of Gate-1's 30-stratified or 8-forced events may legitimately also land in Gate 2's 16; that's fine, the two gates test independent levers — call-count vs. media_resolution — and a shared window doesn't bias either comparison).

Deliberately **oversamples failed/occluded attempts** relative to raw prevalence — the entire premise of Gate 2 is "does HIGH resolution rescue the occlusion-limited cases," so the harder cases must be in the sample, not diluted out by the ~93% that are clean `successful` finishes.

Verified `outcome == "unsuccessful"` events in each class (the harder, more-occluded cases — grip/finish frequently contested or hidden):
- `submission_arm_lock` (1 of 24): `66afe5ab1c62a1db7a53b80b` (Dalton Terei vs Mansur Makhmakhanov), t=28325
- `submission_leg_lock` (5 of 52): `65e3a69da2434704339e4460` t=333, `65f648c51dcf397e6900a827` t=241, `6643b593532ae8f232692d70` t=2647, `66a6ff021c62a1db7a5133c0` t=12775, `66d4e7091c7649ed30cbef2b` t=164

Selection rule per class:
1. Force-include ALL `unsuccessful` events in that class (1 for arm_lock, 5 for leg_lock).
2. Fill the remainder of the quota (8 − forced) from the `successful` pool, sorted `(match_id, t_s)` ascending, via `random.Random(20260726).sample(...)` (same seed/procedure as §A.2, applied to a disjoint set of draws — arm_lock's successful-pool sample happens before leg_lock's in the script, both using fresh `.sample()` calls on the one seeded `Random` instance so the draw sequence is still fully deterministic end-to-end).

Result: arm_lock quota 8 = 1 forced + 7 sampled from 23 successful. leg_lock quota 8 = 5 forced + 3 sampled from 47 successful.

### A.4 Window cut (MANDATORY: identical for every arm in a gate)

No gold event carries a `start_s`/`end_s` span — only a point `t_s`. Rather than invent a new window shape, mirror production's own `HighlightAnalyzeConfig` defaults exactly, treating the gold point as a highlight whose own bounds are `[t_s, t_s]`:

```
window_start_s = max(0.0, t_s - preroll_s)   # preroll_s = 5.0 (HighlightAnalyzeConfig default)
window_end_s   = t_s + postroll_s            # postroll_s = 4.0 (HighlightAnalyzeConfig default)
```

A 9-second native-video clip, backward-loaded (setup precedes the point event; outcome resolves shortly after). **Both arms in Gate 1, and both validator conditions in Gate 2, receive the exact same `[window_start_s, window_end_s]` for a given gold event** — this is non-negotiable; it is the only thing that makes the A/B a controlled comparison of axis-count (Gate 1) or media_resolution (Gate 2) rather than a comparison confounded by different input video.

No clamp needed at the video's end (postroll never needs an upper-bound clamp against duration for this corpus's events — none of the selected `t_s` are within 4s of a video's known end; the driver should still defensively clamp to `min(window_end_s, video_duration_s)` if a duration lookup is available via `eval/adcc/predict/generate_video_durations.py`'s cached durations, and log a warning — never silently skip — if it ever fires).

`video_url` from the gold snapshot is `/embed/<id>` form — MUST be run through `eval/adcc/predict/youtube_url.normalize_youtube_watch_url` before any Gemini call (this is the exact live bug that module's docstring documents; do not re-introduce it).

---

## B. GATE 1 ARMS

### B.1 Arm S — baseline (re-run, not the uncited 63% prose number)

One call: `service.pipelines.simplified_tags.SimplifiedTagsTimeAnalyzer.analyze_chunk(youtube_url, window_start_s, window_end_s, None, fps=10, media_resolution=None)` — `media_resolution=None` resolves to `MEDIA_RESOLUTION_LOW` via `media_resolution_enum`, matching this call's real production default (never explicitly set anywhere in the pipeline today). Response is the `simplified-tags-time-v1` `{"clips": [...]}` shape — zero, one, or several tagged sub-events, each with its own **chunk-relative** `start_s`/`end_s`, `action_class`, `outcome`, `confidence`.

Convert every returned clip to an absolute prediction event for the matcher:
```
pred.start_s = window_start_s + clip["start_s"]
pred.end_s   = window_start_s + clip["end_s"]
pred.action_class = clip["action_class"]
pred.confidence = clip["confidence"]
```

**Ruling — Arm S re-run IS the official baseline.** The 63% `action_class` @ fps=10 prose number (INS-096/097/102/103 lineage) is reported in `GATE1_REPORT.md` as a **footnote for continuity only** — it has no backing artifact and cannot be recomputed, so it is never used as the pass/fail bar (see B.4).

### B.2 Arm T — position + technique + validator (production's real emission contract)

Three calls per window, in this order, all against the identical window from §A.4:
1. **Position** (`highlight_axes.build_position_prompt`/`build_position_schema`, `PositionAxisConfig` defaults: `media_resolution=None`→LOW, `thinking=None`→off).
2. **Technique** (`highlight_axes.build_technique_prompt`/`build_technique_schema`, `TechniqueAxisConfig` defaults: `media_resolution=None`→LOW, `thinking=None`→off).
3. **Validator** (`highlight_axes.build_validator_prompt`/`build_validator_schema`, fed the position+technique labels/justifications from steps 1–2, `ValidatorAxisConfig` defaults: `media_resolution="high"` — this is production's real, shipped default, not a knob this gate varies), `max_validator_iterations=1` (v1 shipped default).

**Ruling — the validator IS in scope, not a separate lever.** The gate text's "two Gemini axis calls (position, technique)" is shorthand for the whole v2 `highlight_analyze` redesign; production's `HighlightAnalyzeConfig` cannot run position+technique without the validator (`max_validator_iterations >= 1` is structurally required — see `executors.highlight_analyze_node`'s `for iteration in range(1, cfg.max_validator_iterations + 1)` with no skip path). Testing position+technique alone and pretending the validator doesn't exist would score a config that will never ship. Arm T = the real 3-call contract, and its cost (3 calls/window vs. Arm S's 1) is exactly the "3–4x" the gate text asks about (the missing 4th unit is `highlight_critique`, bypassed here because Gate 1 tests fixed, already-located windows — critique's own accuracy is a Gate-5-adjacent boundary/localization question, not this gate's).

**Ruling — score the validator-FINAL verdict, never the raw technique-call output.** Per `executors.highlight_analyze_node`'s real reconciliation logic (verified read, lines ~1440–1500):
- `verdict == "agree"` → `final_action_class = technique_label` (unchanged).
- `verdict == "disagree"` → `final_action_class` changes **only if** `wrong_label == "action_class"` (single-field-correction contract — a validator that disagrees about `position` must NOT silently overwrite `action_class`); otherwise unchanged.
- `verdict == "ditch"` → **zero emitted events for this window.** Score as `fn_silent` for the gold event (a real miss — the redesign chose not to report anything here). Do not special-case ditches out of the denominator; a design that ditches real events must pay for it in recall, exactly as production would.

Arm T therefore emits **at most one** prediction event per window: `{start_s: window_start_s, end_s: window_end_s, action_class: final_action_class}` (no confidence field exists on this path — leave `confidence` absent; Arm T is excluded from any confidence-threshold/PR-curve comparison, only the single fixed operating point is scored).

### B.3 Matching / scoring mechanics

Per window `i` (gold event `g_i`), mint a **synthetic match_id** `f"{g_i['match_id']}__w{i}"` for BOTH the gold event and the arm's prediction(s) before calling `matcher.match_all` — mirrors `metrics.bootstrap_match_ci`'s existing synthetic-id technique. This is mandatory: several selected windows share a real `match_id` (e.g. multiple guard_pass events from the same video), and without per-window isolation the Hungarian assignment could let Window A's gold event get "matched" by Window B's prediction — a real bug the harness's own real-match_id grouping doesn't protect against here, since we're intentionally scoring isolated windows, not whole matches.

Run `metrics.attach_compat` → `metrics.filter_scoreable` (drops `transition`/`guard_pull` from Arm S's occasionally-hallucinated extra clips, per `non_scoreable_prediction_classes` in `gracie_mapping/v1.yaml` — gold-side filtering is a no-op here since every selected gold event is `scoreable=True` by construction) → `matcher.match_all(..., tau_t=5.0)` → `metrics.summarize`, once per arm, over the full n=38-window pool.

### B.4 PASS rule (numeric, decisive)

Let `hit_S` = count of the 38 windows where Arm S produced a `tp` for the gold event (i.e. gold event index appears in `outcome.tp`); `hit_T` analogously for Arm T. Let `fp_dup_rate_X = fp_duplicate_count_X / total_emitted_predictions_X`.

**PASS requires BOTH:**
1. `hit_T − hit_S ≥ 4` (≥ ~10.5 percentage points on n=38 — the smallest margin distinguishable from sampling noise at this n that still represents a real, not marginal, improvement).
2. `fp_dup_rate_T ≤ fp_dup_rate_S + 0.05` (absolute 5-point ceiling on how much the duplicate-FP rate is allowed to rise — Arm T structurally emits at most 1 prediction/window so its raw `fp_duplicate` count cannot itself explode the way a hallucination-prone Arm S list-of-clips call could, but a real check must still be run and reported, never assumed).

Report the exact **two-sided sign test p-value** on the paired per-window (T-hit, S-hit) outcomes (ties excluded) as supplementary/informational context (n=38 has limited power — do not gate on it, but do not omit it either; a large margin with a non-significant p-value is itself a finding worth flagging in the report, not hidden).

**NO-GO**: `hit_T − hit_S < 4`, OR `fp_dup_rate_T > fp_dup_rate_S + 0.05`. Either failure alone is sufficient for NO-GO — report which.

---

## C. GATE 2 ARMS

**Ruling — only the validator's `media_resolution` is varied.** Position/technique stay at their production default (LOW) in both conditions — varying them too would test a config nobody proposes shipping (the gate text's "HIGH `media_resolution` on the occlusion subset" names the validator specifically, and recon confirms the validator is the *only* axis that defaults HIGH in production; position/technique default LOW and this design does not question that separately).

**Efficient design — position/technique run ONCE per clip, validator runs TWICE.** Since position/technique outputs are the validator's *inputs*, not its output, there is no need to re-run them per validator-resolution condition (they'd be measuring pure sampling noise across two independent Gemini calls with identical inputs, not the effect of validator resolution). Per clip:
1. Position call (LOW) — once.
2. Technique call (LOW) — once.
3. Validator call, `media_resolution="low"` (the non-default value — Python `Literal["low","medium","high"]`, matches `media_resolution_enum`'s LOW mapping) — using the SAME position/technique labels/justifications from steps 1–2.
4. Validator call, `media_resolution="high"` — same inputs, independently invoked (not chained; validator-high does not see validator-low's output).

4 calls/clip × 16 clips = 64 calls (§D).

### C.1 Scoring — exact-class agreement, NOT the Hungarian matcher

Per the module-level warning above: the family-compat rule in `gracie_mapping/v1.yaml` would score a `submission_leg_lock` gold event "matched" by a validator output of `submission_arm_lock` as a TP — exactly the confusion this gate exists to detect. `matcher.py` is not used for Gate 2. Instead, per clip, per validator-resolution condition:

```
correct = (final_action_class == gold.action_class)   # exact string equality, not family membership
```

A `"ditch"` verdict scores as `correct = False` (the validator declined to report the real event that's known to have happened — a miss, not a neutral non-event, since every Gate-2 clip is drawn from `scoreable=True` hard-track gold).

### C.2 Per-clip reporting (mandatory — n is tiny, no bare aggregate)

`GATE2_REPORT.md` MUST contain a full per-clip table, one row per of the 16 clips:

| match | gold_class | gold_outcome | pos_label | tech_label(=technique final_action_class before validator) | val_low_verdict | val_low_final_class | correct_low | val_high_verdict | val_high_final_class | correct_high |
|---|---|---|---|---|---|---|---|---|---|---|

...plus the aggregate `accuracy_low = correct_low_count/16`, `accuracy_high = correct_high_count/16`, `net_delta = correct_high_count − correct_low_count`.

### C.3 PASS rule (numeric, decisive)

**PASS** (HIGH is justified — keep it as the shipped default): `net_delta ≥ 2` (≥ 12.5 percentage points on n=16 — treated as "measurable" for this deliberately tiny, directional-per-the-gate-text sample; anything smaller is indistinguishable from single-clip noise at this n).

**NO-GO** (HIGH is not justified — drop the validator to default/LOW, a real cost win): `net_delta < 2`, including `net_delta ≤ 0`.

Report this as **directional** in the headline (per the gate text's own framing), but the PASS/NO-GO call itself is binary per the threshold above — no hedging in the verdict line, only in the surrounding caveat that n=16 is small and a future larger occlusion-subset re-run is reasonable post-launch tuning, not a blocker.

---

## D. SETTINGS TUPLE + SPEND

| | fps | media_resolution | model | thinking | τ_t | max_validator_iterations |
|---|---|---|---|---|---|---|
| Gate 1 — Arm S | 10 | None (→LOW) | `gemini-3.1-flash-lite` | None (→off) | 5.0s | n/a |
| Gate 1 — Arm T position | 10 | None (→LOW) | `gemini-3.1-flash-lite` | None (→off) | 5.0s | n/a |
| Gate 1 — Arm T technique | 10 | None (→LOW) | `gemini-3.1-flash-lite` | None (→off) | 5.0s | n/a |
| Gate 1 — Arm T validator | 10 | `"high"` (production default) | `gemini-3.1-flash-lite` | None (→off) | 5.0s | 1 |
| Gate 2 — position/technique | 10 | None (→LOW) | `gemini-3.1-flash-lite` | None (→off) | n/a (§C.1, no matcher) | n/a |
| Gate 2 — validator-low | 10 | `"low"` | `gemini-3.1-flash-lite` | None (→off) | n/a | 1 |
| Gate 2 — validator-high | 10 | `"high"` | `gemini-3.1-flash-lite` | None (→off) | n/a | 1 |

`thinking` is held constant at the production default (off) throughout — it is not a lever either gate is asked to test; varying it would be a 3rd uncontrolled variable in an already-tight budget.

**Call budget:**
- Gate 1: 38 windows × 4 calls/window (1 Arm S + 3 Arm T) = **152 calls**.
- Gate 2: 16 clips × 4 calls/clip (position + technique + validator-low + validator-high) = **64 calls**.
- **Total: 216 calls**, under the ~250 cap, leaving a **34-call buffer**.
- Mandatory smoke test before the full batch: 2 Gate-1 windows (both arms, 8 calls) + 1 Gate-2 clip (4 calls) = 12 calls, drawn from the 34-call buffer, to catch a schema/prompt/URL-normalization bug before spending the other ~204 calls. Remaining ~22-call buffer covers replacement draws for any window whose call fails irrecoverably (§D.1) — up to ~5 replacements at 4 calls each.
- $ estimate (informational, via `eval/adcc/predict/cost_model.estimate_highlight_call_cost_usd`, 9s clips, LOW≈70 tok/frame except the validator-high calls at 280 tok/frame): sub-$1 total: 216 calls × 9s × (≤$0.0000335-ish per-sec ceiling at HIGH) + per-call flat overhead is on the order of a few cents to low single-digit dollars. Not a budget concern at this n; do not over-invest in precision here — report the actual dollar total the driver's own call log sums to (real, not estimated) in `GATE1_REPORT.md`/`GATE2_REPORT.md`, sourced from each call's `usage_metadata`.

### D.1 Call-failure policy — no silent fallback

Every call already goes through `service.pipelines.gemini_retry.call_with_retry` (5 attempts, exponential backoff + jitter, transient-only per `TRANSIENT_HTTP_CODES`) via `SimplifiedTagsTimeAnalyzer.analyze_chunk` — this is production's existing retry policy, reused as-is. **Do not add a second retry layer on top of it.**

If a call still fails after that exhausts (transient exhaustion) or raises a non-transient error (bad schema, auth, model-not-found):
- Record `status: "call_failed"` for that window/clip/arm, with the real exception message verbatim — never a fabricated verdict, never a silent "treat as ditch"/"treat as miss" substitution.
- **Failure counting is per WINDOW/CLIP, not per raw call** (§G.2 — amended 2026-07-26 to resolve a smoke-test ambiguity). A window/clip counts as ONE failure if ANY of its arm's constituent calls (`arm_s`; or `position`/`technique`/`validator` for Arm T; or `position`/`technique`/`validator_low`/`validator_high` for Gate 2) comes back `call_failed` — regardless of how many of that window's calls were attempted before the failure. Arm T's calls are sequentially dependent (technique/validator have no defined meaning without position), so a partial 2-of-3 result is not a partial score, it's an excluded window. Exclude that window/clip from the arm's scored N for that gate's aggregate (report the exclusion explicitly: "N=X windows scored, Y excluded (call_failed): [window_ids]").
- **Halt thresholds** (§G.2 — the smoke test found the original wording of this bullet self-contradictory, ">1 of 16" vs. "any Gate-2 failure at all"; this replaces both): Gate 1 halts when failed-window-count **> 3 of 38** (the 10% line). Gate 2 halts on **ANY** failed clip (threshold 0, trigger at 1+) — deliberately stricter than a flat 10%-of-16 rule, because Gate 2's 16-clip pool is not an interchangeable random sample: several slots are FORCED inclusions of scarce `unsuccessful`-outcome events (§A.3), and losing one silently would delete exactly the hard/occluded population the gate exists to measure. The engineer's smoke-test implementation (halt-on-first-failure for Gate 2, `GATE2_FAILURE_HALT_THRESHOLD = 0` in `gate_common.py`) is confirmed correct and is now the canonical rule — no further doc ambiguity remains here.
- **Replacement-window drawing** (§G.2 — the original "next item in the seeded sequence" phrasing was not implementable: `random.Random(seed).sample(pool, k)` and `.sample(pool, k+1)` do not share a common prefix, so there is no well-defined "next" item, and re-deriving via a full-pool permutation would silently disturb the ALREADY-FROZEN, already-sha256'd primary draw in `gate1_subset_manifest.json`/`gate2_subset_manifest.json`). The implementable rule: a replacement is a **separate, freshly-seeded sub-draw over the stratum's remaining (not-yet-selected) pool**, using a distinct seed `SEED_REPLACEMENT = 20260727` (`SEED + 1`, same date-derived convention). For a stratum/class pool `P` with an already-frozen primary draw `D`: `remaining = sorted(P - D, key=_event_key)`; draw one replacement via a **freshly re-instantiated** `random.Random(20260727).sample(remaining, 1)[0]`. If more than one replacement is needed in the same stratum, process them in ascending order of the failed window's `window_id`, each time re-instantiating `Random(20260727)` fresh against the then-current (shrinking) `remaining` pool — fully deterministic and reproducible regardless of how many replacements end up needed, and never touches the original committed draw. For a Gate-2 FORCED slot specifically, a replacement must first be drawn from the same class's remaining `unsuccessful` sub-pool (preserving the oversampling intent) and only fall back to the `successful` sub-pool if that is exhausted (document which pool a given replacement came from in the manifest). Funded from the buffer described in §D.

---

## E. GATE 5 RULINGS

### E.1 Excluding k=0 (scope-start) from boundary counting — **correct, confirmed.**

`k=0` is the origin of the analysis scope (`grid_start`), not an internal chunk boundary — there is no "before" the start of the scope for an event to straddle across. Only `k ≥ 1` (multiples of 720s past `grid_start`) are real internal seams the zero-overlap chunking can cut an event across. Excluding k=0 is not a judgment call with a wrong alternative; it's the only definition that means anything.

### E.2 The `[t_s−W, t_s+5]` evidence-span model — **acceptable; W=60s is the decision-relevant number.**

Backward-loaded (`W` seconds before the point, a fixed 5s after) correctly encodes the gate's own stated concern — "setup evidence is lost, not symmetric noise" — rather than a naive symmetric window. Grounding the choice of `W` in the actual pipeline: `HighlightCritiqueConfig.critique_backpad_s` (default 6.0s, allowed range 0–30s) is production's real backward-lookback knob for correcting a scan's timestamp drift, and `HighlightAnalyzeConfig.preroll_s` (default 5.0s, max 15s) is the real pre-highlight context window sent to Gemini. The true "how far back could a real production run's evidence-gathering plausibly reach" envelope tops out around `critique_backpad_s`'s own **max** of 30s in the worst case, not 120s — so **W=30 is the tightest defensible floor, and W=120 materially overstates the true evidence-reach the production pipeline actually has**. W=60 is chosen as the decision-relevant number: a 2x safety margin over `critique_backpad_s`'s own allowed max (covering scan-pass timestamp drift beyond critique's own correction budget), while still being far short of W=120's inflated 3.8%/21-event figure. At W=60: **8/557 events (1.4%) straddle a 720s boundary.**

### E.3 Conditional verdict rule

**Primary path (this design's default — see §A.2 Step 2): measure the real delta, don't guess at a threshold in the abstract.** The 8 W=60 straddling events are forced into the Gate-1 subset specifically so this doesn't have to be answered blind. After Gate 1 runs, partition its n=38 windows by `gate5_straddler` and compare Arm T's per-window hit rate (§B.3's `tp` outcome) on the 8 straddlers vs. the 30 non-straddlers.

**Flip-to-NO-GO threshold**: straddler-subset hit rate more than **25 percentage points** below non-straddler-subset hit rate (absolute) → NO-GO, meaning the zero-overlap 720s chunking must be revisited (add cross-chunk overlap + dedup) before cutover. Rationale for 25pp specifically: at n=8, a single-event swing is 12.5pp — 25pp requires roughly a 2-event gap, the smallest gap distinguishable from single-event noise at this n while still being a "material fraction," matching the gate's own "material fraction... loses accuracy" framing.

**Fallback path** (only if call failures — §D.1 — drop the surviving straddler count below 5 of the original 8, making even a 2-event swing statistically meaningless): close Gate 5 on **count alone** — 1.4% (8/557) at the decision-relevant W=60 is small enough to PASS provisionally — **plus** a mandatory post-cutover follow-up ticket: once the full 557-event corpus eventually runs through the production path (a separate, later, much larger batch — not part of this 216-call budget), pull exactly the 8 straddling events already enumerated in `GATE5_REPORT.md` and compare their recall against a random non-straddling sample of equal size. This is a monitoring item, not a pre-cutover blocker, given the corpus-wide rate is low — but it must be filed as a tracked ticket, not silently dropped, if the fallback path is the one actually used.

---

## F. OUTPUT CONTRACT

### F.1 File layout (`eval/adcc/s12_gates/`)

```
gate1_subset_manifest.json      # §A.2 — 38 windows, gate5_straddler flag, sha256 of gold snapshot + itself
gate1_predictions_arm_s.json    # raw+parsed Arm S call records, one per window
gate1_predictions_arm_t.json    # raw+parsed Arm T call records (position/technique/validator), one per window
gate1_scoring.json              # matcher/metrics.summarize output per arm + paired comparison stats (§B.4)
GATE1_REPORT.md                 # headline table, PASS/NO-GO, sign-test p, fp_duplicate rates, footnoted 63% number
gate2_subset_manifest.json      # §A.3 — 16 clips
gate2_predictions.json          # raw+parsed position/technique/validator-low/validator-high per clip
gate2_scoring.json              # exact-class agreement per clip (§C.1) + aggregate
GATE2_REPORT.md                 # mandatory per-clip table (§C.2), PASS/NO-GO
GATE5_ADDENDUM.md               # straddler vs non-straddler Arm T hit-rate comparison (§E.3), sourced from gate1_scoring.json
gate_budget_ledger.json         # every call made across both gates: op, window_id, arm, tokens (from usage_metadata), $ (real, not estimated), ok/call_failed
```

`gate5_boundary_straddle.py`/`GATE5_REPORT.md`/`gate5_results.json` (already committed) are NOT modified — `GATE5_ADDENDUM.md` is additive.

### F.2 Per-window/clip record schema (both `gate1_predictions_arm_*.json` and `gate2_predictions.json`)

```jsonc
{
  "window_id": "<match_id>__w<index>",       // Gate 1; Gate 2 uses "<match_id>__occ<index>"
  "match_id": "...",
  "gate5_straddler": false,                   // Gate 1 only
  "gold_event": {
    "t_s": 727, "action_class": "submission_arm_lock", "outcome": "successful",
    "position_bucket": "open_guard", "track": "hard", "scoreable": true
  },
  "video_url_watch": "https://www.youtube.com/watch?v=3gotA-Y0gXQ",
  "window_start_s": 722.0, "window_end_s": 731.0,
  "settings": {"fps": 10, "model": "gemini-3.1-flash-lite", "thinking": null,
               "media_resolution": {"position": null, "technique": null, "validator": "high"}},
  "calls": [
    {
      "call_name": "position",                // "arm_s" | "position" | "technique" | "validator" | "validator_low" | "validator_high"
      "status": "ok",                          // "ok" | "call_failed"
      "error": null,
      "raw_response_text": "...",              // pre-parse, as returned by analyze_chunk/last_raw_response_text
      "parsed": {"position": "open_guard", "justification": "..."},
      "usage_metadata": {"prompt_token_count": 0, "candidates_token_count": 0, "total_token_count": 0},
      "latency_ms": 0.0
    }
  ],
  "emitted_events": [                          // what actually gets fed to the matcher (Gate 1) — [] on ditch
    {"start_s": 722.0, "end_s": 731.0, "action_class": "submission_arm_lock", "confidence": null}
  ],
  "status": "ok"                               // "ok" | "call_failed" | "ditched" (Arm T / Gate 2 only)
}
```

### F.3 `GATE1_REPORT.md` mandatory contents
- Settings tuple table (§D), subset composition table (§A.2), full call budget actually spent (real, from `gate_budget_ledger.json`).
- Per-arm `metrics.summarize` output: micro P/R/F1, `fp_breakdown.duplicate` rate, totals.
- `hit_S`, `hit_T`, `hit_T − hit_S`, PASS/NO-GO per §B.4's numeric rule, sign-test p-value.
- The 63% prose baseline, footnoted, explicitly labeled "no backing artifact, reported for continuity only, not the pass bar."
- List of any excluded (`call_failed`) windows and whether a replacement was drawn.
- The `whole_match`/`uniform_span` null baselines (`highlight_baselines.py`) computed over the SAME 38-window pool, per the harness's own "never a headline F1 with no null-model comparison" discipline — reported alongside, not gating.

### F.4 `GATE2_REPORT.md` mandatory contents
- Full per-clip table (§C.2) — no bare aggregate permitted.
- `accuracy_low`, `accuracy_high`, `net_delta`, PASS/NO-GO per §C.3.
- Explicit "directional, n=16" caveat in the same paragraph as the verdict, not buried elsewhere.

### F.5 `GATE5_ADDENDUM.md` mandatory contents
- Straddler (n=8) vs non-straddler (n=30) Arm T hit-rate comparison, sourced from `gate1_scoring.json`.
- Verdict per §E.3's threshold, or the fallback path's count-only closure + follow-up ticket text if fewer than 5 straddlers survived call failures.

---

## Summary for the engineer

Implement a driver under `eval/adcc/s12_gates/` (new module, e.g. `run_gate1.py`/`run_gate2.py`, mirroring `run_predictions.py`'s dry-run-by-default + `--execute` gate discipline — default to printing the manifest + budget plan with ZERO Gemini calls, require `--execute` to spend). Read `SimplifiedTagsTimeAnalyzer` and `highlight_axes.py`/`_call_axis_json` directly (already-committed, read-only reuse) rather than reimplementing the schemas/prompts. Everything else in this doc is fully specified — subset membership is a deterministic function of the seed and the tables above; there is no remaining judgment call left for implementation to make.

---

## G. AMENDMENTS (2026-07-26, post-smoke-test — Bezos pass-with-conditions + engineer handoff)

§D.1's per-window-vs-per-call counting, the Gate-2 halt threshold, and the replacement-draw procedure are corrected in place above (search "§G.2" in §D.1) — do not use any pre-2026-07-26 reading of those three points. This section records the one new ruling the smoke test surfaced: the `pwGbW5GZgfc` video blocker.

### G.1 Blocker — video `pwGbW5GZgfc` (8h29m ADCC Trials Finals VOD) fails native YouTube ingestion at every offset/fps

**Confirmed root cause class**: an undocumented Gemini native-`file_uri`-YouTube-ingestion duration cap somewhere between ~3.6h (control video succeeds) and ~8.5h (this video 400s on every probed offset/fps, near-start and mid-video alike) — not an offset-specific or fps-specific bug. This video is the source for Gate-1 straddler `w37` (`66b009481c62a1db7a53d504`, back_take, t_s=30443) and Gate-2's forced-unsuccessful `occ0` (arm_lock, `66afe5ab1c62a1db7a53b80b`) and `occ11` (leg_lock, `66a6ff021c62a1db7a5133c0`).

**Ruling: (c) — route ONLY these 3 windows through the Gemini Files API upload lane.** Rejected alternatives, on the merits:
- **(a) drop the 3 windows** — rejected as first choice. Gate 2 would lose its ONLY forced-unsuccessful `submission_arm_lock` (the arm_lock stratum would silently become 100% `successful`, deleting exactly the occluded/failed-attempt population Gate 2 exists to oversample, §A.3). Gate 1's straddler addendum would drop from n=8 to n=7 (survivable, but avoidable). A cheap, principled fix exists (12 calls); dropping data for expedience when one does is the weaker call.
- **(b) redraw replacements from the seeded pools** — rejected for the two Gate-2 forced slots specifically. Unsuccessful-outcome events are scarce by construction (1 arm_lock / 5 leg_lock in the entire 552-event hard-track corpus, §A.3) — "next in the seeded successful pool" is not a neutral substitution here, it changes what the stratum measures (occlusion-on-failure → occlusion-on-a-clean-finish). (For Gate-1's `w37` specifically, redrawing would be more defensible — the straddler set is a fixed enumeration from `GATE5_REPORT.md`, not a stratified sample, so "drop to n=7" is a clean fallback with no substitution question — but (c) fixes the shared root cause for all 3 windows at once, so there is no reason to solve `w37` differently from `occ0`/`occ11`.)
- **(d)** — no better option surfaced. Files-API upload is the one path in this codebase built exactly for "give Gemini a video by a means other than a YouTube URL" (`service/pipelines/gemini_upload.py`, wt-s12 worktree — `upload_video_to_gemini` / `poll_until_active` / `delete_gemini_file`, no fallback logic, fails loud on `FAILED`/timeout).

**Implementation constraints (binding):**
1. **Copy `gemini_upload.py` verbatim** from `whole-video-analysis-wt-s12/service/pipelines/gemini_upload.py` into this worktree (`eval/adcc/s12_gates/gemini_upload.py` or `service/pipelines/gemini_upload.py` here — engineer's choice of exact path, not the copy decision itself). It is safe to duplicate: zero `shared_lib` imports, zero other production-module imports, pure `google.genai` Files API mechanics. Comment the copy's provenance (source path + `git log -1 --oneline` of that file in wt-s12 at copy time) so it is traceable, not a silent fork. This is a **temporary, disclosed duplication** — when wt-s12 eventually merges, this worktree's copy should be deleted or re-pointed at the merged module; note that in the copy's header comment too.
2. **Local clip prep — reuse, don't reinvent, the codec-normalization the harness already has.** `yt-dlp --download-sections "*{window_start_s-2}-{window_end_s+2}" -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]"` (2s buffer past each edge of the 9s analyze window, so a keyframe-boundary cut never clips the actual window short; capped at 1080p — no accuracy-relevant benefit above that at either `media_resolution` tier per `cost_model.py`'s fixed 70/280 tokens-per-frame pricing, which is resolution-tier-based not pixel-count-based). Then run the downloaded section through `eval/adcc/predict/media_resolver.probe_is_h264` + `build_mp4_derivative` (already in this worktree, already the harness's own passthrough-or-transcode discipline) to guarantee H.264/AAC/faststart MP4 regardless of source codec. This does not eliminate every possible confound (Gemini's own internal YouTube-fetch transcode is opaque and can't be replicated exactly) but it removes every confound this codebase has the means to remove, using its own existing conventions rather than a fresh one-off pipeline.
3. **Cross-arm consistency (mandatory for paired validity).** One upload per window (not per call) — the resulting `files/xxxx` URI is reused across ALL of that window's calls (Arm S's one call; Arm T's position/technique/validator; or Gate 2's position/technique/validator_low/validator_high), exactly mirroring how one `youtube_url` string is reused across a window's calls today. This preserves the A/B: both arms still see byte-identical video content for a given window, even though the ingestion mechanism differs from the other 35 (Gate 1) / 13 (Gate 2) windows.

   **[ENGINEER CORRECTION 2026-07-25]** The implicit assumption above and in the engineer's own earlier handoff report — that `SimplifiedTagsTimeAnalyzer.analyze_chunk`/`_call_axis_json` are "agnostic to what kind of `file_uri` they're given," i.e. a Files API URI can be dropped into the existing `youtube_url` parameter with no other change — is **disproven**. Two real, load-bearing differences were found and fixed during the reroute smoke test (see `files_api_reroute.py`'s module docstring for the full writeup): (1) `file_data.mime_type` must be set for a non-YouTube `file_uri` (undocumented in this worktree's `analyze_chunk`, present in wt-s12's later fork); (2) `video_metadata.start_offset`/`end_offset` must be **file-relative** (the uploaded, already-cropped file's own 0:00), not the original video's absolute match-clock second — sending absolute offsets against a ~13s uploaded file returned `500 INTERNAL` on every attempt. Both are fixed at the call site (`ReroutedUpload.call_window_bounds()` for the offset conversion; a local `FilesApiTimeAnalyzer` subclass for `mime_type`) and confirmed working via a real Gemini call (2026-07-26 smoke test, `occ0`: `position`/`technique` both `status=ok`, `technique` result exactly matched gold). Gate 3 (`GATES_3_4_DESIGN.md` §H.2) makes the identical assumption and will need the same fix.
4. **Tag and disclose, never blend silently.** Every call record and ledger entry for these 3 windows carries `"ingestion_path": "files_api"` (vs. `"youtube_uri"` elsewhere). They stay IN the headline hit-count/accuracy pools (no exclusion) — a paired per-window test is robust to a per-window nuisance variable shared by both arms of that window — but `GATE1_REPORT.md`/`GATE2_REPORT.md` MUST additionally report the headline figures **recomputed with these tagged windows excluded**, and state explicitly whether the PASS/NO-GO verdict is sensitive to their inclusion. If exclusion flips the verdict, that is a reportable finding, not a footnote.
5. **Poll/upload latency recording**: log `upload_latency_ms` (upload call return) and `active_wait_ms` (`poll_until_active`'s total elapsed) per window in the ledger, alongside the existing per-call fields — this is exactly the data Gate 3 also needs (§H below), captured once, reused twice.
6. **Spend delta: zero.** These 3 windows' calls were already counted in the original 216 (12 of them: 3 windows × 4 calls). Re-routing their ingestion mechanism does not add or remove a Gemini call — Gates 1+2 stay at 216 total.
7. **Synergy with Gate 3, bounded.** These 3 forced uploads ARE a free, notable data point for Gate 3's broader "is Files-API viable" question — Files-API successfully ingesting a video that 400s on every native-YouTube attempt is itself a finding worth stating in `GATE3_REPORT.md`. But they do **not** count toward Gate 3's core n=8 paired-comparison sample (§H.1): by definition there is no successful YouTube-URI baseline for these 3 windows to pair against (that's the whole reason they needed rerouting), so there is nothing to compute an agreement-rate or accuracy-delta against. Report them as a separate, clearly-labeled "bonus finding: Files-API rescues an ingestion-path-broken video," never blended into the n=8 fidelity numbers.
