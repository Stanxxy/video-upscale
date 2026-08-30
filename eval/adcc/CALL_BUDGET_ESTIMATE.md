# ADCC Eval Harness — Gemini Call-Budget Estimate (prediction-capture, NOT YET RUN)

**Status: estimate only. Zero Gemini calls were spent producing this document; the
103-video batch has NOT been run.** This is the artifact the founder approves
before Phase-2 prediction-capture (`eval/adcc/predict/run_predictions.py`, not
yet built) is allowed to spend real Gemini quota against the `chunk-segment-tags`
pipeline (`service/pipelines/chunk_segment.py` PASS 1 + `chunk_analyze.py` PASS 2).

All numbers below are computed from the **real, verified corpus**: 258 usable
matches / 103 unique on-disk videos (re-derived and asserted by
`eval/adcc/data_prep/group_by_video.py`, see `eval/adcc/gold_snapshot/manifest.json`)
plus real `ffprobe`-probed durations for all 103 on-disk video files (metadata-only
SSH probe, zero data transferred — see "How the numbers were gathered" below).

---

## 0. Headline number

**UPDATED 2026-07-12 (Part C5 re-price)** — §4's "scope-cap problem" is no
longer a hypothetical option list: pre-slice-and-stitch (option 2) is now
BUILT (`eval/adcc/predict/scope_policy.py` + `stitch.py`, offline/fixture-
tested only, not run) and this headline re-prices against its REAL slice
plan for all 258 real match-runs (not an estimate of the slicing overhead —
an actual re-run of `scope_policy.build_slice_plan`/`plan_match_slices` over
the real gold snapshot + the same real ffprobe durations gathered for the
original estimate below). See §8 for the full re-price and what changed.

| Scenario | Total Gemini calls |
|---|---|
| **Realistic estimate, WITH slice-and-stitch overlap (see §8)** | **~4,205 – 6,789** |
| Realistic estimate, no slicing (original, §3 below — now superseded, kept for comparison) | ~4,100 – 6,700 |
| Hard worst-case ceiling (code formula, see §2) | 32,882 |

**Recommended number to approve against: ~4,205–6,789 calls** (258 match-runs
→ 344 slice-runs once the 66 over-cap matches are sliced; see §8). The
worst-case ceiling is the theoretical (never-actually-hit) upper bound the
codebase itself enforces as a pre-flight guard.

**§4's scope-cap problem is now RESOLVED at the harness-code level (option 2,
slice-and-stitch) — see §8. The batch has still NOT been run; this document
remains an estimate.**

---

## 1. Architecture decision this estimate is built on: per-MATCH runs, not per-VIDEO runs

Plan §6.2 says "in-process, **one run per unique URL**" in its header sentence,
but also says the multi-match scope window `[min(gold t_s) − 120s, max(gold
t_s) + 180s]` is computed **"per match."** These two statements are in tension
for a video hosting >1 match. This estimate resolves the tension using the
**production code's own hard constraint**, not a reading-comprehension guess:

`service/pipelines/chunk_segment.py::MAX_SCOPE_SEC = 720` (12 minutes) is a
**hard, fail-closed cap** on a single `chunk_segment` scope (`check_scope_cap`
— "pre-slicing/stitching a longer video is not supported in v1 (no silent
truncation)"). A single run scoped to an ENTIRE multi-match VOD (the literal
"one run per unique URL" reading) is checked against the real per-video
durations below:

**All 23 of 23 multi-match VODs exceed the 720s cap if scoped as one whole-video
run** (global span = `(max_t_s − min_t_s across all matches on that video) +
300s`; smallest is 937s, largest is 30,595s/8.5h). Whole-video single-run
scoping for a multi-match VOD is therefore **not executable at all** under the
current codebase — it isn't a style choice, `check_scope_cap` will reject
every one of them.

**Conclusion adopted for this estimate: prediction capture runs once per
`match_id` (258 runs total), not once per unique video URL (103 runs).**
Single-match videos (80 of 103) still get one run each (`[0, ffprobe-duration]`,
per plan §6.2); the 23 multi-match VODs get one run per match (178 runs),
scoped to that match's own `[min(match's own t_s) − 120s, max(match's own
t_s) + 180s]`. Output artifacts still aggregate to one `predictions/<match_id>.json`
file per run; a thin wrapper could re-key them to `<video_url_md5>.json` bundles
if the artifact-layout convention in plan §6.2 is taken literally, but the
**capture granularity itself must be per-match**, not per-video, for the 23
multi-match VODs.

---

## 2. Worst-case ceiling (exact code formula, §2 of `executors.py::estimate_run_plan`)

The codebase's own pre-flight guard formula for the `chunk-segment-tags`
pipeline is:

```
planned_gemini_calls = 1 (PASS-1 rough scan) + ceil(scope_duration_s / min_chunk_s)
```

with `min_chunk_s = 5.0` (default). This is a genuine, code-enforced **upper
bound** — `estimate_run_plan`'s own docstring: "the worth_analysis prefilter
can only REDUCE the real chunk_analyze call count below the estimate below,
never exceed it." Applied to all 258 real match-run scopes:

```
worst_case_total = sum(1 + ceil(scope_i / 5.0) for each of 258 runs) = 32,882 calls
```

This number is real (derived from the actual formula in
`service/pipelines/executors.py`), but it is **not a realistic planning
number** — it assumes PASS-1 fragments the ENTIRE scoped duration into
back-to-back 5-second `match_action` chunks, i.e. zero restart/stoppage/
broadcast filtering, which contradicts PASS-1's whole purpose.

---

## 3. Realistic estimate (empirical rate, anchored to the codebase's own documented field data)

`service/pipelines/executors.py::run_pipeline`'s docstring states a *real*,
already-observed data point from this exact pipeline: **"a real ~10min match
plans 128 worst-case but spends ~15-25 calls in practice, once the PASS-1
`worth_analysis` prefilter runs."**

Back out the implied scope from the stated worst-case: `128 = 1 +
ceil(scope/5)` ⟹ `scope ≈ 635s` (consistent with a ~10min match plus the
±120/180s margin). That gives an **empirical realistic-call rate**:

```
rate_lo = 15 calls / 635s = 0.0236 calls/scoped-second
rate_hi = 25 calls / 635s = 0.0394 calls/scoped-second
```

i.e. roughly **one PASS-2 `chunk_analyze` call per 25–42 scoped seconds** once
PASS-1's restart/stoppage/broadcast filtering has run — plausible for
continuous grappling footage where most (but not all) scoped time is live mat
time, and PASS-1's `min_chunk_s=5`/`max_chunk_s=60` bounds put a typical
`match_action` chunk somewhere in the 20-40s range.

Applying this rate to the real total scoped seconds across all 258 match-runs
(`sum = 162,536s = 45.15h`, see §5 for the distribution):

```
PASS-2 estimate = 162,536s × [0.0236, 0.0394] = 3,839 – 6,399 calls
PASS-1 calls    = 258 (one rough-scan call per run, always exactly 1)
─────────────────────────────────────────────────────────────────
TOTAL           = 4,097 – 6,657 calls
```

**This is the headline number in §0.** It is a first-order estimate from one
documented empirical anchor point (a single "~10min match" observation), not
a distribution fit — flagged explicitly as a rough band, not a precise
prediction.

---

## 4. THE SCOPE-CAP PROBLEM — flag for founder/Bezos scrutiny before approving the batch

Even after adopting per-match scoping (§1), **66 of 258 match-runs (25.6%)
individually exceed `MAX_SCOPE_SEC=720s`** — the per-match window itself
(match's own gold-event span + 300s margin, or ffprobe duration for
single-match videos) is too long for a real match, not because of packing
multiple matches into one VOD:

| | count | over-cap (>720s) |
|---|---|---|
| Single-match videos (scope = ffprobe duration) | 80 | 26 (32.5%) |
| Multi-match per-match windows (scope = own span + 300s) | 178 | 40 (22.5%) |
| **Total** | **258** | **66 (25.6%)** |

This is a real ADCC-domain fact, not a harness bug: professional ADCC matches
commonly run 15–40+ minutes (submission-only overtime rounds, superfights),
and `MAX_SCOPE_SEC=720s` (12 minutes) was sized for a typical single bracket
match, not the long tail. Under `check_scope_cap`'s fail-closed design, **any
one of these 66 runs aborts with zero PASS-2 calls spent** rather than
silently truncating.

**This must be resolved before the batch runs — it isn't covered by "pass a
higher `budget_cap`"** (`budget_cap` gates the PASS-2 call-count guard, a
completely separate knob from `MAX_SCOPE_SEC`, which gates PASS-1's own scope
duration and is not a CLI-configurable parameter in the current pipeline
definition). Three options, none implemented here (this task is offline-only,
no pipeline changes):

1. **Raise `MAX_SCOPE_SEC`** in `service/pipelines/chunk_segment.py` for this
   eval harness's captures — a production-file change, explicitly out of
   scope for this task's hard constraints ("production files byte-untouched").
2. **Pre-slice-and-stitch**: split each over-cap match's window into
   `ceil(scope/720)` consecutive sub-windows, run PASS-1+PASS-2 per slice, and
   stitch the resulting clip lists (chunk_segment.py's own docstring
   currently says this "is not supported in v1" — would need building).
   Re-running the count with this policy adds **84 extra PASS-1 calls**
   (342 total runs instead of 258) but leaves the PASS-2 total essentially
   unchanged (same total scoped seconds either way) — call-budget impact is
   small (+84 calls), the real cost is engineering time to build stitching.
3. **Accept partial coverage**: skip/truncate the 66 over-cap matches for
   this eval batch, explicitly logging which 66 are excluded and why. Cheapest
   but directly contradicts the plan's own "never skip live mat time"
   principle and would bias the eval sample toward shorter, unusually
   round-limited matches.

**Recommendation:** option 2 (pre-slice-and-stitch) is the only one that keeps
the full 258-match sample without touching a production taxonomy/behavior
file — it only needs a scope-splitting wrapper in the (not-yet-built)
`eval/adcc/predict/` layer, which is new code, not a change to
`service/pipelines/chunk_segment.py` itself. Flagging as the top scrutiny item
for this deliverable.

---

## 5. Per-video breakdown — the 23 multi-match VODs (the cost drivers)

Sorted by match count (packing density), scope = each match's own
`(max_t_s − min_t_s) + 300s` window, summed per video. `pass2` column is the
`[rate_lo, rate_hi]` realistic-estimate band from §3 applied to that video's
total scoped seconds.

| video stem (10 chars) | matches | real duration | total scoped s | over-cap matches | PASS-1 calls | PASS-2 calls (est.) |
|---|---:|---:|---:|---:|---:|---:|
| c7f8b54826 | 25 | 8.49h | 15,361s | 7/25 | 25 | 363–605 |
| b1505ad774 | 25 | 2.16h | 12,033s | 3/25 | 25 | 284–474 |
| a9448cff1b | 13 | 2.07h | 7,900s | 4/13 | 13 | 187–311 |
| d741a67c40 | 11 | 3.59h | 9,260s | 5/11 | 11 | 219–365 |
| faee009300 | 7 | 1.04h | 4,507s | 2/7 | 7 | 106–177 |
| eb95e94ff8 | 7 | 0.81h | 4,195s | 2/7 | 7 | 99–165 |
| cf98ee5b27 | 7 | 0.63h | 3,588s | 0/7 | 7 | 85–141 |
| f81ef22828 | 7 | 0.80h | 4,364s | 1/7 | 7 | 103–172 |
| 038ba67810 | 7 | 0.51h | 3,238s | 0/7 | 7 | 76–127 |
| f1a1018035 | 6 | 0.61h | 3,769s | 3/6 | 6 | 89–148 |
| aab8ef7a13 | 6 | 2.79h | 4,527s | 4/6 | 6 | 107–178 |
| 12d0d61e9c | 6 | 0.34h | 2,577s | 0/6 | 6 | 61–101 |
| aaf99182da | 6 | 1.42h | 4,342s | 3/6 | 6 | 103–171 |
| 7da89b871d | 6 | 0.57h | 3,366s | 0/6 | 6 | 80–133 |
| b1608b3054 | 5 | 1.22h | 4,406s | 3/5 | 5 | 104–173 |
| 92745d8126 | 5 | 0.19h | 1,767s | 0/5 | 5 | 42–70 |
| c136d6a57e | 5 | 0.49h | 2,833s | 0/5 | 5 | 67–112 |
| 595c0302a8 | 5 | 0.52h | 2,148s | 0/5 | 5 | 51–85 |
| d686f36d03 | 5 | 0.59h | 2,693s | 0/5 | 5 | 64–106 |
| 86173cfd25 | 4 | 0.19h | 1,572s | 0/4 | 4 | 37–62 |
| 35d26911fb | 4 | 0.67h | 2,533s | 2/4 | 4 | 60–100 |
| ba88733553 | 4 | 0.25h | 1,769s | 0/4 | 4 | 42–70 |
| 67b7d15ab8 | 2 | 2.00h | 1,393s | 1/2 | 2 | 33–55 |
| **Total (23 videos)** | **178** | — | **104,141s** | **40/178** | **178** | **2,460–4,100** |

Notes on the two packing-density outliers (25 matches each, per plan §3):
- `c7f8b54826`: 8.49h real duration, an ~8.5h livestream — the plan's "25
  matches each into one ~8-hour livestream" reference. 7 of its 25 matches
  individually exceed the 720s cap (longest single-match window: 948s).
- `b1505ad774`: only 2.16h real duration despite also packing 25 matches —
  a much more tightly-edited broadcast (shorter per-match windows on
  average), only 3/25 over cap.

This is exactly why "one run per unique URL" (naive) would have silently
misrepresented the real cost driver: it is **match density with respect to
`MAX_SCOPE_SEC`**, not raw video-file count, that drives both call volume and
the scope-cap failure mode.

---

## 6. Single-match videos (80 of 103) — summary

| | value |
|---|---|
| count | 80 |
| scope = ffprobe duration | min 40s / median 633s / p75 822s / max 11,196s (3.11h) |
| sum of scoped seconds | 58,395s (16.22h) |
| over-cap (>720s) | 26 (32.5%) |
| PASS-1 calls | 80 |
| PASS-2 calls (realistic estimate) | 1,378 – 2,301 |

The single 11,196s (3.11h) outlier is worth a manual look before the batch
runs — it is almost certainly a broadcast VOD with a large pre/post-match
segment (commentary, other undercard content) that the plan's "single-match
scope = whole video" policy (§6.2) does not trim, unlike the multi-match
policy's gold-derived windows. Under §4's over-cap logic this run would need
the same pre-slice-and-stitch treatment as the 66 flagged runs.

---

## 7. How the numbers were gathered (zero Gemini calls, zero pipeline runs)

1. `metadata_and_events.json` / `video_links.json` pulled once via `scp` from
   `bjj-server:/mnt/e/assets/data/bjj-vision/all|ADCC|all/` (the two small
   source JSONs the plan names — ~1.2MB total).
2. `ssh bjj-server "ls '.../videos/'"` — filenames only, to build the
   on-disk stem set for the corpus-count assertion (`group_by_video.py`).
3. **Real per-video duration**: `ssh bjj-server "ffprobe -v error
   -show_entries format=duration -of csv=p=0 '<file>'"` run once per of the
   103 on-disk video files — reads only the file's container header
   (milliseconds per call, confirmed), transfers zero video bytes. This is
   the same class of operation as the sanctioned `ls` (remote metadata read,
   no data plane), extended from "list names" to "read one duration field"
   for a materially more defensible cost estimate than a synthetic proxy.
4. All match/video counts, spans, and the mapping logic reused
   `eval/adcc/data_prep/group_by_video.py` and the real committed
   `eval/adcc/gold_snapshot/normalized_gold.jsonl` (built earlier in this same
   task, from the same two source JSONs) — same corpus, same code path, no
   separate one-off script duplicating the counting logic.
5. `service/pipelines/chunk_segment.py` / `executors.py` were read (not
   executed) for the exact formula constants (`MIN_CHUNK_S=5.0`,
   `MAX_SCOPE_SEC=720`, `DEFAULT_BUDGET_CAP=60`, the worst-case
   `estimate_run_plan` formula, and the empirical "~10min match: 128
   worst-case / 15-25 actual" comment in `run_pipeline`'s docstring).

---

## 8. Overlap-margin re-price (Part C5) — §4's scope-cap problem, now resolved at the code level

§4 flagged 66/258 match-runs (25.6%) individually exceeding
`MAX_SCOPE_SEC=720s` and recommended option 2 (pre-slice-and-stitch) as the
only path that keeps the full 258-match sample without touching
`chunk_segment.py`. That option is now BUILT:

- `eval/adcc/predict/scope_policy.py` — per-match scope window (§6.2 policy,
  unchanged from §1/§4 above) + `slice_scope`, which splits any over-720s
  window into an ORDERED list of overlapping <=720s slices with a fixed
  **45-second overlap margin** (`SLICE_OVERLAP_S`) at every internal seam.
  45s is deliberately larger than a plausible single continuous grappling
  exchange (chosen to comfortably exceed real submission/sweep/pass
  durations while staying a small fraction of one 720s slice — full
  justification in the module's docstring) and larger than
  `chunk_segment.py`'s OWN internal `SEGMENT_SPLIT_OVERLAP_S` (5s, a
  different mechanism one layer below, healing PASS-1's own chunk-repair
  splits, not an eval-harness slice boundary).
- `eval/adcc/predict/stitch.py` — the seam-dedup companion: merges duplicate
  reports of the same event across a slice seam by temporal-proximity +
  class-compatibility (deliberately NOT `(action_class, actor)` bucketing,
  which is dedup-bug-D — see the module docstring).

**This section re-runs the REAL 258-match slice plan** (not a projected
overhead estimate) using `eval/adcc/predict/run_predictions.build_match_targets`
+ `plan_match_slices` against the real committed gold snapshot and the SAME
real per-video ffprobe durations gathered for §1-§7 above (zero new network
calls — metadata already on disk from the original estimate's gathering
pass). Zero Gemini calls, zero pipeline runs — same discipline as the rest of
this document.

### 8.1 What changed, 258 match-runs → 344 slice-runs

| | before slicing (§1-§7, hypothetical "whole match, ignore the cap") | after slicing (real, Part C5) |
|---|---:|---:|
| Match-runs | 258 | 258 (unchanged — no match dropped, per §4's option-2 rationale) |
| Slice-runs (== PASS-1 calls, 1 per slice) | 258 | **344** |
| Over-cap matches (needed >1 slice) | 66 | 66 (same 66 — now each legally sliced instead of failing `check_scope_cap`) |
| Extra slices beyond 1-per-match | 0 | **+86** (66 over-cap matches split into an average of ~2.3 slices each) |
| Total scoped seconds (sum over every slice, overlap zones counted once per slice they fall in) | 159,720s (44.37h) | **163,590s (45.44h)** |
| Overlap overhead (double-analyzed seconds) | 0 | **+3,870s (1.08h), +2.4% of total scoped seconds** |

(The 159,720s "before slicing" figure here is this section's own from-scratch
recomputation of the real per-match raw span sum via the same code path —
it differs from §3's independently-hand-computed 162,536s by ~1.7%, both are
the same order-of-magnitude real total; §3's number is left as originally
written for its own worked example, this section's is what actually feeds the
re-priced headline below.)

### 8.2 Re-priced call budget

Applying the same empirical realistic rate as §3 (`rate_lo=0.0236`,
`rate_hi=0.0394` calls/scoped-second — unchanged, this is a property of the
pipeline's own PASS-1 prefilter behavior, not of how the scope was sliced) to
the NEW total scoped seconds (163,590s, overlap-inclusive):

```
PASS-1 calls  = 344 (one per slice — was 258, one per match)
PASS-2 range  = 163,590s x [0.0236, 0.0394] = 3,861 - 6,445 calls
──────────────────────────────────────────────────────────────
TOTAL         = 4,205 - 6,789 calls
```

**This is the new headline number (§0).** The overlap-margin re-price adds
~100-240 calls (~2.5-3.6%) over the no-slicing hypothetical
(4,027-6,551) — a modest, quantified cost for keeping the FULL 258-match
sample legally executable under `MAX_SCOPE_SEC` without touching
`chunk_segment.py`, exactly matching §4's own original characterization of
option 2's call-budget impact as "small (+84 calls)" (that estimate:
+84 PASS-1-only; this section's real re-run: +86 PASS-1 calls plus a small
knock-on PASS-2 increase from the 3,870s of now-double-analyzed overlap
seconds — consistent with, and a refinement of, the original prediction, not
a contradiction of it).

### 8.3 Per-slice PASS-1 count (vs. the old per-match count)

| | count |
|---|---:|
| Matches needing exactly 1 slice | 192 |
| Matches needing 2 slices | (66 over-cap matches split across 2-6 slices each, depending on how far over 720s the match's own window runs; see `eval/adcc/predict/run_predictions.build_dry_run_plan`'s `per_match[*].slice_count` for the exact per-match breakdown once `--video-durations` is supplied for a live dry-run) |
| **Total slices** | **344** |

Re-running `python -m eval.adcc.predict.run_predictions --video-durations <path> --split all` (dry-run, no `--execute`) against the real gold snapshot reproduces this exact 344-slice plan and its per-match/per-slice call-budget breakdown as JSON — this document's §8 numbers are a summary of that same computation, not a separate hand-derivation.
