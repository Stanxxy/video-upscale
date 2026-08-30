# Gates 3 & 4 — Concrete Experimental Design

Author: Yann LeCun (`/scientist`) | Date: 2026-07-26 | Status: DESIGN FROZEN — ready for engineer implementation after the Gate-1/2 full batch fires. Zero Gemini calls spent writing this.

Companion to `GATES_1_2_DESIGN.md` (read that first — subset-freeze conventions, seed discipline, call-failure policy, and the §G blocker ruling are reused here, not re-derived). Grounds Gate 3 in `whole-video-analysis-wt-s12`'s Files-API upload lane (`service/pipelines/gemini_upload.py`) and Gate 4 in wt-s12's real S12 Phase-1b actor axis (`service/pipelines/highlight_axes.build_actor_prompt`/`build_actor_schema`, `ActorAxisConfig`, `AXIS2_ACTOR_SENTINELS`) — **neither of these exists in the `wt-vlmstudio` worktree's fork of `highlight_axes.py`** (verified: it has only position/technique/validator). Both drivers below therefore run **from the `whole-video-analysis-wt-s12` worktree**, not `wt-vlmstudio`.

Total new spend: **28 (Gate 3) + 46 (Gate 4) = 74 calls**, under the CEO's ~150 cap for Gates 3+4 combined, leaving a ~76-call buffer for smoke tests and failure replacements.

**AMENDED 2026-07-26 per `VERDICTS.md` §4, after Gate 1 returned NO-GO**: Gate 3's primary architecture-under-test is repointed from Arm T to Arm S (the presumptive shipping baseline post-Gate-1), reducing its cost to 21 calls (8 Arm-S primary + 9 Arm-T spot-check + 4 fps-honoring). Gate 4 is unchanged. Revised combined total: **67 calls**. §H below is the original pre-amendment text — read it together with `VERDICTS.md` §4 for the corrected pairing and the two Files-API implementation bugs (`mime_type`, file-relative offsets) that section also fixes.

---

## H. GATE 3 — Files-API fidelity probe

### H.1 Clip selection — n=8, paired against the ALREADY-CAPTURED Gate-1 YouTube-URI results

**Do not draw a fresh, unrelated sample.** Gate 1's 38-window batch already produced real YouTube-URI Arm-T (position/technique/validator) outputs, scored against gold. Re-running the SAME 8 windows through Files-API and comparing against those already-captured outputs gives a true same-window paired comparison at zero extra "control arm" spend — exactly what the CEO asked for.

Selection rule: from `gate1_predictions_arm_t.json`, take the windows where (a) `status != "call_failed"` (a real YouTube-URI baseline exists to pair against) and (b) `ingestion_path != "files_api"` (excludes the 3 §G.1-rerouted windows — Gate 3's own §H.4 handles those separately, they are not eligible for the core n=8, per `GATES_1_2_DESIGN.md` §G.1 point 7). Sort the eligible window_ids ascending, draw `random.Random(20260726).sample(eligible, 8)` — a fresh, independently-seeded step (same seed value, new `Random` instance, per the Bezos-condition-5 "one fresh instance per named step" convention already governing Gates 1/2). No stratification by `action_class` is imposed at n=8 (small enough that any reasonable draw is adequate for an ingestion-fidelity question, which is not class-sensitive the way Gate 1/2's accuracy questions are) — but re-draw once if the initial draw happens to put >3 of the 8 in one `action_class` (a purely defensive anti-degenerate-sample check, not a hard stratification requirement).

### H.2 Call shape — the real Arm-T contract, Files-API side, paired field-by-field

For each of the 8 selected windows:
1. Download the exact `[window_start_s − 2, window_end_s + 2]` section via `yt-dlp --download-sections` (§G.1 point 2's exact format string and 2s buffer — reused verbatim, not re-derived).
2. Normalize to H.264/AAC MP4 via `media_resolver.probe_is_h264` + `build_mp4_derivative` (same reuse).
3. `upload_video_to_gemini` → `poll_until_active` (`service/pipelines/gemini_upload.py`, copied per §G.1 point 1 if not already present from the Gate-1/2 blocker fix — if the engineer already did that copy for §G.1, this is the same file, no second copy).
4. Run the SAME 3-call Arm-T sequence as Gate 1 (position → technique → validator, identical settings tuple: fps=10, `gemini-3.1-flash-lite`, media_resolution LOW/LOW/HIGH, thinking off, `max_validator_iterations=1`) against the uploaded file's `files/xxxx` URI instead of the YouTube watch URL.
5. `delete_gemini_file` (best-effort cleanup, per the module's own docstring — never blocks or fails the run).

Cost: 8 windows × 3 calls = **24 calls**.

### H.3 INS-096 fps-honoring re-confirmation on the Files-API path

INS-096 established that `video_metadata.fps` is honored for native `file_uri` YouTube calls. Files API uploads use the identical `types.Part(file_data=FileData(file_uri=...), video_metadata=VideoMetadata(fps=...))` shape (`SimplifiedTagsTimeAnalyzer.analyze_chunk`'s `fps` parameter is agnostic to what kind of `file_uri` it's given) — but this has never been directly confirmed for an uploaded file specifically, and Gemini's server-side handling of an uploaded asset is not guaranteed identical to its YouTube-fetch path.

Method (small, targeted, not a repeat of the original INS-096 methodology at scale — a re-confirmation, not a fresh discovery): take 2 of the 8 already-uploaded Files-API clips (the first 2 by window_id in the §H.1 draw), and for each run the **technique-axis call only** (cheapest single-signal probe) at `fps=1` and `fps=10` against the same uploaded file — 2 clips × 2 fps = **4 calls**. Compare the two responses per clip: a detectable difference in `action_class` and/or a `justification` that references distinguishably different motion/frame-level detail between the fps=1 and fps=10 responses counts as "fps is honored" for that clip. **Confirmation bar**: at least 1 of the 2 clips shows a detectable difference (this mirrors how INS-096 was originally established — a single clear positive is sufficient to re-confirm an already-established finding at small n; an absence across both clips is a NO-GO-adjacent flag requiring investigation, not proof of non-fixed-fps, since n=2 cannot prove a negative).

Cost: **4 calls**. Total Gate 3 core spend: 24 + 4 = **28 calls**.

### H.4 Bonus finding (zero incremental cost) — the 3 §G.1-rerouted windows

Report, separately and clearly labeled "bonus finding, not part of the n=8 fidelity sample": for `w37`/`occ0`/`occ11`, Files-API succeeded where native YouTube ingestion 400s unconditionally. This is evidence Files-API is a viable **fallback/workaround** for the undocumented duration-cap bug, independent of whether it's equally *faithful* — the two are different claims and must not be blended into the same number.

### H.5 "No material regression" — numeric

Two metrics, both reported per-window (mandatory table, n=8 is tiny):

**(a) Cross-path agreement** — `agreement_rate` = fraction of the 8 windows where Files-API's `final_action_class` (validator-reconciled, same reconciliation rule as `GATES_1_2_DESIGN.md` §B.2) **exactly** equals the already-captured YouTube-URI arm's `final_action_class` for the SAME window. This is the more sensitive, more direct fidelity signal (does the alternate path reproduce the SAME judgment on the SAME 9s of video) — not confounded by whether either path happens to be right about gold.

**(b) Accuracy-vs-gold, both paths** — `accuracy_youtube` = # of the 8 windows where the (already-known, from Gate 1) YouTube-URI `final_action_class` matches gold's `action_class`; `accuracy_files_api` = the analogous count for the new Files-API run.

**PASS ("no material regression")** requires BOTH:
- `agreement_rate ≥ 6/8` (75%) — the two paths must mostly agree with each other, not merely land on similar aggregate accuracy while disagreeing per-window (which would itself flag path instability even with matching aggregate accuracy).
- `accuracy_files_api ≥ accuracy_youtube − 1` (Files-API may be at most one window's worth, ~12.5pp, worse than the YouTube-URI baseline on this 8-window sample — the smallest gap distinguishable from single-window noise at this n).

**NO-GO**: either condition fails. If NO-GO, the production S3→Files-API upload lane needs investigation before cutover — every prior accuracy claim (INS-096/097/102/103 lineage, and Gate 1/2's own headline numbers) was measured on YouTube-URI ingestion and would not transfer to the real production path.

### H.6 shared_lib version — does NOT matter here

`gemini_upload.py` and every module Gate 3's driver touches (`simplified_tags`, `highlight_axes` position/technique/validator builders, `_call_axis_json`, `gemini_retry`) import **zero** `shared_lib` symbols — confirmed by reading `gemini_upload.py` (pure `google.genai` wrapper) and `highlight_axes.py`'s imports (only `google.genai.types` and its own sibling `simplified_tags` module for the position/technique/validator axes — `AXIS2_ACTOR_SENTINELS`, the one `shared_lib` import in that file, is only reached by the **actor axis** builder, which Gate 3 never calls). wt-s12's `shared_lib==1.3.0` vs. wt-vlmstudio's `1.2.0` is a non-issue for Gate 3. (It matters for Gate 4 — see §I.5.)

### H.7 Output contract

`eval/adcc/s12_gates/gate3_predictions.json` (per-window: both paths' raw+parsed calls, `ingestion_path` tag, upload/poll latency), `gate3_scoring.json` (agreement_rate, accuracy_youtube, accuracy_files_api, per-window table), `GATE3_REPORT.md` (mandatory per-window table per §H.5, PASS/NO-GO, §H.3's fps-honoring confirmation, §H.4's bonus finding).

---

## I. GATE 4 — Attribution accuracy (actor axis + reference images)

### I.1 The honest framing this design is built around

**There is no player-identity ground truth anywhere in the gold corpus.** `highlight_gold.jsonl`'s event schema is `{action_class, outcome, position_bucket, raw_*, t_s, track, scoreable}` — no `player_id`/`actor` field exists on any gold event. The original handoff text already anticipates this ("There is no pre-set numeric bar — this gate produces the number that lets the founder set one"). **Gate 4 therefore cannot produce a true attribution-accuracy-vs-gold number, and this design does not pretend otherwise.** What it CAN produce, honestly:
- **Self-consistency signal**: does the model attribute the same clip-level actor identity to the same underlying athlete across MULTIPLE highlights of the SAME match (a `cross-highlight flip rate`)?
- **Honesty-channel signals**: sentinel rate (the model declining to name an athlete), `identity_uncertain` rate (the model's own confidence flag).
- **The INS-073 re-confirmation**: does providing ANY reference image measurably change these rates vs. providing none (the original finding: actor collapses to a positional enum WITHOUT reference images)?

No real athlete photographs exist anywhere in this project (confirmed by the CEO's brief). The reference images this design produces are **self-referential crops from the SAME match's own video** — a deliberate, disclosed bootstrap, not a substitute for real photo references. Every metric below is scoped to what this bootstrap can actually support; anything it can't (true name-level accuracy) is explicitly marked "not measurable with this corpus."

### I.2 Match selection — n=8, deterministic

Universe: gold matches with ≥2 `track=="hard"` events (152 of 258 matches, verified). Draw: sort by `match_id` ascending, `random.Random(20260726).sample(eligible, 8)` — a third independently-named seeded step (same convention as §H.1/Gates-1-2's stratified core). The frozen draw (reproducible from the seed and the rule above — do not hand-pick a different one):

| match_id | name | video (watch id) | hard events |
|---|---|---|---|
| 650d20c745120c8433e9d018 | Nick Rodriguez vs John Hansen | ATkSimsKwtg | 3 |
| 650d20c745120c8433e9d027 | Rafaela Guedes vs Eleftheria Christodolou | n_dkNnfjx4s | 7 |
| 65396bbbb7741492b97777fc | Gordon Ryan vs Lachlan Giles | KoPxgjLbJmA | 5 |
| 6567972aa4a13c99c4fe36ca | Declan Moody vs Caleb Branson | oHb7bR0epUI | 5 |
| 65dbe62da2434704339da256 | Tommy Langaker vs Mateusz Flaga | BvAQDVP29E0 | 3 |
| 662ae0ce532ae8f232657f96 | Margot Ciccarelli vs Cara du Plessis | srZxKZ70eW0 | 6 |
| 664b79b5532ae8f2326a66b7 | William Tackett vs Jacob Rodriguez | vCyb33erhsc | 3 |
| 6665f04d4828f704f1b3afd8 | Pierre-Olivier Leclerc vs Jeff Childs | dsldlavgLz4 | 2 |

Per match, take the **first 3** hard-track events sorted by `t_s` ascending (2 for the last row, which only has 2) — **23 highlight-windows total**, same window cut as §A.4 (`[t_s−5, t_s+4]`).

### I.3 Reference image production — self-referential still-frame halves, disclosed as a proxy

For each match, extract ONE still frame at `reference_offset_s`:
```
first_window_start_s = the earliest of that match's 3 (or 2) selected highlight windows' window_start_s
reference_offset_s = 3.0 if first_window_start_s >= 8.0 else first_window_start_s / 2.0
```
(ensures the reference frame is always strictly before the earliest analyzed highlight, so the "reference" isn't drawn from inside a technique exchange). Extract via ffmpeg (`-ss reference_offset_s -frames:v 1`) from the same yt-dlp'd/normalized local file used for the match's clips (reuse §G.1/§H.2's download+normalize step — download a slightly wider local section covering `[0, last_selected_window_end_s]` once per match, not once per highlight).

Split the extracted frame into LEFT half and RIGHT half (full height, `x∈[0,W/2)` and `x∈[W/2,W)`), JPEG-encode each half — **this is a coarse deterministic proxy for a real per-athlete bounding-box crop, not a person-detector output** (no detector is run; flag this explicitly in `GATE4_REPORT.md` as a known simplification). Label them `player_id="ref_left"`/`"ref_right"` — **NOT the real athlete names from gold's `name` field**, because there is no verified mapping from "which half of frame at t=reference_offset_s" to "which name in the `A vs B` string" (the two athletes' left/right placement relative to camera is not derivable from the gold corpus without watching the frame, which this deterministic pipeline does not do). This is the honest resolution of the CEO's "how reference crops get produced" question: crops are cheap and deterministic to produce, but a real name-to-crop LABEL is not, without a human-verification step this design does not budget for. Build `player_references = [{"player_id": "ref_left", "player_name": "ref_left", "image_bytes": <jpeg bytes>, "mime_type": "image/jpeg"}, {"player_id": "ref_right", ...}]` — the exact shape `RunContext.player_references` uses in wt-s12 (`service/pipelines/executors.py:173`/`highlight_axes.build_actor_prompt`'s docstring).

### I.4 Call shape — the real S12 Phase-1b actor axis, with/without refs (INS-073 arm)

For each of the 23 highlight-windows, TWO calls, both against the identical window (native video, no reference images needed for the video part itself — only the actor call's inline reference `Part`s differ):
1. **WITH refs**: `highlight_axes.build_actor_schema(["ref_left", "ref_right"])` / `build_actor_prompt(description=None, player_references=[...both refs...])`, called via `_call_axis_json(..., extra_parts=[Part(inline_data=Blob(data=ref["image_bytes"], mime_type=ref["mime_type"])) for ref in refs])` — the exact wt-s12 `executors.py:1215-1223` construction, reused not reimplemented.
2. **WITHOUT refs**: same schema built with `player_choices=[]` (so the enum collapses to `AXIS2_ACTOR_SENTINELS` only — a structurally-forced "must pick a sentinel" call, exactly reproducing the INS-073 scenario), `build_actor_prompt(description=None, player_references=[])` (the module's own "no reference images are available" preamble fires automatically), `extra_parts=None`.

Settings: fps=10, `gemini-3.1-flash-lite`, media_resolution = `ActorAxisConfig`'s production default (`None`→LOW, mixin default, not separately justified here — Gate 4 is not testing resolution), thinking off. `description=None` throughout (Gate 4 bypasses scan, same as Gates 1–3; `build_actor_prompt`'s own "unknown — no description was provided" fallback handles this, identical convention to position/technique).

Cost: 23 windows × 2 calls = **46 calls**.

### I.5 shared_lib — 1.3.0 in wt-s12 is correct, not stale, no action needed

`AXIS2_ACTOR_SENTINELS` is imported from `shared_lib.models.simplified_taxonomy` (a compliant leaf import, per the project's shared_lib boundary rule) in both `service/pipelines/highlight_axes.py` and `executors.py` — this is the one place in Gates 1–4 that genuinely touches `shared_lib`. wt-s12's `1.3.0` is that worktree's own current, intentional version (it is a full production-wiring branch, ahead of the `1.2.0` snapshot referenced in the Phase-1a note, which describes an earlier point in wt-s12's own history — not a drift between two branches that are supposed to match). **No version objection stands.** Sanity-check before running: `cd whole-video-analysis-wt-s12 && .venv/bin/python -c "from service.pipelines.highlight_axes import build_actor_schema, build_actor_prompt; from shared_lib.models.simplified_taxonomy import AXIS2_ACTOR_SENTINELS; print(AXIS2_ACTOR_SENTINELS)"` — a cheap, mandatory smoke import before spending any of the 46 calls.

### I.6 Metrics

Per-window, per-condition (with/without refs), record: `actor` (raw value: `ref_left`/`ref_right`/sentinel), `identity_uncertain` (bool), `justification`.

- **Sentinel rate** (per condition): fraction of the 23 calls where `actor ∈ AXIS2_ACTOR_SENTINELS`. **INS-073 re-confirmation**: `sentinel_rate_without_refs` should be ≈100% by construction (no real choices exist in the enum without refs — this is a structural check, not really a "finding," included for completeness/contract-verification). The live comparison is `sentinel_rate_with_refs` vs. a genuinely-substantive baseline: if `sentinel_rate_with_refs` stays high (say >50%) even WITH two labeled reference crops in hand, that is itself informative (reference crops this coarse — half-frame splits, no verified name mapping — may not be visually distinctive enough to ground identity, a separate finding from "actor needs refs to resolve at all").
- **`identity_uncertain` rate** (with-refs condition only — meaningless without refs, since every without-refs call is forced to a sentinel already).
- **Cross-highlight flip rate** (with-refs condition only, per match): for a match with ≥2 non-sentinel `actor` calls, does the SAME `ref_left`/`ref_right` value get credited with the match's `successful` outcomes consistently, or does it flip between highlights in a way inconsistent with one athlete's narrative arc? Concretely: `flip_rate` = fraction of matches (of those with ≥2 non-sentinel highlights) where the attributed actor for `outcome=="successful"` events is NOT the same `ref_left`/`ref_right` value across all of them. **Caveat, stated plainly in the report**: a genuinely back-and-forth match (sweep exchanges going both directions) makes some flipping *correct*, not an error — this metric is a coarse instability signal, not a precision measurement, and must be reported with that caveat attached, never as a bare percentage.
- **"Accuracy"**: explicitly **not reported** as a headline number — there is no ground truth to measure it against (§I.1). `GATE4_REPORT.md` must say this in the same sentence as any accuracy-adjacent figure, not bury it in a footnote.

### I.7 Output contract

`gate4_predictions.json` (per-window, both conditions, raw+parsed), `gate4_scoring.json` (sentinel/uncertain/flip rates), `GATE4_REPORT.md` — mandatory per-match table (window_id, gold action_class/outcome, with-refs actor/identity_uncertain, without-refs actor), the three rates from §I.6, and a **prominent, unmissable statement** that no accuracy-vs-gold number exists or is claimed here, per §I.1.

### I.8 What this gate does NOT settle (say so, don't hide it)

This design gives the founder sentinel/uncertainty/flip-rate signals and an INS-073 re-confirmation — enough to judge "does the actor axis behave sanely and honestly when it has SOME visual anchor vs. none." It does **not** give a true per-player attribution accuracy number, because no real reference photographs or ground-truth identity labels exist in this project. If the founder wants that number, it requires a separate, later effort: real athlete reference photos (or human-verified name-to-crop labels) and either hand-labeled ground truth on a gold subset or a much larger cross-highlight consistency study — out of scope for this ~150-call budget and flagged here as a genuine, not-yet-closed gap, not solved by this design.
