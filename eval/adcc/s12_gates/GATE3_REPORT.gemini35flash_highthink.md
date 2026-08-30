# Gate 3 Report — Files-API ingestion-path fidelity (Arm-S primary, per VERDICTS.md §4 amendment)

**§G.2 REPLACEMENT IN THIS RUN — NOT a silent substitution.** `650d20c745120c8433e9d059__w17` was replaced by `6661e503716f12758093d038__w1`. Reason: w17 (650d20c745120c8433e9d059__w17, guard_pass, video FO4jG04aGqc) was dropped from the §H.1 draw: its source video is YouTube-unavailable ('ERROR: [youtube] FO4jG04aGqc: Video unavailable. It was blocked due to the claimed content by SME.', yt-dlp-reproduced 2026-07-26) — an external, permanent failure, not a Files-API/Gemini issue. Replaced per GATES_1_2_DESIGN.md §G.2's procedure: a FRESH random.Random(gate_common.SEED_REPLACEMENT=20260727) instance sampling 1 window from the remaining 29-window eligible pool (the original 37-window §H.1-eligible set minus the 8 already drawn) selected 6661e503716f12758093d038__w1 (back_take, video dsldlavgLz4). CEO-authorized (option a), Bezos-reviewed, 2026-07-26 — never a silent substitution.

## Upload metrics (upload_latency_ms / active_wait_ms per window)

| window | file_name | upload_latency_ms | active_wait_ms | sec_start_s | is_replacement |
|---|---|---|---|---|---|
| 650d20c745120c8433e9cff3__w32 | files/3zul1u9c24vp | 0 | 0 | 754.0 | False |
| 650d20c745120c8433e9d00e__w6 | files/n2z42v8zfbu3 | 0 | 0 | 135.0 | False |
| 650d20c745120c8433e9d024__w0 | files/odfxkaaqwbac | 0 | 0 | 12617.0 | False |
| 650d20c745120c8433e9d031__w27 | files/vypvf6ljn958 | 0 | 0 | 247.0 | False |
| 653715dad9868c25fd158a0b__w20 | files/e1l711xajv9d | 0 | 0 | 98.0 | False |
| 65dbd239a2434704339da0e4__w8 | files/463sue0tgtwb | 0 | 0 | 272.0 | False |
| 65de3657a2434704339dc36b__w12 | files/r8pch0b5uxwx | 0 | 0 | 4891.0 | False |
| 6661e503716f12758093d038__w1 | files/ydlthd6jgv4w | 0 | 0 | 2845.0 | True |

## Settings tuple

| | fps | media_resolution | model | thinking | ingestion (Files-API side) |
|---|---|---|---|---|---|
| Arm S (primary) | 10 | None (LOW) | gemini-3.5-flash | off | files_api (mime_type+file-relative offsets) |
| Arm T (spot-check) position/technique | 10 | None (LOW) | gemini-3.5-flash | off | files_api |
| Arm T (spot-check) validator | 10 | high (prod default) | gemini-3.5-flash | off | files_api |
| fps-honoring (technique only) | 1 and 10 | None (LOW) | gemini-3.5-flash | off | files_api |

## PRIMARY — Arm S, YouTube-URI (Gate 1, already-captured) vs Files-API (new)

| window | gold_class | youtube_final_class | youtube_hit | files_api_final_class | files_api_hit | agree |
|---|---|---|---|---|---|---|
| 650d20c745120c8433e9cff3__w32 | back_take | back_take | True | back_take | True | True |
| 650d20c745120c8433e9d00e__w6 | back_take | back_take | True | submission_choke | False | False |
| 650d20c745120c8433e9d024__w0 | back_take | submission_choke | False | submission_choke | False | True |
| 650d20c745120c8433e9d031__w27 | sweep | guard_pass | False | submission_leg_lock | False | False |
| 653715dad9868c25fd158a0b__w20 | submission_choke | submission_choke | True | submission_choke | True | True |
| 65dbd239a2434704339da0e4__w8 | guard_pass | submission_leg_lock | False | guard_pass | True | False |
| 65de3657a2434704339dc36b__w12 | guard_pass | submission_choke | False | guard_pass | True | False |
| 6661e503716f12758093d038__w1 **[§G.2 REPLACEMENT]** | back_take | guard_pass | False | guard_pass | False | True |

N scored: 8/8. agreement_rate=50.0% (4/8), accuracy_youtube=37.5% (3), accuracy_files_api=50.0% (4), accuracy_delta=12.5%.

**Design's own numeric PASS bar (§H.5, reported mechanically — NOT a rendered gate verdict, that is LeCun's call):** condition (a) agreement_rate>=75%: False; condition (b) accuracy_files_api >= accuracy_youtube-1: True; both required: **False**.

## SECONDARY — Arm T spot-check (n=3, reduced per VERDICTS.md §4, insurance only)

| window | gold_class | youtube_final_class | youtube_hit | files_api_final_class | files_api_hit | agree |
|---|---|---|---|---|---|---|
| 650d20c745120c8433e9cff3__w32 | back_take | back_take | True | back_take | True | True |
| 650d20c745120c8433e9d00e__w6 | back_take | submission_choke | False | back_take | True | False |
| 650d20c745120c8433e9d024__w0 | back_take | submission_choke | False | submission_choke | False | True |

N scored: 3/3, agreement: 2/3. NOT part of the n=8 fidelity sample — insurance-only secondary check.

## §H.3 — INS-096 fps-honoring re-confirmation on the Files-API path

**Operationalization note (design says 'compare the two responses'; this scorer's precise rule, flagged):** `detectable_diff_strong` = the two fps' `action_class` differ (a real judgment change — the strong, primary signal). `detectable_diff_weak` = only the `justification` text differs (action_class the same) — reported as supplementary color, NOT treated as dispositive on its own, since `temperature=0.2` calls can produce different wording by sampling noise alone even with zero real fps effect; using text-difference as the primary bar would make this check trivially always pass.

| window | fps1_action_class | fps10_action_class | detectable_diff (action_class) | detectable_diff (justification text) |
|---|---|---|---|---|
| 650d20c745120c8433e9cff3__w32 | back_take | back_take | False | True |
| 650d20c745120c8433e9d00e__w6 | back_take | back_take | False | True |

0/2 clips show a strong (action_class) detectable difference. Design's confirmation bar (>=1 of 2): **NOT CONFIRMED**.

## §H.4 — Bonus finding (zero incremental cost, NOT part of the n=8 sample)

Bonus finding (§H.4) — NOT part of the n=8 fidelity sample. For w37/occ0/occ11, Files-API succeeded where native YouTube ingestion 400'd unconditionally on every probed offset/fps. Evidence that Files-API is a viable fallback/workaround for the undocumented duration-cap bug — a claim about VIABILITY, not FIDELITY; not blended into the n=8 agreement_rate/accuracy numbers above.
Windows: `['66b009481c62a1db7a53d504__w37', '66afe5ab1c62a1db7a53b80b__occ0', '66a6ff021c62a1db7a5133c0__occ11']`. Source artifacts: `['gate1_predictions_arm_s.json', 'gate1_predictions_arm_t.json', 'gate2_predictions.json', 'reroute_manifest.json']`.
