# Gate 4 Report — Attribution (actor axis + self-referential reference-image crops)

**NO ACCURACY-VS-GOLD NUMBER EXISTS OR IS CLAIMED ANYWHERE IN THIS REPORT (§I.1).** There is no player-identity ground truth anywhere in the gold corpus — every metric below is a self-consistency/honesty-channel signal, not a correctness measurement. Reference crops are SELF-REFERENTIAL left/right half-frame stills from the SAME match's own video, a disclosed coarse proxy for a real per-athlete bounding-box crop — NOT a person-detector output, and NOT verified-mapped to the real athlete names in gold's `name` field (labeled `ref_left`/`ref_right` only).

## Settings tuple

fps=10, model=gemini-3.5-flash, media_resolution=None (LOW, ActorAxisConfig default), thinking=off, description=None throughout.

## Per-match table

| window | gold_class | gold_outcome | with_refs actor | with_refs identity_uncertain | without_refs actor |
|---|---|---|---|---|---|
| 650d20c745120c8433e9d018__g4_0 | guard_pass | successful | ref_right | False | unclear |
| 650d20c745120c8433e9d018__g4_1 | back_take | successful | ref_left | False | unclear |
| 650d20c745120c8433e9d018__g4_2 | submission_choke | successful | ref_left | False | unclear |
| 650d20c745120c8433e9d027__g4_0 | sweep | successful | ref_left | False | unclear |
| 650d20c745120c8433e9d027__g4_1 | sweep | successful | ref_left | False | unclear |
| 650d20c745120c8433e9d027__g4_2 | guard_pass | successful | ref_left | False | unclear |
| 65396bbbb7741492b97777fc__g4_0 | guard_pass | successful | ref_left | False | unclear |
| 65396bbbb7741492b97777fc__g4_1 | guard_pass | successful | ref_left | False | unclear |
| 65396bbbb7741492b97777fc__g4_2 | guard_pass | successful | ref_left | False | unclear |
| 6567972aa4a13c99c4fe36ca__g4_0 | guard_pass | successful | ref_right | False | unclear |
| 6567972aa4a13c99c4fe36ca__g4_1 | guard_pass | successful | ref_left | False | unclear |
| 6567972aa4a13c99c4fe36ca__g4_2 | guard_pass | successful | ref_right | False | unclear |
| 65dbe62da2434704339da256__g4_0 | sweep | successful | ref_left | False | unclear |
| 65dbe62da2434704339da256__g4_1 | back_take | successful | ref_left | False | unclear |
| 65dbe62da2434704339da256__g4_2 | submission_arm_lock | successful | ref_left | False | unclear |
| 662ae0ce532ae8f232657f96__g4_0 | sweep | successful | ref_right | False | unclear |
| 662ae0ce532ae8f232657f96__g4_1 | guard_pass | successful | ref_right | False | unclear |
| 662ae0ce532ae8f232657f96__g4_2 | back_take | successful | ref_right | False | unclear |
| 664b79b5532ae8f2326a66b7__g4_0 | guard_pass | successful | ref_right | False | unclear |
| 664b79b5532ae8f2326a66b7__g4_1 | back_take | successful | ref_left | False | unclear |
| 664b79b5532ae8f2326a66b7__g4_2 | submission_leg_lock | successful | ref_left | False | unclear |
| 6665f04d4828f704f1b3afd8__g4_0 | sweep | successful | ref_left | False | unclear |
| 6665f04d4828f704f1b3afd8__g4_1 | submission_leg_lock | successful | ref_left | False | unclear |

N scored: 23/23 (excluded call_failed: []).

## Rates

**sentinel_rate_with_refs** = 0.0% (0/23). INS-073 re-confirmation: **sentinel_rate_without_refs** = 100.0% (23/23) — expected ~100% by construction (no real choices exist in the enum without refs); a structural check, not a finding.

**identity_uncertain_rate** (with-refs only) = 0.0% (0/23).

**cross_highlight_flip_rate** = 37.5% (3/8 eligible matches — see module docstring for the exact eligibility rule: >=2 `outcome==successful` windows that both resolved to a non-sentinel actor). **Caveat, stated plainly (design §I.6): a genuinely back-and-forth match makes SOME flipping correct, not an error — this is a coarse instability signal, not a precision measurement.**

### Per-match flip detail

| match_id | n_successful_non_sentinel | actors | eligible | flipped |
|---|---|---|---|---|
| 650d20c745120c8433e9d018 | 3 | ['ref_right', 'ref_left', 'ref_left'] | True | True |
| 650d20c745120c8433e9d027 | 3 | ['ref_left', 'ref_left', 'ref_left'] | True | False |
| 65396bbbb7741492b97777fc | 3 | ['ref_left', 'ref_left', 'ref_left'] | True | False |
| 6567972aa4a13c99c4fe36ca | 3 | ['ref_right', 'ref_left', 'ref_right'] | True | True |
| 65dbe62da2434704339da256 | 3 | ['ref_left', 'ref_left', 'ref_left'] | True | False |
| 662ae0ce532ae8f232657f96 | 3 | ['ref_right', 'ref_right', 'ref_right'] | True | False |
| 664b79b5532ae8f2326a66b7 | 3 | ['ref_right', 'ref_left', 'ref_left'] | True | True |
| 6665f04d4828f704f1b3afd8 | 2 | ['ref_left', 'ref_left'] | True | False |

## What this gate does NOT settle (§I.8)

This design gives sentinel/uncertainty/flip-rate signals and an INS-073 re-confirmation — enough to judge whether the actor axis behaves sanely and honestly when it has SOME visual anchor vs. none. It does NOT give a true per-player attribution accuracy number. That would require real athlete reference photos (or human-verified name-to-crop labels) and hand-labeled ground truth — out of scope here, a genuine not-yet-closed gap, not solved by this design.
