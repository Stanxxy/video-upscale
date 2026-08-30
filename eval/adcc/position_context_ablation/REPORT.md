# ADCC oracle-position context ablation

This bounded paired experiment tests whether supplying a coarse, oracle-style ADCC position bucket helps Gemini 3.7 Flash classify a technique in an already-located fixed window. It does **not** test Position Scan, candidate recall, player identity, or VTG boundaries.

## Result

- Paired successful windows: 12/12 (failed pairs: 0)
- Baseline exact action: 6/12
- Position-context exact action: 7/12
- Net exact-action delta: +1
- Discordant pairs (context-only / baseline-only): 1 / 0
- Two-sided paired sign-test p: 1.0
- Measured calls: 24; estimated Standard paid-tier spend: $0.217584

## Per-window classifications

| Window | Supplied position | Gold action | Baseline | Position context |
|---|---|---|---|---|
| 650d20c745120c8433e9d024__w0 | turtle | back_take | guard_pass | submission_choke |
| 6661e503716f12758093d038__w1 | mount | back_take | guard_pass | guard_pass |
| 66138fe05da956d5ef54f21b__w7 | open_guard | guard_pass | guard_pass | guard_pass |
| 65dbd239a2434704339da0e4__w8 | side_control | guard_pass | guard_pass | guard_pass |
| 650d20c745120c8433e9d00a__w18 | half_guard | submission_arm_lock | submission_arm_lock | submission_arm_lock |
| 6470211a3ec8dca9d1a32710__w30 | open_guard | submission_arm_lock | takedown_attempt | submission_choke |
| 650d20c745120c8433e9d01b__w19 | back_control | submission_choke | submission_choke | submission_choke |
| 653715dad9868c25fd158a0b__w20 | back_control | submission_choke | transition | submission_choke |
| 6666381f4828f704f1b3bead__w23 | open_guard | submission_leg_lock | submission_leg_lock | submission_leg_lock |
| 652c0e04a947037aeb7eb302__w24 | open_guard | submission_leg_lock | submission_leg_lock | submission_leg_lock |
| 650d20c745120c8433e9d004__w26 | open_guard | sweep | transition | submission_leg_lock |
| 650d20c745120c8433e9d031__w27 | open_guard | sweep | guard_pass | submission_leg_lock |

## Interpretation boundary

- coarse position_bucket is supplied as oracle-style context, not predicted by Position Scan
- fixed nine-second windows test classification only, not discovery recall or VTG boundaries
- ADCC artifact has no two-athlete reference crops, so actor identity is not scored
- single call per condition leaves ordinary model sampling variance unresolved
