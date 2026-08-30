# Gate 2 Report — validator media_resolution (LOW vs HIGH) on the 16-clip occlusion subset

**RUN LABEL: `smoke`** — n_scored=2 of 16 manifest clips. This is NOT the full 16-clip batch; the verdict below demonstrates the scoring PIPELINE only, not a real gate decision.

## Settings tuple (design §D)

| | fps | media_resolution | model | thinking |
|---|---|---|---|---|
| position/technique | 10 | None (LOW) | gemini-3.1-flash-lite | off |
| validator-low | 10 | low | gemini-3.1-flash-lite | off |
| validator-high | 10 | high | gemini-3.1-flash-lite | off |

## Per-clip table (§C.2, mandatory — no bare aggregate)

| match | gold_class | gold_outcome | pos_label | tech_label | val_low_verdict | val_low_final_class | correct_low | val_high_verdict | val_high_final_class | correct_high |
|---|---|---|---|---|---|---|---|---|---|---|
| 66afe5ab1c62a1db7a53b80b | submission_arm_lock | unsuccessful | open_guard | submission_arm_lock | agree | submission_arm_lock | True | agree | submission_arm_lock | True |
| 650d20c745120c8433e9cfdb | submission_arm_lock | successful | half_guard | submission_leg_lock | agree | submission_leg_lock | False | agree | submission_leg_lock | False |
| 65f7217fc98f78bbaa6af549 | submission_arm_lock | successful | _(call_failed / not run)_ | | | | | | | |
| 65dbe62da2434704339da256 | submission_arm_lock | successful | _(call_failed / not run)_ | | | | | | | |
| 650d20c745120c8433e9d011 | submission_arm_lock | successful | _(call_failed / not run)_ | | | | | | | |
| 650d20c745120c8433e9d005 | submission_arm_lock | successful | _(call_failed / not run)_ | | | | | | | |
| 650d20c745120c8433e9d022 | submission_arm_lock | successful | _(call_failed / not run)_ | | | | | | | |
| 650d20c745120c8433e9cfd4 | submission_arm_lock | successful | _(call_failed / not run)_ | | | | | | | |
| 65e3a69da2434704339e4460 | submission_leg_lock | unsuccessful | _(call_failed / not run)_ | | | | | | | |
| 65f648c51dcf397e6900a827 | submission_leg_lock | unsuccessful | _(call_failed / not run)_ | | | | | | | |
| 6643b593532ae8f232692d70 | submission_leg_lock | unsuccessful | _(call_failed / not run)_ | | | | | | | |
| 66a6ff021c62a1db7a5133c0 | submission_leg_lock | unsuccessful | _(call_failed / not run)_ | | | | | | | |
| 66d4e7091c7649ed30cbef2b | submission_leg_lock | unsuccessful | _(call_failed / not run)_ | | | | | | | |
| 652c0e04a947037aeb7eb302 | submission_leg_lock | successful | _(call_failed / not run)_ | | | | | | | |
| 65dfb6dba2434704339de920 | submission_leg_lock | successful | _(call_failed / not run)_ | | | | | | | |
| 66941978a7bafe01741514db | submission_leg_lock | successful | _(call_failed / not run)_ | | | | | | | |

N scored: 2/16 (excluded call_failed/unrun: ['65f7217fc98f78bbaa6af549__occ2', '65dbe62da2434704339da256__occ3', '650d20c745120c8433e9d011__occ4', '650d20c745120c8433e9d005__occ5', '650d20c745120c8433e9d022__occ6', '650d20c745120c8433e9cfd4__occ7', '65e3a69da2434704339e4460__occ8', '65f648c51dcf397e6900a827__occ9', '6643b593532ae8f232692d70__occ10', '66a6ff021c62a1db7a5133c0__occ11', '66d4e7091c7649ed30cbef2b__occ12', '652c0e04a947037aeb7eb302__occ13', '65dfb6dba2434704339de920__occ14', '66941978a7bafe01741514db__occ15']).

## Verdict

accuracy_low=50.0% (1/2), accuracy_high=50.0% (1/2), net_delta=0 (pass threshold: >= 2) -> **NO_GO (directional, n=2)**

**Directional, small-n caveat:** this gate's n=16-by-design sample is deliberately tiny (design §C.3) — a `net_delta` swing of 1-2 clips is the resolution limit at this n. The PASS/NO-GO call above is binary per the design's own threshold, but should be read as directional evidence, not a high-confidence estimate; a larger occlusion-subset re-run is reasonable post-launch tuning, not a cutover blocker.

## §G.1 point 4 dual reporting — headline recomputed excluding files_api-ingested clips

Unlike Gate 1 (whose sole reroute, `w37`, is a straddler already outside the headline), Gate 2's two reroutes (`occ0`/`occ11`) are BOTH inside the main 16-clip pool AND are both forced `unsuccessful`-outcome slots (§A.3's oversampled occluded population) — exclusion is not a structural no-op here.

files_api-ingested clip(s) in this run: `['66afe5ab1c62a1db7a53b80b__occ0']`.

EXCLUDING files_api clips (n_clips=15, n_scored=1): accuracy_low=0.0%, accuracy_high=0.0%, net_delta=0 -> **verdict: NO_GO (directional, n=1)**

**VERDICT SENSITIVITY: 1 files_api clip(s) excluded (of 16) -> verdict 'NO_GO' (full) vs 'NO_GO' (excl. files_api). Verdict does NOT flip on exclusion.**
