# Gate 2 Report — validator media_resolution (LOW vs HIGH) on the 16-clip occlusion subset

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
| 65f7217fc98f78bbaa6af549 | submission_arm_lock | successful | standing | submission_arm_lock | agree | submission_arm_lock | True | agree | submission_arm_lock | True |
| 65dbe62da2434704339da256 | submission_arm_lock | successful | open_guard | sweep | agree | sweep | False | disagree | transition | False |
| 650d20c745120c8433e9d011 | submission_arm_lock | successful | open_guard | submission_leg_lock | agree | submission_leg_lock | False | agree | submission_leg_lock | False |
| 650d20c745120c8433e9d005 | submission_arm_lock | successful | half_guard | submission_leg_lock | agree | submission_leg_lock | False | agree | submission_leg_lock | False |
| 650d20c745120c8433e9d022 | submission_arm_lock | successful | closed_guard | submission_leg_lock | agree | submission_leg_lock | False | disagree | sweep | False |
| 650d20c745120c8433e9cfd4 | submission_arm_lock | successful | open_guard | submission_leg_lock | agree | submission_leg_lock | False | agree | submission_leg_lock | False |
| 65e3a69da2434704339e4460 | submission_leg_lock | unsuccessful | open_guard | submission_arm_lock | agree | submission_arm_lock | False | agree | submission_arm_lock | False |
| 65f648c51dcf397e6900a827 | submission_leg_lock | unsuccessful | open_guard | submission_leg_lock | agree | submission_leg_lock | True | agree | submission_leg_lock | True |
| 6643b593532ae8f232692d70 | submission_leg_lock | unsuccessful | open_guard | submission_leg_lock | agree | submission_leg_lock | True | agree | submission_leg_lock | True |
| 66a6ff021c62a1db7a5133c0 | submission_leg_lock | unsuccessful | half_guard | submission_leg_lock | agree | submission_leg_lock | True | agree | submission_leg_lock | True |
| 66d4e7091c7649ed30cbef2b | submission_leg_lock | unsuccessful | closed_guard | submission_choke | disagree | submission_choke | False | disagree | submission_choke | False |
| 652c0e04a947037aeb7eb302 | submission_leg_lock | successful | open_guard | submission_leg_lock | agree | submission_leg_lock | True | agree | submission_leg_lock | True |
| 65dfb6dba2434704339de920 | submission_leg_lock | successful | open_guard | submission_leg_lock | agree | submission_leg_lock | True | agree | submission_leg_lock | True |
| 66941978a7bafe01741514db | submission_leg_lock | successful | open_guard | submission_leg_lock | agree | submission_leg_lock | True | agree | submission_leg_lock | True |

N scored: 16/16 (excluded call_failed/unrun: []).

## Verdict

accuracy_low=50.0% (8/16), accuracy_high=50.0% (8/16), net_delta=0 (pass threshold: >= 2) -> **NO_GO (directional, n=16)**

**Directional, small-n caveat:** this gate's n=16-by-design sample is deliberately tiny (design §C.3) — a `net_delta` swing of 1-2 clips is the resolution limit at this n. The PASS/NO-GO call above is binary per the design's own threshold, but should be read as directional evidence, not a high-confidence estimate; a larger occlusion-subset re-run is reasonable post-launch tuning, not a cutover blocker.

## §G.1 point 4 dual reporting — headline recomputed excluding files_api-ingested clips

Unlike Gate 1 (whose sole reroute, `w37`, is a straddler already outside the headline), Gate 2's two reroutes (`occ0`/`occ11`) are BOTH inside the main 16-clip pool AND are both forced `unsuccessful`-outcome slots (§A.3's oversampled occluded population) — exclusion is not a structural no-op here.

files_api-ingested clip(s) in this run: `['66a6ff021c62a1db7a5133c0__occ11', '66afe5ab1c62a1db7a53b80b__occ0']`.

EXCLUDING files_api clips (n_clips=14, n_scored=14): accuracy_low=42.9%, accuracy_high=42.9%, net_delta=0 -> **verdict: NO_GO (directional, n=14)**

**VERDICT SENSITIVITY: 2 files_api clip(s) excluded (of 16) -> verdict 'NO_GO' (full) vs 'NO_GO' (excl. files_api). Verdict does NOT flip on exclusion.**
