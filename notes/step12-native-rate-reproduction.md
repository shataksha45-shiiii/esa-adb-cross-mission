# Step 12: official baseline reproduced at the native rate; the grid explains the gap (roadmap item 2) (9 Nov 2026)

## Why
In Step 11, the official GlobalSTD rule re-run in our pipeline scored **0.400** on Mission 2, against **0.064** published by ESA-ADB. Two possible reasons:
- something in our pipeline is wrong (metric, labels, split);
- only our 5-minute grid differs.

We had to know which before trusting any number.

## What we did
`src/repro_globalstd_native.py`:
- loads Mission 2's 47 raw target channels;
- applies **ESA-ADB's own preprocessing**: zero-order hold onto an **18-second grid** (copied from their `Mission2_semiunsupervised_prep_from_raw.py`), and the same train/test split;
- runs the official GlobalSTD rule;
- repeats on 1-minute and 5-minute grids;
- scores everything with the **official ESA-ADB metric**.

Output: `results/repro_globalstd_native_M2.csv` (+ `_log.txt`).

## Result
| Grid | tol | **Ours (official metric)** | Published (ESA-ADB v2, Table 2) | Alarm time | Separate alarm events | FP events |
|---|---|---|---|---|---|---|
| **18 s (native)** | 3 | **0.0075** | 0.008 | 9.0% | 61,840 | 55,923 |
| **18 s (native)** | 5 | **0.0638** | 0.064 | 3.1% | 11,090 | 6,552 |
| 1 min | 3 | 0.0163 | — | 9.1% | 27,803 | 25,504 |
| 1 min | 5 | 0.1283 | — | 3.1% | 4,770 | 3,016 |
| 5 min (ours) | 3 | 0.0619 | — | 9.2% | 7,070 | 6,329 |
| 5 min (ours) | 5 | 0.3997 | — | 3.1% | 1,254 | 660 |

**Verdict: reproduced.** At the native 18-second grid, our numbers match the published ones to the third decimal: 0.0638 vs 0.064, and 0.0075 vs 0.008. So our evaluation pipeline (metric, labels, test definition, train split) agrees with ESA-ADB's. **The whole gap comes from the grid.**

## Why the grid changes the score so much (plain words)
- The **alarm time is the same on every grid**: 3.1% for tol 5. The detector flags the same stretches of time.
- What changes is **how many separate alarm events** that time is cut into. At 18 s, a noisy channel flickers in and out of the band and creates 11,090 separate alarms. At 5 min most flickers are never sampled, leaving 1,254.
- The corrected event-wise F0.5 counts **events**: every separate alarm that touches no labelled event is one false positive. So more fragments mean more false positives, lower precision and a lower F0.5. Recall barely changes (0.99 → 0.98).

**A finding about the metric:** corrected event-wise F0.5 depends strongly on the sampling grid. The same detector moves from 0.064 to 0.400 with no change in what it detects. Any comparison must use the same grid.

## What this means for us
1. **Our pipeline is validated.** It reproduces an official baseline exactly at the native rate.
2. **Our absolute numbers (5-min grid) are inflated relative to published native-rate numbers.** Never compare our 0.846 with ESA-ADB's 0.079 or PitIA's 0.882.
3. **Fair comparisons are within one grid.** All our cross-mission comparisons (Steps 6–10) are, so they stand.
4. **Our method's event builder** (gap merging and min_len) also merges fragments, as the coarse grid does. To claim anything against published methods, our method would have to be run at 18 s. That is possible future work: the AE on 3 million points × 47 channels is heavy but feasible.
5. For the paper: report the grid sensitivity as a methodological note. It also helps other users of ESA-ADB.

## What broke
Nothing. One flagged difference from ESA-ADB's preprocessing: we skip their "restore annotated samples" step, and we use one common grid start for all channels. Both matter little here; the numbers match to 3 decimals.

## Key terms
- **Zero-order hold (ZOH):** the value at a grid time is the last raw sample at or before it.
- **Native rate:** the grid ESA-ADB uses (18 s for M2, 30 s for M1).
- **Alarm fragmentation:** one long alarm broken into many short ones; each piece outside a true event counts as a separate false positive.

## Self-test
1. The alarm time is 3.1% on every grid. Why does F0.5 still change 6×?
2. Why does this result make us *more* confident in our cross-mission findings, not less?
3. What would we need to do to compare our method with PitIA fairly?
