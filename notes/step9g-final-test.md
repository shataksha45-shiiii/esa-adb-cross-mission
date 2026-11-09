# Step 9g: the one-time test of the label-free method (v3) (9 Nov 2026)

Code: `src/run_step9.py --test`. Output: `results/final_step9.csv`, `results/final_step9_log.txt`.
Settings: v3 (replayed real source events + the v2 rule), chosen per pair with **no target labels**.
5 seeds, test labels as defined by the official ESA-ADB code (events overlapping the test period).
Our F0.5 and the official ESA-ADB code agree to within 1e-17.
**Test labels have now been used 3 times (Steps 7, 8, 9). This is the final Step 9 number: no changes based on it.**

## Result table (event-wise F0.5, mean ± std over 5 seeds, official metric)

| Pair | Step 8: source-tuned settings | **Step 9: label-free (v3)** | Step 8: target-tuned settings (uses target labels) |
|---|---|---|---|
| **M1→M2** | 0.087 ± 0.013 | **0.156 ± 0.036** | 0.846 (M2→M2) |
| **M2→M1** | 0.066 ± 0.002 | **0.081 ± 0.003** | 0.481 (M1→M1) |
| M1→M1, label-free | — | 0.096 ± 0.005 | 0.481 |
| M2→M2, label-free | — | 0.139 ± 0.036 | 0.846 |

Other official metrics (mean over seeds, label-free v3):

| Pair | Alarming precision | EW precision | EW recall | Affiliation F0.5 |
|---|---|---|---|---|
| M1→M2 | 0.989 | 0.987 | 0.036 | 0.145 |
| M2→M1 | 0.717 | 0.071 | 0.200 | 0.415 |
| M1→M1 | 0.762 | 0.085 | 0.204 | 0.424 |
| M2→M2 | 1.000 | 0.974 | 0.031 | 0.130 |

## What it means (plain words)
- **Both transfer directions improve over the old transfer**: M1→M2 nearly doubles (0.087 → 0.156), and M2→M1 rises slightly (0.066 → 0.081). The improvement is real but **small**.
- **The dev check was far too optimistic for M1 as the target** (dev 0.507 vs test 0.081).
- On M2 as the target: very precise (almost no false alarms) but recall is about 3%. Still too strict, as the dev check showed.

## Why M1 collapsed on test (checked WITHOUT labels)
With the M2→M1 settings on the M1 test period (seed-0 model):

| Part | Alarm events | Alarm time |
|---|---|---|
| AE only | 12 | 0.1% |
| \|z\| only | **183** | **11.4%** |
| Both (as tested) | 234 | 11.6% |

Share of (time, channel) with |z| > 5: **train 0.10%, test 1.03%**, which is 10× more in the test years.

- **Level shifts in the test years:** M1's test period covers years 7–14 of the mission. The monthly plot (`figures/fig4_drift.png`) shows it is **not** a slow, smooth drift but **episodes**: a burst in mid-2009 and a sustained shift from mid-2013 (about 5% of channel-samples beyond |z| > 5 every month). The level-based |z| detector fires through all of them.
- The autoencoder looks at **centered** windows (shape, not level), so it is barely affected.
- The dev check couldn't see this, because both CV halves come from the training years, before most of the drift.
- The label-free method switched |z| on for M1 targets (q2 = 1e-6). The label-tuned M1 settings in Step 8 had it off, which is why Step 8's M1→M1 survived the test.

## Lessons (for the paper's limitations and for future work)
1. Label-free calibration on the training period can't see **future level shifts / regime changes**. A level detector is fragile over years of mission life.
2. The fakes are still easier than real events, and on M2 that leaves the method too strict (recall about 3%).
3. A dev score from CV inside the training period overestimates the test score when the mission drifts.

## Rules from here
- No changes to Step 9 because of these test numbers.
- Any future method (for example a drift guard that switches |z| off when the level statistics shift) must be designed on training data only. And since test labels have now been used 3 times, any new test number must be reported as **post-hoc** in the paper.

## Self-test
1. Why did the dev check overestimate M1 so badly?
2. Why is the AE robust to slow drift while |z| isn't?
3. How did we find the cause without using test labels?
