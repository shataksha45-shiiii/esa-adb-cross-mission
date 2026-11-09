# Step 13: ablations, confidence intervals, sensitivity (roadmap items 3–5) (9 Nov 2026)

Protocol frozen before any number: `notes/step13-protocol-frozen.md`. Code: `src/step13.py`; `src/inject.py --roll-win/--amp-floor/--out`.
Results:
- `results/step13_points.csv` (every method × pair × seed);
- `results/step13_ci.csv` (bootstrap CIs);
- `results/step13_repeats_choice.csv`;
- new injection runs in `results/inject_v5_win15/`, `results/inject_v5_win60/`, `results/inject_v3_floor02/`.

**All numbers are post-hoc (5th use of the test labels). Analysis only: the selected method stays C3, and nothing was changed because of these numbers.**

## Check first
The per-month split of the score reproduces our evaluator exactly (difference 0.0 on 3 methods; `--check`). Our evaluator matches the official ESA-ADB code to 1e-17, so the bootstrap uses the true metric.

## Item 4: confidence intervals
Method: block bootstrap over test **months** (2,000 resamples, mean over 5 seeds). 95% CI in brackets.

| Method | M1→M2 | M2→M1 | M1→M1 | M2→M2 |
|---|---|---|---|---|
| B0 target-label-tuned (upper bound) | 0.849 [0.805, 0.880] | 0.278 [0.145, 0.397] | 0.481 [0.350, 0.587] | 0.846 [0.802, 0.879] |
| B1 source-label-tuned | 0.087 [0.027, 0.144] | 0.066 [0.034, 0.152] | — | — |
| B2 fixed default | 0.892 [0.861, 0.922] | 0.040 [0.023, 0.074] | 0.048 | 0.890 |
| B3 Heydari static | 0.934 [0.913, 0.953] | 0.007 [0.002, 0.012] | 0.084 | 0.965 |
| C1 v3 (Step 9) | 0.156 [0.070, 0.236] | 0.081 [0.037, 0.225] | 0.096 | 0.139 |
| **C3 fake-recall (selected)** | **0.386 [0.283, 0.469]** | 0.081 [0.037, 0.225] | 0.089 | 0.778 |
| C5 + rolling \|z\| (post-hoc) | 0.386 [0.283, 0.469] | 0.261 [0.122, 0.419] | 0.344 | 0.868 |

Paired differences (same month resamples). "Clear" = the 95% CI excludes 0.

| Comparison | M1→M2 | M2→M1 |
|---|---|---|
| C3 − C1 | **+0.230 [+0.153, +0.315] clear** | 0.000 (same setting) |
| C3 − B1 | **+0.298 [+0.215, +0.379] clear** | +0.015 [−0.068, +0.150] not clear |
| C5 − C1 | **+0.230 clear** | **+0.180 [+0.002, +0.330] clear** (barely) |
| C5 − B3 (Heydari) | −0.549 clear (Heydari better) | **+0.255 [+0.113, +0.413] clear** (C5 better) |

What it means:
- **C3's M1→M2 gain is real, not noise.**
- C3 does nothing for M2→M1.
- C5 beats C1 in all four pairs, but the M2→M1 margin barely clears 0.
- **"No method wins both directions" holds with CIs:** Heydari is clearly better with M2 as the target, C5 clearly better with M1 as the target.
- M1-target CIs are wide (e.g. 0.12–0.42). M1 has few events per month, so any M1-target number is uncertain.

## Item 3: ablations (remove one part at a time; test F0.5, mean of 5 seeds)
| Variant | C3: M1→M2 | C3: M2→M1 | C3: M2→M2 | C5: M1→M2 | C5: M2→M1 | C5: M2→M2 |
|---|---|---|---|---|---|---|
| Full method | 0.386 | 0.081 | 0.778 | 0.386 | 0.261 | 0.868 |
| \|z\| off (AE only) | 0.386 | 0.144 | 0.308 | 0.386 | 0.178 | 0.027 |
| AE off (\|z\| only) | **0.000** | 0.092 | 0.766 | **0.000** | 0.238 | 0.868 |
| No event builder | **0.190** | **0.028** | **0.467** | **0.190** | 0.220 | 0.902 |
| k-of-n = 1 | 0.386 | 0.090 | 0.824 | 0.386 | 0.154 | 0.868 |
| Quantile instead of EVT | 0.357 | 0.141 | 0.292 | 0.357 | 0.350 | 0.235 |

What each part does:
1. **The event builder (gap-merging + min_len) is the most important part.** Removing it hurts in 5 of 6 cells (e.g. 0.386 → 0.190). This is the same effect as Step 12: merging fragmented alarms removes many false-positive events.
2. **The AE is essential for transfer to M2.** Without it, M1→M2 drops to 0. The M1-trained model is what transfers, which supports the paper's main claim.
3. **The \|z\| level detector is essential when M2 is the target** (C5 M2→M2: 0.868 → 0.027 without it). Static \|z\| **hurts** when M1 is the target (C3 M2→M1: turning it off raises 0.081 to 0.144) because of the M1 level shifts. Rolling \|z\| (C5) is what lets it help on M1 too.
4. **EVT vs a plain quantile:** EVT matters when the chosen tail probability q is tiny (1e-7, 1e-8). A plain training quantile can't go beyond the largest training score, so the detector almost never fires. That is why M2→M2 drops to 0.29 / 0.24. With M1 as the target, the plain quantile happened to do better (0.141, 0.350). **EVT isn't uniformly better**; it mainly allows extreme thresholds.
5. **k-of-n voting** matters little here; k = 1 is slightly better on M2→M2.

Calibration ladder (same model, test):

| Rule | M1→M2 | M2→M1 |
|---|---|---|
| source-tuned | 0.087 | 0.066 |
| v1 random fakes | 0.089 | **0.234** |
| v2 random fakes + N̂ | 0.089 | 0.081 |
| v3 replayed + N̂ (C1) | 0.156 | 0.081 |
| C3 fake-recall | 0.386 | 0.081 |
| C5 + rolling | 0.386 | 0.261 |

Note: v1, designed first, scores 0.234 on M2→M1 on test, though the dev check had rejected it. Another sign that dev doesn't predict test with M1 as the target.

## Item 5: sensitivity of fixed knobs
| Knob | Result | Robust? |
|---|---|---|
| **Injection repeats** (3 → 1 or 2) | The chosen setting changes in most cases (`step13_repeats_choice.csv`): C3 M1→M2 changes for 6 of 6 subsets; C5 M2→M2 for 6 of 6. Test F swings a lot: C3 M1→M2 is 0.386 with 3 repeats, 0.594 with 2, 0.867 with 1; C5 M2→M2 is 0.868 with 3 and 0.364 with 1 or 2. | **No.** The label-free selection is noisy: which fake-anomaly draw you get decides the setting. |
| **Rolling window** (C5: 15 / 30 / 60 days) | M2→M1: 0.457 / 0.261 / 0.293. M1→M1: 0.472 / 0.344 / 0.255. M2-target cells barely change. | **No for M1 as the target.** The window matters (a shorter window follows level shifts faster). 15 d looks best, but that is **post-hoc** and must not be chosen from the test. |
| **Replay amplitude floor** (0.05 → 0.2) | C1 and C3 chose the same settings except C1 M1→M1 (0.096 → 0.088). | **Yes.** |

## What this changes in the paper (and what it doesn't)
- **Supported with CIs:** the C3 gain on M1→M2; no single method winning both directions; rolling normalisation helping M1 as the target (vs C1).
- **New limitations to report:**
  1. **Selection noise:** 3 injection repeats are too few, so label-free selection is unstable. Fix for the future: many more repeats, or choosing a setting that is good across draws instead of the single best.
  2. The rolling window is a sensitive knob.
  3. M1-target numbers have wide CIs.
- **New mechanistic findings (ablations):**
  - the event builder matters most (alarm fragmentation, as in Step 12);
  - the AE is what transfers;
  - the \|z\| detector helps on dense M2 and hurts on drifting M1 unless it is rolling.
- **Not changed:** C3 stays the selected method. C5 and the 15-day window stay post-hoc observations.

## Self-test
1. Why do we resample whole months, not single events, in the bootstrap?
2. The event builder ablation drops M1→M2 from 0.386 to 0.190. Connect this to Step 12.
3. Why can a plain quantile threshold never fire at q = 1e-8 with 700k training points?
4. Why is "15-day window is best" not something we can claim?
5. What does the repeats result say about how much to trust any single label-free choice?
