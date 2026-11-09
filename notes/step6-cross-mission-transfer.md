# Step 6: Cross-mission transfer (M1 <-> M2) and first result table

## What we did
`src/run_transfer.py` runs the full pipeline for every (train mission A, test mission B) pair:

1. Normaliser fitted on **B's training period** (B's own statistics, no labels).
2. Model trained on **A's training period** only, no labels. (A = B is the in-mission reference.)
3. Scores on all of B, per channel.
4. EVT threshold per channel from **B's training-period scores**.
5. Alarms on **B's test period** -> event builder -> one mission-level alarm vector.
6. Corrected event-wise F0.5 (Step 2) on B's test labels. **Labels are used only here.**

Repeated for 3 seeds (0, 1, 2). Raw results: `results/transfer_seed{0,1,2}.csv`.

## What "label-free cross-mission" means here
The model never sees any label and never sees mission B's data during training. For B it only gets (a) scaling constants and (b) threshold calibration, both from B's unlabelled training period.

## Results, headline setting (q = 1e-5, fixed in advance), corrected event-wise F0.5, mean of 3 seeds

**Rare Event + Anomaly** (std over seeds in brackets; baseline has no randomness)

| Test mission | Detector | F0.5 | Precision | Recall | TP / FP / FN |
|---|---|---|---|---|---|
| M1 (91 test events) | AE trained on M1 (in-mission) | 0.240 (0.077) | 0.241 | 0.264 | 24 / 88 / 67 |
| M1 | AE trained on M2 (transfer M2->M1) | 0.101 (0.012) | 0.087 | 0.311 | 28 / 289 / 63 |
| M1 | baseline max abs z | 0.123 | 0.107 | 0.308 | 28 / 172 / 63 |
| M2 (362 test events) | AE trained on M2 (in-mission) | 0.040 (0.000) | 1.000 | 0.008 | 3 / 0 / 359 |
| M2 | AE trained on M1 (transfer M1->M2) | 0.049 (0.007) | 1.000 | 0.010 | 4 / 0 / 358 |
| M2 | baseline max abs z | 0.070 | 0.370 | 0.017 | 6 / 10 / 356 |

**Anomaly only** (M1: 55 events; M2: only 6 events)

| Test mission | Detector | F0.5 |
|---|---|---|
| M1 | AE on M1 / AE on M2 / baseline | 0.108 (0.035) / 0.045 (0.006) / 0.081 |
| M2 | AE on M2 / AE on M1 / baseline | 0.000 / 0.000 / 0.099 |

(TP/FP/FN values are 3-seed means, rounded.)

Reference ceilings from Step 2 (an oracle that alarms exactly on each event): F0.5 = 0.986 (M1), 0.9975 (M2).

## Sensitivity to q (uses test labels, NOT for choosing the headline), Rare Event + Anomaly F0.5, 3-seed mean

| Test | Detector | q=1e-7 | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|---|---|
| M1 | AE on M1 | 0.330 | 0.313 | 0.240 | 0.101 | 0.025 |
| M1 | AE on M2 | 0.096 | 0.096 | 0.101 | 0.025 | 0.058 |
| M1 | baseline | 0.119 | 0.119 | 0.123 | 0.122 | 0.000 |
| M2 | AE on M1 | 0.009 | 0.014 | 0.049 | 0.145 | 0.314 |
| M2 | AE on M2 | 0.014 | 0.027 | 0.040 | 0.073 | 0.258 |
| M2 | baseline | 0.070 | 0.070 | 0.070 | 0.070 | 0.208 |

## What we can and cannot say (honest reading)
- Everything is far below the oracle ceiling. This is a **working, leakage-guarded first pipeline, not a good detector yet.**
- **M2->M1 transfer drops** from 0.240 (in-mission) to 0.101, and it is not better than the no-learning baseline (0.123).
- **M1->M2 transfer is about the same as in-mission** (0.049 vs 0.040), but both are near zero and below the baseline (0.070). The M2 recall is about 1%: at q = 1e-5 the detector almost never fires on M2's many short "Rare Events" (median length about 47 min). The M1->M2 vs M2->M2 gap is 0.009 with std 0.007 on 3 seeds: **not a reliable difference**.
- M1's in-mission F0.5 varies across seeds (std 0.077 on mean 0.240), so a single seed is not trustworthy.
- In the sensitivity table the best q differs per mission (small q for M1, large q for M2). A single fixed q is therefore a weak spot of the current design. That observation comes from looking at test labels, so it is a diagnosis, not a result.
- We **cannot** claim a cross-mission advantage from these numbers.

## Not comparable to the paper's numbers
The paper reports its algorithms' scores under its own setup (30 s / 0.056 Hz grids, multivariate or channel-subset models, target-channel handling, their event filtering). Ours is a simplified setup (5-minute grid, single-channel model, OR across channels, events starting in test). We did not reproduce their reported numbers and do not compare against them.

## Everything we chose ourselves (list to defend in the paper)
| Choice | Value | Source |
|---|---|---|
| Resampling | 5-minute ZOH | ours (paper: 30 s / ~18 s) |
| Split dates | 2007-01-01 / 2001-10-01 | derived from "84 / 21 months" |
| Test events | StartTime >= split | ours |
| Normaliser | median/MAD + clip 20 | substitution (paper: nominal mean/std) |
| Window / model | 48 steps; 48-64-8-64-48 | ours, untuned |
| Training | 6000 steps, 3 seeds, no validation | ours |
| EVT | p0 = 0.98, q = 1e-5 | q fixed a priori |
| Event builder | gap 12, min length 3, OR | ours, untuned |

## What broke
- Running everything for 3 seeds took a few minutes, no failures. The only crash in this whole build was the official metric on an empty prediction (Step 2).

## Natural next steps (not done)
1. Use the paper's validation period (last 3 months of train) to pick q, W and the event rules **without test labels**.
2. Try the paper's finer sampling and check how much the 5-minute grid costs.
3. Per-channel or per-group thresholds; a score that handles level shifts and short spikes; require agreement of several channels.
4. More seeds and a confidence interval before any claim.

## Self-test
1. Which two data sources from mission B does a transferred model see, and why does neither involve labels?
2. Why can't the q-sensitivity table be used to pick the headline q?
3. If M1->M2 F0.5 equals M2->M2 F0.5, what does that say about transfer, and what does it not say?
