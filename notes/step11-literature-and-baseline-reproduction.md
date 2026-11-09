# Step 11: literature settled + reproducing an official ESA-ADB baseline (roadmap items 1, 2, 9) (9 Nov 2026)

## What we did
1. Read the ESA-ADB paper's results tables (arXiv 2406.17826 **v2**, PDF text extracted locally) to settle the "0.241 vs 0.079" question.
2. Read the methods of the closest prior work in full text:
   - Heydari et al. 2026 (Results section);
   - Goswami et al. 2023;
   - Perini et al. 2023.
3. Re-implemented the official **GlobalSTD** baseline inside our pipeline. We copied the official rule exactly and compared with the published numbers.
4. Decided the paper's claim (roadmap item 9).

## 1. ESA-ADB baseline numbers: settled
ESA-ADB v2, Table 2, all target channels, corrected event-wise F0.5, all events (anomalies + rare events):

| Mission | Best overall | Best purely unsupervised |
|---|---|---|
| M1 | **0.061** (Telemanom-ESA-Pruned, semi-supervised forecaster) | ≤ 0.008 (Telemanom-ESA 0.008, DC-VAE-ESA STD5 0.007, GlobalSTD5 0.003) |
| M2 | **0.079** (PCC) | 0.079 (PCC). Telemanom-ESA-Pruned 0.071, GlobalSTD5 0.064 |

- **0.079 is correct.** The **0.241** quoted by arXiv 2609.10017 for M2 does not appear as an M2 event-wise score in Table 2. In Table 2, 0.241 appears as M1's channel-aware F0.5 for GlobalSTD5. It also appears in Supplementary Table 9 (M1, lightweight subset, anomalies only) as recall and as channel-aware F0.5. So 2609.10017 most likely mis-cited it. (We checked v2 only; v1 not checked.)
- On the lightweight subsets the numbers are much higher (M1 0.786 Telemanom-ESA-Pruned; M2 0.949 Windowed iForest). That subset is the "0.949" that GitHub repos reproduce.
- **Anomalies only** (Supplementary Table 9, all channels): M1 best 0.039; M2 ≤ 0.001 (only 9 test anomalies).
- Split: each mission's first half is the training set, as in our data.py.

## 2. Closest prior work, read in full text
- **Heydari, Mostafavi, Nyarko (IEEE Access 2026).**
  - What they do: fixed robust-z top-3 scorer, alert policy (cooldown / max-on / duty cap), workload metrics (alerts per hour); no F0.5.
  - Results:
    - Direct threshold transfer overshoots the alert budget about 28× in both directions.
    - Score-distribution mapping helps only M1→M2.
    - Fixed early-window commissioning fails in both directions.
    - Rolling label-free adaptation helps but is capped by the duty limit.
    - In M2→M1, under source normalisation, 100% of nominal scores saturate.
  - Agreement with us: **M2→M1 is the hard direction**, and non-stationary target tails break early-window calibration.
  - What they don't do: learned models, the model-vs-settings split, injection, F0.5.
- **Goswami et al. (ICLR 2023).**
  - Method: inject 9 types of synthetic anomalies (e.g. scale, noise, cutoff, wander, average) into unlabelled series. Treat the injected spots as pseudo-positives and pick the model with the best adjusted F1. Combine several proxy metrics by robust rank aggregation.
  - Data: 9 UCR series + SMD. Single datasets; no cross-dataset transfer.
  - They name two weaknesses: real anomalies may be of other types, and existing anomalies count as negatives. **These are exactly the two problems our v2 (hidden-event estimate) and v3 (replay real source events) address.** So our injection work is "Goswami's idea, extended to cross-mission transfer, with two fixes".
- **Perini, Bürkner, Klami (ICML 2023), γGMM.** Fits a Dirichlet-process Gaussian mixture to the scores of several detectors to get a posterior over the contamination factor (share of anomalies), then thresholds at the posterior mean. Tested on 22 tabular datasets, not time series. It is the closest prior art to our N̂ (hidden-event count). A natural extra baseline for N̂.
- **PitIA (BiDS'25):** original still not found. Keep citing it as "reported in 2609.10017".

## 3. Reproducing GlobalSTD in our pipeline (roadmap item 2)
Code: `src/repro_globalstd.py`. Output: `results/repro_globalstd.csv`.
The rule is copied from `ESA-ADB/TimeEval-algorithms/std/algorithm.py`:
- per-channel mean and std over unlabelled training points;
- alarm when a value leaves mean ± tol·std;
- no post-processing;
- mission alarm = OR over target channels.

| Mission | tol | Published (ESA-ADB) | **Ours (5-min grid)** | Our TP / FP | Alarm time |
|---|---|---|---|---|---|
| M1 | 3 | < 0.001 | 0.003 | 66 / 21,761 | 22.8% |
| M1 | 5 | 0.003 | 0.009 | 59 / 7,225 | 16.2% |
| M2 | 3 | 0.008 | 0.062 | 358 / 6,328 | 9.2% |
| M2 | 5 | 0.064 | **0.400** | 355 / 660 | 3.1% |

**Verdict: not reproduced.**
- M1 agrees in kind: both near zero, swamped by false alarms.
- M2 is about **6× higher** in our pipeline.

Most likely reason (a hypothesis, **not yet tested**): our **5-minute grid**, which keeps the last value per bin.
- ESA-ADB scores the native-rate data (roughly 18–30 s).
- At the native rate, a noisy channel crosses the threshold many separate times. Each crossing outside a labelled event is a separate false-positive event, and precision collapses.
- On a 5-minute grid, most of those short crossings are never sampled. So there are far fewer false-positive events.

**Consequence for every comparison we make:**
- Our absolute F0.5 values are **not directly comparable** to numbers published at the native rate. That includes ESA-ADB, PitIA and 2609.10017. Our grid makes noise-sensitive detectors look better.
- "Ours 0.846 vs ESA-ADB 0.079 on M2" must not be claimed as beating ESA-ADB.
- Fair comparisons are **within our pipeline**: GlobalSTD5 here scores 0.400 (M2) and 0.009 (M1), against our label-tuned 0.846 / 0.481.
- The cross-mission findings (relative comparisons on the same grid) are unaffected.

**Confirmed in Step 12:** at the native 18 s grid GlobalSTD5 scores 0.0638 (published 0.064). The grid explains the whole gap.

## 4. The paper's claim (roadmap item 9): decided
**An analysis paper, not a "new SOTA method" paper.** Working title: *"The model transfers, the settings don't: a cross-mission study of label-free calibration for spacecraft anomaly detection on ESA-ADB."*

Claims we can defend:
1. **Diagnosis:** a learned detector from the other mission scores as well as the target's own (0.860 vs 0.858 on M2). Transfer fails in the settings layer.
2. **No single calibration works in both directions.**
   - Event-dense M2 rewards loose, level-based alarms: fixed default 0.892, Heydari-style 0.934.
   - Drifting M1 punishes them: 0.007–0.040.
   - Rolling normalisation fixes the drift side (0.081 → 0.261, close to the label-tuned 0.278).
3. **Label-free calibration by replayed injection** (Goswami-style, extended with the hidden-event estimate and fake-recall matching) improves one direction (M1→M2 0.087 → 0.386) and not the other.
4. **Dev in the training period doesn't predict test under drift** (gaps up to −0.43).
5. **Methodology warnings:**
   - Our resampling inflates absolute scores relative to the native rate.
   - 0.241 is a mis-cited baseline.
   - Anomalies-only scores are near zero everywhere.

Not claimable: beating ESA-ADB or PitIA; a method that wins both directions.

## Self-test
1. Why can a 5-minute grid make GlobalSTD look 6× better on M2?
2. Which two weaknesses of Goswami's injection did our v2 and v3 address?
3. Why is "0.846 vs 0.079" not a fair comparison, while "0.846 vs GlobalSTD-in-our-pipeline 0.400" is?
4. What would confirm the resampling explanation?
