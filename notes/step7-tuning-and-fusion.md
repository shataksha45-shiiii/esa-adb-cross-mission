# Step 7: Tuning without test leakage, and a fused detector

## What we did
1. Built a **tuning bench** (`src/tune.py`) that only ever looks at the **source mission's training period** and its labels.
2. Compared 25 detector variants on it, each with 162–1134 post-processing settings.
3. Picked one detector family and per-source settings, then ran the test set **once** (`src/run_final.py`, 5 seeds).

## Why
Step 6 picked every setting by hand. Its q-sensitivity table showed the right q differs by mission, but we can't pick q from test labels (that's leakage). We needed an honest way to choose settings.

## Key terms in plain words
- **Blocked 2-fold cross-validation (CV)**: split the training period in two halves *in time*. Fit on half 1, evaluate on half 2; then swap. "Blocked" means we don't shuffle, so the future never leaks into the past.
- **Selection optimism**: when you try many settings and report the best, the best looks better than it really is. CV numbers below are for *comparing* settings, not for reporting.
- **Centered window**: subtract the window's first value before the autoencoder sees it. The model then judges the *shape* of the last 2 h, not the absolute level, so slow drifts stop causing alarms.
- **Max-error score**: instead of the mean squared error over the window, take the largest single-step error. A short spike is not diluted by 23 normal steps.
- **Fusion**: per channel, alarm = (autoencoder alarm) OR (|z| alarm). Each part has its own EVT threshold (q for the AE, q2 for |z|; q2 = 0 means off).
- **k-of-n**: the mission alarms only if at least k channels alarm at the same time. Cuts single-channel false alarms.

## Protocol (the rule we followed)
- Settings for source A are chosen by CV F0.5 on **A's training labels only**.
- Transfer A→B uses A's model **and** A's settings. B contributes only unlabelled training data (normaliser + EVT calibration).
- Test labels are used once, in the final table. The baseline gets **exactly the same tuning**, otherwise beating it means nothing.

## CV results that drove the choice (best of grid, optimistic)
| Detector | M1 CV F0.5 | M2 CV F0.5 |
|---|---|---|
| baseline max abs z, W=48 | 0.418 | 0.455 |
| baseline max abs z, W=12 | 0.416 | 0.871 |
| Step-6 AE (W=48, mean error) | 0.686 | 0.530 |
| AE W=24, centered, max error | 0.784 | 0.665 |
| **fused: AE W=24 centered max + abs z W=12** | **0.784** (tuner switched abs z off) | **0.873** (abs z does most work) |

## Final test table (Rare Event + Anomaly, corrected event-wise F0.5; mean ± std over 5 seeds)
| Train → Test | Step 6 AE (q=1e-5) | Baseline, source-tuned | **Fused AE, source-tuned** |
|---|---|---|---|
| M1 → M1 (in-mission) | 0.240 ± 0.077 | 0.068 | **0.518 ± 0.044** |
| M2 → M2 (in-mission) | 0.040 | 0.862 | **0.862 ± 0.001** |
| M1 → M2 (transfer) | 0.049 | 0.027 | 0.040 ± 0.000 |
| M2 → M1 (transfer) | 0.101 | 0.036 | 0.041 ± 0.001 |

(Step 6 numbers are 3-seed means from `notes/step6`. Raw: `results/final_step7.csv`.)

## Honest reading
- **In-mission improved a lot.** M1 more than doubled (0.24 → 0.52) with lower seed variance. M2 went from 0.04 to 0.86.
- **On M2 the autoencoder adds nothing.** The source-tuned baseline also scores 0.862. M2's rare events are big level excursions that |z| over 1 h already catches.
- **Transfer target NOT met.** Both transfer rows are about 0.04: barely above the tuned baseline and far below in-mission.
- **Why transfer fails (diagnosis, not yet proven):** the *settings* don't transfer.
  - M1's settings switch |z| off, so on M2 we miss the level events (3 of 362 caught).
  - M2's settings switch |z| on with k = 2, which on M1 gives about 690 false-alarm events.
  - We have not yet tested whether the *model* itself transfers when the settings fit.
- **Step 6's transfer numbers looked better (0.10 / 0.05) only because q was fixed by hand.** That isn't a fairer comparison; it's a different protocol.

## What broke and how fixed
- The first background run failed because `results/tune/` didn't exist when `tee` opened its log file. Fixed with `mkdir -p` before starting.
- The new fast evaluator was checked against `evaluator.ew_scores` on 4 cases: identical TP/FP/FN/TNR/F.

## Code map (bench code written by Claude; the core ideas are for you to type into your own files)
- `tune.py: fast_ew`: same maths as `ew_scores`, using `np.searchsorted` instead of double loops (about 100× faster).
- `tune.py: train_ae(..., center=True)`: `x = x - x[:, :1]` subtracts each window's first value.
- `tune.py: score_ae(..., reduce="max")`: `err.max(1)` takes the worst step in the window.
- `tune.py: evt_fit / evt_thr`: fit the GPD once per channel, then get the threshold for any q from the formula in Step 5, without refitting.
- `run_final.py: alarm_vector`: OR of the parts → `build_events` per channel → count channels → `>= k`.

**To type yourself:** centering + max error in `detector.py`, the k-of-n rule in `events.py`.

## Natural next steps
1. **Diagnose transfer** on training data only: is it the model or the settings? Run the M2-trained AE on M1's training CV using M1's settings, and vice versa.
2. **Label-free target adaptation**, a candidate paper contribution: choose q / q2 / k on the target mission from *unlabelled* statistics, e.g. match the source's alarm rate on the target's training period.
   - Caveat: with only two missions, any transfer-method design is developed while looking at both. Say so in the paper.
3. Finer grid (30 s), because M1's 9 events under 5 minutes are invisible at 5 min.

## Self-test
1. Why is the CV F0.5 (0.78) so much higher than the test F0.5 (0.52) for M1? Name two reasons.
2. Why must the baseline be tuned with the same protocol before we claim to beat it?
3. The M2-tuned settings produce 690 false alarms on M1. Which single setting would you suspect first, and why?

---

# Step 7b: Why transfer fails — model or settings? (`src/diag_transfer.py`)

## What we did
On the **target** mission's training-period CV (same 2 folds as `tune.py`; no test data) we swapped one ingredient at a time:
- **model**: trained on the target itself ("own") vs trained on the other mission ("transferred")
- **settings** (q, q2, gap, min_len, k): tuned on the target ("own") vs tuned on the other mission ("transfer", what the real transfer row uses)

Mean of 2 seeds × 2 folds. This uses the target's TRAINING labels to *evaluate*, so it is a diagnosis only; it never picks the reported config.

## Result (CV F0.5)
| Target | own model + own settings | transferred model + own settings | own model + transferred settings | transferred model + transferred settings |
|---|---|---|---|---|
| M1 | 0.781 | 0.640 | 0.231 | 0.220 |
| M2 | 0.873 | 0.871 | 0.164 | 0.186 |

## Reading
- **The settings are the main problem.**
  - Swapping only the settings costs 0.55 on M1 and 0.71 on M2.
  - Swapping only the model costs 0.14 on M1 and 0.00 on M2.
- **The model transfers well.**
  - The M1-trained autoencoder on M2, with M2-fitting settings, scores 0.871, the same as M2's own model.
  - Even the autoencoder alone (|z| part off) reaches 0.786 on M2 with the best grid setting (optimistic).
  - So the "useless autoencoder on M2" from Step 7 came from its settings, not its learning.
- **The failure modes.**
  - M2's settings on M1 (|z| on, k = 2, gap 36) produce about 137 false alarms per fold.
  - M1's settings on M2 (autoencoder only, strict q = 1e-5, k = 1) catch only 4–5% of the events.
  - The autoencoder on M2 wants a loose q (1e-2) *and* k = 3, i.e. many channels at once.

## What this means for the paper
The cross-mission problem here is **setting calibration, not representation**. The next method should choose q / q2 / k on the target mission **without labels**, e.g. from the target's unlabelled training scores (alarm rate, number of alarm events per month, channel-agreement statistics). That's a clean, testable contribution.

## Self-test
4. Which cell of the table shows the model transfers, and which shows the settings don't?
5. Why is it OK to use the target's *training* labels here, but not to pick the final config with them?
