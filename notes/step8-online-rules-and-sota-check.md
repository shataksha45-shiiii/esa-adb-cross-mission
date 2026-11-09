# Step 8: Making the pipeline genuinely online, and checking it against published results

## What we did
1. Read the ESA-ADB paper (arXiv 2406.17826 v2) Table 2 and protocol sections ourselves, from the PDF text.
2. Found and fixed **two places where our pipeline used future data**, which the paper forbids:
   - "detect anomalies in the test set in an online manner (without using future samples from the test set)", p. 30.
3. Re-tuned on training labels only (2-fold blocked CV, as in Step 7), with the fixed rules.
4. Ran the test set with the **official test-event definition**, scored by our evaluator **and** the official ESA-ADB `ESAScores` code. The two agree to every digit.

## The two future leaks (and fixes)
| Leak | What it did | Fix |
|---|---|---|
| Event builder (Step 5) | Filled gaps between alarms *backwards* (up to 3 h) and back-dated an alarm to its first step once it lasted `min_len` steps | `tune.build_events_causal`: an alarm turns on only at the step where `min_len` consecutive exceedances are reached, and stays on for `gap` steps after the last exceedance |
| 5-min bins (Step 1) | `resample('5min').last()` labels bin [00:00, 00:05) as 00:00 but stores the value seen at about 00:04:59, i.e. 5 min of future | `data.load_mission` labels each bin by its **end** |

Causality was checked: changing the future never changes past outputs (tested at 14 cut points).
Effect on M1 CV: 0.784 → 0.608. **The leaks really were inflating the score.** M2 barely moved (0.873 → 0.861).

## Official test definition (from `ESA-ADB/notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py`)
- The test set = samples after 2007-01-01 (M1). This confirms our split date.
- Labels come from the test file itself, so every event **overlapping** the test period counts, clipped at the split. Before, we only kept events *starting* in test. Fixed.

## Final test results (all target channels; Rare Event + Anomaly; corrected event-wise F0.5; 5 seeds)
| Train → Test | Fused AE (ours) | Our \|z\| baseline, tuned the same way | Best in ESA-ADB paper, Table 2, full channel set |
|---|---|---|---|
| M1 → M1 | **0.481 ± 0.041** | 0.071 | 0.061 (Telemanom-ESA-Pruned) |
| M2 → M2 | **0.846 ± 0.002** | 0.843 | 0.079 (PCC) |
| M1 → M2 (transfer) | 0.087 ± 0.013 | 0.040 | no transfer experiments exist |
| M2 → M1 (transfer) | 0.066 ± 0.002 | 0.052 | no transfer experiments exist |

Raw data: `results/final_step8.csv`. Official-code values are in the `official_F` column (baseline and seed 0).

## Other published numbers (honest status)
| Source | Setting | M1 | M2 | Verified by us? |
|---|---|---|---|---|
| ESA-ADB paper v2, Table 2 | static split, all channels, mostly untuned | 0.061 | 0.079 | **yes**, read from the PDF |
| PITIA (Tejedor et al. 2025), as cited by arXiv 2609.10017 | static split | 0.424 | 0.882 | **no**, secondhand only |
| arXiv 2609.10017 (SPAICE 2026 poster) | **monthly retraining**, different protocol | 0.700 | 0.698 | read the HTML; not comparable (different protocol) |
| arXiv 2607.07335 | **uncorrected** event F0.5, supervised | n/a | n/a | not comparable (different metric) |

Note: 2609.10017 cites ESA-ADB's best M2 full-set score as 0.241; the v2 PDF we read shows 0.079. **Unresolved**; check the v1 PDF before citing.

## Honest reading — what we can and cannot claim
**Can claim (if it survives review):**
- Under the official static split, all target channels, and the official metric, our fully online, label-free-trained detector scores 0.48 on M1, about 8× the best result in the ESA-ADB paper.
- It also beats PITIA's reported 0.424 on M1, but only if that number is confirmed.
- First cross-mission (M1 ↔ M2) results on ESA-ADB. Both transfer rows beat the equally tuned baseline. The model itself transfers: the Step 7b diagnosis showed it, and the remaining gap comes from calibrating the settings.

**Cannot claim:**
- "Beats all current standards." On M2 our 0.846 is **below** PITIA's reported 0.882 (unverified). On M2 the autoencoder adds almost nothing over the tuned |z| baseline (0.846 vs 0.843).
- That the gain over the ESA-ADB paper comes from the *model*. On M2 it comes from tuning the post-processing; ESA-ADB's baselines were mostly not tuned that way. A reviewer will say "you tuned, they didn't". Our answer is the tuned-baseline column.
- That transfer works well. 0.07–0.09 is far below in-mission performance.

**Must disclose in the paper:**
- Hyperparameters were selected with **training-period labels** (2-fold CV). Models and thresholds never see labels.
- Test labels were evaluated twice: Step 7, then Step 8 after the leak fixes. That was a protocol correction, not tuning. Settings never came from test results.
- Our input is target channels only, on a 5-min grid. ESA-ADB's deep models also used non-target channels and telecommands.
- Choices made with the M1 / M2 diagnosis in view (Step 7b): with two missions, method design inevitably saw both.

## Self-test
1. Why did fixing the event builder lower M1 but not M2? (Hint: what does a 3 h backward gap fill do to a long M1 event vs a 47 min M2 event?)
2. Why is "labelling a bin by its start" a leak, even though it's only 5 minutes?
3. A reviewer says our M2 result is "just tuning". Which column answers them, and what does it show?
