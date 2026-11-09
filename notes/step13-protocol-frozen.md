# Step 13 protocol: ablations, confidence intervals, sensitivity (roadmap items 3–5) (FROZEN 9 Nov 2026, before any Step 13 number)

## Purpose and the one rule that matters
These are **analysis** runs. They explain *why* the Step 10 results look the way they do, and how sure we can be.
**No method, setting or claim is changed because of a Step 13 number.** The selected method stays C3 (Step 10). C5 stays a post-hoc observation.
All numbers are on the test period (5th use of the test labels), so all are **post-hoc** and labelled that way.

## Item 4: uncertainty (confidence intervals)
- **Seed variation:** 5 AE seeds, as before.
- **Block bootstrap over test months.** Events close in time are not independent, so we resample whole calendar months of the test period with replacement (2,000 resamples).
  - Each resample recomputes corrected event-wise F0.5 from per-month totals:
    - GT events (TP/FN) are assigned to the month they start in;
    - alarm events (FP, and alarm time outside GT) are assigned to the month they start in;
    - nominal time is assigned to the month it falls in.
  - The F0.5 of a resample is the mean over the 5 seeds.
  - **95% CI** = 2.5th–97.5th percentile.
- **Paired difference CIs** use the same month resamples for both methods, for the comparisons:
  - C3 − C1;
  - C5 − C1;
  - C3 − B1;
  - C5 − B3 (Heydari static).
  - A difference is called **clear** only if its 95% CI excludes 0.
- Check: with no resampling, the per-month decomposition must reproduce fast_ew's F exactly.

## Item 3: ablations (remove or replace one part at a time)
Applied to **C3** (selected) and **C5** (best post-hoc), all 4 pairs, 5 seeds:
| ID | Change | Question it answers |
|---|---|---|
| ab-AE | \|z\| part off (q2 = 0) | What does the level detector add? |
| ab-Z | AE part off (q = 0) | What does the AE add? |
| ab-EB | event builder off (gap = 0, min_len = 1) | What do gap-merging and min_len do? |
| ab-K | k-of-n vote set to 1 | What does channel voting do? |
| ab-Q | EVT threshold replaced by the empirical training quantile at the same q | Does EVT matter, vs a plain quantile? |

Also a **calibration-rule ladder**, all on test with the same model:
- source-tuned (B1);
- fixed default (B2);
- v1 random fakes + best injected F;
- v2 random fakes + N̂ rule;
- v3 replayed fakes + N̂ rule (C1);
- C3 fake-recall matching;
- C5 (+ rolling |z|).

v1 and v2 have never been scored on test; they are added here for completeness, and nothing is selected from them.

## Item 5: sensitivity of fixed knobs (values chosen now)
| Knob | Default | Tried | Applies to | Cost |
|---|---|---|---|---|
| Injection repeats | 3 | 1, 2 (subsets of the existing runs) | C1, C3, C5 | none |
| Rolling window | 30 days | 15, 60 days (new inject v5 runs) | C5 | about 1 h each |
| Replay amplitude floor | 0.05 | 0.2 (new inject v3 run) | C1, C3 | about 1 h |

Report:
1. whether the chosen setting changes;
2. if it does, the test F0.5 of the new choice.

**Robust** = the chosen setting or the test F changes by less than the seed std.

## Reporting
Everything goes into `notes/step13-*.md` and `results/step13_*.csv`, including results that make our methods look worse.
