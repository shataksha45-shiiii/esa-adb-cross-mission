# Step 9e: re-run of Step 8a (alarm-rate matching) with the online builder (9 Nov 2026)

## What we did
`src/adapt_stats.py` imports `pred_from` from `diag_transfer.py`, so it now uses the online event builder automatically. No code change was needed. We re-ran it.
Output: `results/adapt_stats.csv`. The old offline version is kept as `results/adapt_stats_offline_step8a.csv`.
Diagnosis only: F uses the target's training labels.

## The question
**Alarm-rate matching** means: pick the setting whose alarm events per month match some target rate r. Which r works?
For each case we average F over the 5 settings closest to r (in log scale).

| Target | Model | r = source's true rate | r = target's true rate (oracle) | Best possible |
|---|---|---|---|---|
| M1 | own | 0.163 | 0.388 | 0.608 |
| M1 | from M2 | 0.153 | 0.343 | 0.549 |
| M2 | own | 0.270 | 0.623 | 0.861 |
| M2 | from M1 | 0.261 | 0.654 | 0.860 |

(True rates: M1 1.25 events per month, M2 13.4 events per month, from training labels.)

## What it means
- Copying the source's rate fails, as found in Step 8a, and it still fails with the online builder.
- **Knowing the target's real event rate would recover most of the gap** (M2: 0.26 → 0.65).
- So the key thing to estimate without labels is **how many real events the target has**. That is exactly N̂ in v2/v3 (Step 9c), which shows why getting N̂ right matters.

## Self-test
1. Why does copying the source's rate fail between M1 and M2?
2. What does "oracle" mean here, and why can't we use it in the paper?
3. How does this result motivate v2/v3?
