# Step 10 protocol: baselines + improvement round on M1/M2 only (FROZEN 9 Nov 2026, before any Step 10 number)

## Why this file exists
We want to improve transfer **without fooling ourselves**. Two ways to fool ourselves:
- **Overfitting the dev check:** try many ideas, keep whichever scores best on the target's training labels. The best of many tries looks better than it really is (selection optimism). Step 9 showed this: dev said 0.507 for M2→M1, test gave 0.081.
- **Overfitting the test:** look at the test and change things. The test labels have already been scored 3 times.

The rules below are written **before** seeing any Step 10 number.

## Rules
1. **All development uses the training periods only.** We use the same 2-fold blocked CV as Steps 7–9, with seed 0. The dev score is the corrected event-wise F0.5 on the **target's training labels**. This is a diagnosis; we disclose it in the paper.
2. **The candidate list is fixed now** (below). No new candidates are added after seeing dev numbers. Each candidate's knobs are fixed now; none of them is tuned on dev.
3. **Selection rule:** among our label-free candidates C1–C5, pick the one with the highest
   **min( dev F of M1→M2 , dev F of M2→M1 )**, i.e. its worse transfer direction. Using the worse direction stops us from picking a method that helps one direction by luck.
   - Tie (within 0.02 of the best): pick the simpler one. The order of simplicity is C2 < C1 < C3 < C4 < C5.
4. **Test: run once**, 5 seeds, official metric, every candidate and baseline. This is the 4th use of the test labels, so **all Step 10 test numbers are post-hoc** and are labelled that way in the paper. The selected candidate is the headline; the rest are shown for transparency, with no re-selection on test.
5. We report the dev→test gap for every candidate. A large gap means dev doesn't predict test, and that becomes a finding.
6. We also report the **anomalies-only** F0.5 (category "Anomaly"; Rare Events ignored) on test, next to the usual Anomaly + Rare Event score.

## Baselines (reported, not selected)
| ID | What | Uses |
|---|---|---|
| B0 | Target-label-tuned settings (upper bound, Step 8 CFG of the target) | target labels (oracle) |
| B1 | Source-label-tuned settings applied to the target (Step 8 transfer) | source labels |
| B2 | **No calibration**, a fixed default: fused AE + \|z\|, q = q2 = 1e-4, gap 12, min_len 3, k-of-n 1 (middle of the grid, chosen now without data) | nothing |
| B3 | **Heydari et al. 2026, static (R2lf-style).** Score s(t) = mean of the top-3 \|robust z\| across channels (z from the target's fit-half median/MAD). Threshold on the fit half picked so the alarm-event rate matches a budget = **source's true event rate** (events/month, source training labels). Causal event builder gap 12, min_len 1. | source event rate |
| B4 | **Heydari et al. 2026, rolling.** Same as B3, but each channel's z uses a trailing 30-day median/MAD (past target data only, updated daily, scale floor 1e-3, clipped ±20) | source event rate |
| B5 | Plain \|z\| max baseline with source-tuned settings (Step 8 "baseline") | source labels |

Differences from Heydari (flagged; our choices):
- We use each mission's target channels (58 / 47), not the 77 shared channels.
- We use our 5-min grid and causal event builder, not their cooldown/max-on policy.
- They sweep the budget; we fix it to the source event rate so there is one number to compare.

## Our label-free candidates (C1–C5)
All use the source AE (seed 0 for calibration, 5 seeds at test) and the target's **unlabelled** training data.
| ID | Settings chosen by | \|z\| part |
|---|---|---|
| C1 | v3 (Step 9 final): replayed source events + the N̂ rule | static |
| C2 | v3 rule restricted to **AE-only** settings (q2 = 0). The AE sees centered windows, so it is robust to level shifts. | off |
| C3 | **Fake-recall matching (new).** On the source, measure the fake recall R_S* that the source's **label-tuned** setting (CFG[S]) achieves on replayed source events (from the S→S injection run, source data + source labels only). On the target, pick the setting with the **fewest background alarms** among those with fake recall ≥ R_S*. If none qualify, pick the highest fake recall. Idea: transfer the operating point in "fraction of fakes caught" units, not in q units. | static |
| C4 | v3 rule, with the \|z\| part computed on **rolling re-centered z**: subtract each channel's trailing 30-day median (past data only, updated daily; no rescaling). The injection calibration is re-run with this \|z\| ("inject_v5"). | rolling |
| C5 | Fake-recall matching (C3 rule) on the inject_v5 runs. R_S* comes from the v5 S→S run at CFG[S]. **Flag:** CFG[S] was tuned with static \|z\|. | rolling |

Fixed knobs, chosen now and not tuned:
- Rolling window 30 days (Heydari's commissioning window length), update every day.
- The first day of a series has no history, so it uses center 0, which is the static normaliser.

## Diagnostics (not candidates, never selected)
- Oracle alarm-rate matching (target's true rate): shows how much an event-count estimate could give.
- Best grid setting on target labels: shows the headroom.

## What would count as a real improvement
On test, the selected candidate beats C1 (Step 9) in **both** transfer directions by more than the seed std.
Anything less is reported as "no clear improvement".

## Disclosure added during implementation (9 Nov, before any dev number)
While checking that rolling_center runs correctly, I looked at one **unlabelled** test-period statistic: the share of |z| > 5.
- Static z: train 0.10%, test 1.03% (already known from Step 9g).
- Rolling re-centred z: train 0.10%, test 0.54%.

So rolling re-centring removes about half of the drift excess. No label was used, and no design choice was changed because of this.

## Selection (applied 9 Nov, BEFORE the test run)
Dev worst-direction F0.5, min(M1→M2, M2→M1):

| Candidate | min |
|---|---|
| C1 | 0.380 |
| C2 | 0.380 |
| **C3** | **0.507** |
| C4 | 0.380 |
| C5 | 0.469 |

**Selected: C3 (fake-recall matching, static |z|).** No candidate is within 0.02 of it, so the tie rule doesn't apply. Full dev table: results/dev_step10.csv.

Dev observations recorded before the test:
- B3 (Heydari static) is very strong on M2 as the target: 0.932 for M1→M2, above the label-tuned 0.860. It fails on M1 as the target: 0.039 for M2→M1.
- B4 (Heydari rolling) raises no alarms on M1. **Probable implementation artefact:** the 1e-3 scale floor on near-constant or binary channels makes the top-3 \|z\| saturate, so the budget-matched threshold sits at the maximum. Heydari works on raw telemetry with their own ε. Report B4 with this flag; don't claim it as Heydari's true performance.
