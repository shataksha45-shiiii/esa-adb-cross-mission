# Step 10: baselines + improvement round, one post-hoc test (9 Nov 2026)

Protocol frozen before any number: `notes/step10-protocol-frozen.md`. Code: `src/step10.py`, `src/rolling.py`; `src/inject.py --version 5`.
Results: `results/dev_step10.csv`, `results/final_step10.csv` (+ `_log.txt`).
**Test labels: 4th use. Every number below is POST-HOC.** Our F0.5 matches the official ESA-ADB code to 1.4e-17.

## What we did (plain words)
1. Wrote down the rules first: the list of candidates, a selection rule, and what would count as a win.
2. Added baselines:
   - a fixed default (no tuning at all);
   - Heydari et al. 2026, static and rolling versions;
   - plain |z|.
3. Added 3 new label-free candidates:
   - **fake-recall matching** (C3): transfer the operating point as "fraction of fakes caught", not as a threshold;
   - a **rolling re-centred |z|** that follows level shifts (C4);
   - both together (C5).
4. Scored everything on dev (training periods only), picked by the rule, then ran the test once.

## Selected by the frozen rule: C3 (fake-recall matching)
Dev, worse direction: C3 0.507 > C5 0.469 > C1/C2/C4 0.380.

## Test results (official corrected event-wise F0.5, mean ± std over 5 seeds; B3–B5 have no AE, so 1 run)
| Method | M1→M2 | M2→M1 | Worse direction | M1→M1 (label-free) | M2→M2 (label-free) |
|---|---|---|---|---|---|
| B0 target-label-tuned (upper bound) | 0.849 | 0.278 | — | 0.481 | 0.846 |
| B1 source-label-tuned (old transfer) | 0.087 ± 0.013 | 0.066 ± 0.002 | 0.066 | — | — |
| B2 fixed default (no tuning) | **0.892** | 0.040 | 0.040 | 0.048 | 0.890 |
| B3 Heydari static | **0.934** | 0.007 | 0.007 | 0.084 | **0.965** |
| B4 Heydari rolling (our implementation, flagged) | 0.930 | 0.000 | 0.000 | 0.000 | 0.898 |
| B5 plain \|z\|, source-tuned | 0.040 | 0.052 | 0.040 | — | — |
| C1 v3 (Step 9) | 0.156 ± 0.036 | 0.081 ± 0.003 | 0.081 | 0.096 | 0.139 |
| C2 v3, AE-only | 0.156 ± 0.036 | **0.273 ± 0.029** | 0.156 | 0.312 | 0.139 |
| **C3 fake-recall match (selected)** | **0.386 ± 0.019** | 0.081 ± 0.003 | 0.081 | 0.089 | 0.778 |
| C4 v3 + rolling \|z\| | 0.156 ± 0.036 | 0.261 ± 0.013 | 0.156 | 0.236 | 0.139 |
| C5 fake-recall + rolling \|z\| | **0.386 ± 0.019** | **0.261 ± 0.013** | **0.261** | **0.344** | **0.868** |

Anomalies only (Rare Events ignored) is low everywhere:
- Best transfer: B3 M1→M2 0.498.
- C5: 0.009 (M1→M2), 0.121 (M2→M1).
- C3: 0.009 (M1→M2), 0.055 (M2→M1).
- Upper bound B0: 0.270 (M1→M2), 0.219 (M2→M1).

## Verdict against the pre-registered criterion
"The selected candidate beats C1 in both directions by more than the seed std."
- M1→M2: **yes**. 0.156 → 0.386 (std 0.02–0.04).
- M2→M1: **no**. 0.081 → 0.081; C3 chose the same setting as C1 there.
- **Pre-registered result: no clear improvement in both directions. A clear gain in one.**

## What the test shows (post-hoc observations, NOT pre-registered claims)
1. **The rolling |z| fixes M1 as the target.** M2→M1 rises from 0.081 to 0.261 (C4/C5), and label-free M1→M1 from 0.096 to 0.344 (C5). That's close to the label-tuned upper bound for M2→M1 (0.278). So the level-shift diagnosis from Step 9g was right. Dev couldn't show it, because the level shifts happen only in the test years.
2. **C5 is the best worst-case method on test** (0.261), but the dev rule picked C3. Dev doesn't predict test well when the mission drifts (dev–test gaps up to −0.43). We must report C3 as the selected method and C5 as a post-hoc finding.
3. **On M2 as the target, simple methods beat everything,** including M2's own label-tuned settings:
   - fixed default 0.892, Heydari static 0.934–0.965, label-tuned 0.846–0.849.
   - M2 has many frequent, easy events, so looser settings win, and label tuning on the training years was too strict for M2's test years.
   - Our label-free C3 (0.386) is still too strict there.
4. **No method wins both directions.** M2 as the target rewards loose, level-based alarms. M1 as the target punishes them, because its long-term drift trips them.
5. **Anomalies-only scores are low for every method.** Most of the high all-events scores on M2 come from Rare Events (commanded, nominal events), not from anomalies.

## What this means for the paper
We're in the "analysis paper" row:
- The model transfers; the settings don't.
- The right setting depends on the target: event density pulls one way, drift pulls the other.
- Rolling normalisation fixes the drift side; looser alarms fix the dense-event side.
- An in-period dev check is not reliable under drift.

A method that wins both directions would need a new design: for example, C5's rolling |z| with a looser operating point on dense targets. Because the test labels have now been used 4 times, any such method needs a fresh evaluation that doesn't touch the test labels. Options:
- a blocked "pseudo-test" at the end of the training period;
- the deferred Mission3.

## What broke and how it was fixed
- **zsh word-splitting in the run loop** (`for p in "1 2"` passed "1 2" as one argument). Fixed by looping over "12 21 11 22" and slicing the characters.
- **The official ESA-ADB metric crashes on an empty prediction** (`np.max` of an empty array), triggered by B4 on M1. This happened before any score was printed or saved. Fixed: an empty prediction gets F = 0 from our evaluator, which is the value by definition (no true positives).
- **B4's zero alarms on M1** are probably caused by our 1e-3 scale floor on near-constant channels. They are not evidence about Heydari's real method.

## Key terms
- **Fake-recall matching:** pick target settings that catch the same share of injected fake events as the source's best settings caught on the source.
- **Rolling re-centring:** subtract each channel's median over the past 30 days, so a lasting level shift stops looking abnormal after a while.
- **Post-hoc:** a result found after looking at the test; it can suggest ideas but can't be a pre-registered claim.
- **Worst-case (min) selection:** judge a method by its weaker direction, so one lucky direction can't carry it.

## Self-test
1. Why did the dev rule pick C3 when C5 did better on test?
2. Why does the rolling |z| help on M1 but not change M2 much?
3. Why can a fixed default with no tuning beat label-tuned settings on M2's test years?
4. Why must C5 be reported as a post-hoc observation, not as our method?
