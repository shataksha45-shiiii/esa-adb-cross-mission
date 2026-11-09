# Step 9c: v2 (counting hidden real events) and why M2 still fails (9 Nov 2026)

Design: `notes/step9-design-frozen.md`, section "v2", frozen before any v2 number.
Code: `src/inject_v2.py`. It re-uses the v1 runs, so no new injections were needed. Dev check: `src/run_step9.py --dev` → `results/dev_step9_v2.csv`.

## What we did, in plain words
v1 treated every alarm in the unlabelled background as false. v2 says: "some of those alarms are real events we don't have labels for, so estimate how many there are."

The idea: a setting with recall R would catch about R × N of the N hidden real events. So the background alarm count B can't be less than R × N, which gives

$$
N \le \frac{B(\theta)}{R(\theta)} \quad \text{for every setting } \theta
\qquad\Rightarrow\qquad
\hat N = \min_\theta \frac{B(\theta)}{R(\theta)}
$$

- B(θ) = number of alarms in the background that touch no fake event
- R(θ) = recall on fake events
- N̂ = estimated number of real events

Then the estimated precision is P̂ = R·N̂ / B, and we pick the setting with the best estimated F0.5.

## Key terms
- **PU learning (positive-unlabelled)**: some data is known positive (our fakes), and the rest is a mix of normal and unknown positives.
- **SCAR assumption ("selected completely at random")**: the known positives behave like the unknown ones. Here: fake events are as easy to catch as real ones.
- **Representative injection**: fakes that look like real anomalies (similar size, length, channels).

## Dev result (target training labels, diagnosis only)
| Pair | v1 | **v2** | Source settings (old transfer) | Upper bound |
|---|---|---|---|---|
| **M2→M1** | 0.400 | **0.507** | 0.217 | 0.503 |
| **M1→M1** (label-free) | 0.387 | **0.478** | — | 0.573 |
| M1→M2 | 0.129 | 0.129 | 0.242 | 0.860 |
| M2→M2 (label-free) | 0.425 | 0.230 | — | 0.861 |

M2→M1 now reaches the label-tuned upper bound. Targets on M2 still fail.

## Allowed validation (source labels only): does N̂ count correctly?
| Mission | N̂ (estimated real events per eval half) | True |
|---|---|---|
| M1 | 25.8 | 52.5 |
| M2 | 53.9 | 141.0 |

It undercounts by 2–2.6×.

## Diagnosis: why
1. The minimum of B/R is reached at **strict** settings. There, fake events are still caught (R_fake = 0.55 on M2) but real ones aren't (R_real = 0.06). So R is too big, and N̂ = B/R comes out too small.
2. Why are fakes easier? We measured the real event sizes in the source data (largest |z − value just before the event|, per event and channel; source training labels):

| Mission | 10% | 25% | median | 75% | 90% |
|---|---|---|---|---|---|
| M1 real | 0.11 | 0.34 | **0.79** | 2.0 | 9.1 |
| M2 real | 0.07 | 0.46 | **3.6** | 19.6 | 20.8 |
| Our fakes (v1/v2) | 2 to 10, log-uniform | | 4.5 | | |

   Most real M1 events are **smaller** than our fakes. M2 events are split: many tiny, and many huge (about 20, the clip limit).
   **The SCAR assumption is broken: the fake events aren't representative.**
3. At settings that really work on M2 (best AE-only setting from the diagnosis), fake and real recall do agree (0.46–0.72 vs 0.61). So the method is fine there; the estimate of N is what's broken.

## What we did NOT do
We did not try random new settings to see what scores better on dev. The next fix (v3) targets the measured cause: fake amplitudes.

## Self-test
1. What does N̂ estimate, and why is it a minimum?
2. What is the SCAR assumption, and which measurement showed it was broken?
3. Why does an undercounted N̂ push the method towards strict settings?
