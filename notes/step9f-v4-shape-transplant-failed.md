# Step 9f: v4, transplant real shapes rescaled to normal variability (FAILED) (9 Nov 2026)

Design: `notes/step9-design-frozen.md`, section "v4", frozen before any v4 number. It was declared the last version.
Code: `src/inject.py --version 4` (`variability`, `source_prior_v4`, `inject_v4`). Results: `results/inject_v4/`, `results/dev_step9_v4.csv`.

## What v4 did, in plain words
It copied each real source anomaly's actual curve (shape and sign), then rescaled it:

$$
z'_{s'+u} = z_{s'+u} + d_{s+u}\cdot\frac{D_k(L)}{D_j(L)}
$$

- d = the real deviation curve on source channel j
- D_c(L) = how far channel c normally wanders in L steps (median over random windows)
- k = the target channel that receives it

The aim was for "as unusual as on the source" to become "as unusual on the target".

## Result: worse than v3 everywhere
| Pair | v3 | **v4** | Upper bound |
|---|---|---|---|
| M1→M2 | 0.380 | 0.203 | 0.860 |
| M2→M1 | 0.507 | 0.423 | 0.503 |
| M1→M1 | 0.478 | 0.387 | 0.573 |
| M2→M2 | 0.299 | 0.115 | 0.861 |

Source-only validation (allowed): N̂ moved **further** from the truth. M1: 28.2 vs v3 33.5 (true 52.5). M2: 7.0 vs v3 34.5 (true 141).

Fake vs real recall got worse, so the fakes became even easier:
| Pair | v3 fake / real | v4 fake / real |
|---|---|---|
| M2→M2 | 0.66 / 0.09 | 0.83 / 0.03 |
| M2→M1 | 0.78 / 0.24 | 0.81 / 0.13 |

## Why it failed
Many source channels are almost constant, so their normal wander D_j(L) sits at the 0.05 floor. Dividing by 0.05 multiplies the anomaly by up to 20× or more when it is pasted onto a channel that moves normally.
The idea "rescale relative to normal variability" makes sense, but the ratio is unstable when the denominator is tiny. **The hypothesis behind v4 is not confirmed.**

## Decision (rule frozen in advance, uses no target labels)
"Test v4 unless it is clearly worse than v3 on the source-only validation; then test v3."
v4 is clearly worse on the source-only validation (N̂ is worse on both missions), so **v3 goes to the test**.
No v5: we said v4 was the last version, to avoid overfitting the dev labels.

## Self-test
1. Why does dividing by a near-zero D make fakes easier?
2. Which rule decided between v3 and v4, and why does it use no target labels?
3. Why do we stop at v4 instead of trying a v5?
