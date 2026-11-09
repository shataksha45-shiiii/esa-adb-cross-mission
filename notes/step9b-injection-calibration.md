# Step 9b: label-free calibration by synthetic anomaly injection (9 Nov 2026)

Design written and frozen **before** any result: `notes/step9-design-frozen.md`.
Code: `src/inject.py` (calibration), `src/run_step9.py --dev` (diagnosis). Test set **not** used yet.

## What we did, in plain words
1. Take the target mission's training data and ignore its labels.
2. Paste in fake anomalies: level shift, spike burst, noise burst, stuck sensor, or ramp.
   How long they last, how many channels each touches, and how often they happen are copied from the **source** mission's labelled events.
3. Run the source-trained model over this data, try about 1,100 post-processing settings, and keep the one with the best F0.5 on the **fake** anomalies.
4. Dev check: score that setting on the target's **real training labels**. This is diagnosis only: we don't change anything because of it.

## Key terms
- **Injection**: inserting artificial anomalies at known places, so we have labels without asking anyone.
- **Prior**: what we assume before seeing the target, here "anomalies look like the source's anomalies".
- **Contamination**: real anomalies hidden in the unlabelled data we treat as normal. Any alarm on them counts as a false alarm.

## Code, key lines (`src/inject.py`)
```python
n = int(round(prior["rate"] * (tns[hi-1] - tns[lo]) / MONTH_NS))   # how many fake events: source rate x months
L, k = prior["pairs"][rng.integers(len(prior["pairs"]))]          # copy (duration, n_channels) of a random source event
if all(e + DAY <= a or s >= b + DAY for a, b in placed): break     # no overlap, at least 1 day apart
A = exp(uniform(log 2, log 10))                                   # amplitude, log-uniform 2..10 robust-z
Z[s:e, j] += sg * A                                                # e.g. "shift": add a step of size A
np.clip(Z, -20, 20, out=Z)                                         # same clipping as the normaliser
```
The EVT thresholds and the normaliser are fitted on the **clean** fit half. Injections only go into the eval half, the same 2-fold split as `tune.py`.

## Chosen settings
| Pair | q | q2 (\|z\| part) | gap | min_len | k-of-n |
|---|---|---|---|---|---|
| M1→M1 | 1e-8 | off | 36 | 1 | 2 |
| M1→M2 | 1e-4 | off | 12 | 6 | 2 |
| M2→M1 | 1e-5 | off | 12 | 3 | 1 |
| M2→M2 | 1e-3 | off | 12 | 3 | 3 |
| (label-tuned M1) | 1e-5 | off | 12 | 1 | 2 |
| (label-tuned M2) | 1e-4 | 1e-4 | 36 | 3 | 2 |

## Dev result (target training labels, 2-fold CV, seed 0)
| Pair | Injection (label-free) | Current transfer (source settings) | Target-tuned (upper bound) |
|---|---|---|---|
| **M2→M1** | **0.400** | 0.217 | 0.503 |
| M1→M2 | 0.129 | 0.242 | 0.860 |
| M1→M1 (fully label-free) | 0.387 | — | 0.573 |
| M2→M2 (fully label-free) | 0.425 | — | 0.861 |

## What it means
- **It works on M1 as target.** M2→M1 nearly doubles (0.217 → 0.400) and closes about 64% of the gap to the upper bound. The 155 false alarms drop to 11.
- **It fails on M2 as target.** It's worse than doing nothing (0.242 → 0.129). It always switches the |z| detector off and is too strict (recall 3–15%).
- **Why it fails on M2 (the weakness we predicted in the design):** M2 has many real events (about 13 per month) inside the "unlabelled normal" background. Whenever the |z| detector fires on them, it gets punished as if they were false alarms. So the method learns to turn |z| off, even though |z| is what catches M2's big level-change events.

## What broke and how it was fixed
- The logs stayed empty while running: Python buffers `print` output when it goes to a file. Nothing was actually broken; the logs filled in when each run finished. Fix for next time: `python -u` or `flush=True`.

## Self-test
1. Why do fake anomalies give us "labels for free"?
2. Why does contamination push the method towards stricter settings?
3. Why did this hurt M2 much more than M1?
