# Step 9d: v3, fakes copied from real source events (9 Nov 2026)

Design: `notes/step9-design-frozen.md`, section "v3", frozen before any v3 number.
Code: `src/inject.py --version 3` (`source_prior_v3`), selection with `src/inject_v2.py` (`cfg_v2(s, t, "inject_v3")`).
Results: `results/inject_v3/`, `results/dev_step9_v3.csv`. Test set not used.

## What changed, in plain words
v1/v2 invented fake anomaly sizes (2 to 10). v3 copies a **real** source event instead: its length, how many channels it touched, and how big the change was on each of those channels.

Amplitude of a real event on one channel:

$$
A = \max_{s \le t < e} \big| z_t - z_{s-1} \big|
$$

- z = normalised value (source normaliser, fitted on source train)
- s, e = event start and end steps
- z_{s−1} = the value just before the event starts

## Code, key lines (`src/inject.py`)
```python
for _, g in tr.groupby("ID"):                                      # one real source event
    dur = ceil((end - start) / 5 min), capped at 3 days
    amps.append(max(abs(Z[a:b, j] - Z[a-1, j]).max(), 0.05))       # its size on channel j (floor 0.05)
...
L, amps = prior["events"][rng.integers(len(prior["events"]))]      # pick a real event to replay
A = float(amps[n_j])                                               # n-th fake channel gets the n-th real size
```

## Results (dev check = target training labels; diagnosis only)
| Pair | Old transfer | v1 | v2 | **v3** | Upper bound |
|---|---|---|---|---|---|
| M1→M2 | 0.242 | 0.129 | 0.129 | **0.380** | 0.860 |
| M2→M1 | 0.217 | 0.400 | 0.507 | **0.507** | 0.503 |
| M1→M1, label-free | — | 0.387 | 0.478 | 0.478 | 0.573 |
| M2→M2, label-free | — | 0.425 | 0.230 | 0.299 | 0.861 |

**Both transfer directions now beat the old transfer**: M1→M2 0.242 → 0.380, and M2→M1 0.217 → 0.507.
Targets on M2 are still far below the upper bound.

## Allowed validation (source labels only)
| | N̂ v2 | N̂ v3 | True |
|---|---|---|---|
| M1 (M1→M1) | 25.8 | 33.5 | 52.5 |
| M2 (M2→M2) | 53.9 | 34.5 | 141.0 |

## Diagnosis: fakes are still too easy, especially M2-style ones
Recall at the chosen setting:

| Pair | On fakes | On real events |
|---|---|---|
| M1→M1 | 0.38 | 0.21 |
| M1→M2 | 0.23 | 0.13 |
| M2→M1 | 0.78 | 0.24 |
| M2→M2 | 0.66 | **0.09** |

Likely cause (**hypothesis, not yet verified**): our size measure |z − value before| ignores how much that channel normally moves. Many M2 events have size about 20, but probably on channels that swing that much in normal operation too. Pasted onto a random quiet channel, the same size becomes an obvious anomaly. Also, our fake *types* are random, while real events probably follow specific shapes.

## Self-test
1. What is the one thing v3 changed compared with v2?
2. Why does "fakes easier than real events" make N̂ too small?
3. Why could a size of 20 be normal on one channel but huge on another?
