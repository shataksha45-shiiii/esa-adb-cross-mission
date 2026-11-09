# Step 9a: re-run of the Step 7b diagnosis with the online event builder (9 Nov 2026)

## What we did
`src/diag_transfer.py` asks: does transfer fail because of the **model** or the **settings**?
It still used the old offline event builder, which used future data. We swapped one import:

```python
from tune import build_events_causal as build_events   # online builder: no future samples
```

Everything else is unchanged. The old outputs were kept as `results/diag_transfer_offline_step7b.csv` and `results/diag_transfer_offline_step7b_log.txt`.
New outputs: `results/diag_transfer.csv` and `results/diag_transfer_log.txt`.
Evaluation uses the target's **training** labels (2-fold CV), 2 seeds. This is diagnosis only; no setting gets picked from it.

## Result (mean CV F0.5)

| Target | Model | Settings | Online (new) | Offline (old) |
|---|---|---|---|---|
| M1 | own | own (M1) | **0.554** | 0.781 |
| M1 | from M2 | own (M1) | **0.473** | 0.640 |
| M1 | from M2 | M2's (real transfer) | 0.212 | 0.220 |
| M2 | own | own (M2) | **0.858** | 0.873 |
| M2 | from M1 | own (M2) | **0.860** | 0.871 |
| M2 | from M1 | M1's (real transfer) | 0.248 | 0.186 |

## What it means (plain words)
- **On M2 the model transfers perfectly.** The M1-trained model scores 0.860 with good settings, and M2's own model scores 0.858.
- **On M1 the model transfers partly.** The M2 model loses about 0.08 (0.554 → 0.473). The settings lose much more (0.473 → 0.212).
- The Step 7b conclusion still holds online: **settings are the main problem**, which is why Step 9b calibrates the settings.
- Going online lowered M1 a lot (0.78 → 0.55) and M2 barely (0.87 → 0.86). This matches what Step 8 found.

## Self-test
1. Which row proves the model transfers to M2?
2. On M1, how much of the transfer loss is model and how much is settings?
3. Why is it fine to use the target's training labels here but not to pick settings?
