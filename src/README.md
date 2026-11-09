# src/: what each file is for

**Current pipeline** (used by the results in the main README):

| File | Role | Introduced |
|---|---|---|
| `loader.py` | read one raw ESA-ADB channel | Step 1 |
| `preprocess.py` | raw channel → 5-min grid (zero-order hold), cached | Step 1 |
| `data.py` | load a mission as one matrix; labels; train/test split; bins labelled by their END (online rule) | Step 1, 8 |
| `normaliser.py` | per-channel robust normaliser, fitted on training data only | Step 3 |
| `evaluator.py` | corrected event-wise F0.5 (readable version) + wrapper around the official ESA-ADB code | Step 2 |
| `tune.py` | AE and \|z\| scorers, EVT threshold, **causal event builder**, fast evaluator, blocked-CV tuning bench | Step 7, 8 |
| `run_final.py` | source-tuned transfer table; holds the tuned settings `CFG`, `BASE` | Step 8 |
| `diag_transfer.py` | model-vs-settings diagnosis; `pred_from` (fused alarm with the causal builder) | Step 7, 9 |
| `adapt_stats.py` | alarm-rate matching diagnostic | Step 8, 9 |
| `inject.py` | label-free calibration by synthetic injection (`--version 1, 3, 4, 5`; `--roll-win`, `--amp-floor`) | Step 9, 10, 13 |
| `inject_v2.py` | hidden-event estimate N̂ and the v2/v3 selection rule | Step 9 |
| `run_step9.py` | Step 9 dev check and final test | Step 9 |
| `rolling.py` | rolling (causal) re-centring / robust z | Step 10 |
| `step10.py` | baselines (fixed default, Heydari-style, \|z\|) + fake-recall matching; dev + test | Step 10 |
| `repro_globalstd.py`, `repro_globalstd_native.py` | official GlobalSTD baseline in our pipeline, at 5 min and at the native 18 s | Step 11, 12 |
| `step13.py` | ablations, month-block bootstrap CIs, sensitivity | Step 13 |
| `step14.py` | all official metrics + operator view | Step 14 |

**Learning files and the early offline pipeline (kept on purpose):**

| File | Status |
|---|---|
| `detector.py`, `events.py`, `threshold.py` | First versions from Steps 4–5. Shataksha re-types the core ideas here (CLAUDE plan), so they stay. `events.build_events` is the **offline** builder (it looks at future samples). Never use it for reported numbers; use `tune.build_events_causal`. |
| `run_transfer.py` | Step 6 first transfer table (offline builder). **Superseded** by `run_final.py` (Step 8). Kept so that `results/transfer_seed*.csv` stay reproducible. |

Method versions v1–v4 are flags of `inject.py` (`--version`) and the rule in `inject_v2.py`. They are not separate files, so they stay where they are. Their history and why each was replaced: `notes/step9-design-frozen.md`.
