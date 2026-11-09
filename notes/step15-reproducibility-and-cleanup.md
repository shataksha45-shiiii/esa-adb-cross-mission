# Step 15: reproducibility and repo cleanup (roadmap items 11–15) (10 Nov 2026)

## What we did, and why
A reviewer, or an internship supervisor, should be able to rerun every number. A future version of us should be able to trust the code. So we added:

| Item | What | Files |
|---|---|---|
| 14 | **Unit tests** (69, about 2 s, no ESA data needed except one) | `tests/test_metric.py`, `tests/test_online_and_leaks.py` |
| 11 | **One command per result:** `bash reproduce.sh <stage>` | `reproduce.sh` |
| 13 | Pinned `requirements.txt` (incl. `portion` for the official metric, `pytest`); `setup_env.sh` installs from it; data guide; README "Reproduce" rewritten | `requirements.txt`, `setup_env.sh`, `DATA_README.md`, `README.md` |
| 12 | A map of every source file: role, the step that introduced it, superseded or not | `src/README.md` |
| 15 | MIT licence checked; `CITATION.cff` updated (version 0.3.0, repo URL, keywords, Heydari 2026 reference) | `LICENSE`, `CITATION.cff` |

## The tests (plain words)
1. **The metric is right.** On random synthetic labels and alarms:
   - our readable evaluator = the **official ESA-ADB code** (5 cases, all events and anomalies only, to 1e-12);
   - the fast evaluator used for tuning = the readable one (8 cases);
   - the per-month split used for the bootstrap CIs = the readable one (8 cases).
2. **The online rule holds.**
   - Event builder: cutting off the future never changes past output (40 cuts × 10 seeds × 4 settings).
   - Event builder: an alarm never starts before `min_len` consecutive exceedances have been seen.
   - Rolling normalisation: changing future values never changes past outputs.
3. **No leaks.** Changing the test period does not change the normaliser's training output.
4. **Grid timing.** Every 5-min value is labelled by the end of its bin (needs the cache).

Run: `bash reproduce.sh test`, or `python -m pytest -q tests`.

## What broke while writing the tests
The first run of `test_fast_ew…` failed: TP 0 vs 2. The cause was **time units**.
- In pandas 3, `pd.date_range("2005-01-01", …)` uses **microseconds**.
- Our evaluators convert times with `.astype(int64)` and assume **nanoseconds**.

The real data was never affected: `data.load_mission` builds its grid from an ns timestamp, and its dtype is `datetime64[ns]`, as checked. Fixes:
- the test now builds its grid in ns;
- `data.py` now **asserts** ns, so this can never fail silently in the pipeline.

## A deliberate deviation from the roadmap (item 12)
The roadmap said "move v1–v4 into `archive/`". We did **not** move files:
- v1–v4 are flags of `inject.py` plus the rule in `inject_v2.py`, not separate files;
- the Step 6 script (`run_transfer.py`) and the learning files (`detector.py`, `events.py`, `threshold.py`) are imported by other scripts, or are the files Shataksha will retype.

Moving them would break reproduction of the earlier notes. Instead, `src/README.md` marks what is current, what is superseded, and the one trap: the offline `events.build_events` must never be used for reported numbers.

## What is verified and what is not
- Verified: `bash reproduce.sh test` (69 passed); the script's syntax; the stage map.
- **Not verified:** a full end-to-end rerun (about 12 h). The stage commands are the ones that produced the committed results (same scripts and flags), but nobody has rerun them from a clean checkout yet. Worth doing once before the paper is submitted, ideally on a second machine.

## Self-test
1. Why test the evaluator against the official code on *synthetic* data, when we already matched it on real test data?
2. What does "prefix invariance" check about the event builder, and why does it matter for the ESA-ADB online rule?
3. Why did the microsecond/nanosecond bug not affect any reported number?
