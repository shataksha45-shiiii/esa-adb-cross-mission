#!/bin/bash
# One command per result. Usage:  bash reproduce.sh <stage> [<stage> ...]   or   bash reproduce.sh all
# Stages run in the order below; each later stage needs the outputs of the earlier ones.
# Runtimes are for an Apple-silicon laptop (torch on "mps"). Seeds are fixed inside the scripts.
#
#   stage       produces                                                    about
#   test        unit tests (metric = official code, online rules, leaks)    10 s
#   preprocess  cache/M{1,2}/ 5-min grid                                    one-off, ~1 h
#   step8       results/tune/M*_causal_*.csv (settings), final_step8.csv    ~15 min   (README 5.1 "source-tuned")
#   step9       results/inject*/ (v1, v3, v4), dev_step9_v4.csv, final_step9.csv     ~4 h  (README 5.1, 5.2)
#   step10      results/inject_v5/, dev_step10.csv, final_step10.csv       ~2 h      (README 5.3)
#   step12      repro_globalstd.csv, repro_globalstd_native_M2.csv         ~10 min   (README 7, notes/step12)
#   step13      inject_v5_win15/60, inject_v3_floor02, step13_*.csv        ~4 h      (README 5.4)
#   step14      step14_metrics.csv                                         ~45 min   (README 5.5)
#   figures     figures/fig1-fig8 (.png, step14 figures also .pdf)         ~5 min
#
# Test labels are scored by step8, step9, step10, step12, step13, step14 (see the honesty ledger in README 7).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
PY="python"

pairs() { for p in 12 21 11 22; do echo "${p:0:1} ${p:1:1}"; done; }
inject() {   # inject <version> <outdir> [extra args...]
  local v=$1 out=$2; shift 2
  mkdir -p "results/$out"
  pairs | while read -r s t; do
    $PY src/inject.py --source "$s" --target "$t" --version "$v" --out "$out" "$@" > "results/$out/S${s}_T${t}_log.txt"
  done
}

stage_test()       { $PY -m pytest -q tests; }
stage_preprocess() { $PY src/preprocess.py; }
stage_step8() {
  for m in 1 2; do
    $PY src/tune.py --mission $m --tag causal_base12 --det '{"kind":"maxabs","w":12}'
    $PY src/tune.py --mission $m --tag causal_base48 --det '{"kind":"maxabs","w":48}'
    $PY src/tune.py --mission $m --tag causal_fuse \
      --det '{"kind":"fuse","parts":[{"kind":"ae","w":24,"hidden":64,"code":8,"reduce":"max","center":true},{"kind":"maxabs","w":12}]}'
  done
  # the best rows printed above are the settings hard-coded in src/run_final.py (CFG, BASE)
  $PY src/run_final.py | tee results/final_step8_log.txt
}
stage_step9() {
  inject 1 inject
  inject 3 inject_v3
  inject 4 inject_v4
  $PY src/run_step9.py --dev  | tee results/dev_step9_v4_log.txt
  $PY src/run_step9.py --test | tee results/final_step9_log.txt
}
stage_step10() {
  inject 5 inject_v5
  $PY src/step10.py --dev  | tee results/dev_step10_log.txt
  $PY src/step10.py --test | tee results/final_step10_log.txt
}
stage_step12() {
  $PY src/repro_globalstd.py
  $PY src/repro_globalstd_native.py | tee results/repro_globalstd_native_M2_log.txt
}
stage_step13() {
  inject 5 inject_v5_win15 --roll-win 15
  inject 5 inject_v5_win60 --roll-win 60
  inject 3 inject_v3_floor02 --amp-floor 0.2
  $PY src/step13.py --run       | tee results/step13_run_log.txt
  $PY src/step13.py --bootstrap | tee results/step13_ci_log.txt
}
stage_step14() { $PY src/step14.py | tee results/step14_log.txt; }
stage_figures() { $PY scripts/make_figures.py; $PY scripts/make_figures_step14.py; }

ALL="test preprocess step8 step9 step10 step12 step13 step14 figures"
[ $# -eq 0 ] && { sed -n '2,20p' "$0"; exit 1; }
[ "$1" = "all" ] && set -- $ALL
for st in "$@"; do
  echo "=== $st ==="
  "stage_$st"
done
