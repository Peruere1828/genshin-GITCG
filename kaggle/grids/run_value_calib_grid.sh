#!/usr/bin/env bash
# H3: does a different value objective calibrate better? (same store/seed/budget as orl_bc_e15)
#
# Arms: identity+mse (baseline, already in data/checkpoints/orl_bc_e15.pt),
# sigmoid+brier (proper score), sigmoid+bce (harder miss penalty),
# identity+mse with target smoothing (a cheap over-confidence fix).
# Measure with: python -m scripts.eval_value_calibration --store data/store/sa_all.pt ...
#
# Usage: setsid nohup bash kaggle/grids/run_value_calib_grid.sh > data/logs/value_calib_grid.log 2>&1 &
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
THREADS="${THREADS:-4}"
EPOCHS="${EPOCHS:-15}"
LIMIT="${LIMIT:-30000}"

run() {
  local name="$1"; shift
  echo "=== $(date -Is) $name: $*"
  "$PY" -m scripts.datafirst_bc_train --store-cache data/store/sa_all.pt \
    --d-model 256 --batch-size 256 --epochs "$EPOCHS" \
    --train-limit "$LIMIT" --val-limit 4000 --threads "$THREADS" \
    --out "data/checkpoints/vc_${name}.pt" --report "reports/train/vc_${name}.json" "$@"
  echo "=== $(date -Is) done $name"
}

run sig_brier --value-activation sigmoid --value-loss brier
run sig_bce --value-activation sigmoid --value-loss bce
run smooth --value-smooth 0.05
echo "=== $(date -Is) value-calibration grid done"
