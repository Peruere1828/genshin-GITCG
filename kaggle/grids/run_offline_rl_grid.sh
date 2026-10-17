#!/usr/bin/env bash
# Local CPU A/B grid: BC vs advantage-weighted BC vs action value (H2 signal test).
#
# Same store, same seed, same subsample and epoch budget — only the policy target
# changes, so the run-to-run difference is attributable to the target. Output
# checkpoints go to data/checkpoints/ and reports to reports/train/; certify the
# winner with scripts/eval_protocol (engine CPU) once the grid finishes.
#
# Usage: setsid nohup bash kaggle/grids/run_offline_rl_grid.sh > data/logs/offline_rl_grid.log 2>&1 &
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
STORE="${STORE:-data/store/sa_all.pt}"
LIMIT="${LIMIT:-30000}"
VAL_LIMIT="${VAL_LIMIT:-4000}"
EPOCHS="${EPOCHS:-15}"
D_MODEL="${D_MODEL:-256}"
BATCH="${BATCH:-256}"
THREADS="${THREADS:-6}"
TAG="${TAG:-orl}"

run() {
  local name="$1"; shift
  echo "=== $(date -Is) $name: $*"
  "$PY" -m scripts.datafirst_bc_train --store-cache "$STORE" \
    --d-model "$D_MODEL" --batch-size "$BATCH" --epochs "$EPOCHS" \
    --train-limit "$LIMIT" --val-limit "$VAL_LIMIT" --threads "$THREADS" \
    --out "data/checkpoints/${TAG}_${name}.pt" \
    --report "reports/train/${TAG}_${name}.json" "$@"
  echo "=== $(date -Is) done $name"
}

run bc_e${EPOCHS}  --loss ce
run awr_b1_e${EPOCHS} --loss awr --awr-beta 1.0
run awr_b2_e${EPOCHS} --loss awr --awr-beta 2.0
run q_disc_e${EPOCHS} --loss q --q-target discount
echo "=== $(date -Is) offline-RL grid done"
