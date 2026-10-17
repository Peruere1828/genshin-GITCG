#!/usr/bin/env bash
# Certify the Kaggle-trained full-corpus BC checkpoints on the fixed pool (H1 axis).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DECK="${DECK:-superconduct_aggro}"; SEEDS="${SEEDS:-10}"; WORKERS="${WORKERS:-12}"
for spec in "kg_bc_e15:data/checkpoints/kaggle/sa_bc_d256_e15.pt" "kg_bc_e50:data/checkpoints/kaggle/sa_bc_d256_e50.pt"; do
  tag="${spec%%:*}"; ckpt="${spec#*:}"
  echo "=== $(date -Is) certifying $tag"
  "$PY" -m scripts.eval_protocol --contestant "neural:${ckpt}" --contestant-deck "$DECK" \
    --seeds "$SEEDS" --workers "$WORKERS" --strict-ci --baseline "expert:${DECK}" --tag "$tag"
done
echo "=== $(date -Is) done"
