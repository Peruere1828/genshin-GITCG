#!/usr/bin/env bash
# H1/H2 at scale: certify the 150-epoch full-corpus Kaggle arms on the fixed pool.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DECK="${DECK:-superconduct_aggro}"; SEEDS="${SEEDS:-10}"; WORKERS="${WORKERS:-12}"
for n in sa_bc_d256_e150 sa_awr_b05_e150 sa_awr_b1_e150 sa_awr_b2_e150 sa_q_disc_e150; do
  echo "=== $(date -Is) certifying $n"
  "$PY" -m scripts.eval_protocol --contestant "neural:data/checkpoints/kaggle/${n}.pt" \
    --contestant-deck "$DECK" --seeds "$SEEDS" --workers "$WORKERS" --strict-ci \
    --baseline "expert:${DECK}" --tag "kg150_${n}"
done
echo "=== $(date -Is) done"
