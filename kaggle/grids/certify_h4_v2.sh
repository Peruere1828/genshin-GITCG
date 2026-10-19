#!/usr/bin/env bash
# H4 v2: does the training budget help on the mixed corpus (4-deck + targeted)?
# v1 (200k/15ep) reached per-deck parity; these arms use the full 542k for 50/150 ep.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
WORKERS="${WORKERS:-6}"; SEEDS="${SEEDS:-10}"
declare -a JOBS=(
  "h4m_e50:data/checkpoints/kaggle/h4m_bc_d256_e50.pt:superconduct_aggro dvalin_bonk"
  "h4m_e150:data/checkpoints/kaggle/h4m_bc_d256_e150.pt:superconduct_aggro dvalin_bonk"
)
for job in "${JOBS[@]}"; do
  tag="${job%%:*}"; rest="${job#*:}"; ckpt="${rest%%:*}"; decks="${rest#*:}"
  for deck in $decks; do
    echo "=== $(date -Is) $tag on $deck"
    "$PY" -m scripts.eval_protocol --contestant "neural:${ckpt}" --contestant-deck "$deck" \
      --seeds "$SEEDS" --workers "$WORKERS" --strict-ci --baseline "expert:${deck}" \
      --tag "h4v2_${tag}_${deck}"
  done
done
echo "=== $(date -Is) H4 v2 certification done"
