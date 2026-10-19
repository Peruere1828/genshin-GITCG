#!/usr/bin/env bash
# Validate the per-deck checkpoint selection on FRESH seeds (no selection bias).
#
# The v1/v2 mixed-corpus arms were compared on seeds 0-9, and which arm wins differs
# by deck (v1 wins double_geo 0.7250 vs 0.6587; v2 wins dvalin 0.7575 vs 0.6438 and
# unyielding 0.8225 vs 0.7837; SA is a tie). That selection must be re-verified on
# seeds nobody looked at, otherwise it is just overfitting the report table.
#
# Usage: bash kaggle/grids/certify_selected_fresh_seeds.sh [seed-start] [seeds]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
SEED_START="${1:-50}"; SEEDS="${2:-10}"; WORKERS="${WORKERS:-10}"
declare -a JOBS=(
  "v1:data/checkpoints/h4_mixed_d256_e15.pt:superconduct_aggro double_geo_navia"
  "v2:data/checkpoints/kaggle/h4m_bc_d256_e150.pt:dvalin_bonk unyielding_geo"
)
for job in "${JOBS[@]}"; do
  tag="${job%%:*}"; rest="${job#*:}"; ckpt="${rest%%:*}"; decks="${rest#*:}"
  for deck in $decks; do
    echo "=== $(date -Is) selection $tag on $deck (seeds $SEED_START..$((SEED_START + SEEDS - 1)))"
    "$PY" -m scripts.eval_protocol --contestant "neural:${ckpt}" --contestant-deck "$deck" \
      --seeds "$SEEDS" --seed-start "$SEED_START" --workers "$WORKERS" --strict-ci \
      --baseline "expert:${deck}" --tag "sel_fresh_${tag}_${deck}"
  done
done
echo "=== $(date -Is) fresh-seed selection validation done"
