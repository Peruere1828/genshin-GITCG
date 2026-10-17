#!/usr/bin/env bash
# H4: certify the multi-deck (mixed + targeted) model per deck on the fixed pool.
#
# The floor gate (M4: no opponent < 45%) is per (deck, opponent), so a multi-deck
# model must be reported per deck and compared against that deck's scripted expert
# (paired, same seeds). Reports land in reports/explore/h4_mixed_<deck>.json.
#
# Usage: bash kaggle/grids/certify_h4_mixed.sh <checkpoint> [seeds] [tag]
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
CKPT="${1:-data/checkpoints/h4_mixed_d256_e15.pt}"
SEEDS="${2:-10}"
TAG="${3:-h4_mixed}"
WORKERS="${WORKERS:-12}"
DECKS="${DECKS:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia}"

for deck in $DECKS; do
  echo "=== $(date -Is) certifying $TAG on $deck (seeds=$SEEDS)"
  "$PY" -m scripts.eval_protocol \
    --contestant "neural:${CKPT}" --contestant-deck "$deck" \
    --seeds "$SEEDS" --workers "$WORKERS" --strict-ci \
    --baseline "expert:${deck}" --tag "${TAG}_${deck}"
done
echo "=== $(date -Is) done"
