#!/usr/bin/env bash
# Milestone: certify the single 20-deck model per deck against that deck's scripted expert.
# Usage: bash kaggle/grids/certify_all_decks.sh <ckpt> <tag> [deck ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
CKPT="${1:?checkpoint}"; TAG="${2:?tag}"; shift 2
DECKS="${*:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia klee_mavuika_mualani superconduct_mika_attack aywolfskirk nahida_kokomi_raiden}"
WORKERS="${WORKERS:-12}"; SEEDS="${SEEDS:-10}"
for deck in $DECKS; do
  echo "=== $(date -Is) $TAG on $deck"
  "$PY" -m scripts.eval_protocol --contestant "neural:${CKPT}" --contestant-deck "$deck" \
    --seeds "$SEEDS" --workers "$WORKERS" --strict-ci --baseline "expert:${deck}" \
    --tag "${TAG}_${deck}"
done
echo "=== $(date -Is) all-deck certification done"
