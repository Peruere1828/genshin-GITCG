#!/usr/bin/env bash
# H3 gate test: does a *calibrated* value head make MCCFR search stop being harmful?
#
# Same protocol/budget as A4's cheap ladder row (i8/d1/c4/ms8, 6-opponent subset,
# 6 seeds = 72 paired games, ~0.44 s/search in A4) so the numbers are directly
# comparable with reports/search/a4_search_budget_ladder.md:
#   A4 expert (SUB, n=36): 0.431 | A4 cfr d1: Δ=-0.042 (ns) | A4 full pool: significantly negative
#
# Arms (each writes reports/search/h3_*.json + data/search/*.jsonl, resumable):
#   1. expert base + calibrated value frontier  (--value-spec)
#   2. net base + calibrated value frontier     (--checkpoint + --value-spec)
#
# Usage: bash kaggle/grids/run_h3_search_ladder.sh <value-ckpt> [tag-prefix]
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
VALUE_CKPT="${1:-data/checkpoints/vc_sig_brier.pt}"
TAG_PREFIX="${2:-h3}"
SUB="${SUB:-unyielding_geo double_geo_navia electrocharged_moon dvalin_bonk nahida_kokomi_raiden natlan_battleship}"
COMMON="--opponents $SUB --seeds 6 --iterations 8 --depth 1 --candidate-limit 4 --max-searches 8 --workers 6 --bridge-workers 1"

echo "=== $(date -Is) expert base + value frontier ($VALUE_CKPT)"
$PY -m scripts.run_cfr_ladder $COMMON --value-spec "neural:${VALUE_CKPT}" \
  --tag "${TAG_PREFIX}_expert_val"

echo "=== $(date -Is) net base + value frontier"
$PY -m scripts.run_cfr_ladder $COMMON --checkpoint "$VALUE_CKPT" --value-spec "neural:${VALUE_CKPT}" \
  --tag "${TAG_PREFIX}_net_val"

echo "=== $(date -Is) done"

echo "=== $(date -Is) deeper budget (d2): does the value frontier matter at all?"
$PY -m scripts.run_cfr_ladder --opponents $SUB --seeds 6 --iterations 8 --depth 2 --candidate-limit 4 \
  --max-searches 4 --workers 6 --bridge-workers 1 \
  --value-spec "neural:${VALUE_CKPT}" --tag "${TAG_PREFIX}_expert_val_d2"
echo "=== $(date -Is) d2 done"
