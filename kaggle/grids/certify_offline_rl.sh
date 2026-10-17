#!/usr/bin/env bash
# Certify the offline-RL grid arms on the fixed scripted pool (paired seeds, strict CI).
#
# Pure-policy mode only (PLAN §6.3): each arm is a `neural:<ckpt>`; the baseline is
# the same-deck scripted expert, so the report carries both the pooled rate and the
# paired Δ (non-inferiority) that the data-first reports use.
#
# Usage: bash kaggle/grids/certify_offline_rl.sh [tag_prefix] [seeds]
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DECK="${DECK:-superconduct_aggro}"
TAG_PREFIX="${1:-orl}"
SEEDS="${2:-10}"
WORKERS="${WORKERS:-12}"
ARMS="${ARMS:-bc_e15 awr_b1_e15 awr_b2_e15 q_disc_e15}"

for name in $ARMS; do
  ckpt="data/checkpoints/${TAG_PREFIX}_${name}.pt"
  [ -f "$ckpt" ] || { echo "=== skip $name (no $ckpt)"; continue; }
  echo "=== $(date -Is) certifying $name ($DECK, seeds=$SEEDS)"
  "$PY" -m scripts.eval_protocol \
    --contestant "neural:${ckpt}" --contestant-deck "$DECK" \
    --seeds "$SEEDS" --workers "$WORKERS" --strict-ci \
    --baseline "expert:${DECK}" --tag "${TAG_PREFIX}_${name}"
done
echo "=== $(date -Is) certification done"
