#!/usr/bin/env bash
# Re-run a round's policy-gradient update on already-collected data (bug-fix rerun).
#
# Round 1's first attempt produced NaN on the first batch (advantage normalisation
# divided by a ~0 std) and its certification at seeds 70-79 measured the destroyed
# network. This reruns the update with the guarded normaliser + gradient clipping and
# certifies on *never-used* seeds (80-89), keeping the pre-registration honest.
#
# Usage: bash kaggle/grids/rerun_pg_round.sh <store> <tag> <eval-seed-start> <init-ckpt> [decks...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
STORE="${1:?store}"; TAG="${2:?tag}"; EVAL_SEED_START="${3:?eval seed start}"; INIT="${4:?init checkpoint}"; shift 4
DECKS="${*:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia}"
EPOCHS="${EPOCHS:-2}"; LR="${LR:-1e-4}"; GRAD_CLIP="${GRAD_CLIP:-1.0}"; WORKERS="${WORKERS:-12}"
LOSS="${LOSS:-pg}"; EXTRA_ARGS="${EXTRA_ARGS:-}"
CKPT="data/checkpoints/${TAG}.pt"

echo "=== $(date -Is) update $TAG (loss=$LOSS store=$STORE init=$INIT ep=$EPOCHS lr=$LR clip=$GRAD_CLIP $EXTRA_ARGS)"
"$PY" -m scripts.datafirst_bc_train --store-cache "$STORE" --init-from "$INIT" \
  --d-model 256 --epochs "$EPOCHS" --lr "$LR" --batch-size 256 --threads 10 --seed 0 \
  --loss "$LOSS" --pg-entropy 0.01 --grad-clip "$GRAD_CLIP" $EXTRA_ARGS \
  --out "$CKPT" --report "reports/train/${TAG}.json"

for deck in $DECKS; do
  echo "=== $(date -Is) certify $TAG on $deck (seeds $EVAL_SEED_START+)"
  "$PY" -m scripts.eval_protocol --contestant "neural:${CKPT}" --contestant-deck "$deck" \
    --seeds 10 --seed-start "$EVAL_SEED_START" --workers "$WORKERS" --strict-ci \
    --baseline "expert:${deck}" --tag "${TAG}_${deck}"
done
echo "=== $(date -Is) $TAG done"
