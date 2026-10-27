#!/usr/bin/env bash
# Single-deck exploitation round (used for double_geo, the one deck where on-policy
# fine-tuning has replicated a significant gain three times on fresh seeds:
# +0.100 (110-119), +0.058 (130-139), +0.070 (140-149)).
#
# Round 3 asks whether the gain keeps growing with another iteration or has saturated.
# Pre-registered: certify on seeds 150-159 (never used); accept iff CI_low >= -0.02
# and the point estimate > 0.
#
# Usage: bash kaggle/grids/run_single_deck_rl.sh <deck> <tag> <teacher-ckpt> <seed-start> <seed-count> <eval-seed-start>
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DECK="${1:?deck}"; TAG="${2:?tag}"; TEACHER="${3:?teacher}"; SEED_START="${4:?seed start}"
SEED_COUNT="${5:?seed count}"; EVAL_SEED_START="${6:?eval seed start}"
WORKERS="${WORKERS:-10}"; EPOCHS="${EPOCHS:-1}"; LR="${LR:-3e-5}"
# A2: keep the identity branch when the teacher/base checkpoint has it (0 = legacy).
ID_BUCKETS="${ID_BUCKETS:-0}"; ID_EMB_DIM="${ID_EMB_DIM:-32}"

echo "=== $(date -Is) collect $DECK ($SEED_COUNT seeds, teacher=$TEACHER)"
"$PY" -m train.collect --deck "$DECK" --all-opponents \
  --seed-start "$SEED_START" --seed-count "$SEED_COUNT" --workers "$WORKERS" \
  --teacher-spec "neural:${TEACHER}#t=1.0" --tag "${TAG}_${DECK}" --format compact --no-privileged-state

echo "=== $(date -Is) store + PG for $DECK"
"$PY" -m scripts.datafirst_bc_store --replays "data/replays/${TAG}_${DECK}" \
  --store-cache "data/store/${TAG}_${DECK}.pt" --val-every 10 \
  --report "reports/train/${TAG}_${DECK}_store.json"
"$PY" -m scripts.datafirst_bc_train --store-cache "data/store/${TAG}_${DECK}.pt" --init-from "$TEACHER" \
  --d-model 256 --epochs "$EPOCHS" --lr "$LR" --batch-size 256 --threads 10 --seed 0 \
  --id-buckets "$ID_BUCKETS" --id-emb-dim "$ID_EMB_DIM" \
  --loss pg --pg-entropy 0.01 --grad-clip 1.0 \
  --out "data/checkpoints/${TAG}_${DECK}.pt" --report "reports/train/${TAG}_${DECK}.json"

echo "=== $(date -Is) certify $DECK on seeds $EVAL_SEED_START+"
"$PY" -m scripts.eval_protocol --contestant "neural:data/checkpoints/${TAG}_${DECK}.pt" \
  --contestant-deck "$DECK" --seeds 10 --seed-start "$EVAL_SEED_START" --workers 12 --strict-ci \
  --baseline "expert:${DECK}" --tag "${TAG}_${DECK}"
echo "=== $(date -Is) done"
