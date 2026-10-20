#!/usr/bin/env bash
# Online RL round: self-play collection -> policy gradient update -> fresh-seed cert.
#
# Implements the on-policy half of the mentor guidance (2026-10-20): treat the
# deterministic scripts as the environment and optimise win rate directly, instead of
# imitating the expert (`--loss pg` = REINFORCE with the value head as baseline).
#
# PRE-REGISTERED ACCEPTANCE (fixed before looking at any result):
#   * evaluation seeds are disjoint from every seed used before (rl round r uses
#     --seed-start 60+10*r on the protocol, collection uses 4000+1000*r);
#   * accept the round only if, on those fresh seeds, the paired Δ vs the same-deck
#     scripted expert has CI_low >= -0.02 AND point estimate > 0 for the deck set;
#   * any deck that regresses > 2pp is reported as a regression (no cherry-picking).
#
# Usage:
#   bash kaggle/grids/run_online_rl_round.sh <round> <start-ckpt> [decks...]
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
ROUND="${1:?round index (1,2,...)}"
START_CKPT="${2:?starting checkpoint}"
shift 2
DECKS="${*:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia}"
TEMP="${TEMP:-1.0}"
SEED_COUNT="${SEED_COUNT:-25}"
COLLECT_WORKERS="${COLLECT_WORKERS:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-12}"
EVAL_SEEDS="${EVAL_SEEDS:-10}"
PG_EPOCHS="${PG_EPOCHS:-2}"
PG_LR="${PG_LR:-3e-4}"
COLLECT_SEED_START=$((4000 + 1000 * ROUND))
EVAL_SEED_START=$((60 + 10 * ROUND))
TAG="rl_r${ROUND}"
CKPT_OUT="data/checkpoints/${TAG}.pt"

echo "=== $(date -Is) ROUND $ROUND (collect seeds $COLLECT_SEED_START+, eval seeds $EVAL_SEED_START+)"
echo "=== pre-registered accept rule: paired Δ CI_low >= -0.02 AND point estimate > 0 on fresh seeds"

REPLAY_DIRS=""
for deck in $DECKS; do
  echo "=== $(date -Is) collect: $deck vs pool (teacher=${START_CKPT}#t=${TEMP})"
  "$PY" -m train.collect --deck "$deck" --all-opponents \
    --seed-start "$COLLECT_SEED_START" --seed-count "$SEED_COUNT" --workers "$COLLECT_WORKERS" \
    --teacher-spec "neural:${START_CKPT}#t=${TEMP}" \
    --tag "${TAG}_${deck}" --format compact --no-privileged-state
  REPLAY_DIRS="$REPLAY_DIRS data/replays/${TAG}_${deck}"
done

echo "=== $(date -Is) build store"
"$PY" -m scripts.datafirst_bc_store --replays $REPLAY_DIRS \
  --store-cache "data/store/${TAG}.pt" --val-every 10 \
  --report "reports/train/${TAG}_store.json"

echo "=== $(date -Is) policy-gradient update (${PG_EPOCHS} ep, lr ${PG_LR})"
"$PY" -m scripts.datafirst_bc_train --store-cache "data/store/${TAG}.pt" \
  --d-model 256 --epochs "$PG_EPOCHS" --lr "$PG_LR" --batch-size 256 --threads 10 --seed 0 \
  --loss pg --pg-entropy 0.01 \
  --out "$CKPT_OUT" --report "reports/train/${TAG}.json"

for deck in $DECKS; do
  echo "=== $(date -Is) certify $TAG on $deck (fresh seeds $EVAL_SEED_START+)"
  "$PY" -m scripts.eval_protocol --contestant "neural:${CKPT_OUT}" --contestant-deck "$deck" \
    --seeds "$EVAL_SEEDS" --seed-start "$EVAL_SEED_START" --workers "$EVAL_WORKERS" --strict-ci \
    --baseline "expert:${deck}" --tag "${TAG}_${deck}"
done
echo "=== $(date -Is) round $ROUND done (checkpoint: $CKPT_OUT)"
