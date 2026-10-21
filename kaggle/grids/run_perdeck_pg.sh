#!/usr/bin/env bash
# Per-deck exploitation round: fine-tune the shared anchor on ONE deck's self-play,
# with the on-policy objective, then certify that deck only.
#
# Motivated by round 1e: one shared PG step on 4 decks helped double_geo (+0.100,
# CI [+0.047,+0.153]) but hurt dvalin (-0.062) and superconduct (-0.146) on fresh
# seeds — i.e. the shared trunk cannot move per-deck without cross-deck interference.
# Splitting the update per deck removes that interference at the cost of one model
# per deck (deck-level routing is legitimate; matchup-level routing was noise).
#
# Pre-registered: certify each deck on seeds 130-139 (never used), paired vs that
# deck's expert; accept a deck if CI_low >= -0.02 and the point estimate > 0.
#
# Usage: bash kaggle/grids/run_perdeck_pg.sh <round-tag> <init-ckpt> [decks...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
TAG="${1:?round tag}"; INIT="${2:?anchor checkpoint}"; shift 2
DECKS="${*:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia}"
EPOCHS="${EPOCHS:-1}"; LR="${LR:-3e-5}"; EVAL_SEED_START="${EVAL_SEED_START:-130}"
REPLAY_TAG="${REPLAY_TAG:-$TAG}"   # data dir prefix (may differ from the artifact tag)

for deck in $DECKS; do
  replayed="data/replays/${REPLAY_TAG}_${deck}"
  [ -d "$replayed" ] || { echo "=== skip $deck (no $replayed)"; continue; }
  store="data/store/${TAG}_${deck}.pt"
  ckpt="data/checkpoints/${TAG}_${deck}.pt"
  echo "=== $(date -Is) store $deck"
  "$PY" -m scripts.datafirst_bc_store --replays "$replayed" --store-cache "$store" \
    --val-every 10 --report "reports/train/${TAG}_${deck}_store.json"
  echo "=== $(date -Is) per-deck update $deck (loss=pg ep=$EPOCHS lr=$LR)"
  "$PY" -m scripts.datafirst_bc_train --store-cache "$store" --init-from "$INIT" \
    --d-model 256 --epochs "$EPOCHS" --lr "$LR" --batch-size 256 --threads 10 --seed 0 \
    --loss pg --pg-entropy 0.01 --grad-clip 1.0 \
    --out "$ckpt" --report "reports/train/${TAG}_${deck}.json"
  echo "=== $(date -Is) certify $deck"
  "$PY" -m scripts.eval_protocol --contestant "neural:${ckpt}" --contestant-deck "$deck" \
    --seeds 10 --seed-start "$EVAL_SEED_START" --workers 12 --strict-ci \
    --baseline "expert:${deck}" --tag "${TAG}_${deck}"
done
echo "=== $(date -Is) per-deck round done"
