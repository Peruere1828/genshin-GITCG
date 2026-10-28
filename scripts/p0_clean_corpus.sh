#!/usr/bin/env bash
# P0 clean corpus: re-collect the 4-deck A2 corpus with exact definition ids.
#
# The old dfbc_* compact corpus was written with fp16 definition columns, which
# is not reversible for six-digit card ids (A2 post-mortem). This script uses
# the current collector (exact int id fields) and builds a store whose meta
# records definition_ids_present, so an identity-embedding run can prove it is
# not training on a corrupted corpus.
#
# Usage:
#   PY=/path/to/conda/envs/gitcg/bin/python bash scripts/p0_clean_corpus.sh
#   DECKS="dvalin_bonk double_geo_navia" SEEDS=20 bash scripts/p0_clean_corpus.sh
set -euo pipefail

PY="${PY:-python}"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

DECKS="${DECKS:-dvalin_bonk unyielding_geo double_geo_navia natlan_battleship}"
SEEDS="${SEEDS:-20}"
WORKERS="${WORKERS:-8}"
TAG_PREFIX="${TAG_PREFIX:-p0_clean}"
STORE="${STORE:-data/store/p0_clean_four_decks.pt}"
REPLAY_DIRS=()

for deck in $DECKS; do
  tag="${TAG_PREFIX}_${deck}_c${SEEDS}"
  echo "[p0-clean] collect $deck -> $tag ($(date -Is))"
  "$PY" -m train.collect \
    --deck "$deck" --all-opponents \
    --seed-start 0 --seed-count "$SEEDS" --workers "$WORKERS" \
    --tag "$tag" --format compact --no-privileged-state
  REPLAY_DIRS+=("data/replays/$tag")
done

echo "[p0-clean] build store $STORE ($(date -Is))"
"$PY" -m scripts.datafirst_bc_store \
  --replays "${REPLAY_DIRS[@]}" \
  --store-cache "$STORE" \
  --val-every 10 \
  --report "${STORE%.pt}.json"

echo "[p0-clean] done; check meta.definition_ids_present == meta.events before id training"
