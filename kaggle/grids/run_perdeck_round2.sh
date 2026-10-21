#!/usr/bin/env bash
# Per-deck exploitation, round 2: 4x the data and iterate from each deck's best policy.
#
# Round 1 (rl_pd1) gave the first *replicated* improvement over a scripted expert:
# double_geo +0.100 [+0.047,+0.153] on seeds 110-119 and +0.0575 [+0.008,+0.107] on
# seeds 130-139 (never used before), plus unyielding at parity with **0** matchups
# below 45%. SA/dvalin regressed, so round 2 (a) collects 4x more games per deck
# (100 seeds instead of 25) to cut PG variance, and (b) samples from each deck's
# current best checkpoint, not the shared anchor.
#
# Pre-registered: certify on seeds 140-149; accept a deck iff CI_low >= -0.02 and the
# point estimate > 0. Any deck that regresses >2pp is reported.
#
# Usage: bash kaggle/grids/run_perdeck_round2.sh [anchor]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
ANCHOR="${1:-data/checkpoints/h4_mixed_d256_e15.pt}"
DECKS="${DECKS:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia}"
SEED_START="${SEED_START:-6000}"
SEED_COUNT="${SEED_COUNT:-100}"
WORKERS="${WORKERS:-10}"
TAG="rl_r2"

for deck in $DECKS; do
  teacher="$ANCHOR"
  best="data/checkpoints/rl_pd1_${deck}.pt"
  [ -f "$best" ] && teacher="$best"
  echo "=== $(date -Is) collect $deck (teacher=$teacher, seeds $SEED_START+$SEED_COUNT)"
  "$PY" -m train.collect --deck "$deck" --all-opponents \
    --seed-start "$SEED_START" --seed-count "$SEED_COUNT" --workers "$WORKERS" \
    --teacher-spec "neural:${teacher}#t=1.0" \
    --tag "${TAG}_${deck}" --format compact --no-privileged-state
done

echo "=== $(date -Is) per-deck PG on the round-2 data (init from each deck's best)"
TAG="$TAG" REPLAY_TAG="$TAG" EVAL_SEED_START=140 bash kaggle/grids/run_perdeck_pg.sh "$TAG" "$ANCHOR"
echo "=== $(date -Is) round 2 done"
