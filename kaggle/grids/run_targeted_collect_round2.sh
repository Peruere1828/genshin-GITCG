#!/usr/bin/env bash
# Targeted collection, round 2: the remaining floor-violating matchups (H4/S5 reports).
#
# Round 1 covered unyielding_geo / electrocharged_moon / nahida_kokomi_raiden /
# aywolfskirk. The H4 per-deck floors still lose to these five, so collect them too.
# Seeds start at 2000 (round 1 used 1000..1299, dfbc_* used 0..79).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DECKS="${DECKS:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia}"
OPPONENTS="${OPPONENTS:-shenhe_double_swirl double_geo_navia superconduct_mika_attack dvalin_bonk yelan_clorinde_ororon}"
SEED_START="${SEED_START:-2000}"
SEED_COUNT="${SEED_COUNT:-250}"
WORKERS="${WORKERS:-6}"
TAG_PREFIX="${TAG_PREFIX:-tgt2}"

for deck in $DECKS; do
  echo "=== $(date -Is) collecting $deck vs [$OPPONENTS] seeds $SEED_START..$((SEED_START + SEED_COUNT - 1))"
  "$PY" -m train.collect --deck "$deck" --opponents $OPPONENTS \
    --seed-start "$SEED_START" --seed-count "$SEED_COUNT" --workers "$WORKERS" \
    --tag "${TAG_PREFIX}_${deck}" --format compact --no-privileged-state
  echo "=== $(date -Is) done $deck"
done
echo "=== $(date -Is) round-2 targeted collection done"
