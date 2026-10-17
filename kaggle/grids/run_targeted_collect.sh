#!/usr/bin/env bash
# Targeted (worst-matchup) collection for the H4 floor arm — see
# reports/explore/plan_strength_2day_kaggle.md (T0.5).
#
# Each trainee deck is collected against the scripted decks that suppress it
# (long tail: rock / electro / dendro stall + the aywolfskirk freeze deck), so the
# per-opponent floor gate (no matchup < 45%) gets data where it currently fails.
# Seeds start at 1000 so this never overlaps the dfbc_* corpora (seeds 0..79).
#
# Usage: setsid nohup bash kaggle/grids/run_targeted_collect.sh > data/logs/collect_tgt.log 2>&1 &
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DECKS="${DECKS:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia}"
OPPONENTS="${OPPONENTS:-unyielding_geo electrocharged_moon nahida_kokomi_raiden aywolfskirk}"
SEED_START="${SEED_START:-1000}"
SEED_COUNT="${SEED_COUNT:-300}"
WORKERS="${WORKERS:-10}"

for deck in $DECKS; do
  echo "=== $(date -Is) collecting $deck vs [$OPPONENTS] seeds $SEED_START..$((SEED_START + SEED_COUNT - 1))"
  "$PY" -m train.collect --deck "$deck" --opponents $OPPONENTS \
    --seed-start "$SEED_START" --seed-count "$SEED_COUNT" --workers "$WORKERS" \
    --tag "tgt_${deck}" --format compact --no-privileged-state
  echo "=== $(date -Is) done $deck"
done
echo "=== $(date -Is) all targeted collection done"
