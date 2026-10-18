#!/usr/bin/env bash
# Corpus breadth: collect the remaining 16 decks against the full 20-opponent pool.
#
# Rationale (evidence): the only measured strength gain so far came from *data
# composition* — the 4-deck mixed corpus reached per-deck parity with the scripted
# experts and lifted superconduct_aggro from 0.476 to 0.5075 (reports/explore/
# plan_strength_2day_kaggle.md). The mixed model also beat its teacher on matchups
# that only *other decks'* games cover, i.e. cross-deck transfer is real. This job
# widens that axis to all 20 decks so a single model can be trained for (and
# evaluated on) every deck in the pool — which is also what the web/decklab lines
# need.
#
# Layout: `data/replays/all_<deck>/` — 20 opponents x 80 seeds (0..79), compact,
# resumable; the store builder dedups by (deck, opponent, seed).
#
# Usage: setsid nohup bash kaggle/grids/run_corpus_all_decks.sh > data/logs/corpus_all_decks.log 2>&1 &
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
WORKERS="${WORKERS:-8}"
SEED_COUNT="${SEED_COUNT:-80}"
# the 4 decks already covered by the dfbc_* corpora are excluded
DECKS="${DECKS:-natlan_battleship klee_mavuika_mualani superconduct_mika_attack dual_mualani_stacks \
mualani_double_pyro yelan_clorinde_ororon electrocharged_moon aywolfskirk shenhe_double_swirl \
electro_manifestation_nuke beidou_skirk_ayaka skirk_chasca_freeze skirk_ayaka_navia \
ice_water_battleship nahida_kokomi_raiden trial_skirk_ayaka_navia}"
# optional: wait for another collection to finish first (avoid CPU oversubscription)
WAIT_FOR="${WAIT_FOR:-}"
if [ -n "$WAIT_FOR" ]; then
  echo "=== $(date -Is) waiting for $WAIT_FOR"
  while pgrep -f "$WAIT_FOR" >/dev/null 2>&1; do sleep 60; done
fi

for deck in $DECKS; do
  echo "=== $(date -Is) $deck: 20 opponents x $SEED_COUNT seeds, $WORKERS workers"
  "$PY" -m train.collect --deck "$deck" --all-opponents \
    --seed-start 0 --seed-count "$SEED_COUNT" --workers "$WORKERS" \
    --tag "all_${deck}" --format compact --no-privileged-state
  echo "=== $(date -Is) done $deck"
done
echo "=== $(date -Is) all-deck corpus collection done"
