#!/usr/bin/env bash
# Expert iteration, cheap version: self-play -> keep the wins -> BC -> certify.
#
# Why this and not more BC on expert data: every imitation run so far plateaus at
# the teacher (parity at best, reports/explore/plan_strength_2day_kaggle.md), and
# AWR's soft version of "weight by outcome" was neutral-to-harmful. The remaining
# cheap non-imitation signal is *selection*: let the current policy play (with
# temperature for diversity), keep the games it won, and imitate only those. With a
# policy already at teacher parity on four decks, the wins are the better
# realizations of its own behaviour, so this can in principle exceed the teacher.
#
# Chain: wait for round-2 targeted collection -> collect self-play -> build the
# win-filtered store -> BC -> certify the worst decks against the teacher.
#
# Usage: setsid nohup bash kaggle/grids/run_selfplay_filtered_bc.sh > data/logs/selfplay.log 2>&1 &
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
TEACHER="${TEACHER:-data/checkpoints/h4_mixed_d256_e15.pt}"
TEMP="${TEMP:-1.0}"
DECKS="${DECKS:-superconduct_aggro dvalin_bonk unyielding_geo double_geo_navia}"
SEED_START="${SEED_START:-3000}"
SEED_COUNT="${SEED_COUNT:-100}"
COLLECT_WORKERS="${COLLECT_WORKERS:-6}"
EPOCHS="${EPOCHS:-15}"
WAIT_FOR="${WAIT_FOR:-run_targeted_collect_round2}"

if [ -n "$WAIT_FOR" ]; then
  echo "=== $(date -Is) waiting for $WAIT_FOR"
  while pgrep -f "$WAIT_FOR" >/dev/null 2>&1; do sleep 60; done
fi

REPLAY_DIRS=""
for deck in $DECKS; do
  echo "=== $(date -Is) self-play collection: $deck (teacher=$TEACHER#t=$TEMP)"
  "$PY" -m train.collect --deck "$deck" --all-opponents \
    --seed-start "$SEED_START" --seed-count "$SEED_COUNT" --workers "$COLLECT_WORKERS" \
    --teacher-spec "neural:${TEACHER}#t=${TEMP}" \
    --tag "sp_${deck}" --format compact --no-privileged-state
  REPLAY_DIRS="$REPLAY_DIRS data/replays/sp_${deck}"
  echo "=== $(date -Is) done $deck"
done

echo "=== $(date -Is) building win-filtered store from:$REPLAY_DIRS"
"$PY" -m scripts.datafirst_bc_store --replays $REPLAY_DIRS \
  --store-cache data/store/selfplay_wins.pt --val-every 10 --min-outcome 1.0 \
  --report reports/train/selfplay_wins_store.json

echo "=== $(date -Is) BC on the win-filtered self-play store"
"$PY" -m scripts.datafirst_bc_train --store-cache data/store/selfplay_wins.pt \
  --d-model 256 --epochs "$EPOCHS" --batch-size 256 --threads 10 --seed 0 \
  --out data/checkpoints/selfplay_wins_d256_e${EPOCHS}.pt \
  --report reports/train/selfplay_wins_d256_e${EPOCHS}.json

for deck in superconduct_aggro dvalin_bonk; do
  echo "=== $(date -Is) certifying selfplay_wins on $deck"
  "$PY" -m scripts.eval_protocol \
    --contestant "neural:data/checkpoints/selfplay_wins_d256_e${EPOCHS}.pt" \
    --contestant-deck "$deck" --seeds 10 --workers 12 --strict-ci \
    --baseline "expert:${deck}" --tag "sp_wins_${deck}"
done
echo "=== $(date -Is) self-play filtered-BC round done"
