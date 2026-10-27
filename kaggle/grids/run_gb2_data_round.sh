#!/usr/bin/env bash
# Second data round for the near-parity Group B decks.
#
# mika (-0.0275), natlan (-0.0288) and aywolf (-0.0338) sit just outside the
# pre-registered ladder gate (delta >= -0.02). Their gaps are imitation error, not
# strategy error -- BC's ceiling is the scripted expert itself -- so the sanctioned
# fix is more games per matchup (2,000 -> 4,000), then re-certify.
#
# Queued behind the round-2 ladders so the two jobs do not fight for the 12 cores.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"

echo "=== $(date -Is) waiting for the round-2 ladders"
while pgrep -f "run_ladder_round2_gc.sh" >/dev/null 2>&1; do sleep 60; done

collect() { # deck seed-start
  echo "=== $(date -Is) collect 2nd batch $1 (seeds $2+100)"
  "$PY" -m train.collect --deck "$1" --all-opponents \
    --seed-start "$2" --seed-count 100 --workers 12 \
    --teacher-spec "expert:$1" --tag "gb2_$1" --format compact --no-privileged-state
}

collect superconduct_mika_attack 13400
collect natlan_battleship 13500
collect aywolfskirk 13600

echo "=== $(date -Is) merging stores (2,000 + 2,000 games per deck)"
merge() { # deck
  "$PY" -m scripts.datafirst_bc_store \
    --replays "data/replays/gb_$1" "data/replays/gb2_$1" \
    --store-cache "data/store/gbm_$1.pt" --val-every 10 \
    --report "reports/train/gbm_$1_store.json"
}
merge superconduct_mika_attack
merge natlan_battleship
merge aywolfskirk

echo "=== $(date -Is) staging corpus-gb2 for upload"
mkdir -p build/kaggle/bundle-gb2
cat > build/kaggle/bundle-gb2/dataset-metadata.json <<'JSON'
{
  "title": "genshin-gitcg-corpus-gb2",
  "id": "fmyh1828/genshin-gitcg-corpus-gb2",
  "licenses": [{"name": "AGPL-3.0"}],
  "description": "Private merged stores for Group B round 2 (mika, natlan, aywolfskirk): 4,000 expert games each (200 seeds x 20 opponents). Numeric model features only."
}
JSON
for deck in superconduct_mika_attack natlan_battleship aywolfskirk; do
  ln -f "data/store/gbm_$deck.pt" build/kaggle/bundle-gb2/
done
du -sh build/kaggle/bundle-gb2/*.pt

export HTTPS_PROXY="${KAGGLE_PROXY:-http://127.0.0.1:7890}"
export HTTP_PROXY="$HTTPS_PROXY" NO_PROXY="127.0.0.1,localhost"
echo "=== $(date -Is) uploading corpus-gb2"
/home/ubuntuaaa/miniconda3/bin/kaggle datasets create -p build/kaggle/bundle-gb2 2>&1 | grep -v outdated | tail -2

echo "=== $(date -Is) done (push the gb2 kernel next: KAGGLE_MODE=gb2)"
