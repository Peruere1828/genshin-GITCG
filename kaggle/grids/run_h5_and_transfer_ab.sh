#!/usr/bin/env bash
# H5 (matchup-weighted floor attack) + the cross-deck-transfer A/B, chained.
#
# Answers two questions with paired `eval_protocol` runs:
#   1. H5: does deficit reweighting (upweighting matchups the *data* loses) move the
#      worst per-opponent rates?  -> certify H5 vs H4 (same corpus/budget, unweighted).
#   2. Transfer: is the H4 gain on superconduct_aggro (+0.03 over the best SA-only BC)
#      really cross-deck data, or just "more data"? -> train SA-only at the *same*
#      200k/15ep budget and certify it on the same deck.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
WAIT_FOR="${WAIT_FOR:-h5_mixed_deficit}"
SEEDS="${SEEDS:-10}"
WORKERS="${WORKERS:-12}"

if [ -n "$WAIT_FOR" ]; then
  echo "=== $(date -Is) waiting for $WAIT_FOR"
  while pgrep -f "$WAIT_FOR" >/dev/null 2>&1; do sleep 60; done
fi

echo "=== $(date -Is) SA-only control at the same budget (200k/15ep)"
"$PY" -m scripts.datafirst_bc_train --store-cache data/store/sa_all.pt \
  --d-model 256 --epochs 15 --batch-size 256 --train-limit 200000 --val-limit 8000 \
  --threads 10 --seed 0 --out data/checkpoints/h5_saonly_200k_e15.pt \
  --report reports/train/h5_saonly_200k_e15.json

for pair in "h5_mixed_deficit:data/checkpoints/h5_mixed_deficit_e15.pt:superconduct_aggro:dvalin_bonk" \
            "h5_saonly:data/checkpoints/h5_saonly_200k_e15.pt:superconduct_aggro"; do
  tag="${pair%%:*}"; rest="${pair#*:}"; ckpt="${rest%%:*}"; decks="${rest#*:}"
  for deck in ${decks//:/ }; do
    echo "=== $(date -Is) certifying $tag on $deck"
    "$PY" -m scripts.eval_protocol --contestant "neural:${ckpt}" --contestant-deck "$deck" \
      --seeds "$SEEDS" --workers "$WORKERS" --strict-ci --baseline "expert:${deck}" \
      --tag "h5_${tag}_${deck}"
  done
done
echo "=== $(date -Is) H5 + transfer A/B done"
