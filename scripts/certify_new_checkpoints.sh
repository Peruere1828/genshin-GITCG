#!/usr/bin/env bash
# Certify freshly trained checkpoints (e.g. the A2 sweep from the 4090) on the decks
# that matter, with seed ranges that have never been used for these models.
#
# The remote machine only trains; evaluation stays here because this box owns the
# pre-registered seed ledger and the engine build. One command per checkpoint set:
#
#   bash scripts/certify_new_checkpoints.sh data/checkpoints/a2
#
# Every checkpoint in the directory is run against its own fresh seed bands:
#   checkpoint i -> dgeo 900+30i, ungeo 910+30i, dvalin 920+30i   (10 seeds each)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DIR="${1:?directory with checkpoints}"
DECKS="${DECKS:-double_geo_navia unyielding_geo dvalin_bonk}"
BASE_SEED="${BASE_SEED:-900}"
WORKERS="${WORKERS:-12}"

i=0
for ckpt in "$DIR"/*.pt; do
  [ -f "$ckpt" ] || continue
  name="$(basename "$ckpt" .pt)"
  offset=$((i * 30))
  j=0
  for deck in $DECKS; do
    start=$((BASE_SEED + offset + j * 10))
    tag="a2_${name}_$(echo "$deck" | cut -c1-12)"
    echo "=== $(date -Is) $name on $deck (seeds ${start}-$((start + 9)))"
    "$PY" -m scripts.eval_protocol --contestant "neural:${ckpt}" --contestant-deck "$deck" \
      --seeds 10 --seed-start "$start" --workers "$WORKERS" --strict-ci \
      --baseline "expert:${deck}" --tag "$tag" | grep -E "pooled=|gates:|paired vs baseline"
    j=$((j + 1))
  done
  i=$((i + 1))
done

echo "=== $(date -Is) done; table:"
"$PY" -m scripts.specialist_table | head -12
