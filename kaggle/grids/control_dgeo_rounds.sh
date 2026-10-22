#!/usr/bin/env bash
# Controls for double_geo's growing gain: separate "the round helped" from "the seed
# set was easier".
#  1. r2 checkpoint on the SAME seeds r3 was measured on (150-159) — like-for-like;
#  2. r3 checkpoint on a brand-new set (160-169) — a 4th independent replication.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DECK="double_geo_navia"
echo "=== $(date -Is) control: r2 on seeds 150-159 (same set as r3)"
"$PY" -m scripts.eval_protocol --contestant "neural:data/checkpoints/rl_r2_double_geo_navia.pt" \
  --contestant-deck "$DECK" --seeds 10 --seed-start 150 --workers 12 --strict-ci \
  --baseline "expert:${DECK}" --tag "ctrl_r2_on150_${DECK}"
echo "=== $(date -Is) r3 on fresh seeds 160-169"
"$PY" -m scripts.eval_protocol --contestant "neural:data/checkpoints/rl_dgeo3_double_geo_navia.pt" \
  --contestant-deck "$DECK" --seeds 10 --seed-start 160 --workers 12 --strict-ci \
  --baseline "expert:${DECK}" --tag "rl_dgeo3_on160_${DECK}"
echo "=== $(date -Is) controls done"
