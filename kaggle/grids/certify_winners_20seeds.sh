#!/usr/bin/env bash
# Tighter headline numbers for the two specialists that beat the whole scripted pool
# by a wide margin. 20 seeds per deck (800 paired games) on seed ranges never used for
# either deck; same pre-registered acceptance rule as every other certification.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
cert() { # deck ckpt seed-start tag
  echo "=== $(date -Is) certify $1 on seeds $3+ (20 seeds)"
  "$PY" -m scripts.eval_protocol --contestant "neural:$2" --contestant-deck "$1" \
    --seeds 20 --seed-start "$3" --workers 12 --strict-ci \
    --baseline "expert:$1" --tag "$4"
}
cert double_geo_navia  data/checkpoints/rl_dgeo3_double_geo_navia.pt 600 final20_dgeo
cert unyielding_geo   data/checkpoints/rl_ungeo3_unyielding_geo.pt  620 final20_ungeo
echo "=== $(date -Is) done"
