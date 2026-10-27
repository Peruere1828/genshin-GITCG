#!/usr/bin/env bash
# A2 identity-embedding certification on the same fresh seeds.
#
# Runs the pre-A2 baseline once per deck, then each candidate, all against the
# 20-opponent pool with the same seeds.  The paired delta vs the baseline is
# written next to the eval report, so the merge decision never uses val_acc alone.
#
# Usage:
#   PY=/root/blockdata/conda/gitcg/bin/python bash scripts/certify_a2.sh
#   SEEDS=20 SEED_START=900 bash scripts/certify_a2.sh
#   CANDIDATES="a0i:data/checkpoints/a2/A0i.pt a2ft:data/checkpoints/a2/A2ft.pt" \
#     bash scripts/certify_a2.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python}"
SEEDS="${SEEDS:-10}"
SEED_START="${SEED_START:-900}"
WORKERS="${WORKERS:-10}"
DECKS="${DECKS:-double_geo_navia unyielding_geo dvalin_bonk}"
BASELINE="${BASELINE:-data/checkpoints/kaggle/h4m_bc_d256_e150.pt}"
CANDIDATES="${CANDIDATES:-a0i:data/checkpoints/a2/A0i.pt}"

run_one() { # tag contestant deck
  local tag="$1" contestant="$2" deck="$3"
  echo "=== $(date -Is) $tag ($deck)"
  "$PY" -m scripts.eval_protocol --contestant "neural:${contestant}" \
    --contestant-deck "$deck" --seeds "$SEEDS" --seed-start "$SEED_START" \
    --workers "$WORKERS" --strict-ci --baseline "expert:${deck}" --tag "$tag" \
    || { echo "FAILED $tag"; return 1; }
}

for deck in $DECKS; do
  base_tag="a0_${deck}_s${SEED_START}"
  run_one "$base_tag" "$BASELINE" "$deck" || continue
  for spec in $CANDIDATES; do
    name="${spec%%:*}"; ckpt="${spec#*:}"
    tag="${name}_${deck}_s${SEED_START}"
    run_one "$tag" "$ckpt" "$deck" || continue
    "$PY" -m scripts.compare_eval_runs "$base_tag" "$tag" \
      --json-out "reports/explore/${tag}_vs_a0.json" || true
  done
done
echo "=== $(date -Is) A2 certification done"
