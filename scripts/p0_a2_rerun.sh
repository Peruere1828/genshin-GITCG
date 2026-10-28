#!/usr/bin/env bash
# P0-D: re-run the A2 identity-embedding A/B on the clean (exact-id) corpus.
#
# Prerequisite:
#   bash scripts/p0_clean_corpus.sh          # 4 decks x 20 opponents x 20 seeds
#   -> data/store/p0_clean_four_decks.pt     # definition_ids_present == events
#
# This script trains A0 (no ids) and A0i (ids) from the same clean store, then
# certifies both on dgeo/dvalin with fresh paired seeds. Run it on the A800
# window (GPU training); evaluation stays on the local CPU node by copying the
# checkpoints back and running the printed eval commands there.
set -euo pipefail

PY="${PY:-python}"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

STORE="${STORE:-data/store/p0_clean_four_decks.pt}"
OUTDIR="${OUTDIR:-data/checkpoints/p0_a2}"
EPOCHS="${EPOCHS:-150}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
mkdir -p "$OUTDIR"

if [ ! -f "$STORE" ]; then
  echo "missing $STORE; run: bash scripts/p0_clean_corpus.sh" >&2
  exit 2
fi

echo "[p0-a2] A0 (no identity branch) $(date -Is)"
"$PY" -m scripts.datafirst_bc_train \
  --store-cache "$STORE" --arch cvpn --d-model 256 --epochs "$EPOCHS" \
  --lr 1e-3 --batch-size 512 --threads 8 --device "$DEVICE" --seed "$SEED" \
  --loss ce --val-limit 8000 --save-best --select-metric val_ce \
  --out "$OUTDIR/A0.pt" --report "$OUTDIR/A0.json"

echo "[p0-a2] A0i (identity branch, exact ids) $(date -Is)"
"$PY" -m scripts.datafirst_bc_train \
  --store-cache "$STORE" --arch cvpn --d-model 256 --epochs "$EPOCHS" \
  --lr 1e-3 --batch-size 512 --threads 8 --device "$DEVICE" --seed "$SEED" \
  --loss ce --id-buckets 1024 --id-emb-dim 32 --val-limit 8000 \
  --save-best --select-metric val_ce \
  --out "$OUTDIR/A0i.pt" --report "$OUTDIR/A0i.json"

cat <<EOF
[p0-a2] training done. Copy $OUTDIR to the CPU eval node, then run:

for deck in double_geo_navia dvalin_bonk; do
  for arm in A0 A0i; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    "$PY" -m scripts.eval_protocol \\
      --contestant "neural:$OUTDIR/\${arm}.pt" --contestant-deck "\$deck" \\
      --seeds 20 --seed-start 900 --workers 10 --strict-ci \\
      --baseline "expert:\$deck" --tag "p0a2_\${arm}_\${deck}_s900"
  done
done

# Paired A0 vs A0i (same seeds/opponents/sides):
"$PY" -m scripts.compare_eval_runs p0a2_A0_double_geo_navia_s900 p0a2_A0i_double_geo_navia_s900
"$PY" -m scripts.compare_eval_runs p0a2_A0_dvalin_bonk_s900 p0a2_A0i_dvalin_bonk_s900
EOF
