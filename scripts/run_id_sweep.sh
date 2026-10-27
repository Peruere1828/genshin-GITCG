#!/usr/bin/env bash
# A2 sweep: does learned card/entity identity help, at matched data and epochs?
#
# Four runs, same corpus and epoch budget, so the only differences are architecture
# (mean-pool CVPN vs transformer) and identity embeddings (off/on):
#
#   A0   cvpn d256                     the shipped baseline (0.20 M params)
#   A0i  cvpn d256            + ids    isolates the embedding effect on the same trunk
#   A1   transformer 6L d384           capacity/structure without identity
#   A2   transformer 6L d384  + ids    the combination the 4090 should be able to push
#
# Usage:
#   STORE=data/store/mixed_four_decks.pt bash scripts/run_id_sweep.sh
#   EPOCHS=150 ID_BUCKETS=1024 RUNS="A0 A0i A1 A2" bash scripts/run_id_sweep.sh
#
# Judgement rule (pre-registered): val_acc + win rate vs the 20 scripted experts on a
# fresh seed set; a run only counts if the paired delta CI excludes zero. val_acc alone
# is capped by the scripted expert, so the win rate decides.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python}"
STORE="${STORE:-data/store/mixed_four_decks.pt}"
OUT="${OUT:-data/checkpoints/a2}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-512}"
LR="${LR:-1e-3}"
ID_BUCKETS="${ID_BUCKETS:-1024}"
ID_EMB_DIM="${ID_EMB_DIM:-32}"
THREADS="${THREADS:-8}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-A0 A0i A1 A2}"
mkdir -p "$OUT"

run() { # name arch d_model n_layers n_heads id_buckets
  local name="$1" arch="$2" d="$3" layers="$4" heads="$5" ids="$6"
  local ckpt="$OUT/${name}.pt"
  if [ -f "$ckpt" ]; then echo "=== $name already done"; return 0; fi
  echo "=== $(date -Is) $name (arch=$arch d=$d L=$layers ids=$ids epochs=$EPOCHS)"
  "$PY" -m scripts.datafirst_bc_train \
    --store-cache "$STORE" --arch "$arch" --d-model "$d" \
    --n-layers "$layers" --n-heads "$heads" \
    --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH" --threads "$THREADS" \
    --device "$DEVICE" --seed 0 --loss ce \
    --id-buckets "$ids" --id-emb-dim "$ID_EMB_DIM" \
    --out "$ckpt" --report "$OUT/${name}.json" 2>&1 | grep -E "\[train\]|\[done\]|error" | tail -2
}

for r in $RUNS; do
  case "$r" in
    A0)  run A0  cvpn        256 2 4 0 ;;
    A0i) run A0i cvpn        256 2 4 "$ID_BUCKETS" ;;
    A1)  run A1  transformer 384 6 8 0 ;;
    A2)  run A2  transformer 384 6 8 "$ID_BUCKETS" ;;
    A3)  run A3  transformer 512 8 8 "$ID_BUCKETS" ;;
    *) echo "unknown run $r"; exit 2 ;;
  esac
done

echo "=== $(date -Is) sweep done; checkpoints in $OUT"
"$PY" - <<'EOF'
import glob, json, os
rows = []
for p in sorted(glob.glob(os.path.join(os.environ.get("OUT", "data/checkpoints/a2"), "*.json"))):
    d = json.load(open(p))
    h = d.get("history") or [{}]
    rows.append((os.path.basename(p).replace(".json", ""), d.get("n_train"), h[-1].get("val_acc"),
                 h[-1].get("val_ce"), round(d.get("train_seconds", 0) / 60, 1)))
print(f"{'run':6s} {'n_train':>8s} {'val_acc':>8s} {'val_ce':>8s} {'min':>6s}")
for r in rows:
    print(f"{r[0]:6s} {str(r[1]):>8s} {str(r[2]):>8s} {str(r[3]):>8s} {str(r[4]):>6s}")
EOF
