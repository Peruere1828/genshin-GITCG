#!/usr/bin/env bash
# After the wave kernel finishes: pull the artefacts, install the checkpoints, and
# certify every new per-deck base on its own deck with a fresh seed set.
#
# Certifying the base is the gate for the ladder: only decks whose BC base reaches
# parity (or better) are eligible for PG rounds, so this script produces the next
# ladder candidate list instead of guessing.
#
# Usage: bash kaggle/grids/after_wave.sh [kernel-ref] [out-dir]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
K="${KAGGLE:-/home/ubuntuaaa/miniconda3/bin/kaggle}"
REF="${1:-fmyh1828/gitcg-train-arm}"
OUT="${2:-build/kaggle/out_wave}"
export HTTPS_PROXY="${KAGGLE_PROXY:-http://127.0.0.1:7890}"
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY="127.0.0.1,localhost"

echo "=== $(date -Is) waiting for $REF"
while true; do
  st="$("$K" kernels status "$REF" 2>/dev/null | grep -v outdated | tail -1)"
  case "$st" in
    *COMPLETE*) echo "=== $(date -Is) complete"; break ;;
    *ERROR*|*CANCEL*) echo "=== $(date -Is) finished with: $st"; break ;;
  esac
  sleep 120
done

rm -rf "$OUT"
"$K" kernels output "$REF" -p "$OUT" 2>&1 | grep -v outdated | tail -2

mkdir -p data/checkpoints/kaggle
for f in "$OUT"/checkpoints/*.pt; do
  [ -f "$f" ] || continue
  cp -f "$f" data/checkpoints/kaggle/
  echo "[install] $(basename "$f")"
done
cp -f "$OUT"/reports/*.json reports/train/ 2>/dev/null || true

# base checkpoint -> deck -> fresh eval seed start
PHASE="${PHASE:-all}"   # all | gb | gc -- the two waves run as separate kernels
cb() { # checkpoint-prefix deck seed-start tag
  case "$PHASE:$1" in
    all:*) ;;
    gb:gb_*) ;;
    gc:gc_*) ;;
    *) return 0 ;;
  esac
  local ckpt="data/checkpoints/kaggle/$1.pt"
  [ -f "$ckpt" ] || { echo "=== skip $2 ($1.pt missing)"; return 0; }
  echo "=== $(date -Is) certify base $1 on $2 (seeds $3+)"
  "$PY" -m scripts.eval_protocol --contestant "neural:$ckpt" --contestant-deck "$2" \
    --seeds 10 --seed-start "$3" --workers 12 --strict-ci \
    --baseline "expert:$2" --tag "$4"
}

cb gb_mika_bc_d256_e150    superconduct_mika_attack 660 base_gb_mika
cb gb_aywolf_bc_d256_e150  aywolfskirk              670 base_gb_aywolf
cb gb_natlan_bc_d256_e150  natlan_battleship        680 base_gb_natlan
cb gc_klee_bc_d256_e150    klee_mavuika_mualani     700 base_gc_klee
cb gc_nahida_bc_d256_e150  nahida_kokomi_raiden     710 base_gc_nahida
cb gc_shenhe_bc_d256_e150  shenhe_double_swirl      720 base_gc_shenhe
cb gc_electro_bc_d256_e150 electro_manifestation_nuke 730 base_gc_electro
cb gc_mualani_bc_d256_e150 mualani_double_pyro     740 base_gc_mualani
cb gc_dual_bc_d256_e150    dual_mualani_stacks      750 base_gc_dual

echo "=== $(date -Is) base certification done"
"$PY" -m scripts.specialist_table | tail -8
