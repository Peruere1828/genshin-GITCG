#!/usr/bin/env bash
# D30 gap-closing screen, LOCAL half (engine + eval) for the hist A/B.
#
# Stage 1 (store):  build the two compact stores from the paired collections
# Stage 2 (train):  ONLY a CPU fallback -- the sanctioned training plane is Kaggle
#                   (`KAGGLE_MODE=hist1 kaggle/grids/push_hist1.sh`)
# Stage 3 (eval):   400 paired games per arm on a NEVER-USED seed range
#
# Usage:
#   bash kaggle/grids/hist_local_screen.sh store
#   bash kaggle/grids/hist_local_screen.sh train-local       # 6 CPU runs, ~10 min
#   bash kaggle/grids/hist_local_screen.sh eval  <ckpt> <tag>
#   bash kaggle/grids/hist_local_screen.sh eval-kaggle       # pull dir -> eval every ckpt
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
DECK="dvalin_bonk"
TEACHER="data/checkpoints/rl_dvalin1_dvalin_bonk.pt"
SEED_START=9100
WORKERS="${WORKERS:-12}"
EVAL_SEEDS="${EVAL_SEEDS:-10}"     # 10 seeds x 20 opponents x 2 sides = 400 paired games
EVAL_SEED_START="${EVAL_SEED_START:-9200}"
STAGE="${1:-store}"

build_store() {
  local tag="$1"
  "$PY" -m scripts.datafirst_bc_store --replays "data/replays/${tag}_${DECK}" \
    --store-cache "data/store/${tag}_${DECK}.pt" --val-every 10 --seed-start "$SEED_START" \
    --report "reports/train/${tag}_${DECK}_store.json"
}

train_one() {
  local name="$1" store="$2" loss="$3" extra="${4:-}"
  # shellcheck disable=SC2086
  "$PY" -m scripts.datafirst_bc_train --store-cache "data/store/${store}" \
    --init-from "$TEACHER" --d-model 256 --batch-size 256 --epochs 1 --lr 3e-5 \
    --loss "$loss" --pg-entropy 0.01 --grad-clip 1.0 --seed 0 --threads "$WORKERS" \
    $extra --out "data/checkpoints/${name}.pt" --report "reports/train/${name}.json"
}

eval_one() {
  local ckpt="$1" tag="$2"
  "$PY" -m scripts.eval_protocol --contestant "neural:${ckpt}" --contestant-deck "$DECK" \
    --seeds "$EVAL_SEEDS" --seed-start "$EVAL_SEED_START" --workers "$WORKERS" --strict-ci \
    --baseline "expert:${DECK}" --tag "$tag"
}

case "$STAGE" in
  store)
    build_store hh_a0 && build_store hh_a1
    ;;
  train-local)
    for arm in a0 a1 a2; do
      store="hh_a1_${DECK}.pt"; [ "$arm" = "a0" ] && store="hh_a0_${DECK}.pt"
      priv=""; [ "$arm" = "a2" ] && priv="--privileged"
      train_one "hh_${arm}_${DECK}_local_s0" "$store" pg "$priv"
    done
    ;;
  eval)
    eval_one "${2:?ckpt}" "${3:?tag}"
    ;;
  eval-kaggle)
    out="${KAGGLE_OUT:-build/kaggle/out_hist1}"
    shopt -s nullglob
    for ckpt in "$out"/*/checkpoints/*.pt "$out"/checkpoints/*.pt; do
      name="$(basename "$ckpt" .pt)"
      name="${name%.part}"
      eval_one "$ckpt" "hist1_${name%%_${DECK}*}_${DECK}"
    done
    ;;
  *)
    echo "unknown stage: $STAGE" >&2; exit 2;;
esac
