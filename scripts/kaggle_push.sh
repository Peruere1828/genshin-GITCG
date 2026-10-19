#!/usr/bin/env bash
# Push the Kaggle training-arm kernel (script kernel) from this repo.
#
# Kaggle's blob upload endpoint is googleapis.com, which is unreachable from this
# network unless the local proxy is used, so every kaggle call below runs with
# HTTPS_PROXY/HTTP_PROXY set (override with KAGGLE_PROXY=...).
#
# Usage:
#   scripts/kaggle_push.sh              # refresh the code dataset, then push the kernel
#   REFRESH_CODE=0 scripts/kaggle_push.sh
#   KAGGLE_ACCEL= scripts/kaggle_push.sh    # push without an accelerator override
#
# Env knobs: KAGGLE_PROXY (default http://127.0.0.1:7890), KAGGLE_ACCEL
# (default NvidiaTeslaT4 == GPU T4 x2), KAGGLE_TIMEOUT seconds (default 39600 = 11 h).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle || echo "$HOME/miniconda3/bin/kaggle")}"
PROXY="${KAGGLE_PROXY:-http://127.0.0.1:7890}"
ACCEL="${KAGGLE_ACCEL-NvidiaTeslaT4}"
TIMEOUT="${KAGGLE_TIMEOUT:-39600}"
CODE_DS="${CODE_DS:-fmyh1828/genshin-gitcg-code}"
REFRESH_CODE="${REFRESH_CODE:-1}"
KAGGLE_MODE="${KAGGLE_MODE:-auto}"   # auto | probe | smoke | runs (pinned into the staging kernel folder)
PROBE="${PROBE:-0}"                  # 1 = push the CPU env-probe kernel instead of the GPU one

export HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY" NO_PROXY="127.0.0.1,localhost"

cd "$REPO"
if [ "$KAGGLE_MODE" != "auto" ]; then
  REFRESH_CODE=1   # the mode ships inside the code dataset (see train_arm.resolve_pinned_mode)
fi
{ echo "$KAGGLE_MODE"; [ -n "${KAGGLE_WORKERS:-}" ] && echo "workers=$KAGGLE_WORKERS"; } > build/kaggle/bundle-code/mode.txt
if [ "$REFRESH_CODE" = "1" ]; then
  echo "[push] refreshing code dataset $CODE_DS (mode=$KAGGLE_MODE)"
  tar czf build/kaggle/bundle-code/gitcg-code.tar.gz \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pt' --exclude='.pytest_cache' \
    PLAN.md AGENTS.md README.md NOTICE.md LICENSE pyproject.toml \
    common reps envs train scripts agents eval coach decklab npc configs tests web kaggle \
    data/assets/README.md data/assets/expert_system_assets_chs_latest.json \
    -C build/kaggle/bundle-code mode.txt
  if "$KAGGLE_BIN" datasets status "$CODE_DS" >/dev/null 2>&1; then
    "$KAGGLE_BIN" datasets version -p build/kaggle/bundle-code -r skip \
      -m "commit $(git rev-parse --short HEAD)"
  else
    "$KAGGLE_BIN" datasets create -p build/kaggle/bundle-code -r skip
  fi
  # the kernel must not be pushed against a dataset version still processing
  for _ in $(seq 1 40); do
    state="$("$KAGGLE_BIN" datasets status "$CODE_DS" 2>/dev/null | head -1)"
    [ "$state" = "ready" ] && break
    echo "[push] waiting for $CODE_DS ($state)"; sleep 15
  done
fi

echo "[push] kernel push (accelerator='$ACCEL', timeout=${TIMEOUT}s, mode=$KAGGLE_MODE)"
KERNEL_DIR="$REPO/build/kaggle/kernel"
rm -rf "$KERNEL_DIR"
mkdir -p "$KERNEL_DIR"
if [ "$PROBE" = "1" ]; then
  # CPU-only environment probe (no GPU quota): separate kernel id, same code file.
  KERNEL_REF="${PROBE_KERNEL:-fmyh1828/gitcg-env-probe}"
  ACCEL=""
  TIMEOUT="900"
  cp kaggle/train_arm.py "$KERNEL_DIR/"
  cp kaggle/probe/kernel-metadata.json "$KERNEL_DIR/kernel-metadata.json"
  echo "probe" > "$KERNEL_DIR/mode.txt"
else
  KERNEL_REF="${KERNEL:-fmyh1828/gitcg-train-arm}"
  cp kaggle/train_arm.py kaggle/kernel-metadata.json "$KERNEL_DIR/"
  echo "$KAGGLE_MODE" > "$KERNEL_DIR/mode.txt"   # staging copy, not the tracked folder
fi
if [ -n "$ACCEL" ]; then
  "$KAGGLE_BIN" kernels push -p "$KERNEL_DIR" --accelerator "$ACCEL" -t "$TIMEOUT"
else
  "$KAGGLE_BIN" kernels push -p "$KERNEL_DIR" -t "$TIMEOUT"
fi
"$KAGGLE_BIN" kernels status "$KERNEL_REF" || true
