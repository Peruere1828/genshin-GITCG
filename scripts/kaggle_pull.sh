#!/usr/bin/env bash
# Kaggle training-arm kernel: status / logs / pull output back into the repo.
#
# Usage:
#   scripts/kaggle_pull.sh status        # one-line run status
#   scripts/kaggle_pull.sh logs [lines]  # tail the kernel log
#   scripts/kaggle_pull.sh out           # download outputs into build/kaggle/out
#   scripts/kaggle_pull.sh collect       # out + copy checkpoints/reports into the repo
#
# Env knobs: KAGGLE_PROXY (default http://127.0.0.1:7890), KERNEL (default fmyh1828/gitcg-train-arm).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle || echo "$HOME/miniconda3/bin/kaggle")}"
PROXY="${KAGGLE_PROXY:-http://127.0.0.1:7890}"
KERNEL="${KERNEL:-fmyh1828/gitcg-train-arm}"
OUT="${OUT:-$REPO/build/kaggle/out}"

export HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY" NO_PROXY="127.0.0.1,localhost"

cmd="${1:-status}"
case "$cmd" in
  status)
    "$KAGGLE_BIN" kernels status "$KERNEL"
    ;;
  logs)
    "$KAGGLE_BIN" kernels logs "$KERNEL" 2>/dev/null | tail -n "${2:-40}"
    ;;
  out|collect)
    mkdir -p "$OUT"
    "$KAGGLE_BIN" kernels output "$KERNEL" -p "$OUT" -o
    if [ "$cmd" = "collect" ]; then
      mkdir -p "$REPO/data/checkpoints/kaggle" "$REPO/reports/train" "$REPO/data/kaggle_logs"
      cp -v "$OUT"/checkpoints/*.pt "$REPO/data/checkpoints/kaggle/" 2>/dev/null || true
      for f in "$OUT"/reports/*.json; do
        [ -e "$f" ] || continue
        cp -v "$f" "$REPO/reports/train/kaggle_$(basename "$f")"
      done
      cp -v "$OUT"/logs/*.log "$REPO/data/kaggle_logs/" 2>/dev/null || true
      cp -v "$OUT"/summary.json "$REPO/build/kaggle/last_summary.json" 2>/dev/null || true
    fi
    ;;
  *)
    echo "usage: $0 {status|logs [lines]|out|collect}" >&2
    exit 2
    ;;
esac
