#!/usr/bin/env bash
# Push the floor-repair corpus (hist5) and run the `hist5` grid on Kaggle.
#
# Store = 2,000-game base + 400 expert-taught games vs dvalin's two structural weak
# matchups; plus the warm-start teacher (referenced as data/store/<name> because the
# kernel's link_stores symlinks every mounted *.pt into repo/data/store).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle || echo "$HOME/miniconda3/bin/kaggle")}"
PROXY="${KAGGLE_PROXY:-http://127.0.0.1:7890}"
DS="${HIST5_DS:-fmyh1828/genshin-gitcg-corpus-hist5}"
BUNDLE="build/kaggle/bundle-hist5"
DATA_ONLY=0
[ "${1:-}" = "--data-only" ] && DATA_ONLY=1

export HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY" NO_PROXY="127.0.0.1,localhost"

for f in data/store/r5_dvalin_selfplay.pt data/checkpoints/tg3_bc_e20_s0.pt; do
  [ -f "$f" ] || { echo "[abort] missing $f"; exit 1; }
done

rm -rf "$BUNDLE"; mkdir -p "$BUNDLE"
ln data/store/r5_dvalin_selfplay.pt "$BUNDLE/"
ln data/checkpoints/tg3_bc_e20_s0.pt "$BUNDLE/"
cat > "$BUNDLE/dataset-metadata.json" <<JSON
{"title": "gitcg corpus hist5", "id": "$DS", "licenses": [{"name": "CC0-1.0"}]}
JSON
du -sh "$BUNDLE"/*

if "$KAGGLE_BIN" datasets status "$DS" >/dev/null 2>&1; then
  "$KAGGLE_BIN" datasets version -p "$BUNDLE" -r skip -m "floor-repair corpus $(git rev-parse --short HEAD)"
else
  "$KAGGLE_BIN" datasets create -p "$BUNDLE" -r skip
fi
for _ in $(seq 1 60); do
  state="$("$KAGGLE_BIN" datasets status "$DS" 2>/dev/null | grep -v outdated | head -1 | tr -d '[:space:]' || true)"
  [ "$state" = "ready" ] && break
  echo "[push] waiting for $DS (${state:-unknown})"; sleep 15
done

[ "$DATA_ONLY" = "1" ] && { echo "[push] data only, done"; exit 0; }
KAGGLE_MODE=hist5 bash scripts/kaggle_push.sh
