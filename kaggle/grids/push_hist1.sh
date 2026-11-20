#!/usr/bin/env bash
# Push the D30 gap-closing corpus (history A/B) and run the `hist1` grid on Kaggle.
#
# Data plane stays local (engine/cffi on CPU); Kaggle only trains. See
# docs/environments/kaggle-t4.md and reports/explore/hist_priv_prereg_20261120.md.
#
# Usage: bash kaggle/grids/push_hist1.sh [--data-only]
#   --data-only   refresh the corpus dataset, do not push the kernel
#
# Requires (already produced locally):
#   data/store/hh_a0_dvalin_bonk.pt   historyless control store
#   data/store/hh_a1_dvalin_bonk.pt   history store (same games)
#   data/checkpoints/rl_dvalin1_dvalin_bonk.pt   warm-start teacher
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle || echo "$HOME/miniconda3/bin/kaggle")}"
PROXY="${KAGGLE_PROXY:-http://127.0.0.1:7890}"
DS="${HIST_DS:-fmyh1828/genshin-gitcg-corpus-hist}"
BUNDLE="build/kaggle/bundle-hist"
DATA_ONLY=0
[ "${1:-}" = "--data-only" ] && DATA_ONLY=1

export HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY" NO_PROXY="127.0.0.1,localhost"

for f in data/store/hh_a0_dvalin_bonk.pt data/store/hh_a1_dvalin_bonk.pt \
         data/checkpoints/rl_dvalin1_dvalin_bonk.pt; do
  [ -f "$f" ] || { echo "[abort] missing $f"; exit 1; }
done

rm -rf "$BUNDLE"; mkdir -p "$BUNDLE"
# Hard links: the bundle must not double the 5-6 GB on disk. ``link_stores`` on the
# kernel side scans every *.pt in the mount and symlinks it into repo/data/store, so
# the teacher checkpoint rides along in the same dataset and is referenced as
# ``data/store/rl_dvalin1_dvalin_bonk.pt`` by the hist1 run specs.
ln data/store/hh_a0_dvalin_bonk.pt "$BUNDLE/"
ln data/store/hh_a1_dvalin_bonk.pt "$BUNDLE/"
ln data/checkpoints/rl_dvalin1_dvalin_bonk.pt "$BUNDLE/"

cat > "$BUNDLE/dataset-metadata.json" <<JSON
{
  "title": "gitcg corpus hist",
  "id": "$DS",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

echo "[push] corpus sizes:"
du -sh "$BUNDLE"/* | sort -h

if "$KAGGLE_BIN" datasets status "$DS" >/dev/null 2>&1; then
  "$KAGGLE_BIN" datasets version -p "$BUNDLE" -r skip \
    -m "hist A/B stores + dvalin teacher $(git rev-parse --short HEAD)"
else
  "$KAGGLE_BIN" datasets create -p "$BUNDLE" -r skip
fi
for _ in $(seq 1 60); do
  state="$("$KAGGLE_BIN" datasets status "$DS" 2>/dev/null | grep -v outdated | head -1 | tr -d '[:space:]')"
  [ "$state" = "ready" ] && break
  echo "[push] waiting for $DS ($state)"; sleep 15
done

[ "$DATA_ONLY" = "1" ] && { echo "[push] data only, done"; exit 0; }

KAGGLE_MODE=hist1 bash scripts/kaggle_push.sh
