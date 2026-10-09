#!/usr/bin/env bash
# Cluster bootstrap for genshin-GITCG on the NJU HPC (LSF, CentOS7 login).
# Idempotent: safe to re-run. Does NOT touch data/checkpoints (append-only rule).
#
# Prerequisites (see scripts/cluster/README.md):
#   * this repo checked out (branches: feat/cluster-prep for code, vendor for the
#     prebuilt gitcg wheel + asset cache);
#   * an existing conda env with torch (default: $HOME/.conda/envs/fa_env).
#
# Usage:  GITCG_VENDOR_DIR=$HOME/genshin-GITCG-vendor bash scripts/cluster/bootstrap.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Modules + existing python env (see scripts/cluster/env.sh for the rationale).
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/cluster/env.sh"

PYDEPS="${GITCG_PYDEPS:-$REPO_ROOT/.pydeps}"
VENDOR="${GITCG_VENDOR_DIR:-$REPO_ROOT/../genshin-GITCG-vendor}"
PIP_INDEX="${GITCG_PIP_INDEX:-https://mirror.nju.edu.cn/pypi/web/simple}"

echo "[bootstrap] repo:    $REPO_ROOT"
echo "[bootstrap] python:  $(python --version 2>&1)"
echo "[bootstrap] vendor:  $VENDOR"

# 1) Python deps into a repo-local dir (do NOT mutate the shared conda env).
mkdir -p "$PYDEPS"
pip install -q --target "$PYDEPS" -i "$PIP_INDEX" \
  "cffi>=1.17.1" "protobuf==5.29.1" tomli pytest

# 2) gitcg prebuilt wheel (not on the internal mirror): extract into PYDEPS.
WHL="$(ls "$VENDOR"/wheels/gitcg-*.whl 2>/dev/null | head -1 || true)"
if [ -n "$WHL" ]; then
  python - "$WHL" "$PYDEPS" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
print("[bootstrap] extracted", sys.argv[1])
PY
else
  echo "[bootstrap] ERROR: gitcg wheel not found under $VENDOR/wheels" >&2
  exit 1
fi

# 3) Offline asset cache (cluster has no external network).
if [ -f "$VENDOR/assets/expert_system_assets_chs_latest.json" ]; then
  mkdir -p data/assets
  cp "$VENDOR/assets/expert_system_assets_chs_latest.json" data/assets/
  echo "[bootstrap] assets cache installed"
fi

# 4) Smoke checks.
python -c "import gitcg; print('[bootstrap] gitcg ok:', gitcg.__file__)"
python - <<'PY'
from agents.scripted.assets import load_assets
catalog = load_assets()
print(f"[bootstrap] assets: {len(catalog.cards)} cards, {len(catalog.characters)} characters")
PY
GITCG_ASSETS_OFFLINE=1 python -m pytest -q -m "not slow" \
  tests/test_info_hiding.py tests/test_action_adapter.py

echo "[bootstrap] done. Source scripts/cluster/env.sh before running anything."
