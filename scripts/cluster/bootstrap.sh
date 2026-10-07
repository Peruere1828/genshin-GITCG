#!/usr/bin/env bash
# Cluster bootstrap for genshin-GITCG (PLAN.md §5 / §10 step 7).
# Idempotent: safe to re-run. Does NOT touch data/checkpoints (append-only rule).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${GITCG_ENV_NAME:-gitcg}"
PY_VERSION="${GITCG_PY_VERSION:-3.12}"

echo "[bootstrap] repo: $REPO_ROOT"
cd "$REPO_ROOT"

# 1) Python env (conda if available, else venv)
if command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  [ -d ".venv" ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
echo "[bootstrap] python: $(python --version 2>&1)"

# 2) Core deps (engine + tests + toy training)
pip install -q --upgrade pip
pip install -q gitcg pytest
pip install -q torch --index-url https://download.pytorch.org/whl/cpu || \
  echo "[bootstrap] WARN: torch install failed; train/ will be unavailable"

# 3) Reference repos (read-only; gitignored). Shallow clone by default.
mkdir -p refs
clone() {
  local url="$1"; local dir="$2"; local depth="${3:-1}"
  if [ ! -d "$dir/.git" ]; then
    git clone --depth "$depth" "$url" "$dir"
  else
    echo "[bootstrap] $dir already present"
  fi
}
clone https://github.com/piovium/genius-invokation refs/genius-invokation
clone https://github.com/piovium/Rebel_base_RL refs/Rebel_base_RL
clone https://github.com/piovium/AI refs/AI

# 4) Card assets cache (data/ is gitignored; workers must not fetch mid-job)
export GITCG_ASSETS_OFFLINE="${GITCG_ASSETS_OFFLINE:-0}"
python - <<'PY'
from agents.scripted.assets import load_assets
catalog = load_assets()
print(f"[bootstrap] assets: {len(catalog.cards)} cards, {len(catalog.characters)} characters")
PY

# 5) Smoke check
python -m pytest -q -m "not slow" tests/test_env_smoke.py tests/test_info_hiding.py

echo "[bootstrap] done. Engine lock:"
cat configs/engine.lock
