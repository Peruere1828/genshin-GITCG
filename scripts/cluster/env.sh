#!/usr/bin/env bash
# NJU HPC (LSF) environment for genshin-GITCG. Source this from job scripts / interactively.
#
# Why the extra steps (verified on the cluster 2026-10-09):
#   * login and compute nodes are CentOS 7 (glibc 2.17), but gitcg 0.21.0 ships a
#     manylinux_2_34 ``libgitcg.so`` -> load the newer glibc module (>= 2.34).
#   * the cluster has no conda channel access, so we reuse an existing conda env
#     that already carries torch/numpy/pip instead of creating one.
#   * the cluster has no external network: gitcg comes from the ``vendor`` branch
#     (prebuilt wheel) and the expert asset cache is copied in by ``bootstrap.sh``.
set -euo pipefail

if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
fi

module purge
module load glibc/2.36-gcc12.1.0
module load anaconda/3

GITCG_REPO="${GITCG_REPO:-$HOME/genshin-GITCG}"
GITCG_BASE_ENV="${GITCG_BASE_ENV:-$HOME/.conda/envs/fa_env}"
GITCG_PYDEPS="${GITCG_PYDEPS:-$GITCG_REPO/.pydeps}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$GITCG_BASE_ENV"

export GITCG_ASSETS_OFFLINE=1
export PYTHONPATH="$GITCG_PYDEPS:$GITCG_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

cd "$GITCG_REPO"
