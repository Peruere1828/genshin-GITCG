#!/usr/bin/env bash
# NJU HPC (LSF) environment for genshin-GITCG. Source this from job scripts / interactively.
#
# Why the extra steps (verified on the cluster 2026-10-09):
#   * login and compute nodes are CentOS 7 (glibc 2.17), but gitcg 0.21.0 ships a
#     manylinux_2_34 ``libgitcg.so`` (needs glibc/libstdc++ from a modern toolchain)
#     -> load the glibc + gcc modules.
#   * the cluster has no conda channel access and ``module load anaconda/3`` does not
#     put conda on PATH in batch: use an existing env's interpreter directly (its
#     site-packages resolve from the interpreter path, no activation needed).
#   * no external network: gitcg comes from the njugit ``vendor`` branch wheel and
#     the expert asset cache is copied in by ``bootstrap.sh``.
set -euo pipefail

if ! type module >/dev/null 2>&1; then
  for _init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
               /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    if [ -f "$_init" ]; then . "$_init"; break; fi
  done
fi

module purge
module load glibc/2.36-gcc12.1.0
module load gcc/12.1.0 2>/dev/null || true
module load anaconda/3 2>/dev/null || true

GITCG_REPO="${GITCG_REPO:-$HOME/genshin-GITCG}"
GITCG_CONDA_BASE="${GITCG_CONDA_BASE:-/fs00/software/anaconda/3}"
GITCG_BASE_ENV="${GITCG_BASE_ENV:-$HOME/.conda/envs/fa_env}"
GITCG_PYDEPS="${GITCG_PYDEPS:-$GITCG_REPO/.pydeps}"

export PATH="$GITCG_BASE_ENV/bin:$GITCG_CONDA_BASE/bin:$PATH"

export GITCG_ASSETS_OFFLINE=1
export PYTHONPATH="$GITCG_PYDEPS:$GITCG_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

cd "$GITCG_REPO"
