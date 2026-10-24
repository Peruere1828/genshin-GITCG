#!/usr/bin/env bash
# Wave 2 of the exploitation ladder, queued behind the Group B screening.
#
# Each deck starts from *its own* best certified base -- the rule bought by dvalin
# (the shared h4_mixed anchor was 4pp behind the script on dvalin, so PG amplified
# noise there; the dense per-deck base is 7pp ahead). Group B bases come from the
# `wave1` Kaggle grid (2,000 expert games per deck), SA's from the dense sa_all corpus.
#
# Pre-registered: each certification uses a fresh seed set, accept iff CI_low >= -0.02
# and the point estimate > 0. Missing bases are skipped, not faked.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
G="bash kaggle/grids/run_single_deck_rl.sh"

echo "=== $(date -Is) waiting for the Group B screening"
while pgrep -f "queue_group_b.sh" >/dev/null 2>&1; do sleep 60; done

ladder() { # deck tag base seed-start eval-seed-start
  local deck="$1" tag="$2" base="$3" seeds="$4" evalstart="$5"
  if [ ! -f "$base" ]; then
    echo "=== $(date -Is) skip $deck ($(basename "$base") not downloaded yet)"
    return 0
  fi
  $G "$deck" "$tag" "$base" "$seeds" 100 "$evalstart"
}

ladder superconduct_aggro rl_sa1 data/checkpoints/kaggle/sa_bc_d256_e150.pt 9500 520
ladder superconduct_mika_attack rl_mika1 data/checkpoints/kaggle/gb_mika_bc_d256_e150.pt 9600 530
ladder aywolfskirk rl_aywolf1 data/checkpoints/kaggle/gb_aywolf_bc_d256_e150.pt 9700 540
ladder natlan_battleship rl_natlan1 data/checkpoints/kaggle/gb_natlan_bc_d256_e150.pt 9800 550
ladder dvalin_bonk rl_dvalin3 data/checkpoints/rl_dvalin2_dvalin_bonk.pt 8700 560

echo "=== $(date -Is) wave 2 done"
