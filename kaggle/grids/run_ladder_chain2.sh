#!/usr/bin/env bash
# Ladder chain 2: keep climbing unyielding, and start dvalin from its *true* best base.
#
# Why dvalin restarts: earlier PG used the shared h4_mixed anchor, which was already
# 4pp behind the script on dvalin, so PG amplified noise (-0.063, -0.128). The best
# certified dvalin arm is `kaggle/h4m_bc_d256_e150.pt` at +0.0712 [+0.028,+0.115]
# (seeds 0-9), i.e. dvalin is parity+ once the base is right. The recipe is
# "start the ladder from the deck's best certified arm", not from one shared anchor.
#
# Pre-registered: every certification uses a fresh seed set; accept iff CI_low >= -0.02
# and the point estimate > 0.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
G="bash kaggle/grids/run_single_deck_rl.sh"

echo "=== $(date -Is) unyielding round 3 (collect 8300+, eval 200-209)"
$G unyielding_geo rl_ungeo3 data/checkpoints/rl_ungeo2_unyielding_geo.pt 8300 100 200

echo "=== $(date -Is) dvalin round 1 from best BC arm (collect 8400+, eval 210-219)"
$G dvalin_bonk rl_dvalin1 data/checkpoints/kaggle/h4m_bc_d256_e150.pt 8400 100 210

echo "=== $(date -Is) dvalin round 2 (collect 8500+, eval 220-229)"
$G dvalin_bonk rl_dvalin2 data/checkpoints/rl_dvalin1_dvalin_bonk.pt 8500 100 220

echo "=== $(date -Is) ladder chain 2 done"
