#!/usr/bin/env bash
# Round 2 for the two decks that reached parity and just started their ladders.
#
# The ladder needs 2-3 rounds before it jumps (dgeo: +0.100 -> +0.058 -> +0.188;
# unyielding: -0.040 -> +0.043 -> +0.105), so a single round that only lands at +0.03
# is not a verdict yet. Both rounds here start from the round-1 policy and use fresh
# seed ranges.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
G="bash kaggle/grids/run_single_deck_rl.sh"

echo "=== $(date -Is) shenhe round 2 (collect 13200+, eval 850-859)"
$G shenhe_double_swirl rl_shenhe2 data/checkpoints/rl_shenhe1_shenhe_double_swirl.pt 13200 100 850

echo "=== $(date -Is) electro round 2 (collect 13300+, eval 860-869)"
$G electro_manifestation_nuke rl_electro2 data/checkpoints/rl_electro1_electro_manifestation_nuke.pt 13300 100 860

echo "=== $(date -Is) round 2 done"
