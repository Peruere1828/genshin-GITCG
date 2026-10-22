#!/usr/bin/env bash
# Continue the exploitation ladder: one more dgeo round (does it keep climbing?) and
# the first two unyielding rounds (does the recipe transfer to another parity deck?).
# Every certification uses seed sets never touched before.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
G="bash kaggle/grids/run_single_deck_rl.sh"

echo "=== $(date -Is) dgeo round 4 (seeds 8000+, eval 170-179)"
$G double_geo_navia rl_dgeo4 data/checkpoints/rl_dgeo3_double_geo_navia.pt 8000 100 170

echo "=== $(date -Is) unyielding round 1 (seeds 8100+, eval 180-189)"
$G unyielding_geo rl_ungeo1 data/checkpoints/h4_mixed_d256_e15.pt 8100 100 180

echo "=== $(date -Is) unyielding round 2 (seeds 8200+, eval 190-199)"
$G unyielding_geo rl_ungeo2 data/checkpoints/rl_ungeo1_unyielding_geo.pt 8200 100 190

echo "=== $(date -Is) chain done"
