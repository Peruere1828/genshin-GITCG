#!/usr/bin/env bash
# Stage 2 of the per-deck rollout, queued behind the current ladder chain.
#
# Evidence (2026-10-22..24): on-policy exploitation only pays off on decks whose base
# already matches the scripted expert, and data *per matchup* is what decides whether a
# base gets there (4-deck corpus at ~250 games/matchup -> val_acc 0.84 and dvalin at
# +0.07; 20-deck corpus at 40 games/matchup -> 0.69 and 7/8 decks worse). So the next
# step is a second deck group with the same density as the first:
#   * collect 2,000 expert games per deck (100 seeds x 20 opponents) for the four
#     decks closest to parity (SA -0.001, mika -0.024, aywolfskirk -0.068, natlan -0.085);
#   * screen the decks that were never certified with the best general base, so the
#     specialist set can be filled in rather than guessed.
# Everything runs under the same seed discipline (fresh collection ranges 9100-9400,
# fresh eval ranges from 400).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"

echo "=== $(date -Is) waiting for the ladder chain to finish"
while pgrep -f "run_ladder_chain2.sh" >/dev/null 2>&1; do sleep 60; done

collect() { # deck seed-start
  local deck="$1" start="$2"
  echo "=== $(date -Is) collect $deck (seeds ${start}+100, expert teacher)"
  "$PY" -m train.collect --deck "$deck" --all-opponents \
    --seed-start "$start" --seed-count 100 --workers 12 \
    --teacher-spec "expert:${deck}" --tag "gb_${deck}" \
    --format compact --no-privileged-state
}

collect superconduct_aggro 9100
collect superconduct_mika_attack 9200
collect aywolfskirk 9300
collect natlan_battleship 9400

BASE="${BASE:-data/checkpoints/kaggle/alld_bc_d256_e150.pt}"
screen() { # deck eval-seed-start
  local deck="$1" start="$2"
  echo "=== $(date -Is) screen $deck with $(basename "$BASE") (seeds ${start}-$((start+9)))"
  "$PY" -m scripts.eval_protocol --contestant "neural:${BASE}" --contestant-deck "$deck" \
    --seeds 10 --seed-start "$start" --workers 12 --strict-ci \
    --baseline "expert:${deck}" --tag "screen_alld_${deck}"
}

screen klee_mavuika_mualani 400
screen nahida_kokomi_raiden 410
screen dual_mualani_stacks 420
screen mualani_double_pyro 430
screen yelan_clorinde_ororon 440
screen electro_manifestation_nuke 450
screen shenhe_double_swirl 460
screen beidou_skirk_ayaka 470
screen skirk_chasca_freeze 480
screen skirk_ayaka_navia 490
screen trial_skirk_ayaka_navia 500
screen ice_water_battleship 510

echo "=== $(date -Is) group B collection + screening done"
