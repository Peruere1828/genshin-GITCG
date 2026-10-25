#!/usr/bin/env bash
# Group C: dense corpora for the uncovered decks whose scripted expert is actually
# strong (>=0.55 pooled against the pool), i.e. the ones where a specialist can win
# real games rather than inherit a losing deck. The weak-deck entries (skirk family
# at 0.19-0.38) are left out for now: their expert itself loses to the pool, so the
# ceiling there is not a strength statement.
#
# 2,000 expert games per deck (100 seeds x 20 opponents), same density as groups A/B.
# Collection seeds 11000-11600 are unused; certification will use its own fresh range.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python}"
collect() { # deck seed-start
  echo "=== $(date -Is) collect $1 (seeds $2+100)"
  "$PY" -m train.collect --deck "$1" --all-opponents \
    --seed-start "$2" --seed-count 100 --workers "${WORKERS:-10}" \
    --teacher-spec "expert:$1" --tag "gc_$1" --format compact --no-privileged-state
}
collect klee_mavuika_mualani 11000
collect nahida_kokomi_raiden 11100
collect shenhe_double_swirl 11200
collect electro_manifestation_nuke 11300
collect mualani_double_pyro 11400
collect dual_mualani_stacks 11500
echo "=== $(date -Is) group C collection done"
