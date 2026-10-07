# NOTICE

This project is licensed AGPL-3.0-only.

## Vendored / ported code

The following directories contain code ported (copied and adapted, not patched
in place) from upstream AGPL-3.0 projects. Each ported file carries a header
comment pointing at its origin.

- `reps/` — ported from **Rebel_base_RL** `research/world_model/src/gitcg_world_model`
  (action representation: schema, snapshot, action hierarchy/taxonomy,
  action adapter, public-state masking, semantic priors, baseline agents, replay).
- `agents/scripted/` — ported from **Rebel_base_RL**
  `research/world_model/src/gitcg_expert_system` (the 20/22 scripted deck rules
  used as fixed evaluation opponents, plus their asset/profile/deck-share-code
  plumbing).

Upstream: https://github.com/piovium/genius-invokation (engine, `gitcg` binding)
and the Rebel_base_RL reference clone under `refs/` (read-only, not committed).

No mihoyo card art / assets are redistributed. Card metadata is fetched on demand
into `data/` (gitignored) via the public GI-TCG assets API.

## Engine reproducibility note

`gitcg`'s deck-pile shuffle uses JavaScript `Math.random()` and is **not**
controlled by the engine's seeded random generator. To keep "same seed -> same
game" we pre-shuffle each pile in Python from a derived seed and pass
`NO_SHUFFLE=1` to the engine (`envs/match.py`). Dice rolls and draws use the
engine's seeded RNG and are already deterministic given the pile order.
