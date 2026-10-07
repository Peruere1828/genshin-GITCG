"""Resolver / search teacher (PLAN.md WS3) -- scaffold + engine-limitation note.

**Blocker (verified 2026-10-07 on gitcg 0.21.0):** branching search (continual
resolving needs to evaluate multiple candidate actions from the same state)
requires cloning a mid-game state. The pybinding exports state via
``State.toJson`` which serializes the log entry with ``canResume: false``
(``packages/core/src/game.ts`` / ``cbinding/js/main.ts``), and a round-trip
through ``State(json=...)`` + ``Game`` does **not** resume play
(``Game.is_resumable() == False``). The TS server pause path does support
``canResume: true``; the Python binding does not expose it.

Consequences for M3:
- A continual-resolving search must either (a) run fully inside a live game
  callback using the *real* engine without cloning (not possible: search needs
  branching), (b) get resumable snapshots exposed by extending the pybinding /
  using the TS server, or (c) use a learned model for rollouts (research).

Until then, the "teacher" for distillation is a non-search policy: the scripted
expert (behavior cloning, implemented in ``train/pipeline.py``) or the current
network's own prior (self-distillation). This module provides the interface the
real resolver will implement, plus a prior-based resolver usable today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reps.action_adapter import BuiltDecisionContext

SEARCH_BLOCKED_REASON = (
    "gitcg 0.21.0 pybinding cannot clone/resume mid-game state "
    "(State.toJson writes canResume:false; Game.is_resumable()==False), so "
    "branching search over candidate actions is unsupported. See train/README.md."
)


class SearchUnavailable(RuntimeError):
    pass


@dataclass
class PriorResolver:
    """No-search resolver: returns the network's greedy action over legal options.

    It also exposes a soft distribution over options (the policy prior) which the
    distillation loop can use as a target. It is intentionally *not* a search.
    """

    model: Any
    encoder: Any
    temperature: float = 1.0

    def distribution(self, built: BuiltDecisionContext) -> list[float]:
        import torch

        codes = list(int(c) for c in built.context.legal_low_level_codes)
        if not codes:
            return []
        observation = self.encoder.encode_context(
            built.context, history=(), include_training_targets=False
        )
        if not observation.option_features:
            return [1.0 / len(codes)] * len(codes)
        from train.model import collate

        tokens, token_mask, options, option_mask = collate([observation])
        with torch.no_grad():
            logits, _ = self.model(tokens, token_mask, options, option_mask)
            probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)[0]
        return [float(p) for p in probs[: len(codes)]]

    def choose(self, built: BuiltDecisionContext) -> int:
        codes = list(int(c) for c in built.context.legal_low_level_codes)
        if not codes:
            return -1
        probs = self.distribution(built)
        return codes[max(range(len(codes)), key=lambda i: probs[i])]


def require_search_available() -> None:
    raise SearchUnavailable(SEARCH_BLOCKED_REASON)
