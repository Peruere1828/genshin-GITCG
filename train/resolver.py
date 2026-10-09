"""Resolver / search teacher (PLAN.md WS3) -- scaffold + engine-bridge note.

**Engine bridge (re-verified 2026-10-09 on gitcg 0.21.0):** the pybinding CAN
snapshot a mid-game state, round-trip it faithfully, and resume it
clone-deterministically -- but resume replays phase-internal work, so a snapshot
is **not** an exact fork of a live decision point (pauses are not all phase
boundaries; ``canResume`` does not reliably mark a replay-safe point). See
``envs/snapshot.py`` (``FORK_LIMITATION``) and ``scripts/probe_engine_snapshot.py``.

Consequence for M3: continual resolving cannot fork the live game through the
pybinding. The D12 fallback, ``train/replay_branch.py``, is the supported
search teacher today: it *replays* to any decision point (the full game is
deterministic), injects a candidate action, and finishes with a rollout policy,
giving a Monte-Carlo action value. A true live-fork bridge (TS server, non-async
serialization contract) remains future work.

``PriorResolver`` (no search, greedy network prior) stays usable for cold start
and self-distillation; the real resolver replaces it behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reps.action_adapter import BuiltDecisionContext

SEARCH_BLOCKED_REASON = (
    "gitcg 0.21.0 pybinding snapshots round-trip faithfully but resume replays "
    "phase-internal work, so a snapshot is not an exact fork of a live decision "
    "point. Use train.replay_branch (D12 replay-branch MC) as the teacher. See "
    "envs/snapshot.py / train/README.md."
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
