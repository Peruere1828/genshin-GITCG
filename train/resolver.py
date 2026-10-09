"""Resolver / search teacher (PLAN.md WS3) -- scaffold + engine-bridge note.

**Engine bridge (verified 2026-10-09 on gitcg 0.21.0, PLAN D13/I10):** snapshots
taken at ``canResume:true`` boundary pauses (``Game.is_resumable()``) are exact
forks -- resume plus the same decisions reproduces the live terminal
byte-for-byte. Two rules: only snapshot at boundary pauses (``canResume:false``
mid-phase pauses replay phase-internal work and are not fork-safe), and mirror
game-level attrs on the fork (``envs/snapshot.fork_game(game_attrs=...)``). See
``envs/snapshot.py`` (``FORK_LIMITATION``) and ``scripts/probe_boundary_fork.py``.

Consequence for M3: continual resolving forks the live game directly (O(1) fork +
injected candidate + rollout). For bulk offline labeling there is also
``train/replay_branch.py`` (replay to the decision point, inject, rollout -> MC
action value) at O(depth) per branch.

``PriorResolver`` (no search, greedy network prior) stays usable for cold start
and self-distillation; the real resolver replaces it behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reps.action_adapter import BuiltDecisionContext

SEARCH_BLOCKED_REASON = (
    "The fork-based search resolver is not implemented yet (PriorResolver is the "
    "no-search placeholder). Forking itself is available and exact: snapshot at "
    "canResume:true boundary pauses and fork via envs/snapshot.fork_game"
    "(game_attrs=...) (PLAN D13/I10). Use train.replay_branch (replay-branch MC) "
    "for offline teacher labeling."
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
