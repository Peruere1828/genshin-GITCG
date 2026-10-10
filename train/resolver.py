"""Resolver / search teacher (PLAN.md WS3).

**Practical fork search (delivered 2026-10-09, L5.4/D13/I10):** the fork-based
resolver now exists as ``envs.fork_search.ForkSearchPolicy``: at a decision it
forks the latest ``is_resumable()`` boundary snapshot (``envs.fork_bridge``,
subprocess pool), injects each candidate, rolls out, and averages outcomes. The
base policy is always one of the evaluated candidates, so a search never ignores
the expert action. ``run_search_match`` drives a full game with it.

**Engine bridge (verified 2026-10-09 on gitcg 0.21.0):** snapshots taken at
``canResume:true`` boundary pauses (``Game.is_resumable()``) are exact forks --
resume plus the same decisions reproduces the live terminal byte-for-byte. Two
rules: only snapshot at boundary pauses (``canResume:false`` mid-phase pauses
replay phase-internal work and are not fork-safe), and mirror game-level attrs on
the fork (``envs/snapshot.fork_game(game_attrs=...)``); the driver and bridge
handle both. See ``scripts/probe_boundary_fork.py``.

For bulk offline labeling there is also ``train/replay_branch.py`` (replay to the
decision point, inject, rollout -> MC action value) at O(depth) per branch; the
fork bridge is the O(1) online counterpart.

``PriorResolver`` (no search, greedy network prior) stays usable for cold start
and self-distillation; the fork resolver is the real search behind the same idea.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reps.action_adapter import BuiltDecisionContext

SEARCH_UNAVAILABLE_REASON = (
    "No fork bridge was provided. Fork search is available: use "
    "envs.fork_search.run_search_match / ForkSearchPolicy with a subprocess "
    "envs.fork_bridge.ForkBridge(workers>=1). Forking must run in a subprocess "
    "(nesting two engines in one live callback corrupts the JS runtime). Use "
    "train.replay_branch for offline teacher labeling."
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


def require_search_available(bridge: Any | None = None) -> None:
    """Raise unless a subprocess fork bridge is available for online search."""
    if bridge is not None and getattr(bridge, "workers", 0) != 0:
        return
    raise SearchUnavailable(SEARCH_UNAVAILABLE_REASON)


def prior_resolver_from_checkpoint(checkpoint: str, *, temperature: float = 1.0) -> PriorResolver:
    """Build a ``PriorResolver`` (policy prior + greedy) from a CVPN checkpoint.

    The encoder is rebuilt from the config saved beside the checkpoint (see
    ``train.model.checkpoint_encoder_config``), falling back to the default encoder
    for older checkpoints. Feed it to ``ForkSearchPolicy.prior``/``base`` or pass it
    as a ``base_policies`` entry of ``envs.fork_search.run_search_match``.
    """
    from train.model import CVPN, checkpoint_encoder_config

    model = CVPN.load(checkpoint)
    config = checkpoint_encoder_config(checkpoint)
    if config is not None:
        from reps.observation_encoder import TokenObservationEncoder

        encoder = TokenObservationEncoder.from_dict(config)
    else:
        from envs.observation import default_encoder

        encoder = default_encoder()
    return PriorResolver(model=model, encoder=encoder, temperature=temperature)


def neural_rollout_spec(checkpoint: str, *, temperature: float | None = None) -> str:
    """Policy spec string for using a checkpoint as a rollout policy in the workers."""
    spec = f"neural:{checkpoint}"
    if temperature is not None:
        spec = f"{spec}#t={temperature}"
    return spec
