"""Neural policy: drive a trained CVPN through the env's ``Policy`` interface.

This closes the loop -- a trained checkpoint can be evaluated in the same arena as
the scripted experts and used for self-play (PLAN.md WS3/WS4 pure-policy mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reps.action_adapter import BuiltDecisionContext
from train.model import collate


@dataclass
class NeuralPolicy:
    model: Any
    encoder: Any
    name: str = "neural"
    device: str = "cpu"
    temperature: float = 0.0
    seed: int | None = None
    last_value: float = field(default=0.0, init=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        import torch

        self._torch = torch
        self.model.eval()

    def choose(self, built: BuiltDecisionContext) -> int:
        codes = list(int(c) for c in built.context.legal_low_level_codes)
        if not codes:
            return -1
        observation = self.encoder.encode_context(
            built.context, history=(), include_training_targets=False
        )
        if not observation.option_features:
            return codes[0]
        tokens, token_mask, options, option_mask = collate([observation])
        with self._torch.no_grad():
            logits, value = self.model(tokens, token_mask, options, option_mask)
        self.last_value = float(value.item())
        self.calls += 1
        if self.temperature and self.temperature > 0:
            probs = self._torch.softmax(logits / self.temperature, dim=-1)
            index = int(self._torch.multinomial(probs, 1).item())
        else:
            index = int(self._torch.argmax(logits, dim=-1).item())
        index = max(0, min(index, len(codes) - 1))
        return codes[index]


def load_neural_policy(
    checkpoint_path: str,
    encoder: Any,
    *,
    name: str | None = None,
    temperature: float = 0.0,
) -> NeuralPolicy:
    from train.model import CVPN

    model = CVPN.load(checkpoint_path)
    return NeuralPolicy(
        model=model,
        encoder=encoder,
        name=name or f"neural:{checkpoint_path}",
        temperature=temperature,
    )
