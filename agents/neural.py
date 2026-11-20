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
    # Set by ``reps.history.attach_history`` when the policy runs inside
    # ``run_match`` (or by the collector's capture wrapper).
    match_history: Any = None
    # Whether this checkpoint was TRAINED with the history channel. Checkpoints
    # saved before 2026-11-20 have no ``include_history`` marker and must keep
    # playing history-free, otherwise every historical measurement silently
    # becomes non-comparable (the model would see 32 tokens it never trained on).
    use_history: bool = False
    last_value: float = field(default=0.0, init=False)
    calls: int = field(default=0, init=False)

    @property
    def uses_match_history(self) -> bool:
        """Opt in to the observation encoder's history channel (per checkpoint)."""
        return bool(self.use_history)

    def __post_init__(self) -> None:
        import torch

        self._torch = torch
        self.model.to(self.device)
        self.model.eval()
        # P0: temperature sampling must be reproducible under paired seeds. The
        # historical implementation used the process-global torch RNG, so two
        # workers / reruns with the same task seed did not sample the same game.
        self._generator = torch.Generator(device=self.device)
        if self.seed is not None:
            self._generator.manual_seed(int(self.seed))

    def choose(self, built: BuiltDecisionContext) -> int:
        codes = list(int(c) for c in built.context.legal_low_level_codes)
        if not codes:
            return -1
        history = (
            self.match_history.snapshot()
            if self.use_history and self.match_history is not None
            else ()
        )
        observation = self.encoder.encode_context(
            built.context, history=history, include_training_targets=False
        )
        if not observation.option_features:
            return codes[0]
        tokens, token_mask, options, option_mask = collate([observation])
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)
        token_mask = token_mask.to(device)
        options = options.to(device)
        option_mask = option_mask.to(device)
        with self._torch.no_grad():
            logits, value = self.model(tokens, token_mask, options, option_mask)
        self.last_value = float(value.item())
        self.calls += 1
        if self.temperature and self.temperature > 0:
            probs = self._torch.softmax(logits / self.temperature, dim=-1)
            index = int(
                self._torch.multinomial(probs, 1, generator=self._generator).item()
            )
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
    device: str = "cpu",
    seed: int | None = None,
    use_history: bool | None = None,
) -> NeuralPolicy:
    from train.model import checkpoint_encoder_config, load_cvpn_model

    model = load_cvpn_model(checkpoint_path)
    if use_history is None:
        # Producer side: training stamps the flag into ``encoder_config``. A
        # checkpoint without the marker predates the history channel and must
        # keep running history-free so old measurements stay reproducible.
        saved = checkpoint_encoder_config(checkpoint_path) or {}
        use_history = bool(saved.get("include_history", False))
    return NeuralPolicy(
        model=model,
        encoder=encoder,
        name=name or f"neural:{checkpoint_path}",
        device=device,
        temperature=temperature,
        seed=seed,
        use_history=bool(use_history),
    )
