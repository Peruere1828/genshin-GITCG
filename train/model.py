"""Toy CVPN (policy + value) network over the tokenized observation (PLAN.md WS3).

This is the L4 *toy* model: a small CPU-trainable network that consumes the ported
`reps.observation_encoder` outputs and produces (a) a logit per legal option and
(b) a scalar value. It is deliberately simple -- the point of L4 is to validate the
collect -> train -> evaluate -> persist loop and that the network can learn, not to
match the full SoG network.

Options are scored relative to their encoded features, so the model is invariant to
the process-global low-level action codes (see AGENTS.md "重放确定性坑").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn

from reps.observation_encoder import EncodedObservation


@dataclass(frozen=True)
class CVPNConfig:
    token_dim: int
    option_dim: int
    d_model: int = 128
    option_hidden: int = 256
    value_hidden: int = 128
    dropout: float = 0.0


class CVPN(nn.Module):
    def __init__(self, config: CVPNConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.token_proj = nn.Sequential(
            nn.Linear(config.token_dim, d), nn.LayerNorm(d), nn.SiLU()
        )
        self.option_proj = nn.Sequential(nn.Linear(config.option_dim, d), nn.SiLU())
        self.option_scorer = nn.Sequential(
            nn.Linear(2 * d, config.option_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.option_hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d, config.value_hidden), nn.SiLU(), nn.Linear(config.value_hidden, 1)
        )

    def forward(
        self,
        tokens: torch.Tensor,          # (B, T, token_dim)
        token_mask: torch.Tensor,      # (B, T) bool
        options: torch.Tensor,         # (B, K, option_dim)
        option_mask: torch.Tensor,     # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.token_proj(tokens)
        mask = token_mask.unsqueeze(-1).to(projected.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        global_vec = (projected * mask).sum(dim=1) / denom  # (B, d)

        option_vec = self.option_proj(options)  # (B, K, d)
        repeated = global_vec.unsqueeze(1).expand(-1, option_vec.size(1), -1)
        logits = self.option_scorer(torch.cat([repeated, option_vec], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~option_mask, float("-inf"))
        value = self.value_head(global_vec).squeeze(-1)  # (B,)
        return logits, value

    # -- persistence ------------------------------------------------------------
    def save(self, path: str, encoder_config: dict | None = None) -> None:
        """Persist weights + config, optionally the encoder config (for reload)."""
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "encoder_config": encoder_config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "CVPN":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(CVPNConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def checkpoint_encoder_config(path: str) -> dict | None:
    """Return the encoder config stored in a checkpoint, or ``None`` if absent.

    Training saves it so a checkpoint can be reloaded as a policy without also
    knowing which encoder vocabulary produced its inputs (``neural:<path>`` spec).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("encoder_config")


def config_for_encoder(encoder, **overrides) -> CVPNConfig:
    base = dict(token_dim=encoder.token_dim, option_dim=encoder.option_dim)
    base.update(overrides)
    return CVPNConfig(**base)


def collate(observations: Sequence[EncodedObservation]) -> tuple[torch.Tensor, ...]:
    """Pad a batch of encoded observations into dense tensors."""
    max_tokens = max(len(obs.token_features) for obs in observations)
    max_options = max(len(obs.option_features) for obs in observations)
    token_dim = len(observations[0].token_features[0])
    option_dim = len(observations[0].option_features[0]) if observations[0].option_features else 0

    b = len(observations)
    tokens = torch.zeros(b, max_tokens, token_dim)
    token_mask = torch.zeros(b, max_tokens, dtype=torch.bool)
    options = torch.zeros(b, max_options, option_dim)
    option_mask = torch.zeros(b, max_options, dtype=torch.bool)
    for i, obs in enumerate(observations):
        t = len(obs.token_features)
        tokens[i, :t] = torch.tensor(obs.token_features, dtype=torch.float32)
        token_mask[i, :t] = True
        k = len(obs.option_features)
        if k:
            options[i, :k] = torch.tensor(obs.option_features, dtype=torch.float32)
            option_mask[i, :k] = True
    return tokens, token_mask, options, option_mask
