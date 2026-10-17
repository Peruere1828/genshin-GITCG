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
from train.belief_head import BeliefHead, BeliefOutput


@dataclass(frozen=True)
class CVPNConfig:
    token_dim: int
    option_dim: int
    d_model: int = 128
    option_hidden: int = 256
    value_hidden: int = 128
    dropout: float = 0.0
    # M3-CPU S-C (G7): optional belief head. ``belief_group_dim`` = the
    # ``reps.belief_groups`` marginal count, ``belief_card_dim`` = the encoder's
    # card-vocabulary size. Both 0 (the default) keeps the historical
    # policy+value network and loads pre-S-C checkpoints unchanged.
    belief_group_dim: int = 0
    belief_card_dim: int = 0
    belief_hidden: int = 128
    # H3 (value calibration): "identity" keeps the historical linear head (default);
    # "sigmoid" makes the head emit a probability, which is what Brier/BCE-trained
    # runs and the search frontier want. Recorded in the checkpoint, so consumers
    # always receive a probability either way.
    value_activation: str = "identity"


def activate_value(value: torch.Tensor, config) -> torch.Tensor:
    """Map the value head's raw output to the space its loss/consumers expect.

    ``identity`` (default) keeps the historical linear scalar; ``sigmoid`` makes it
    a probability in (0, 1) — required for Brier/BCE-trained value heads (H3) and
    safe for the search frontier, which treats the value as P(win).
    """
    activation = getattr(config, "value_activation", "identity")
    if activation == "sigmoid":
        return torch.sigmoid(value)
    if activation in ("identity", "", None):
        return value
    raise ValueError(f"unknown value_activation: {activation!r}")


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
        self.has_belief = bool(config.belief_group_dim or config.belief_card_dim)
        if self.has_belief:
            self.belief_head = BeliefHead(
                d,
                group_dim=config.belief_group_dim,
                card_dim=config.belief_card_dim,
                hidden=config.belief_hidden,
            )

    def forward_all(
        self,
        tokens: torch.Tensor,          # (B, T, token_dim)
        token_mask: torch.Tensor,      # (B, T) bool
        options: torch.Tensor,         # (B, K, option_dim)
        option_mask: torch.Tensor,     # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor, BeliefOutput | None]:
        """Policy logits, value, and the (optional) belief prediction in one pass."""
        projected = self.token_proj(tokens)
        mask = token_mask.unsqueeze(-1).to(projected.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        global_vec = (projected * mask).sum(dim=1) / denom  # (B, d)

        option_vec = self.option_proj(options)  # (B, K, d)
        repeated = global_vec.unsqueeze(1).expand(-1, option_vec.size(1), -1)
        logits = self.option_scorer(torch.cat([repeated, option_vec], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~option_mask, float("-inf"))
        value = activate_value(self.value_head(global_vec).squeeze(-1), self.config)  # (B,)
        belief = self.belief_head(global_vec) if self.has_belief else None
        return logits, value, belief

    def forward(
        self,
        tokens: torch.Tensor,          # (B, T, token_dim)
        token_mask: torch.Tensor,      # (B, T) bool
        options: torch.Tensor,         # (B, K, option_dim)
        option_mask: torch.Tensor,     # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value, _ = self.forward_all(tokens, token_mask, options, option_mask)
        return logits, value

    # -- persistence ------------------------------------------------------------
    def save(self, path: str, encoder_config: dict | None = None) -> None:
        """Persist weights + config, optionally the encoder config (for reload)."""
        torch.save(
            {
                "config": asdict(self.config),
                "arch": "cvpn",
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


def config_for_encoder(encoder, *, belief: bool = False, **overrides) -> CVPNConfig:
    base = dict(token_dim=encoder.token_dim, option_dim=encoder.option_dim)
    if belief:
        base["belief_group_dim"] = int(encoder.belief_group_dim)
        base["belief_card_dim"] = int(encoder.belief_histogram_dim)
    base.update(overrides)
    return CVPNConfig(**base)


def load_cvpn_model(path: str) -> nn.Module:
    """Load any CVPN-interface checkpoint, dispatching on its ``arch`` field.

    ``CVPN`` and ``TransformerCVPN`` share ``forward(tokens, token_mask,
    options, option_mask) -> (logits, value)``, so callers (``neural:<ckpt>``
    specs, arena, eval protocol) can treat them uniformly. Checkpoints saved
    before the ``arch`` field existed default to the mean-pool ``CVPN``.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = payload.get("arch")
    if arch == "transformer_cvpn":
        model: nn.Module = TransformerCVPN(TransformerCVPNConfig(**payload["config"]))
    elif arch in (None, "cvpn"):
        model = CVPN(CVPNConfig(**payload["config"]))
    else:
        raise ValueError(f"unknown checkpoint arch: {arch!r} in {path}")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@dataclass(frozen=True)
class TransformerCVPNConfig:
    """Config for the token-transformer CVPN (M3 exploration, PLAN.md WS3).

    A step up from the mean-pool toy ``CVPN``: self-attention over the token
    sequence lets tokens interact (character/entity/summary tokens), and the
    option head *cross-attends* to the tokens instead of only seeing a pooled
    summary. Still small enough to train on the WSL MX550 (fp32, d_model 128).
    """

    token_dim: int
    option_dim: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dim_feedforward: int = 256
    option_hidden: int = 256
    value_hidden: int = 128
    dropout: float = 0.0
    # M3-CPU S-C (G7) belief head, mirroring :class:`CVPNConfig`.
    belief_group_dim: int = 0
    belief_card_dim: int = 0
    belief_hidden: int = 128
    value_activation: str = "identity"


class TransformerCVPN(nn.Module):
    """Attention-based CVPN: token self-attention + option cross-attention.

    Same interface as :class:`CVPN` (``forward(tokens, token_mask, options,
    option_mask) -> (logits, value)``), so it drops into ``train_bc`` and the
    neural policy unchanged. Option logits for padded options are ``-inf``.
    """

    def __init__(self, config: TransformerCVPNConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        d = config.d_model
        self.token_proj = nn.Sequential(nn.Linear(config.token_dim, d), nn.LayerNorm(d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.option_proj = nn.Sequential(nn.Linear(config.option_dim, d), nn.SiLU())
        self.cross_attn = nn.MultiheadAttention(
            d, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.option_scorer = nn.Sequential(
            nn.Linear(2 * d, config.option_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.option_hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d, config.value_hidden), nn.SiLU(), nn.Linear(config.value_hidden, 1)
        )
        self.has_belief = bool(config.belief_group_dim or config.belief_card_dim)
        if self.has_belief:
            self.belief_head = BeliefHead(
                d,
                group_dim=config.belief_group_dim,
                card_dim=config.belief_card_dim,
                hidden=config.belief_hidden,
            )

    def forward_all(
        self,
        tokens: torch.Tensor,          # (B, T, token_dim)
        token_mask: torch.Tensor,      # (B, T) bool
        options: torch.Tensor,         # (B, K, option_dim)
        option_mask: torch.Tensor,     # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor, BeliefOutput | None]:
        pad = ~token_mask  # True = ignore in attention
        encoded = self.encoder(self.token_proj(tokens), src_key_padding_mask=pad)  # (B,T,d)

        mask = token_mask.unsqueeze(-1).to(encoded.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (encoded * mask).sum(dim=1) / denom  # (B, d)

        option_vec = self.option_proj(options)  # (B, K, d)
        attended, _ = self.cross_attn(
            query=option_vec, key=encoded, value=encoded, key_padding_mask=pad
        )  # (B, K, d)
        logits = self.option_scorer(torch.cat([attended, option_vec], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~option_mask, float("-inf"))
        value = activate_value(self.value_head(pooled).squeeze(-1), self.config)
        belief = self.belief_head(pooled) if self.has_belief else None
        return logits, value, belief

    def forward(
        self,
        tokens: torch.Tensor,          # (B, T, token_dim)
        token_mask: torch.Tensor,      # (B, T) bool
        options: torch.Tensor,         # (B, K, option_dim)
        option_mask: torch.Tensor,     # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value, _ = self.forward_all(tokens, token_mask, options, option_mask)
        return logits, value

    def save(self, path: str, encoder_config: dict | None = None) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "arch": "transformer_cvpn",
                "state_dict": self.state_dict(),
                "encoder_config": encoder_config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TransformerCVPN":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(TransformerCVPNConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def transformer_config_for_encoder(encoder, *, belief: bool = False, **overrides) -> TransformerCVPNConfig:
    base = dict(token_dim=encoder.token_dim, option_dim=encoder.option_dim)
    if belief:
        base["belief_group_dim"] = int(encoder.belief_group_dim)
        base["belief_card_dim"] = int(encoder.belief_histogram_dim)
    base.update(overrides)
    return TransformerCVPNConfig(**base)


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
