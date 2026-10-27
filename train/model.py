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
from typing import Mapping, Sequence

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
    # A2 (2026-10-27): learned card/entity identity. The encoder squashes a definition
    # id into one scalar (``id / 400000``), which is exactly invertible, so the model
    # can bucket it and look it up in an embedding table. ``id_buckets = 0`` keeps the
    # historical architecture and makes every pre-A2 checkpoint load unchanged.
    id_buckets: int = 0
    id_emb_dim: int = 32
    token_def_index: int = -1     # < 0 -> resolve from the encoder layout
    option_def_index: int = -1


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
        self.has_ids = bool(config.id_buckets and config.id_buckets > 1)
        if self.has_ids:
            from reps.observation_encoder import definition_feature_indices

            default_token, default_option = definition_feature_indices()
            self.token_def_index = (
                config.token_def_index if config.token_def_index >= 0 else default_token
            )
            self.option_def_index = (
                config.option_def_index if config.option_def_index >= 0 else default_option
            )
            self.token_id_emb = nn.Embedding(config.id_buckets, config.id_emb_dim, padding_idx=0)
            self.option_id_emb = nn.Embedding(config.id_buckets, config.id_emb_dim, padding_idx=0)
            self.id_proj = nn.Sequential(nn.Linear(config.id_emb_dim, d), nn.SiLU())
            self.option_id_proj = nn.Sequential(nn.Linear(config.id_emb_dim, d), nn.SiLU())

    def _with_ids(self, projected: torch.Tensor, raw: torch.Tensor, *, option: bool) -> torch.Tensor:
        """Add the learned identity embedding of each token/option to its projection."""
        from reps.id_features import option_definition_ids, token_definition_ids

        if not self.has_ids:
            return projected
        if option:
            ids = option_definition_ids(raw, self.option_def_index, self.config.id_buckets)
            return projected + self.option_id_proj(self.option_id_emb(ids))
        ids = token_definition_ids(raw, self.token_def_index, self.config.id_buckets)
        return projected + self.id_proj(self.token_id_emb(ids))

    def forward_all(
        self,
        tokens: torch.Tensor,          # (B, T, token_dim)
        token_mask: torch.Tensor,      # (B, T) bool
        options: torch.Tensor,         # (B, K, option_dim)
        option_mask: torch.Tensor,     # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor, BeliefOutput | None]:
        """Policy logits, value, and the (optional) belief prediction in one pass."""
        projected = self._with_ids(self.token_proj(tokens), tokens, option=False)
        mask = token_mask.unsqueeze(-1).to(projected.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        global_vec = (projected * mask).sum(dim=1) / denom  # (B, d)

        option_vec = self._with_ids(self.option_proj(options), options, option=True)  # (B, K, d)
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


ID_PARAM_PREFIXES = ("token_id_emb.", "option_id_emb.", "id_proj.", "option_id_proj.")


def zero_init_identity_branch(model: nn.Module) -> bool:
    """Make the A2 identity branch a no-op at initialisation.

    Used when a pre-A2 checkpoint is upgraded to ``id_buckets > 0``: the source
    has no identity parameters, so the new branch must contribute exactly zero
    until it is trained.  The last linear layer of both projections is zeroed
    (SiLU(0) == 0); the embedding tables stay randomly initialised and start
    receiving gradients once the projection moves off zero.
    """
    if not getattr(model, "has_ids", False):
        return False
    for name in ("id_proj", "option_id_proj"):
        proj = getattr(model, name, None)
        if proj is None:
            continue
        linears = [m for m in proj.modules() if isinstance(m, nn.Linear)]
        if not linears:
            raise ValueError(f"{name} has no Linear layer")
        nn.init.zeros_(linears[-1].weight)
        nn.init.zeros_(linears[-1].bias)
    return True


def load_state_dict_compatible(
    model: nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> list[str]:
    """Load a pre-A2 checkpoint into an identity-enabled model.

    Missing keys are allowed only for the A2 identity parameters; any other
    mismatch is an architecture error.  Returns the missing identity keys so
    callers can report what was newly initialised.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed = [key for key in missing if key.startswith(ID_PARAM_PREFIXES)]
    disallowed = [key for key in missing if key not in allowed]
    if disallowed or unexpected:
        raise RuntimeError(
            "state_dict mismatch: "
            f"missing={sorted(disallowed)} unexpected={sorted(unexpected)}"
        )
    return allowed


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
    # A2: learned card/entity identity (see CVPNConfig).
    id_buckets: int = 0
    id_emb_dim: int = 32
    token_def_index: int = -1
    option_def_index: int = -1


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
        self.has_ids = bool(config.id_buckets and config.id_buckets > 1)
        if self.has_ids:
            from reps.observation_encoder import definition_feature_indices

            default_token, default_option = definition_feature_indices()
            self.token_def_index = (
                config.token_def_index if config.token_def_index >= 0 else default_token
            )
            self.option_def_index = (
                config.option_def_index if config.option_def_index >= 0 else default_option
            )
            self.token_id_emb = nn.Embedding(config.id_buckets, config.id_emb_dim, padding_idx=0)
            self.option_id_emb = nn.Embedding(config.id_buckets, config.id_emb_dim, padding_idx=0)
            self.id_proj = nn.Sequential(nn.Linear(config.id_emb_dim, d), nn.SiLU())
            self.option_id_proj = nn.Sequential(nn.Linear(config.id_emb_dim, d), nn.SiLU())

    def _with_ids(self, projected: torch.Tensor, raw: torch.Tensor, *, option: bool) -> torch.Tensor:
        """Add the learned identity embedding of each token/option to its projection."""
        from reps.id_features import option_definition_ids, token_definition_ids

        if not self.has_ids:
            return projected
        if option:
            ids = option_definition_ids(raw, self.option_def_index, self.config.id_buckets)
            return projected + self.option_id_proj(self.option_id_emb(ids))
        ids = token_definition_ids(raw, self.token_def_index, self.config.id_buckets)
        return projected + self.id_proj(self.token_id_emb(ids))

    def forward_all(
        self,
        tokens: torch.Tensor,          # (B, T, token_dim)
        token_mask: torch.Tensor,      # (B, T) bool
        options: torch.Tensor,         # (B, K, option_dim)
        option_mask: torch.Tensor,     # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor, BeliefOutput | None]:
        pad = ~token_mask  # True = ignore in attention
        embedded = self._with_ids(self.token_proj(tokens), tokens, option=False)
        encoded = self.encoder(embedded, src_key_padding_mask=pad)  # (B,T,d)

        mask = token_mask.unsqueeze(-1).to(encoded.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (encoded * mask).sum(dim=1) / denom  # (B, d)

        option_vec = self._with_ids(self.option_proj(options), options, option=True)  # (B, K, d)
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
