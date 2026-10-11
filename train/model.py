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
    # MX550 fused-kernel fast path (train/cuda/cvpn_fused.cu). Off by default;
    # falls back to the eager path if the extension cannot be built.
    use_cuda_kernels: bool = False


class CVPN(nn.Module):
    # Row-count crossover below which the fused projection GEMMs beat cuBLAS
    # (measured on the MX550; see reports/explore/mx550_kernels.md).
    _FUSED_PROJ_TOKEN_ROWS = 128
    _FUSED_PROJ_OPTION_ROWS = 256

    def __init__(self, config: CVPNConfig) -> None:
        super().__init__()
        self.config = config
        self.use_cuda_kernels = bool(config.use_cuda_kernels)
        self._cuda_kernels_module = None
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
        if (
            self.use_cuda_kernels
            and tokens.is_cuda
            and self.config.dropout == 0.0
            and not self.training
        ):
            fast = self._forward_cuda_kernels(tokens, token_mask, options, option_mask)
            if fast is not None:
                return fast
        return self._forward_eager(tokens, token_mask, options, option_mask)

    def _forward_cuda_kernels(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """MX550 fused-kernel forward (eval, fp32). Returns ``None`` to fall back."""
        if self._cuda_kernels_module is None:
            from train.cuda_kernels import cvpn_kernels

            self._cuda_kernels_module = cvpn_kernels() or False
        kernels = self._cuda_kernels_module
        if not kernels:
            return None

        tp = self.token_proj
        op = self.option_proj
        token_rows = tokens.size(0) * tokens.size(1)
        option_rows = options.size(0) * options.size(1)
        # Fused projection GEMMs win only at small row counts (below cuBLAS's
        # crossover); use them there, cuBLAS + fused epilogue otherwise.
        if token_rows <= self._FUSED_PROJ_TOKEN_ROWS:
            projected = kernels.token_proj_ln_silu(
                tokens.reshape(-1, tokens.size(-1)),
                tp[0].weight, tp[0].bias, tp[1].weight, tp[1].bias, float(tp[1].eps),
            ).view(tokens.size(0), tokens.size(1), -1)
        else:
            projected = kernels.layernorm_silu(
                tp[0](tokens), tp[1].weight, tp[1].bias, float(tp[1].eps)
            )
        global_vec = kernels.masked_mean_pool(projected, token_mask)

        if option_rows <= self._FUSED_PROJ_OPTION_ROWS:
            option_vec = kernels.linear_silu(
                options.reshape(-1, options.size(-1)), op[0].weight, op[0].bias
            ).view(options.size(0), options.size(1), -1)
        else:
            option_vec = torch.nn.functional.silu(op[0](options))
        repeated = global_vec.unsqueeze(1).expand(-1, option_vec.size(1), -1)
        hidden_pre = self.option_scorer[0](torch.cat([repeated, option_vec], dim=-1))
        last = self.option_scorer[3]
        logits = kernels.scorer_tail(
            hidden_pre, last.weight.squeeze(0), float(last.bias), option_mask
        )
        vh = self.value_head
        value = kernels.value_head(
            global_vec, vh[0].weight, vh[0].bias, vh[2].weight.squeeze(0), float(vh[2].bias)
        )
        return logits, value

    def _forward_eager(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
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


def config_for_encoder(encoder, **overrides) -> CVPNConfig:
    base = dict(token_dim=encoder.token_dim, option_dim=encoder.option_dim)
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

    def forward(
        self,
        tokens: torch.Tensor,          # (B, T, token_dim)
        token_mask: torch.Tensor,      # (B, T) bool
        options: torch.Tensor,         # (B, K, option_dim)
        option_mask: torch.Tensor,     # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        value = self.value_head(pooled).squeeze(-1)
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


def transformer_config_for_encoder(encoder, **overrides) -> TransformerCVPNConfig:
    base = dict(token_dim=encoder.token_dim, option_dim=encoder.option_dim)
    base.update(overrides)
    return TransformerCVPNConfig(**base)


def enable_cuda_kernels(model: "CVPN") -> bool:
    """Turn on the MX550 fused-kernel fast path on ``model`` (returns availability).

    Only affects the eager ``CVPN`` (the transformer has its own path). Falls back
    silently to eager when the extension cannot be built.
    """
    if not isinstance(model, CVPN):
        return False
    from train.cuda_kernels import kernels_available

    model.use_cuda_kernels = True
    model._cuda_kernels_module = None
    return kernels_available()


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
