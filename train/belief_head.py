"""Belief head + proper-scoring belief loss (PLAN.md M3-CPU S-C, gap G7).

The CVPN's policy head answers "what to play"; the value head answers "how good
is this public state for me". The charter's CVPN additionally predicts the
**opponent's hidden-zone posterior** so the search frontier and the continual
resolving belief update have a learned, public-information-only belief to lean on
(``train/CONTINUAL_RESOLVING.md`` §2 row G7).

This module is engine-free and self-contained:

* :class:`BeliefHead` maps the pooled public-state embedding to
  ``belief_groups`` presence logits (the S-A / ``reps.belief_groups`` marginals),
  an opponent-hand card posterior (softmax over the encoder's card vocabulary) and
  two scalars (opponent hand size, burst-ready count);
* :func:`belief_loss` scores those predictions against the **god's-eye**
  ``reps.observation_encoder.BeliefTarget`` labels with proper scoring rules:
  binary cross-entropy for the Bernoulli group/archetype marginals and
  cross-entropy (KL up to the target entropy) for the card-identity posterior,
  plus bounded-scalar MSE.

The target is a *label*, never an input: the encoder derives it from
``context.full_state`` and the model only ever consumes ``player_view``-based
tokens/options, so no hidden truth enters the network (charter §5.2; the
info-leak gate in ``tests/test_info_leak.py`` pins the inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BeliefOutput:
    """Belief-head predictions for a batch (``None`` when that sub-head is off)."""

    group_hand_logits: torch.Tensor | None
    group_deck_logits: torch.Tensor | None
    hand_card_logits: torch.Tensor | None
    hand_size: torch.Tensor
    burst_ready: torch.Tensor


@dataclass
class BeliefTargetBatch:
    """Dense belief labels for a batch (all tensors on one device)."""

    group_hand_presence: torch.Tensor  # (B, G)
    group_deck_presence: torch.Tensor  # (B, G)
    hand_histogram: torch.Tensor  # (B, V) raw counts (unnormalised)
    hand_size: torch.Tensor  # (B,)
    burst_ready: torch.Tensor  # (B,)
    mask: torch.Tensor  # (B,) bool: rows carrying any belief signal


class BeliefHead(nn.Module):
    """Small MLP head over the pooled public embedding.

    ``group_dim`` is ``reps.belief_groups.belief_group_dim()`` (the S-A marginal
    targets); ``card_dim`` is the encoder's card-vocabulary size (the
    opponent-hand card posterior). At least one must be positive -- the head only
    exists when it has something to predict.
    """

    def __init__(
        self,
        d_model: int,
        *,
        group_dim: int = 0,
        card_dim: int = 0,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.group_dim = int(group_dim)
        self.card_dim = int(card_dim)
        if self.group_dim <= 0 and self.card_dim <= 0:
            raise ValueError("BeliefHead needs group_dim > 0 or card_dim > 0")
        self.hidden = int(hidden)
        out_dim = 2 + 2 * self.group_dim + self.card_dim
        self.trunk = nn.Sequential(nn.Linear(int(d_model), self.hidden), nn.SiLU())
        self.proj = nn.Linear(self.hidden, out_dim)

    @property
    def out_dim(self) -> int:
        return 2 + 2 * self.group_dim + self.card_dim

    def forward(self, pooled: torch.Tensor) -> BeliefOutput:  # (B, d_model)
        raw = self.proj(self.trunk(pooled))  # (B, out_dim)
        index = 0
        group_hand = group_deck = hand_card = None
        if self.group_dim > 0:
            group_hand = raw[:, index : index + self.group_dim]
            index += self.group_dim
            group_deck = raw[:, index : index + self.group_dim]
            index += self.group_dim
        if self.card_dim > 0:
            hand_card = raw[:, index : index + self.card_dim]
            index += self.card_dim
        hand_size = raw[:, index]
        burst_ready = raw[:, index + 1]
        return BeliefOutput(group_hand, group_deck, hand_card, hand_size, burst_ready)


def target_batch_from_observations(
    observations: Sequence[object],
    *,
    device: str | torch.device = "cpu",
) -> BeliefTargetBatch:
    """Stack ``EncodedObservation.belief_target`` labels into a dense batch.

    ``observations`` are ``reps.observation_encoder.EncodedObservation`` objects
    (only ``belief_target`` is read). Rows whose belief label is entirely
    degenerate -- e.g. a sample collected with ``include_training_targets=False``
    -- get ``mask=False`` so they do not drag the loss toward a meaningless
    uniform/uninformative target.
    """
    group_hand: list[Sequence[float]] = []
    group_deck: list[Sequence[float]] = []
    histograms: list[Sequence[float]] = []
    hand_sizes: list[float] = []
    burst: list[float] = []
    masks: list[bool] = []
    for observation in observations:
        target = observation.belief_target  # type: ignore[attr-defined]
        group_hand.append(target.group_hand_presence)
        group_deck.append(target.group_remaining_deck_presence)
        histograms.append(target.hand_histogram)
        hand_sizes.append(float(target.hand_size))
        burst.append(float(target.burst_ready_count))
        signal = (
            float(target.hand_size) > 0.0
            or any(float(v) > 0.0 for v in target.group_hand_presence)
            or any(float(v) > 0.0 for v in target.group_remaining_deck_presence)
            or any(float(v) > 0.0 for v in target.hand_histogram)
        )
        masks.append(bool(signal))
    group_dim = len(group_hand[0]) if group_hand else 0
    card_dim = len(histograms[0]) if histograms else 0
    tensor = torch.tensor
    return BeliefTargetBatch(
        group_hand_presence=tensor(group_hand, dtype=torch.float32, device=device)
        if group_dim
        else torch.zeros(len(observations), 0, device=device),
        group_deck_presence=tensor(group_deck, dtype=torch.float32, device=device)
        if group_dim
        else torch.zeros(len(observations), 0, device=device),
        hand_histogram=tensor(histograms, dtype=torch.float32, device=device)
        if card_dim
        else torch.zeros(len(observations), 0, device=device),
        hand_size=tensor(hand_sizes, dtype=torch.float32, device=device),
        burst_ready=tensor(burst, dtype=torch.float32, device=device),
        mask=tensor(masks, dtype=torch.bool, device=device),
    )


def belief_loss(
    prediction: BeliefOutput,
    targets: BeliefTargetBatch,
    *,
    group_coef: float = 1.0,
    card_coef: float = 1.0,
    scalar_coef: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Proper-scoring belief loss (BCE for marginals, CE for card identity, MSE).

    Only sub-heads present in ``prediction`` contribute; rows with
    ``targets.mask == False`` are excluded. Returns a scalar tensor (0 when the
    batch carries no belief signal at all).
    """
    device = prediction.hand_size.device
    mask = targets.mask.to(device).to(prediction.hand_size.dtype)
    denom = mask.sum().clamp(min=1.0)
    loss = prediction.hand_size.new_zeros(())

    if prediction.group_hand_logits is not None and prediction.group_hand_logits.size(-1) > 0:
        hand_target = targets.group_hand_presence.to(device)
        deck_target = targets.group_deck_presence.to(device)
        hand_rows = F.binary_cross_entropy_with_logits(
            prediction.group_hand_logits, hand_target, reduction="none"
        ).mean(dim=-1)
        deck_rows = F.binary_cross_entropy_with_logits(
            prediction.group_deck_logits, deck_target, reduction="none"
        ).mean(dim=-1)
        loss = loss + group_coef * ((hand_rows + deck_rows) * mask).sum() / denom

    if prediction.hand_card_logits is not None and prediction.hand_card_logits.size(-1) > 0:
        raw_hist = targets.hand_histogram.to(device)
        total = raw_hist.sum(dim=-1)
        normalized = raw_hist / total.unsqueeze(-1).clamp(min=eps)
        log_probs = F.log_softmax(prediction.hand_card_logits, dim=-1)
        card_rows = -(normalized * log_probs).sum(dim=-1)
        valid = (total > eps).to(prediction.hand_size.dtype) * mask
        valid_denom = valid.sum().clamp(min=1.0)
        loss = loss + card_coef * (card_rows * valid).sum() / valid_denom

    scalar_rows = F.mse_loss(
        prediction.hand_size, targets.hand_size.to(device), reduction="none"
    ) + F.mse_loss(prediction.burst_ready, targets.burst_ready.to(device), reduction="none")
    loss = loss + scalar_coef * (scalar_rows * mask).sum() / denom
    return loss


__all__ = [
    "BeliefHead",
    "BeliefOutput",
    "BeliefTargetBatch",
    "belief_loss",
    "target_batch_from_observations",
]
