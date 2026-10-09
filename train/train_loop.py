"""Behavior-cloning / distillation training loop for the toy CVPN.

Trains the policy head to imitate a teacher (scripted expert today; a search
resolver later) and regresses the value head to the episode outcome. Small enough
to run on CPU for the L4 toy validation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from train.data import Sample
from train.model import CVPN, collate


@dataclass
class TrainConfig:
    epochs: int = 8
    lr: float = 1e-3
    batch_size: int = 32
    value_coef: float = 0.5
    seed: int = 0
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    log_every: int = 0


@dataclass
class TrainMetrics:
    epochs: int
    n_train: int
    n_val: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "train_loss": round(self.train_loss, 4),
            "train_accuracy": round(self.train_accuracy, 4),
            "val_loss": round(self.val_loss, 4),
            "val_accuracy": round(self.val_accuracy, 4),
        }


def split_samples(
    samples: Sequence[Sample], *, val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    data = list(samples)
    rng = random.Random(seed)
    rng.shuffle(data)
    if val_fraction <= 0 or len(data) <= 4:
        return data, []
    n_val = max(1, int(len(data) * val_fraction))
    return data[n_val:], data[:n_val]


def _batch_loss(model: CVPN, batch: Sequence[Sample], value_coef: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, token_mask, options, option_mask = collate([s.observation for s in batch])
    logits, value = model(tokens, token_mask, options, option_mask)
    teacher = torch.tensor([s.teacher_index for s in batch], dtype=torch.long)
    # Guard: teacher index must be within the option count we actually encoded.
    teacher = teacher.clamp(min=0)
    ce = nn.functional.cross_entropy(logits, teacher)
    targets = torch.tensor([s.value_target for s in batch], dtype=torch.float32)
    mse = nn.functional.mse_loss(value, targets)
    loss = ce + value_coef * mse
    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        acc = (pred == teacher).float().mean()
    return loss, acc, ce.detach()


def _evaluate(model: CVPN, samples: Sequence[Sample], value_coef: float) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    model.eval()
    losses: list[float] = []
    correct = 0
    with torch.no_grad():
        for start in range(0, len(samples), 64):
            batch = samples[start : start + 64]
            loss, acc, _ = _batch_loss(model, batch, value_coef)
            losses.append(float(loss.item()))
            correct += int(round(float(acc.item()) * len(batch)))
    model.train()
    return sum(losses) / len(losses), correct / len(samples)


def evaluate_loss_accuracy(
    model: CVPN, samples: Sequence[Sample], value_coef: float
) -> tuple[float, float]:
    """Mean loss and accuracy of ``model`` on ``samples`` (no gradients).

    Public so the learning-curve sweep can score every grid point on one *fixed*
    holdout instead of a freshly re-drawn validation split (PLAN.md L5.3).
    """
    return _evaluate(model, samples, value_coef)


def train_bc(
    samples: Sequence[Sample],
    model: CVPN,
    config: TrainConfig,
) -> TrainMetrics:
    torch.manual_seed(config.seed)
    train, val = split_samples(samples, val_fraction=config.val_fraction, seed=config.seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    model.train()
    rng = random.Random(config.seed)
    history: list[dict] = []
    last_train_acc = 0.0
    last_train_loss = 0.0
    for epoch in range(config.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        epoch_losses: list[float] = []
        correct = 0
        for start in range(0, len(order), config.batch_size):
            batch = [train[i] for i in order[start : start + config.batch_size]]
            if not batch:
                continue
            optimizer.zero_grad()
            loss, acc, _ = _batch_loss(model, batch, config.value_coef)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            correct += int(round(float(acc.item()) * len(batch)))
        last_train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        last_train_acc = correct / max(1, len(train))
        history.append(
            {"epoch": epoch, "train_loss": round(last_train_loss, 4), "train_acc": round(last_train_acc, 4)}
        )
    val_loss, val_acc = _evaluate(model, val, config.value_coef)
    model.eval()
    val_loss = float(val_loss)
    return TrainMetrics(
        epochs=config.epochs,
        n_train=len(train),
        n_val=len(val),
        train_loss=last_train_loss,
        train_accuracy=last_train_acc,
        val_loss=val_loss,
        val_accuracy=val_acc,
        history=history,
    )
