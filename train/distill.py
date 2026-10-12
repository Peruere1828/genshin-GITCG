"""Search-distillation training loop (PLAN.md M3-CPU S-C, gaps G6 + G7).

The practical resolver's final step (``train/CONTINUAL_RESOLVING.md`` §3 step 3)
is **search-as-teacher**: the sampled GT-CFR solver (``train/cfr_solver.py`` +
``envs/cfr_search.py``) is the decision-maker, and the CVPN is trained to
approximate it. This module distils a solver-annotated replay
(``train.data.Sample`` with ``policy_target`` / ``search_value`` / belief labels)
into the network with three terms:

* **policy** -- soft cross-entropy ``-sum_a pi*(a) log q(a)`` against the solver's
  root average strategy (KL up to the teacher entropy), falling back to the hard
  ``teacher_index`` CE when no strategy was logged (behaviour cloning);
* **value** -- regression to the solver's root value ``search_value`` when
  present, else the episode outcome ``value_target``;
* **belief** -- the proper-scoring belief loss of ``train.belief_head`` (BCE for
  the ``belief_groups`` marginals, CE for the opponent card posterior, MSE for the
  bounded scalars), when the network has a belief head.

Expert BC is therefore the *cold-start / auxiliary* mode: pass a ``DistillConfig``
with ``use_policy_target=False`` and ``use_search_value=False`` and the same data
degenerates to classic one-hot BC on ``teacher_index`` / ``value_target``. The
loop is engine-free (pure torch over encoded samples) and CPU-sized for the
M3-CPU tier.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from train.belief_head import belief_loss, target_batch_from_observations
from train.data import Sample
from train.model import collate
from train.train_loop import split_samples

_EPS = 1.0e-8


@dataclass
class DistillConfig:
    epochs: int = 8
    lr: float = 1e-3
    batch_size: int = 32
    policy_coef: float = 1.0
    value_coef: float = 0.5
    belief_coef: float = 0.5
    seed: int = 0
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    device: str = "cpu"
    log_every: int = 0
    # Ablations: ``False`` degrades the soft/solver targets to the hard BC
    # fallback (the cold-start configuration).
    use_policy_target: bool = True
    use_search_value: bool = True


@dataclass
class DistillMetrics:
    epochs: int
    n_train: int
    n_val: int
    n_search: int
    train_loss: float
    train_policy_acc: float
    train_value_mse: float
    train_belief_loss: float
    val_loss: float
    val_policy_acc: float
    val_value_mse: float
    val_belief_loss: float
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_search": self.n_search,
            "train_loss": round(self.train_loss, 5),
            "train_policy_acc": round(self.train_policy_acc, 4),
            "train_value_mse": round(self.train_value_mse, 5),
            "train_belief_loss": round(self.train_belief_loss, 5),
            "val_loss": round(self.val_loss, 5),
            "val_policy_acc": round(self.val_policy_acc, 4),
            "val_value_mse": round(self.val_value_mse, 5),
            "val_belief_loss": round(self.val_belief_loss, 5),
        }


def _batch_targets(batch: Sequence[Sample], model: nn.Module, config: DistillConfig):
    """Dense tensors for the policy / value targets of one batch."""
    device = next(model.parameters()).device
    n = len(batch)
    # Option count is the padded max of this batch (``collate`` pads to it).
    k = max((len(s.observation.option_features) for s in batch), default=0)
    policy_target = torch.zeros(n, k, device=device)
    has_policy = torch.zeros(n, dtype=torch.bool, device=device)
    option_mask = torch.zeros(n, k, dtype=torch.bool, device=device)
    for i, sample in enumerate(batch):
        n_opts = len(sample.observation.option_features)
        option_mask[i, :n_opts] = True
        if config.use_policy_target and sample.policy_target:
            raw = [max(0.0, float(v)) for v in sample.policy_target[:n_opts]]
            total = sum(raw)
            if total > _EPS:
                policy_target[i, : len(raw)] = torch.tensor(raw, device=device) / total
                has_policy[i] = True
    teacher = torch.tensor(
        [max(0, min(int(s.teacher_index), k - 1)) for s in batch], dtype=torch.long, device=device
    )
    values = torch.tensor(
        [
            float(s.search_value)
            if (config.use_search_value and s.search_value is not None)
            else float(s.value_target)
            for s in batch
        ],
        dtype=torch.float32,
        device=device,
    )
    return policy_target, has_policy, option_mask, teacher, values


def distill_batch_loss(
    model: nn.Module, batch: Sequence[Sample], config: DistillConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted distillation loss + component metrics for one batch.

    ``total = policy_coef * policy + value_coef * value + belief_coef * belief``.
    The policy term is a soft (teacher-strategy) CE when available and a hard
    ``teacher_index`` CE otherwise; padded options are masked out of the soft term
    (a zero target contributes nothing, and never multiplies the ``-inf`` log-prob
    of an illegal action).
    """
    device = next(model.parameters()).device
    tokens, token_mask, options, option_mask = collate(
        [s.observation for s in batch]
    )
    tokens = tokens.to(device)
    token_mask = token_mask.to(device)
    options = options.to(device)
    option_mask = option_mask.to(device)
    logits, value, belief = model.forward_all(tokens, token_mask, options, option_mask)

    policy_target, has_policy, legal_mask, teacher, values = _batch_targets(
        batch, model, config
    )
    # Never let a target sit on a padded/illegal option.
    policy_target = policy_target * legal_mask.to(policy_target.dtype)
    row_total = policy_target.sum(dim=-1, keepdim=True)
    policy_target = policy_target / row_total.clamp(min=_EPS)
    has_policy = has_policy & (row_total.squeeze(-1) > _EPS)

    log_probs = F.log_softmax(logits, dim=-1)
    safe_log_probs = torch.where(policy_target > 0.0, log_probs, torch.zeros_like(log_probs))
    soft_rows = -(policy_target * safe_log_probs).sum(dim=-1)
    hard_rows = F.cross_entropy(logits, teacher, reduction="none")
    policy_rows = torch.where(has_policy, soft_rows, hard_rows)
    policy_loss = policy_rows.mean()

    value_loss = F.mse_loss(value, values)

    belief_value = torch.zeros((), device=device)
    if getattr(model, "has_belief", False) and belief is not None:
        targets = target_batch_from_observations(
            [s.observation for s in batch], device=device
        )
        belief_value = belief_loss(belief, targets)

    total = (
        config.policy_coef * policy_loss
        + config.value_coef * value_loss
        + config.belief_coef * belief_value
    )
    with torch.no_grad():
        teacher_argmax = torch.where(
            has_policy, policy_target.argmax(dim=-1), teacher
        )
        accuracy = (logits.argmax(dim=-1) == teacher_argmax).float().mean()
    info = {
        "policy": float(policy_loss.detach().item()),
        "value": float(value_loss.detach().item()),
        "belief": float(belief_value.detach().item()),
        "acc": float(accuracy.item()),
    }
    return total, info


@torch.no_grad()
def evaluate_distill(
    model: nn.Module, batch: Sequence[Sample], config: DistillConfig
) -> dict[str, float]:
    loss, info = distill_batch_loss(model, batch, config)
    return {"loss": float(loss.item()), **info}


def _mean_info(infos: Sequence[dict[str, float]]) -> dict[str, float]:
    if not infos:
        return {"loss": 0.0, "policy": 0.0, "value": 0.0, "belief": 0.0, "acc": 0.0}
    return {
        key: sum(info[key] for info in infos) / len(infos)
        for key in ("loss", "policy", "value", "belief", "acc")
    }


def train_distill(
    samples: Sequence[Sample],
    model: nn.Module,
    config: DistillConfig,
) -> DistillMetrics:
    """Train ``model`` by distilling the search teacher (policy CE + value + belief)."""
    torch.manual_seed(config.seed)
    model.to(config.device)
    train, val = split_samples(samples, val_fraction=config.val_fraction, seed=config.seed)
    n_search = sum(1 for s in samples if s.has_search_target)
    if not train:
        return DistillMetrics(
            config.epochs, 0, 0, n_search, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    model.train()
    rng = random.Random(config.seed)
    history: list[dict] = []
    last = {
        "loss": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "belief": 0.0,
        "acc": 0.0,
    }
    for epoch in range(config.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        infos: list[dict[str, float]] = []
        for start in range(0, len(order), config.batch_size):
            batch = [train[i] for i in order[start : start + config.batch_size]]
            if not batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, info = distill_batch_loss(model, batch, config)
            loss.backward()
            optimizer.step()
            info = dict(info)
            info["loss"] = float(loss.detach().item())
            infos.append(info)
        last = _mean_info(infos)
        model.eval()
        val_info = _evaluate_all(model, val, config) if val else None
        model.train()
        record = {
            "epoch": epoch,
            "train_loss": round(last["loss"], 5),
            "train_policy_acc": round(last["acc"], 4),
            "train_value_mse": round(last["value"], 5),
            "train_belief_loss": round(last["belief"], 5),
        }
        if val_info is not None:
            record.update(
                {
                    "val_loss": round(val_info["loss"], 5),
                    "val_policy_acc": round(val_info["acc"], 4),
                    "val_value_mse": round(val_info["value"], 5),
                    "val_belief_loss": round(val_info["belief"], 5),
                }
            )
        history.append(record)

    model.eval()
    val_info = _evaluate_all(model, val, config) if val else last
    return DistillMetrics(
        epochs=config.epochs,
        n_train=len(train),
        n_val=len(val),
        n_search=n_search,
        train_loss=last["loss"],
        train_policy_acc=last["acc"],
        train_value_mse=last["value"],
        train_belief_loss=last["belief"],
        val_loss=val_info["loss"],
        val_policy_acc=val_info["acc"],
        val_value_mse=val_info["value"],
        val_belief_loss=val_info["belief"],
        history=history,
    )


def _evaluate_all(
    model: nn.Module, samples: Sequence[Sample], config: DistillConfig
) -> dict[str, float]:
    if not samples:
        return {"loss": 0.0, "policy": 0.0, "value": 0.0, "belief": 0.0, "acc": 0.0}
    infos: list[dict[str, float]] = []
    for start in range(0, len(samples), 64):
        batch = samples[start : start + 64]
        infos.append(evaluate_distill(model, batch, config))
    return _mean_info(infos)


__all__ = [
    "DistillConfig",
    "DistillMetrics",
    "distill_batch_loss",
    "evaluate_distill",
    "train_distill",
]
