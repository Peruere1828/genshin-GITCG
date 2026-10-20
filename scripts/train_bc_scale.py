"""Memory-efficient large-scale behaviour-cloning from ``train.collect`` replay dirs.

The stock ``train.learning_curve`` / ``train.pipeline`` paths call
``train.data.load_samples`` which materialises every sample as an
``EncodedObservation`` (Python tuples of floats).  That is fine for the ~1e3-1e4
sample M3-CPU tier but blows past RAM at the 1e5-1e6 sample scale the "data-first"
reschedule targets.

This script streams the collector's per-game JSONL records, converts each
decision **straight to compact per-sample tensors** (tokens ``(T,65)``, options
``(K,68)``, belief labels), drops the Python objects, and keeps only the tensors
in RAM.  Batches are padded to the *batch* max at train time (originals stay
variable length, so no global ``K=256`` padding blow-up).  Memory is therefore
~16 KB/sample instead of ~200 KB/sample, which is what makes a few 1e5-sample run
fit on a 30 GB box.

Selection is deterministic and order-independent: games are de-duplicated by the
full game identity ``(deck, opponent, seed)`` (A7 merges multi-deck corpora that
each restart ``--seed-start 0``, so the deck must be part of the key), split
train/val by a rule on ``(seed - seed_start)`` and sorted by ``(deck, opponent,
seed, step_index)`` before shuffling, so a rerun reproduces the same batches
regardless of the (completion-order) file layout.

``--max-train-samples`` / ``--max-val-samples`` are honoured deterministically
too (H2): the cap is applied after ordering the games by identity, never in file
order, so the same replay dir yields the same dataset twice. When a cap is set
the loader streams the dirs **twice** — a first pass enumerates each game's
identity and sample count, then a second pass loads only the accepted games up to
their per-game budget — which keeps both determinism and the bounded
(~16 KB/sample) memory footprint. Without a cap a single pass suffices, since the
post-load sort already makes the result order-independent.

Pure-policy BC: hard ``teacher_index`` CE + episode-outcome value regression +
optional belief head, i.e. the cold-start arm of ``train.distill``.

Usage::

    python -m scripts.train_bc_scale --replays data/replays/datafirst_bc_sa \\
        --seed-start 1000000 --val-every 10 --max-train-samples 500000 \\
        --d-model 256 --epochs 30 --out data/checkpoints/big_d256.pt \\
        --report reports/train/big_d256_train.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from common.lock import load_engine_lock, runtime_metadata
from common.paths import ensure_dir
from envs.observation import default_encoder
from train.model import CVPN, TransformerCVPN, config_for_encoder, transformer_config_for_encoder


@dataclass
class Store:
    """Compact per-sample tensors (variable length, no padding)."""

    tokens: list[torch.Tensor]      # each (T_i, 65) float32
    options: list[torch.Tensor]     # each (K_i, 68) float32
    teacher: list[int]
    value: list[float]
    # belief labels (god's-eye targets); empty list when capture_targets=False
    b_hand: list[torch.Tensor]      # (G,) float32
    b_deck: list[torch.Tensor]      # (G,)
    b_hist: list[torch.Tensor]      # (V,)
    b_hsize: list[float]
    b_burst: list[float]
    b_mask: list[bool]
    keys: list[tuple[str, str, int, int]]  # (deck, opponent, seed, step_index) for sorting

    def __len__(self) -> int:
        return len(self.tokens)

    def subset(self, indices: Sequence[int]) -> "Store":
        take = lambda xs: [xs[i] for i in indices]
        return Store(
            tokens=take(self.tokens),
            options=take(self.options),
            teacher=take(self.teacher),
            value=take(self.value),
            b_hand=take(self.b_hand),
            b_deck=take(self.b_deck),
            b_hist=take(self.b_hist),
            b_hsize=take(self.b_hsize),
            b_burst=take(self.b_burst),
            b_mask=take(self.b_mask),
            keys=take(self.keys),
        )


def _is_val(seed: int, seed_start: int, val_every: int) -> bool:
    if val_every <= 0:
        return False
    return ((int(seed) - int(seed_start)) % int(val_every)) == 0


def _game_key(record: dict[str, Any]) -> tuple[str, str, int]:
    """Full game identity ``(deck, opponent, seed)``.

    ``deck`` is part of the key (A7): a multi-deck corpus collected with each deck
    restarting ``--seed-start 0`` has overlapping ``(opponent, seed)`` pairs, so
    dropping the deck would silently dedup distinct games away (H1).
    """
    return (str(record.get("deck", "")), str(record["opponent"]), int(record["seed"]))


def _iter_replay_records(replay_dirs: Sequence[str]) -> Iterator[dict[str, Any]]:
    """Yield parsed per-game records, skipping blanks and torn trailing lines."""
    for replay_dir in replay_dirs:
        samples_path = Path(replay_dir)
        if samples_path.is_dir():
            samples_path = samples_path / "samples.jsonl"
        if not samples_path.exists():
            raise SystemExit(f"no samples.jsonl under {replay_dir}")
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn trailing line


def _plan_budgets(
    replay_dirs: Sequence[str],
    *,
    seed_start: int,
    val_every: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[dict[tuple[str, str, int], int], int, int]:
    """Pass 1: enumerate games and pick a deterministic per-game sample budget.

    Returns ``(budgets, skipped, duplicates)`` where ``budgets`` maps a game's
    full identity to the number of samples to load for it. Games are visited in
    sorted-identity order, so *which* games survive the cap never depends on the
    (completion-order) file layout. Only identities + counts are held (never the
    samples), so the pass stays memory-bounded even on a huge corpus.
    """
    manifest: dict[tuple[str, str, int], int] = {}
    seen: set[tuple[str, str, int]] = set()
    skipped = duplicates = 0
    for record in _iter_replay_records(replay_dirs):
        if record.get("error"):
            skipped += 1
            continue
        key = _game_key(record)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        manifest[key] = len(record.get("samples", ()))
    budgets: dict[tuple[str, str, int], int] = {}
    used_train = used_val = 0
    for key in sorted(manifest):
        is_val = _is_val(key[2], seed_start, val_every)
        cap = max_val_samples if is_val else max_train_samples
        count = manifest[key]
        if cap is None:
            take = count
        else:
            remaining = cap - (used_val if is_val else used_train)
            if remaining <= 0:
                continue
            take = min(count, remaining)
        budgets[key] = take
        if is_val:
            used_val += take
        else:
            used_train += take
    return budgets, skipped, duplicates


def load_store(
    replay_dirs: Sequence[str],
    *,
    seed_start: int,
    val_every: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    device: str = "cpu",
    progress_every: int = 500,
) -> tuple[Store, Store, dict[str, Any]]:
    """Stream replay dirs into compact (train, val) stores.

    Deterministic and order-independent (H2). When a sample cap is set the dirs
    are streamed twice: :func:`_plan_budgets` picks the accepted games (and each
    one's budget) in sorted-identity order, then the load pass reads only those
    games. Without a cap a single pass is enough because the final sort removes
    any dependence on file order.
    """
    train = Store([], [], [], [], [], [], [], [], [], [], [])
    val = Store([], [], [], [], [], [], [], [], [], [], [])
    caps_set = max_train_samples is not None or max_val_samples is not None
    n_skipped = n_dup = 0
    remaining: dict[tuple[str, str, int], int] | None = None
    if caps_set:
        remaining, n_skipped, n_dup = _plan_budgets(
            replay_dirs,
            seed_start=seed_start,
            val_every=val_every,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
        )
    n_games = 0
    seen: set[tuple[str, str, int]] = set()
    t0 = time.perf_counter()
    for record in _iter_replay_records(replay_dirs):
        if record.get("error"):
            if not caps_set:
                n_skipped += 1
            continue
        game_key = _game_key(record)
        if remaining is not None:
            # Pass-1 already deduped and budgeted; anything absent is either a
            # repeat of an accepted game or a game the cap rejected.
            if game_key not in remaining:
                continue
            budget: int | None = remaining.pop(game_key)
        else:
            if game_key in seen:
                n_dup += 1
                continue
            seen.add(game_key)
            budget = None
        deck, opponent, seed = game_key
        is_val = _is_val(seed, seed_start, val_every)
        dest = val if is_val else train
        payloads = record.get("samples", ())
        if budget is not None:
            payloads = payloads[:budget]
        for payload in payloads:
            obs = payload["observation"]
            dest.tokens.append(
                torch.tensor(obs["token_features"], dtype=torch.float32, device=device)
            )
            dest.options.append(
                torch.tensor(obs["option_features"], dtype=torch.float32, device=device)
            )
            dest.teacher.append(int(payload["teacher_index"]))
            dest.value.append(float(payload["value_target"]))
            bt = obs.get("belief_target") or {}
            dest.b_hand.append(
                torch.tensor(bt.get("group_hand_presence", ()), dtype=torch.float32, device=device)
            )
            dest.b_deck.append(
                torch.tensor(bt.get("group_remaining_deck_presence", ()), dtype=torch.float32, device=device)
            )
            dest.b_hist.append(
                torch.tensor(bt.get("hand_histogram", ()), dtype=torch.float32, device=device)
            )
            dest.b_hsize.append(float(bt.get("hand_size", 0.0)))
            dest.b_burst.append(float(bt.get("burst_ready_count", 0.0)))
            sig = (
                float(bt.get("hand_size", 0.0)) > 0.0
                or any(float(v) > 0.0 for v in bt.get("group_hand_presence", ()))
                or any(float(v) > 0.0 for v in bt.get("group_remaining_deck_presence", ()))
                or any(float(v) > 0.0 for v in bt.get("hand_histogram", ()))
            )
            dest.b_mask.append(bool(sig))
            dest.keys.append((deck, opponent, seed, int(payload.get("step_index", 0))))
        n_games += 1
        if progress_every and n_games % progress_every == 0:
            print(
                f"[store] games={n_games} train={len(train)} val={len(val)} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    # deterministic ordering independent of completion order
    for store in (train, val):
        order = sorted(range(len(store)), key=lambda i: store.keys[i])
        _reorder(store, order)
    meta = {
        "replay_dirs": list(replay_dirs),
        "games": n_games,
        "skipped_games": n_skipped,
        "duplicate_games": n_dup,
        "train_samples": len(train),
        "val_samples": len(val),
        "load_seconds": round(time.perf_counter() - t0, 1),
        "seed_start": seed_start,
        "val_every": val_every,
    }
    return train, val, meta


def _reorder(store: Store, order: Sequence[int]) -> None:
    for field in (
        "tokens", "options", "teacher", "value",
        "b_hand", "b_deck", "b_hist", "b_hsize", "b_burst", "b_mask", "keys",
    ):
        seq = getattr(store, field)
        setattr(store, field, [seq[i] for i in order])


def matchup_weights_for_store(store: Store, *, mode: str, strength: float = 1.0,
                              floor: float = 0.45) -> list[float] | None:
    """Per-sample weights that rebalance matchups (H5, floor attack).

    Groups are ``(deck, opponent)`` — everything in the key but the step index — and
    the group's own win rate comes from the stored episode scores. ``None`` means
    "unweighted" so callers can skip the multiply entirely (the default path stays
    byte-for-byte the historical BC loss).

    ``balance`` uses inverse frequency: a matchup with n samples gets weight
    ``mean(n) / n``, so a corpus where the targeted worst matchups carry 4x the
    games does not simply drown the other matchups (and vice versa).
    ``deficit`` is evidence-driven: only groups below the M4 floor get boosted,
    proportionally to how far below they are.
    """
    if mode in ("none", "", None):
        return None
    groups: dict[Any, list[int]] = {}
    scores: dict[Any, list[float]] = {}
    for i, key in enumerate(store.keys):
        group = tuple(key[:-1])
        groups.setdefault(group, []).append(i)
        scores.setdefault(group, []).append(float(store.value[i]))
    counts = {g: len(idx) for g, idx in groups.items()}
    if not counts:
        return None
    if mode == "balance":
        target = sum(counts.values()) / len(counts)
        group_weight = {g: target / n for g, n in counts.items()}
    elif mode == "deficit":
        group_weight = {}
        for group, idx in groups.items():
            rate = sum(scores[group]) / len(idx)
            group_weight[group] = 1.0 + strength * max(0.0, floor - rate)
    else:
        raise ValueError(f"unknown matchup_weight mode: {mode!r}")
    weights = [0.0] * len(store.keys)
    for group, idx in groups.items():
        for i in idx:
            weights[i] = group_weight[group]
    mean = sum(weights) / len(weights)
    return [w / mean for w in weights] if mean > 0 else None


def _apply_sample_weights(loss_per_sample: torch.Tensor, index: Sequence[int],
                          sample_weights: Sequence[float] | None) -> torch.Tensor:
    if sample_weights is None:
        return loss_per_sample.mean()
    w = torch.tensor([float(sample_weights[i]) for i in index],
                     dtype=loss_per_sample.dtype, device=loss_per_sample.device)
    return (w * loss_per_sample).mean()


def _game_last_steps(store: Store) -> dict[tuple[Any, ...], int]:
    """Highest ``step_index`` seen per game (used for discounted MC returns).

    Tolerates both key layouts seen on disk: ``(opponent, seed, step)`` from the
    older store builder and ``(deck, opponent, seed, step)`` from the current one —
    the step is always the last element and the game identity is everything before it.
    """
    last: dict[tuple[Any, ...], int] = {}
    for key in store.keys:
        game = tuple(key[:-1])
        step = int(key[-1])
        if step > last.get(game, -1):
            last[game] = step
    return last


def q_scale_for_store(store: Store, *, target: str, gamma: float) -> list[float] | None:
    """Per-sample ``gamma ** steps_remaining`` factors (``None`` for episode targets).

    ``steps_remaining`` = the game's last step index minus this sample's step index,
    so the last decision of a game gets weight 1.0. This gives the Q target a
    little credit assignment without needing a next-state bootstrap.
    """
    if target != "discount":
        return None
    last = _game_last_steps(store)
    return [
        float(gamma) ** (last[tuple(k[:-1])] - int(k[-1]))
        for k in store.keys
    ]


def _q_targets(values: torch.Tensor, index: Sequence[int], q_scale: Sequence[float] | None) -> torch.Tensor:
    if q_scale is None:
        return values
    scale = torch.tensor(
        [float(q_scale[i]) for i in index], dtype=values.dtype, device=values.device
    )
    return values * scale


def _collate(store: Store, index: Sequence[int], device: str):
    """Pad a batch to its own max and stack (vectorised via ``pad_sequence``)."""
    from torch.nn.utils.rnn import pad_sequence

    toks = [store.tokens[i] for i in index]
    opts = [store.options[i] for i in index]
    b = len(index)
    t_lens = torch.tensor([t.shape[0] for t in toks])
    k_lens = torch.tensor([o.shape[0] for o in opts])
    tokens = pad_sequence(toks, batch_first=True).to(device)
    options = pad_sequence(opts, batch_first=True).to(device)
    token_mask = (
        torch.arange(tokens.shape[1]).unsqueeze(0) < t_lens.unsqueeze(1)
    ).to(device)
    option_mask = (
        torch.arange(options.shape[1]).unsqueeze(0) < k_lens.unsqueeze(1)
    ).to(device)
    teacher = torch.tensor([store.teacher[i] for i in index], dtype=torch.long, device=device)
    values = torch.tensor([store.value[i] for i in index], dtype=torch.float32, device=device)
    return tokens, token_mask, options, option_mask, teacher, values


def _belief_batch(store: Store, index: Sequence[int], device: str):
    if not store.b_hand[index[0]].numel() and not store.b_hist[index[0]].numel():
        return None
    return {
        "b_hand": torch.stack([store.b_hand[i] for i in index]).to(device),
        "b_deck": torch.stack([store.b_deck[i] for i in index]).to(device),
        "b_hist": torch.stack([store.b_hist[i] for i in index]).to(device),
        "b_hsize": torch.tensor([store.b_hsize[i] for i in index], dtype=torch.float32, device=device),
        "b_burst": torch.tensor([store.b_burst[i] for i in index], dtype=torch.float32, device=device),
        "b_mask": torch.tensor([store.b_mask[i] for i in index], dtype=torch.bool, device=device),
    }


def _belief_loss(belief, bt) -> torch.Tensor:
    if belief is None or bt is None:
        return torch.zeros((), device=bt["b_hsize"].device if bt else "cpu")
    from train.belief_head import BeliefTargetBatch, belief_loss

    targets = BeliefTargetBatch(
        group_hand_presence=bt["b_hand"],
        group_deck_presence=bt["b_deck"],
        hand_histogram=bt["b_hist"],
        hand_size=bt["b_hsize"],
        burst_ready=bt["b_burst"],
        mask=bt["b_mask"],
    )
    return belief_loss(belief, targets)


def _value_loss(value: torch.Tensor, targets: torch.Tensor, config, model) -> torch.Tensor:
    """Episode-outcome loss for the value head (H3).

    Contract: ``mse`` regresses whatever the head outputs (the historical path:
    identity activation); ``brier``/``bce`` are proper scoring rules that require a
    probability head (``value_activation="sigmoid"``), and are refused otherwise so a
    raw-logit head can never be silently trained as if it were a probability.
    """
    kind = getattr(config, "value_loss", "mse") if config is not None else "mse"
    smooth = float(getattr(config, "value_smooth", 0.0) or 0.0) if config is not None else 0.0
    target = targets * (1.0 - 2.0 * smooth) + smooth if smooth else targets
    activation = getattr(getattr(model, "config", None), "value_activation", "identity")
    if kind == "mse":
        return F.mse_loss(value, target)
    if activation != "sigmoid":
        raise ValueError(f"value_loss={kind!r} needs value_activation='sigmoid'")
    if kind == "brier":
        return F.mse_loss(value, target)
    if kind == "bce":
        p = value.clamp(1e-4, 1 - 1e-4)
        return -(target * p.log() + (1 - target) * (1 - p).log()).mean()
    raise ValueError(f"unknown value_loss: {kind!r}")


def run_batch(model, store, index, device, value_coef, belief_coef, has_belief,
              config=None, q_scale=None, sample_weights=None):
    tokens, token_mask, options, option_mask, teacher, values = _collate(store, index, device)
    if has_belief:
        logits, value, belief = model.forward_all(tokens, token_mask, options, option_mask)
    else:
        logits, value = model(tokens, token_mask, options, option_mask)
        belief = None
    loss_kind = getattr(config, "loss", "ce") if config is not None else "ce"
    extra: dict[str, float] = {}
    if loss_kind == "q":
        # DouZero-style: only the played (teacher) action's Q is supervised; the
        # shared option features let the unscored actions inherit a ranking.
        target = _q_targets(values, index, q_scale)
        q_played = logits.gather(1, teacher.unsqueeze(1)).squeeze(1)
        policy_loss = F.mse_loss(q_played, target)
        extra["q_mse"] = float(policy_loss.item())
    else:
        raw = F.cross_entropy(logits, teacher, reduction="none")
        if loss_kind == "awr":
            beta = float(getattr(config, "awr_beta", 1.0))
            max_w = float(getattr(config, "awr_max_weight", 10.0))
            # in-batch advantage baseline: the model's own value head (detached)
            adv = values - value.detach()
            weights = torch.exp(beta * adv).clamp(max=max_w)
            weights = weights / weights.mean().clamp(min=1e-6)
            policy_loss = (weights * raw).mean()
            extra["w_mean"] = float(weights.mean().item())
            extra["w_max"] = float(weights.max().item())
        elif loss_kind == "ce":
            # H5: optional matchup reweighting; ``sample_weights=None`` reproduces
            # the historical plain CE mean exactly.
            policy_loss = _apply_sample_weights(raw, index, sample_weights)
        elif loss_kind == "pg":
            # Online RL: ``teacher`` is the action the *sampling* policy took.
            logp = F.log_softmax(logits, dim=-1)
            chosen = logp.gather(1, teacher.unsqueeze(1)).squeeze(1)
            adv = values - value.detach()          # terminal return minus baseline
            if getattr(config, "pg_advantage_norm", True) and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            entropy = -(logp.exp() * logp).sum(dim=-1).mean()
            policy_loss = -(chosen * adv).mean() - float(config.pg_entropy) * entropy
            extra["entropy"] = float(entropy.item())
            extra["adv_mean"] = float(adv.mean().item())
        else:
            raise ValueError(f"unknown loss kind: {loss_kind!r}")
    mse = _value_loss(value, values, config, model)
    bl = _belief_loss(belief, _belief_batch(store, index, device)) if has_belief else torch.zeros((), device=device)
    loss = policy_loss + value_coef * mse + belief_coef * bl
    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == teacher).float().mean()
    return loss, {"ce": float(policy_loss.item()), "value": float(mse.item()),
                  "belief": float(bl.item()), "acc": float(acc.item()), **extra}


@torch.no_grad()
def evaluate(model, store, config, device, has_belief, q_scale=None, sample_weights=None):
    model.eval()
    if len(store) == 0:
        return {"ce": 0.0, "value": 0.0, "belief": 0.0, "acc": 0.0, "n": 0}
    n = len(store)
    agg: dict[str, float] = {}
    for start in range(0, n, config.batch_size):
        index = list(range(start, min(start + config.batch_size, n)))
        _, info = run_batch(model, store, index, device, config.value_coef,
                            config.belief_coef, has_belief, config, q_scale, sample_weights)
        for key, val in info.items():
            agg[key] = agg.get(key, 0.0) + float(val) * len(index)
    for key in agg:
        agg[key] /= n
    agg["n"] = n
    model.train()
    return agg


@dataclass
class Config:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 128
    value_coef: float = 0.5
    belief_coef: float = 0.5
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1
    # --- offline-RL policy targets (H2 of the 2-day strength plan) ----------
    # "ce"  = behaviour cloning (default, unchanged);
    # "awr" = advantage-weighted CE: w = exp(beta * (G - V(s))) from the model's
    #         own (detached) value head, normalised to mean 1 and clipped;
    # "q"   = DouZero-style action value: MSE of Q(s, a_teacher) to the MC return
    #         (optionally discounted by the remaining steps of that game). The
    #         head stays a per-option scorer, so ``argmax`` at inference is a
    #         greedy Q policy (``neural:<ckpt>`` needs no change).
    loss: str = "ce"
    awr_beta: float = 1.0
    awr_max_weight: float = 10.0
    q_target: str = "episode"   # "episode" | "discount"
    q_gamma: float = 0.97
    # --- H3 value calibration -------------------------------------------------
    # "mse"  = historical regression on the episode score (default);
    # "brier"= same target but for a sigmoid value head (proper scoring rule);
    # "bce"  = cross entropy in probability space (log-clamped).
    # ``value_smooth`` pulls the 0/1 targets towards 0.5 (label smoothing), which
    # also fights the optimistic bias a plain MSE head develops.
    value_loss: str = "mse"
    value_smooth: float = 0.0
    # --- H5 matchup weighting (floor attack) ----------------------------------
    # "none"  = every sample counts once (default);
    # "balance" = inverse-frequency across (deck, opponent) groups, so a deck that
    #             was collected 80 seeds × 20 opponents counts each matchup equally;
    # "deficit" = upweight groups whose own episode win rate is below ``floor``
    #             (weight = 1 + strength * (floor - rate)), i.e. mine the matchups
    #             the current data loses. Weights are normalised to mean 1.
    matchup_weight: str = "none"
    matchup_weight_strength: float = 1.0
    matchup_floor: float = 0.45
    # --- online RL (policy gradient on self-play data) -------------------------
    # ``loss="pg"`` treats the stored ``teacher_index`` as the *action sampled by the
    # current policy* (collect with ``neural:<ckpt>#t=<temp>``) and does REINFORCE
    # with the value head as baseline:  -(log pi(a|s) * (G - V(s))) - c * H(pi).
    # This is on-policy, unlike AWR on expert data (which was neutral-to-harmful).
    pg_entropy: float = 0.01
    pg_advantage_norm: bool = True


def train(train_store: Store, val_store: Store, model, config: Config, has_belief: bool):
    torch.manual_seed(config.seed)
    model.to(config.device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    n = len(train_store)
    q_scale = (
        q_scale_for_store(train_store, target=config.q_target, gamma=config.q_gamma)
        if config.loss == "q" else None
    )
    val_q_scale = (
        q_scale_for_store(val_store, target=config.q_target, gamma=config.q_gamma)
        if config.loss == "q" and val_store is not None and len(val_store) else None
    )
    sample_weights = matchup_weights_for_store(
        train_store, mode=config.matchup_weight,
        strength=config.matchup_weight_strength, floor=config.matchup_floor,
    )
    if sample_weights is not None:
        print(f"[train] matchup weighting: {config.matchup_weight} "
              f"(mean 1.0, min {min(sample_weights):.3f}, max {max(sample_weights):.3f})", flush=True)
    order0 = list(range(n))
    import random as _random

    rng = _random.Random(config.seed)
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        order = list(order0)
        rng.shuffle(order)
        agg = {"loss": 0.0, "ce": 0.0, "value": 0.0, "belief": 0.0, "acc": 0.0}
        nb = 0
        for start in range(0, n, config.batch_size):
            index = order[start : start + config.batch_size]
            if not index:
                continue
            opt.zero_grad(set_to_none=True)
            loss, info = run_batch(model, train_store, index, config.device, config.value_coef, config.belief_coef, has_belief, config, q_scale, sample_weights)
            loss.backward()
            opt.step()
            agg["loss"] += float(loss.item())
            for k in ("ce", "value", "belief", "acc"):
                agg[k] += info[k]
            nb += 1
        for k in agg:
            agg[k] /= max(1, nb)
        row = {"epoch": epoch, **{k: round(v, 5) for k, v in agg.items()}}
        if val_store is not None and len(val_store):
            v = evaluate(model, val_store, config, config.device, has_belief, val_q_scale)
            row["val_ce"] = round(v["ce"], 5)
            row["val_acc"] = round(v["acc"], 4)
            row["val_value"] = round(v["value"], 5)
            row["val_belief"] = round(v["belief"], 5)
            row["val_n"] = int(v.get("n", 0))
            for extra in ("q_mse", "w_mean", "w_max"):
                if extra in v:
                    row[f"val_{extra}"] = round(float(v[extra]), 5)
        history.append(row)
        if config.log_every and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(f"[train] {json.dumps(row)}", flush=True)
    return history, round(time.perf_counter() - t0, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", nargs="*", default=None)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--val-every", type=int, default=10, help="every Nth game seed -> val (0=no val)")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None, help="cap the val store used for scoring")
    p.add_argument("--train-limit", type=int, default=None, help="cap train store after load (nested subsets)")
    p.add_argument("--arch", choices=["cvpn", "transformer"], default="cvpn")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--belief", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="checkpoint .pt path")
    p.add_argument("--report", default=None, help="aggregate report .json path")
    p.add_argument(
        "--store-cache",
        default=None,
        help="cache the compact tensor store as a .pt (reused across runs; skips JSON re-parse)",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="build/save the store cache and exit (no training)",
    )
    return p


def load_or_build_store(args):
    if args.store_cache:
        import os

        if os.path.exists(args.store_cache):
            print(f"[store] loading cache {args.store_cache}", flush=True)
            payload = torch.load(args.store_cache, map_location="cpu", weights_only=False)
            return payload["train"], payload["val"], payload["meta"], True
    if not args.replays:
        raise SystemExit("provide --replays or an existing --store-cache")
    train_store, val_store, meta = load_store(
        args.replays,
        seed_start=args.seed_start,
        val_every=args.val_every,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        device=args.device,
    )
    if args.store_cache:
        ensure_dir(Path(args.store_cache).parent)
        torch.save({"train": train_store, "val": val_store, "meta": meta}, args.store_cache)
        print(f"[store] wrote cache {args.store_cache}", flush=True)
    return train_store, val_store, meta, False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = default_encoder()
    train_store, val_store, meta, _cached = load_or_build_store(args)
    if args.build_only:
        print(f"[store] built train={len(train_store)} val={len(val_store)} (build-only)", flush=True)
        return 0
    if args.train_limit is not None and args.train_limit < len(train_store):
        train_store = train_store.subset(list(range(args.train_limit)))
    if args.max_val is not None and len(val_store) > args.max_val:
        val_store = val_store.subset(list(range(args.max_val)))
    print(f"[store] train={len(train_store)} val={len(val_store)} meta={meta}", flush=True)
    if args.arch == "transformer":
        cfg = transformer_config_for_encoder(
            encoder, belief=args.belief, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads,
        )
        model = TransformerCVPN(cfg)
    else:
        cfg = config_for_encoder(encoder, belief=args.belief, d_model=args.d_model)
        model = CVPN(cfg)
    has_belief = bool(getattr(model, "has_belief", False))
    config = Config(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, device=args.device, weight_decay=args.weight_decay,
    )
    history, train_seconds = train(train_store, val_store, model, config, has_belief)
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.out:
        out = Path(args.out)
        ensure_dir(out.parent)
    else:
        out = Path("data/checkpoints") / f"{stamp}_bc_scale.pt"
        ensure_dir(out.parent)
    model.save(str(out), encoder_config=encoder.to_dict())
    report = {
        "kind": "bc_scale",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "arch": args.arch,
        "d_model": args.d_model,
        "belief_head": has_belief,
        "config": asdict(config),
        "store": meta,
        "n_train": len(train_store),
        "n_val": len(val_store),
        "train_seconds": train_seconds,
        "history": history,
        "checkpoint": str(out),
        "engine_lock": load_engine_lock(),
        "runtime": runtime_metadata(),
    }
    if args.report:
        rp = Path(args.report)
        ensure_dir(rp.parent)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] ckpt={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
