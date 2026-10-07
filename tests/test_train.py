"""Toy CVPN pipeline: data collection, model shapes, overfit, and play-through."""

from __future__ import annotations

import pytest

from agents.neural import NeuralPolicy
from envs.decks import deck_spec
from envs.match import run_match
from envs.observation import default_encoder
from envs.policy import expert_policy
from train.data import collect_samples, sample_stats
from train.model import CVPN, collate, config_for_encoder
from train.train_loop import TrainConfig, train_bc


@pytest.fixture(scope="module")
def collected():
    encoder = default_encoder()
    samples = collect_samples(
        "superconduct_aggro", "natlan_battleship", (0,), encoder=encoder
    )
    assert samples
    return encoder, samples


def test_collection_produces_well_formed_samples(collected):
    encoder, samples = collected
    stats = sample_stats(samples)
    assert stats["n"] == len(samples)
    assert stats["max_options"] >= 2
    for sample in samples:
        assert sample.observation.option_features
        assert 0 <= sample.teacher_index < sample.n_options
        assert sample.value_target in (0.0, 0.5, 1.0)


def test_collate_pads_to_batch(collected):
    _encoder, samples = collected
    import torch

    tokens, token_mask, options, option_mask = collate(
        [s.observation for s in samples[:5]]
    )
    assert tokens.shape[0] == 5
    assert token_mask.shape == tokens.shape[:2]
    assert options.shape[0] == 5
    assert option_mask.shape == options.shape[:2]
    assert bool(token_mask.any()) and bool(option_mask.any())
    assert tokens.dtype == torch.float32


@pytest.mark.slow
def test_overfit_small_dataset(collected):
    encoder, samples = collected
    model = CVPN(config_for_encoder(encoder))
    metrics = train_bc(
        samples,
        model,
        TrainConfig(epochs=120, lr=8e-3, batch_size=16, val_fraction=0.0, value_coef=0.0),
    )
    # Majority-class baseline is ~0.3 here; beating 0.6 shows the network learns.
    assert metrics.train_accuracy > 0.6, metrics.as_dict()


@pytest.mark.slow
def test_trained_network_plays_a_legal_game(collected):
    encoder, samples = collected
    model = CVPN(config_for_encoder(encoder))
    train_bc(samples, model, TrainConfig(epochs=30, lr=5e-3, batch_size=16, value_coef=0.0))
    policy = NeuralPolicy(model=model, encoder=encoder)
    record = run_match(
        deck_spec("superconduct_aggro"),
        deck_spec("natlan_battleship"),
        policy,
        expert_policy("natlan_battleship", seed=1),
        seed=42,
    )
    assert record.error is None
    assert record.io_errors0 == ()
    assert record.fallbacks0 == 0
