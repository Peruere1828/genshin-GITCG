"""Whole-game batching (``--batch-games``) for the lower-variance PG update."""

from __future__ import annotations

import random

import torch

from scripts.train_bc_scale import Store, game_batches, game_groups


def _store(games: list[int]) -> Store:
    """One toy store where game g has ``games[g]`` samples."""
    keys: list[tuple[str, str, int, int]] = []
    for seed, steps in enumerate(games):
        for step in range(steps):
            keys.append(("dvalin_bonk", "opp", seed, step))
    n = len(keys)
    empty = torch.empty((0,), dtype=torch.float32)
    return Store(
        tokens=[torch.zeros((1, 65)) for _ in range(n)],
        options=[torch.zeros((1, 68)) for _ in range(n)],
        teacher=[0] * n,
        value=[0.0] * n,
        b_hand=[empty] * n,
        b_deck=[empty] * n,
        b_hist=[empty] * n,
        b_hsize=[0.0] * n,
        b_burst=[0.0] * n,
        b_mask=[False] * n,
        keys=keys,
        priv=[empty] * n,
        has_history=False,
    )


def test_groups_are_contiguous_and_cover_every_sample():
    store = _store([3, 5, 2, 4])
    groups = game_groups(store)
    assert [len(g) for g in groups] == [3, 5, 2, 4]
    flat = [i for g in groups for i in g]
    assert flat == list(range(len(store)))


def test_batches_hold_whole_games_and_meet_the_minimum():
    store = _store([2] * 10)
    batches = list(game_batches(store, 4, max_samples=10_000))
    assert all(len(b) >= 4 for b in batches)
    # every batch is a union of whole games (indices stay contiguous in pairs)
    for batch in batches:
        assert len(batch) % 2 == 0
        for a, b in zip(batch[::2], batch[1::2]):
            assert b == a + 1


def test_sample_cap_splits_before_overflowing():
    store = _store([10] * 10)
    batches = list(game_batches(store, 100, max_samples=25))
    assert batches
    assert all(len(b) <= 30 for b in batches)      # never exceeds cap + one game
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(store)))


def test_shuffle_is_seeded_and_keeps_the_covering():
    store = _store([2] * 8)
    a = list(game_batches(store, 3, rng=random.Random(0)))
    b = list(game_batches(store, 3, rng=random.Random(0)))
    c = list(game_batches(store, 3, rng=random.Random(1)))
    assert a == b
    assert a != c
    assert sorted(i for batch in a for i in batch) == list(range(len(store)))


def test_empty_store_yields_nothing():
    store = _store([])
    assert game_groups(store) == []
    assert list(game_batches(store, 4)) == []


def test_interleaved_layout_groups_by_key_value():
    """A shuffled/limited store is not contiguous -- grouping must still work."""
    store = _store([2, 2, 2])
    # interleave: game 0 step0, game 1 step0, game 2 step0, game 0 step1, ...
    order = [0, 2, 4, 1, 3, 5]
    shuffled = store.subset(order)
    groups = game_groups(shuffled)
    assert sorted(len(g) for g in groups) == [2, 2, 2]
    assert sorted(i for g in groups for i in g) == list(range(6))
