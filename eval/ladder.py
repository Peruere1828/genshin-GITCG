"""Elo ladder over arena match records (PLAN.md WS2).

A simple, order-dependent Elo is enough for an engineering ladder: it gives a
single scalar per deck and makes "did the new checkpoint regress?" visible. The
intended fixed reference set is 20 scripted decks + baselines + historical
checkpoints (see PLAN.md §4).
"""

from __future__ import annotations

from typing import Any, Iterable

DEFAULT_RATING = 1500.0
DEFAULT_K = 16.0


def _expected(score_a: float, score_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((score_b - score_a) / 400.0))


def compute_elo(
    matches: Iterable[dict[str, Any]],
    *,
    k: float = DEFAULT_K,
    initial: float = DEFAULT_RATING,
) -> dict[str, float]:
    """Update Elo ratings over an ordered list of match records."""
    ordered = sorted(matches, key=lambda m: m.get("task_index", m.get("index", 0)))
    ratings: dict[str, float] = {}

    def rating(name: str) -> float:
        return ratings.setdefault(name, initial)

    for match in ordered:
        deck0, deck1 = match["deck0"], match["deck1"]
        winner = match["winner"]
        score0 = 0.5 if winner is None else (1.0 if int(winner) == 0 else 0.0)
        expected0 = _expected(rating(deck0), rating(deck1))
        delta = k * (score0 - expected0)
        ratings[deck0] = rating(deck0) + delta
        ratings[deck1] = rating(deck1) - delta
    return ratings
