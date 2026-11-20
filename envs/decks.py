"""Deck plumbing: expert deck profiles -> DeckSpec / gitcg.Deck.

The 20 scripted expert decks (``agents/scripted``) double as the fixed opponent
pool (PLAN.md WS2) and as a sanity baseline deck pool.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from gitcg import Deck as GitcgDeck

from reps.schema import DeckSpec


@lru_cache(maxsize=1)
def expert_deck_specs() -> tuple[DeckSpec, ...]:
    """Deck specs for every ported scripted expert (ordered by registry)."""
    from agents.scripted.registry import RAW_DECKS
    from agents.scripted.profiles import load_deck_profiles_by_slug

    profiles = load_deck_profiles_by_slug()
    specs: list[DeckSpec] = []
    for entry in RAW_DECKS:
        profile = profiles[entry.slug]
        specs.append(
            DeckSpec(
                name=entry.slug,
                characters=tuple(int(c) for c in profile.characters),
                cards=tuple(int(c) for c in profile.cards),
            )
        )
    return tuple(specs)


@lru_cache(maxsize=1)
def expert_decks_by_name() -> dict[str, DeckSpec]:
    return {spec.name: spec for spec in expert_deck_specs()}


def deck_spec(name: str) -> DeckSpec:
    decks = expert_decks_by_name()
    if name not in decks:
        raise KeyError(f"unknown deck: {name!r}; known: {sorted(decks)}")
    return decks[name]


def to_gitcg_deck(spec: DeckSpec) -> GitcgDeck:
    return GitcgDeck(characters=list(spec.characters), cards=list(spec.cards))


def deck_spec_from_json(payload: dict) -> DeckSpec:
    """Build a ``DeckSpec`` from a plain dict (``{"name","characters","cards"}``).

    Decks are always supplied as input (D28): there is no deck-selection or
    deck-building step, so the only requirement is that the payload is complete
    and well-formed.
    """
    try:
        name = str(payload["name"])
        characters = tuple(int(c) for c in payload["characters"])
        cards = tuple(int(c) for c in payload["cards"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid deck payload: {exc}") from exc
    if len(characters) != 3:
        raise ValueError(f"deck {name!r} must have exactly 3 characters, got {len(characters)}")
    if len(cards) != 30:
        raise ValueError(f"deck {name!r} must have exactly 30 cards, got {len(cards)}")
    return DeckSpec(name=name, characters=characters, cards=cards)


def load_deck_spec(path: str | Path) -> DeckSpec:
    """Load a committed deck JSON file into a ``DeckSpec``."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return deck_spec_from_json(payload)


CARD_KINDS = ("weapon", "artifact", "talent", "support", "event")


@lru_cache(maxsize=1)
def _asset_catalog():
    from agents.scripted.assets import load_assets

    return load_assets()


@lru_cache(maxsize=1)
def _card_kind_index() -> dict[int, str]:
    """Map every known action card id to ``weapon|artifact|talent|support|event``."""
    index: dict[int, str] = {}
    for card_id, card in _asset_catalog().cards.items():
        tags = set(card.tags)
        if "GCG_TAG_WEAPON" in tags:
            index[int(card_id)] = "weapon"
        elif "GCG_TAG_ARTIFACT" in tags:
            index[int(card_id)] = "artifact"
        elif "GCG_TAG_TALENT" in tags:
            index[int(card_id)] = "talent"
        elif {"GCG_TAG_ALLY", "GCG_TAG_PLACE", "GCG_TAG_ITEM"} & tags:
            index[int(card_id)] = "support"
        else:
            index[int(card_id)] = "event"
    return index


def card_kind(card_id: int) -> str:
    """Coarse action-card kind, from the public asset tags."""
    return _card_kind_index().get(int(card_id), "event")


def deck_summary(deck: DeckSpec) -> dict[str, object]:
    """Small reporting helper: per-kind card counts of a deck."""
    kinds: dict[str, int] = {kind: 0 for kind in CARD_KINDS}
    for card_id in deck.cards:
        kinds[card_kind(card_id)] = kinds.get(card_kind(card_id), 0) + 1
    return {
        "name": deck.name,
        "characters": list(deck.characters),
        "n_cards": len(deck.cards),
        "card_kinds": kinds,
        "cards": list(deck.cards),
    }


def pool_slug_for_deck(payload: dict[str, Any]) -> str | None:
    """Return the pool slug when ``payload`` is exactly one of the scripted decks.

    This is deck *identification* (for aligning a scripted rule set to the deck
    that is already specified), never deck selection: it returns ``None`` for
    every deck outside the 20-deck scripted pool.
    """
    characters = [int(c) for c in (payload.get("characters") or [])]
    cards = sorted(int(c) for c in (payload.get("cards") or []))
    if not characters:
        return None
    for spec in expert_deck_specs():
        if list(spec.characters) == characters and sorted(spec.cards) == cards:
            return spec.name
    return None
