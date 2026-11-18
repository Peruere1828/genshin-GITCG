"""Share-code deck importer (``scripts.import_deck``).

Fast and network-free.  The two share codes below are official in-game codes
kept inline on purpose so the tests never depend on a ``gitcg-skills`` checkout.
Skips when the asset dump is absent (it is fetched by ``agents.scripted.assets``
and lives in the gitignored ``data/assets/``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.import_deck import (
    DEFAULT_ASSETS,
    WEAPON_TAGS,
    Entry,
    card_kind,
    character_weapon,
    decode_entry,
    import_entry,
    load_asset_entries,
    main,
    parse_entries,
    share_to_id_map,
    validate_deck,
)

# Official v3.3-era in-game deck share codes.
KEQING_JIHUA = "AbGgDi8OAWDA5WsOBWCA9LQPDxBgyvYMDtBQa/IJD5AwmbMPC4BQ+PcPD8CQ9P0PEBAA"
NVPU_SHUANGYAN = "AkCw9ioPAGCg+IYPCJFhC00QFKDQ9ekMDiCQyuoMDsCgy/EMBpBgt5cLDzAw8fQQD7AA"
CODES = (("keqing_jihua", KEQING_JIHUA), ("nvpu_shuangyan", NVPU_SHUANGYAN))

pytestmark = pytest.mark.skipif(
    not DEFAULT_ASSETS.exists(), reason=f"asset dump absent: {DEFAULT_ASSETS}"
)


@pytest.fixture(scope="module")
def catalog() -> dict[int, dict]:
    return load_asset_entries(DEFAULT_ASSETS)


@pytest.fixture(scope="module")
def share_to_id(catalog: dict[int, dict]) -> dict[int, int]:
    return share_to_id_map(catalog)


def _decode(name: str, code: str, share_to_id: dict[int, int]) -> tuple[list[int], list[int]]:
    decoded = decode_entry(name, code, share_to_id)
    assert decoded is not None, f"{name} should decode"
    return decoded


def _replace_card(cards: list[int], old: int, new: int) -> list[int]:
    return [new if card == old else card for card in cards]


# --------------------------------------------------------------------------- #
# real share codes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,code", CODES)
def test_real_share_codes_decode_to_legal_decks(
    name: str, code: str, catalog: dict[int, dict], share_to_id: dict[int, int]
) -> None:
    characters, cards = _decode(name, code, share_to_id)
    assert len(characters) == 3
    assert len(set(characters)) == 3
    assert len(cards) == 30
    assert max(Counter(cards).values()) <= 2
    assert validate_deck(characters, cards, catalog) == []

    for character in characters:
        item = catalog[character]
        assert item.get("category") == "characters"
        assert item.get("obtainable") is not False


def test_keqing_jihua_decodes_to_expected_team(
    catalog: dict[int, dict], share_to_id: dict[int, int]
) -> None:
    characters, _ = _decode("keqing_jihua", KEQING_JIHUA, share_to_id)
    assert characters == [1401, 1701, 1403]  # Fischl, Collei, Keqing
    names = {catalog[c]["name"] for c in characters}
    assert names == {"菲谢尔", "柯莱", "刻晴"}


def test_missing_share_id_is_named(
    catalog: dict[int, dict],
    share_to_id: dict[int, int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agents.scripted.codec import decode_share_code_raw

    dropped = decode_share_code_raw(KEQING_JIHUA)[-1]
    partial = {key: value for key, value in share_to_id.items() if key != dropped}
    assert decode_entry("dropped", KEQING_JIHUA, partial) is None
    out = capsys.readouterr().out
    assert str(dropped) in out
    assert "not in the asset catalog" in out


# --------------------------------------------------------------------------- #
# deck-building rules (the engine does not enforce these)
# --------------------------------------------------------------------------- #
def test_rule_check_rejects_31_cards(catalog: dict[int, dict]) -> None:
    characters, cards = _decode("keqing_jihua", KEQING_JIHUA, share_to_id_map(catalog))
    errors = validate_deck(characters, cards + [cards[0]], catalog)
    assert any("expected 30 cards, got 31" in error for error in errors), errors


def test_rule_check_rejects_three_copies(catalog: dict[int, dict]) -> None:
    characters, cards = _decode("keqing_jihua", KEQING_JIHUA, share_to_id_map(catalog))
    counts = Counter(cards)
    doubled = next(card for card, copies in counts.items() if copies == 2)
    single = next(
        card for card in cards if counts[card] < 2 and card != doubled
    )
    errors = validate_deck(characters, _replace_card(cards, single, doubled), catalog)
    assert any("3 copies (max 2)" in error for error in errors), errors


def test_rule_check_rejects_foreign_talent(catalog: dict[int, dict]) -> None:
    characters, cards = _decode("keqing_jihua", KEQING_JIHUA, share_to_id_map(catalog))
    team = set(characters)
    foreign = next(
        card_id
        for card_id, item in catalog.items()
        if card_kind(item) == "talent"
        and int(item.get("relatedCharacterId") or 0) not in team
    )
    victim = next(card for card in cards if card_kind(catalog[card]) != "talent")
    errors = validate_deck(characters, _replace_card(cards, victim, foreign), catalog)
    assert any("talent" in error and "not in" in error for error in errors), errors


def test_rule_check_rejects_mismatched_weapon(catalog: dict[int, dict]) -> None:
    characters, cards = _decode("keqing_jihua", KEQING_JIHUA, share_to_id_map(catalog))
    team_tags = {
        WEAPON_TAGS[character_weapon(catalog[character])]
        for character in characters
        if character_weapon(catalog[character]) in WEAPON_TAGS
    }
    foreign = next(
        card_id
        for card_id, item in catalog.items()
        if card_kind(item) == "weapon"
        and not set(item.get("tags") or ()) & team_tags
    )
    victim = next(card for card in cards if card_kind(catalog[card]) == "event")
    errors = validate_deck(characters, _replace_card(cards, victim, foreign), catalog)
    assert any("not usable by any chosen character" in error for error in errors), errors


def test_rule_check_rejects_unplayable_character(catalog: dict[int, dict]) -> None:
    characters, cards = _decode("keqing_jihua", KEQING_JIHUA, share_to_id_map(catalog))
    not_a_character = 311101  # a real action card, not a character card
    assert catalog[not_a_character].get("category") != "characters"
    errors = validate_deck(
        [not_a_character, *characters[1:]], cards, catalog
    )
    assert any("not a playable character card" in error for error in errors), errors


# --------------------------------------------------------------------------- #
# CLI: dry-run / write / empty input
# --------------------------------------------------------------------------- #
def test_dry_run_prints_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "imported_decks.json"
    assert main(["--code", KEQING_JIHUA, "--name", "demo", "--out", str(out)]) == 0
    assert not out.exists()


def test_write_appends_and_updates(tmp_path: Path, catalog: dict[int, dict]) -> None:
    out = tmp_path / "imported_decks.json"
    out.write_text(
        json.dumps({"decks": {"other": {"name": "other", "characters": [], "cards": []}}}),
        encoding="utf-8",
    )
    config = json.loads(out.read_text(encoding="utf-8"))
    share_to_id = share_to_id_map(catalog)

    for name, code in CODES:
        entry = Entry(name=name, code=code, source="test")
        assert import_entry(
            entry,
            entries=catalog,
            share_to_id=share_to_id,
            config=config,
            out_path=out,
            write=True,
            probe=False,
            version=None,
        )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["decks"]) == {"other", *[name for name, _ in CODES]}
    record = payload["decks"]["keqing_jihua"]
    assert record["source"] == "share_code"
    assert record["share_code"] == KEQING_JIHUA
    assert record["characters"] == [1401, 1701, 1403]
    assert len(record["cards"]) == 30
    assert record["composition"]["total"] == 30
    assert sum(
        record["composition"][kind]
        for kind in ("talent", "weapon", "artifact", "support", "event", "other")
    ) == 30

    # Re-importing the same name updates in place instead of duplicating it.
    again = Entry(name="keqing_jihua", code=KEQING_JIHUA, source="test")
    assert import_entry(
        again,
        entries=catalog,
        share_to_id=share_to_id,
        config=payload,
        out_path=out,
        write=True,
        probe=False,
        version=None,
    )
    assert set(json.loads(out.read_text(encoding="utf-8"))["decks"]) == {
        "other",
        *[name for name, _ in CODES],
    }


def test_invalid_entry_is_never_written(tmp_path: Path, catalog: dict[int, dict]) -> None:
    out = tmp_path / "imported_decks.json"
    config = {"decks": {}}
    # A truncated/embedded invalid share code cannot decode at all.
    entry = Entry(name="broken", code="not-a-real-code", source="test")
    assert not import_entry(
        entry,
        entries=catalog,
        share_to_id=share_to_id_map(catalog),
        config=config,
        out_path=out,
        write=True,
        probe=False,
        version=None,
    )
    assert not out.exists()
    assert config["decks"] == {}


def test_parse_entries_ignores_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "decks.txt"
    path.write_text(
        "# comment\n"
        "\n"
        f"keqing_jihua {KEQING_JIHUA}  # inline comment\n"
        f"nvpu_shuangyan {NVPU_SHUANGYAN}\n",
        encoding="utf-8",
    )
    entries = parse_entries(path)
    assert [entry.name for entry in entries] == ["keqing_jihua", "nvpu_shuangyan"]
    assert [entry.code for entry in entries] == [KEQING_JIHUA, NVPU_SHUANGYAN]


def test_file_without_entries_waits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "decks.txt"
    path.write_text("# nothing yet\n\n", encoding="utf-8")
    assert main(["--file", str(path)]) == 0
    assert "waiting for codes" in capsys.readouterr().out


def test_missing_input_file_waits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--file", str(tmp_path / "absent.txt")]) == 0
    assert "waiting for codes" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# optional engine probe (skipped when the gitcg binding is unavailable)
# --------------------------------------------------------------------------- #
def test_probe_loads_deck_in_latest_engine(catalog: dict[int, dict]) -> None:
    pytest.importorskip("gitcg")
    from scripts.import_deck import probe_load

    characters, cards = _decode("keqing_jihua", KEQING_JIHUA, share_to_id_map(catalog))
    ok, message = probe_load(characters, cards, version=None)
    assert ok, message
