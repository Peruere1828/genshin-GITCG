"""Import official TCG deck share codes (self-contained, no era_decks.py).

Input format for ``--file`` (``<name> <share_code>`` per line)::

    # comments and blank lines are ignored
    <name> <share_code>
    <name> <share_code>   # inline comment after the code is allowed

The importer:

1. decodes each code with the repo codec (``agents.scripted.codec``); the raw
   share ids are decoded first so any share id missing from the asset catalog is
   named precisely, then ``decode_share_code`` maps them to definition ids;
2. prints the three characters (id, name, element, weapon) and the 30 action
   cards grouped by type (talent / weapon / artifact / support / event) with
   copy counts;
3. re-checks the TCG deck-building rules the engine does not enforce (33 ids,
   3 distinct playable characters, exactly 30 cards, <= 2 copies per name,
   talents only for the chosen characters, weapons only for a matching weapon
   type, at most 2 weapon cards per character slot).  Character cards must be
   playable by asset metadata (``category == "characters"``,
   ``obtainable != false``), never by an id range;
4. optionally (``--probe``) loads the deck in the engine with
   ``CreateParam(version=...)`` / ``Game`` at ``--version`` (default: ``None`` =
   latest) and, on failure, names every item outside that version with its
   asset ``sinceVersion``;
5. with ``--write``, appends/updates the deck in ``--out`` (default
   ``configs/imported_decks.json``) as ``source="share_code"`` with the original
   share code, the decoded characters/cards and the per-type composition.
   Undecodable or invalid decks are printed but never written.

No network calls and no invented share codes.  If the input file has no
entries, the importer prints ``waiting for codes`` and exits 0.

Usage::

    PY=/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python   # any env with gitcg
    "$PY" -m scripts.import_deck --code "<SHARE_CODE>" --name my_deck
    "$PY" -m scripts.import_deck --file data/decks.txt --write
    "$PY" -m scripts.import_deck --file data/decks.txt --probe --version v3.3.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agents.scripted.codec import decode_share_code, decode_share_code_raw

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = REPO_ROOT / "data" / "assets" / "expert_system_assets_chs_latest.json"
DEFAULT_OUT = REPO_ROOT / "configs" / "imported_decks.json"

DECK_SIZE = 30
CHARACTER_SLOTS = 3
EXPECTED_IDS = CHARACTER_SLOTS + DECK_SIZE
MAX_COPIES = 2
MAX_WEAPONS_PER_SLOT = 2

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
GROUP_ORDER = ("talent", "weapon", "artifact", "support", "event", "other")

WEAPON_TAGS = {
    "CATALYST": "GCG_TAG_WEAPON_CATALYST",
    "BOW": "GCG_TAG_WEAPON_BOW",
    "CLAYMORE": "GCG_TAG_WEAPON_CLAYMORE",
    "POLE": "GCG_TAG_WEAPON_POLE",
    "SWORD": "GCG_TAG_WEAPON_SWORD",
}
WEAPON_TAGS_BY_TAG = {tag: name for name, tag in WEAPON_TAGS.items()}
# Characters such as the Fatui Pyro Agent have a "no weapon type" tag; keep it
# for display, but it never matches a weapon card.
WEAPON_TAGS_BY_TAG["GCG_TAG_WEAPON_NONE"] = "NONE"
ELEMENT_TAGS = {
    "CRYO": "GCG_TAG_ELEMENT_CRYO",
    "HYDRO": "GCG_TAG_ELEMENT_HYDRO",
    "PYRO": "GCG_TAG_ELEMENT_PYRO",
    "ELECTRO": "GCG_TAG_ELEMENT_ELECTRO",
    "ANEMO": "GCG_TAG_ELEMENT_ANEMO",
    "GEO": "GCG_TAG_ELEMENT_GEO",
    "DENDRO": "GCG_TAG_ELEMENT_DENDRO",
}


@dataclass(frozen=True)
class Entry:
    name: str
    code: str
    source: str  # "--code" or "file:line"


# --------------------------------------------------------------------------- #
# asset catalog
# --------------------------------------------------------------------------- #
def load_asset_entries(path: Path = DEFAULT_ASSETS) -> dict[int, dict[str, Any]]:
    """Asset dump (``{success, data: [...]}``) keyed by definition id."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries: dict[int, dict[str, Any]] = {}
    for item in payload.get("data", ()):
        if isinstance(item.get("id"), int):
            entries[int(item["id"])] = item
    return entries


def share_to_id_map(entries: dict[int, dict[str, Any]]) -> dict[int, int]:
    return {
        int(item["shareId"]): int(item["id"])
        for item in entries.values()
        if isinstance(item.get("shareId"), int)
    }


def _item_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("englishName") or item.get("id") or "?")


def card_kind(item: dict[str, Any]) -> str:
    tags = set(item.get("tags") or ())
    if "GCG_TAG_TALENT" in tags:
        return "talent"
    card_type = str(item.get("type") or "")
    if card_type == "GCG_CARD_MODIFY":
        if "GCG_TAG_WEAPON" in tags:
            return "weapon"
        if "GCG_TAG_ARTIFACT" in tags:
            return "artifact"
        return "other"
    if card_type == "GCG_CARD_ASSIST":
        return "support"
    return "event"


def weapon_type(item: dict[str, Any]) -> str:
    for tag in item.get("tags") or ():
        if tag in WEAPON_TAGS_BY_TAG:
            return WEAPON_TAGS_BY_TAG[tag]
    return ""


def character_element(item: dict[str, Any]) -> str:
    for tag in item.get("tags") or ():
        for name, element_tag in ELEMENT_TAGS.items():
            if tag == element_tag:
                return name
    return ""


def character_weapon(item: dict[str, Any]) -> str:
    return weapon_type(item)


def _version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    match = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", str(value).strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def _version_le(value: str | None, limit: tuple[int, ...] | None) -> bool:
    if limit is None:
        return True
    parsed = _version_tuple(value)
    if parsed is None:
        return False
    padded = parsed + (0,) * (len(limit) - len(parsed))
    return padded <= limit


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def decode_entry(
    name: str, code: str, share_to_id: dict[int, int]
) -> tuple[list[int], list[int]] | None:
    """Decode one share code; print a precise reason and return None on failure."""
    try:
        share_ids = decode_share_code_raw(code)
    except Exception as exc:  # base64/length problems, never fatal for the batch
        print(f"[import] {name}: cannot decode share code: {exc}", flush=True)
        return None
    missing = [share_id for share_id in share_ids if share_id not in share_to_id]
    if missing:
        print(
            f"[import] {name}: share id(s) {missing} are not in the asset catalog "
            "(the dump is older/newer than this code)",
            flush=True,
        )
        return None
    if len(share_ids) != EXPECTED_IDS:
        print(
            f"[import] {name}: decoded {len(share_ids)} share ids, expected "
            f"{EXPECTED_IDS} ({CHARACTER_SLOTS} characters + {DECK_SIZE} cards)",
            flush=True,
        )
        return None
    try:
        characters, cards = decode_share_code(code, share_to_id)
    except KeyError as exc:  # defensive: the missing check above should catch it
        print(
            f"[import] {name}: share id {exc.args[0]} has no definition id in the "
            "asset catalog",
            flush=True,
        )
        return None
    return [int(character) for character in characters], [int(card) for card in cards]


# --------------------------------------------------------------------------- #
# metadata + deck-building rules
# --------------------------------------------------------------------------- #
def composition(cards: Iterable[int], entries: dict[int, dict[str, Any]]) -> dict[str, int]:
    counts = {kind: 0 for kind in GROUP_ORDER}
    total = 0
    for card_id in cards:
        counts[card_kind(entries.get(int(card_id), {}))] += 1
        total += 1
    return {**counts, "total": total}


def character_label(character: int, entries: dict[int, dict[str, Any]]) -> str:
    item = entries.get(character, {})
    element = character_element(item) or "?"
    weapon = character_weapon(item) or "?"
    return f"{character} {_item_name(item)} [{element} / {weapon}]"


def card_label(card_id: int, entries: dict[int, dict[str, Any]]) -> str:
    return f"{card_id} {_item_name(entries.get(card_id, {}))}"


def validate_deck(
    characters: Iterable[int],
    cards: Iterable[int],
    entries: dict[int, dict[str, Any]],
) -> list[str]:
    """Deck-building rules the engine does not enforce itself."""
    characters = [int(character) for character in characters]
    cards = [int(card_id) for card_id in cards]
    errors: list[str] = []

    if len(characters) != CHARACTER_SLOTS or len(set(characters)) != CHARACTER_SLOTS:
        errors.append(
            f"expected {CHARACTER_SLOTS} distinct characters, got {characters}"
        )
    for character in characters:
        item = entries.get(character)
        if item is None:
            errors.append(f"character {character} is unknown to the asset catalog")
        elif item.get("category") != "characters" or item.get("obtainable") is False:
            errors.append(
                f"character {character} ({_item_name(item)}) is not a playable "
                "character card (category/obtainable)"
            )

    if len(cards) != DECK_SIZE:
        errors.append(f"expected {DECK_SIZE} cards, got {len(cards)}")

    names: dict[str, int] = {}
    for card_id in cards:
        item = entries.get(card_id)
        if item is None:
            errors.append(f"card {card_id} is unknown to the asset catalog")
            continue
        name = _item_name(item)
        names[name] = names.get(name, 0) + 1
        if card_kind(item) == "other":
            errors.append(f"card {card_id} ({name}) is not a player action card")
    for name, copies in names.items():
        if copies > MAX_COPIES:
            errors.append(f"card name {name!r} has {copies} copies (max {MAX_COPIES})")

    character_set = set(characters)
    weapon_tags = {
        WEAPON_TAGS[character_weapon(entries[character])]
        for character in characters
        if character in entries
        and character_weapon(entries[character]) in WEAPON_TAGS
    }
    weapons: list[int] = []
    for card_id in cards:
        item = entries.get(card_id)
        if item is None:
            continue
        kind = card_kind(item)
        if kind == "talent":
            related = int(item.get("relatedCharacterId") or 0)
            if related not in character_set:
                errors.append(
                    f"talent {card_id} ({_item_name(item)}) is for character "
                    f"{related}, not in {characters}"
                )
        elif kind == "weapon":
            weapons.append(card_id)
            if not set(item.get("tags") or ()) & weapon_tags:
                errors.append(
                    f"weapon {card_id} ({_item_name(item)}) is not usable by any "
                    f"chosen character (attributes: {sorted(weapon_tags)})"
                )
    if len(weapons) > MAX_WEAPONS_PER_SLOT * len(characters):
        errors.append(
            f"{len(weapons)} weapon cards > {MAX_WEAPONS_PER_SLOT} per character slot"
        )
    return errors


# --------------------------------------------------------------------------- #
# optional engine probe
# --------------------------------------------------------------------------- #
def probe_load(
    characters: Iterable[int],
    cards: Iterable[int],
    *,
    version: str | None = None,
) -> tuple[bool, str]:
    """Build one engine ``Game`` and capture the engine's stderr message.

    The gitcg binding raises a bare ``AssertionError``; the useful text
    (``Unknown card id <id>`` / ``Unknown character id <id>``) is printed by the
    embedded JS runtime on fd 2, so fd 2 is piped for the construction.
    """
    from gitcg import CreateParam, Game

    read_fd, write_fd = os.pipe()
    saved_stderr = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    ok = True
    python_error = ""
    try:
        create_param = CreateParam(version=version)
        create_param.set_characters(0, [int(character) for character in characters])
        create_param.set_cards(0, [int(card) for card in cards])
        create_param.set_characters(1, [3001, 3002, 3002])
        create_param.set_cards(1, [311101] * DECK_SIZE)
        try:
            Game(create_param=create_param)
        except Exception as exc:  # the binding swallows the message; see above
            ok = False
            python_error = f"{type(exc).__name__}: {exc}"
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(read_fd)
    message = b"".join(chunks).decode("utf-8", "replace").strip()
    if python_error:
        message = f"{message} | {python_error}".strip(" |")
    return ok, message


def _report_out_of_version(
    label: str,
    ids: Iterable[int],
    entries: dict[int, dict[str, Any]],
    version: str | None,
) -> None:
    limit = _version_tuple(version)
    outside = list(
        dict.fromkeys(
            item_id
            for item_id in ids
            if item_id in entries
            and not _version_le(entries[item_id].get("sinceVersion"), limit)
        )
    )
    if not outside:
        return
    print(f"    {label} outside {version}:", flush=True)
    for item_id in outside:
        item = entries[item_id]
        print(
            f"      {card_label(item_id, entries)} "
            f"(sinceVersion {item.get('sinceVersion') or 'unknown'})",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def print_deck(
    entry: Entry,
    characters: list[int],
    cards: list[int],
    entries: dict[int, dict[str, Any]],
    *,
    rule_errors: list[str],
    probe: tuple[bool, str] | None = None,
    version: str | None = None,
) -> None:
    print(f"[import] {entry.name}  ({entry.source})", flush=True)
    print(
        f"  share code: {entry.code[:12]}...  "
        f"({len(characters)} characters + {len(cards)} cards)",
        flush=True,
    )
    print(f"  characters ({len(characters)}):", flush=True)
    for character in characters:
        print(f"    {character_label(character, entries)}", flush=True)

    grouped: dict[str, list[int]] = {kind: [] for kind in GROUP_ORDER}
    for card_id in cards:
        grouped[card_kind(entries.get(card_id, {}))].append(card_id)
    print(f"  action cards ({len(cards)}):", flush=True)
    for kind in GROUP_ORDER:
        ids = grouped.get(kind, [])
        if not ids:
            continue
        counts: dict[int, int] = {}
        for card_id in ids:
            counts[card_id] = counts.get(card_id, 0) + 1
        print(f"    {kind} ({len(ids)}):", flush=True)
        for card_id in sorted(counts):
            item = entries.get(card_id, {})
            suffix = ""
            if kind == "talent" and item.get("relatedCharacterId"):
                related = int(item["relatedCharacterId"])
                related_item = entries.get(related)
                owner = _item_name(related_item) if related_item else related
                suffix = f"  ({owner} talent)"
            print(f"      {card_label(card_id, entries)}  x{counts[card_id]}{suffix}", flush=True)

    counts = composition(cards, entries)
    print(
        "  composition: "
        + " / ".join(f"{counts[kind]} {kind}" for kind in GROUP_ORDER[:5])
        + f" = {counts['total']}",
        flush=True,
    )

    if probe is not None:
        ok, message = probe
        target = version or "latest"
        print(f"  engine load at {target}: {'OK' if ok else 'FAIL'}", flush=True)
        if message:
            print(f"    engine: {message}", flush=True)
        if not ok:
            _report_out_of_version("characters", characters, entries, version)
            _report_out_of_version("cards", cards, entries, version)

    if rule_errors:
        print("  rule check: FAIL", flush=True)
        for error in rule_errors:
            print(f"    - {error}", flush=True)
    else:
        print("  rule check: OK", flush=True)
    print("", flush=True)


# --------------------------------------------------------------------------- #
# input / output
# --------------------------------------------------------------------------- #
def parse_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    seen: set[str] = set()
    if not path.exists():
        return entries
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        parts = line.split()
        if len(parts) < 2:
            raise SystemExit(
                f"{path}:{lineno}: expected '<name> <share_code>', got {raw!r}"
            )
        name, code = parts[0], parts[1]
        if not NAME_RE.match(name):
            raise SystemExit(
                f"{path}:{lineno}: deck name {name!r} must match {NAME_RE.pattern}"
            )
        if name in seen:
            raise SystemExit(f"{path}:{lineno}: duplicate deck name {name!r}")
        seen.add(name)
        entries.append(Entry(name=name, code=code, source=f"{path}:{lineno}"))
    return entries


def deck_record(
    name: str,
    code: str,
    characters: list[int],
    cards: list[int],
    entries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": name,
        "source": "share_code",
        "share_code": code,
        "characters": [int(character) for character in characters],
        "cards": [int(card_id) for card_id in cards],
        "composition": composition(cards, entries),
    }


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"decks": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    payload.setdefault("decks", {})
    return payload


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def import_entry(
    entry: Entry,
    *,
    entries: dict[int, dict[str, Any]],
    share_to_id: dict[int, int],
    config: dict[str, Any],
    out_path: Path,
    write: bool,
    probe: bool,
    version: str | None,
) -> bool:
    """Decode / print / validate one entry; write it when valid. True == valid."""
    decoded = decode_entry(entry.name, entry.code, share_to_id)
    if decoded is None:
        print(f"[import] {entry.name}: NOT written (undecodable)\n", flush=True)
        return False
    characters, cards = decoded
    rule_errors = validate_deck(characters, cards, entries)

    probe_result: tuple[bool, str] | None = None
    if probe:
        try:
            probe_result = probe_load(characters, cards, version=version)
        except ImportError as exc:
            print(f"[import] {entry.name}: engine probe skipped: {exc}", flush=True)
    print_deck(
        entry,
        characters,
        cards,
        entries,
        rule_errors=rule_errors,
        probe=probe_result,
        version=version,
    )

    valid = not rule_errors and (probe_result is None or probe_result[0])
    if not valid:
        print(f"[import] {entry.name}: NOT written (invalid)\n", flush=True)
        return False

    record = deck_record(entry.name, entry.code, characters, cards, entries)
    if not write:
        print(f"[import] {entry.name}: dry-run, not written\n", flush=True)
        return True
    decks = config.setdefault("decks", {})
    replaced = entry.name in decks
    decks[entry.name] = record
    # Persist after each entry so a later failure does not discard earlier ones.
    write_config(out_path, config)
    print(
        f"[import] {entry.name}: written to {out_path} "
        f"({'updated' if replaced else 'new'})\n",
        flush=True,
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--code", default=None, help="import a single share code")
    parser.add_argument(
        "--name", default="imported_deck", help="deck id used with --code"
    )
    parser.add_argument(
        "--file",
        default=None,
        help="input list, one '<name> <share_code>' per line",
    )
    parser.add_argument("--assets", default=str(DEFAULT_ASSETS), help="asset dump")
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help="JSON output (with --write)"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write valid decks to --out (default: dry-run)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="also load the deck in the engine at --version",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="engine version for --probe (default: None = latest)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assets_path = Path(args.assets)
    if not assets_path.exists():
        print(f"[import] asset dump not found: {assets_path}", flush=True)
        return 1
    entries = load_asset_entries(assets_path)
    share_to_id = share_to_id_map(entries)

    if args.code:
        if not NAME_RE.match(args.name):
            raise SystemExit(f"--name {args.name!r} must match {NAME_RE.pattern}")
        rows = [Entry(name=args.name, code=args.code, source="--code")]
    else:
        if not args.file:
            raise SystemExit("pass --code <SHARE_CODE> or --file <path>")
        input_path = Path(args.file)
        if not input_path.exists():
            print(
                f"[import] waiting for codes: {input_path} does not exist.", flush=True
            )
            return 0
        rows = parse_entries(input_path)
        if not rows:
            print(
                f"[import] waiting for codes: {input_path} has no deck entries.",
                flush=True,
            )
            return 0

    out_path = Path(args.out)
    config = load_config(out_path) if args.write else {"decks": {}}
    print(f"[import] {len(rows)} entr{'y' if len(rows) == 1 else 'ies'}", flush=True)
    valid = 0
    for entry in rows:
        if import_entry(
            entry,
            entries=entries,
            share_to_id=share_to_id,
            config=config,
            out_path=out_path,
            write=bool(args.write),
            probe=bool(args.probe),
            version=args.version,
        ):
            valid += 1

    summary = f"{valid}/{len(rows)} share codes valid, {len(rows) - valid} invalid"
    print(f"[import] summary: {summary}", flush=True)
    if args.write:
        print(
            f"[import] config: {out_path} now has "
            f"{len(config.get('decks', {}))} deck(s)",
            flush=True,
        )
    else:
        print("[import] dry-run: nothing written (pass --write to save)", flush=True)
    return 0 if valid == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
