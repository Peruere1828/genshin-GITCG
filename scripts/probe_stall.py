"""Forensic replay of long / drawn NPC PvE games (the "stall" probe).

A PvE game that runs to the engine round cap comes back with ``winner=None``
and ``rounds=maxRoundsCount`` (default **15**, ``ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT``).
That is not a normal outcome, so this probe replays one ``(stage, seed)`` with a
policy and dumps enough state to tell *why* nobody closed:

* per-round HP of every character on both sides (from the public notification
  state captured before each decision),
* every damage mutation (value, target, old/new HP, reaction, defeated flag),
* every ``skill_used`` / ``declare_end`` / ``set_winner`` mutation with its round,
* the final state: phase, winner, turn, hand/pile sizes, dice, declared-end,
* per-player decision counts (including how often each policy chose
  ``action_declare_end``).

How the trace is captured
-------------------------
``envs.match.run_match`` builds its own ``PolicyPlayer``; the public mutation
stream (where damage values live) is only visible from ``on_notify``. The probe
therefore subclasses ``PolicyPlayer`` and swaps it into ``envs.match`` for the
duration of one replay, so match semantics stay exactly those of ``run_match``.
Policies are wrapped in ``TracePolicy`` only to snapshot the acting player's
public view (``built.context.player_view``) before each decision; the wrapped
policy still makes the choice.

Common stall root causes this exposes: elemental wards / elemental-life immunity
(e.g. Cryo Ward 131031 or an Electro Slime's elemental life) absorbing every hit
against a policy that never switches to the countering element — see the
per-round ``attempted`` vs ``effective`` columns.

Usage::

    PY=/path/to/conda/envs/gitcg/bin/python
    "$PY" -m scripts.probe_stall --stage chefmao --seed 1 --policy heuristic
    "$PY" -m scripts.probe_stall --stage buyun --seed 5 --deck superconduct_aggro \\
        --policy expert:keqing_jihua --json-out data/runs/stall_probe.jsonl
    "$PY" -m scripts.probe_stall --raw data/npc/<run>.jsonl --min-rounds 10
    "$PY" -m scripts.probe_stall --check-round-cap
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import PolicyPlayer, _build_create_param, run_match
from envs.policy import build_policy
from npc.policy import npc_policy_for_stage
from npc.spec import get_stage, stage_names
from npc.stage_rules import stage_rules_json
from reps.schema import DeckSpec

DEFAULT_DECK = "superconduct_aggro"
DEFAULT_MIN_ROUNDS = 10


# --------------------------------------------------------------------------- #
# mutation trace
# --------------------------------------------------------------------------- #
_MUTATIONS: list[dict[str, Any]] = []


def _enum_name(message: Any, field_name: str) -> str:
    """Name of an enum-valued proto field (``"SKILL_TYPE_BURST"`` ...)."""
    try:
        descriptor = message.DESCRIPTOR.fields_by_name[field_name]
        return descriptor.enum_type.values_by_number[int(getattr(message, field_name))].name
    except Exception:  # unknown value / descriptor mismatch on an older wheel
        return str(int(getattr(message, field_name, 0)))


class ProbePlayer(PolicyPlayer):
    """``PolicyPlayer`` that also stores the public mutation stream (P0 only).

    The engine delivers the public notification to both players; one sink is
    enough, and reading player 0 avoids double counting (same convention as
    ``run_match``'s ``event_sink``).
    """

    def on_notify(self, notification: Any) -> None:
        super().on_notify(notification)
        if self.who != 0:
            return
        state = getattr(notification, "state", None)
        round_number = int(getattr(state, "round_number", 0)) if state is not None else 0
        turn = int(getattr(state, "current_turn", 0)) if state is not None else 0
        for mutation in getattr(notification, "mutation", ()):
            which = mutation.WhichOneof("mutation")
            if which is None:
                continue
            record: dict[str, Any] = {
                "round": round_number,
                "turn": turn,
                "kind": which,
            }
            if which == "skill_used":
                skill = mutation.skill_used
                record.update(
                    {
                        "who": int(skill.who),
                        "skill_definition_id": int(skill.skill_definition_id),
                        "skill_type": _enum_name(skill, "skill_type"),
                        "caller_definition_id": int(skill.caller_definition_id),
                    }
                )
            elif which == "damage":
                damage = mutation.damage
                record.update(
                    {
                        "value": int(damage.value),
                        "target_id": int(damage.target_id),
                        "target_definition_id": int(damage.target_definition_id),
                        "source_id": int(damage.source_id),
                        "source_definition_id": int(damage.source_definition_id),
                        "reaction": int(damage.reaction_type),
                        "old_health": int(damage.old_health),
                        "new_health": int(damage.new_health),
                        "cause_defeated": bool(damage.cause_defeated),
                        "heal_kind": int(damage.heal_kind),
                    }
                )
            elif which == "set_player_flag":
                flag = mutation.set_player_flag
                record.update(
                    {
                        "who": int(flag.who),
                        "flag": _enum_name(flag, "flag_name"),
                        "value": bool(flag.flag_value),
                    }
                )
            elif which == "player_status_change":
                status = mutation.player_status_change
                record.update({"who": int(status.who), "status": int(status.status)})
            elif which == "change_phase":
                phase = mutation.change_phase
                record.update(
                    {
                        "phase": _enum_name(phase, "new_phase"),
                        "has_change": bool(phase.has_change),
                    }
                )
            elif which == "set_winner":
                record.update({"winner": int(mutation.set_winner.winner)})
            elif which == "switch_active":
                switch = mutation.switch_active
                record.update(
                    {
                        "who": int(switch.who),
                        "character_definition_id": int(switch.character_definition_id),
                    }
                )
            elif which == "switch_turn":
                record.update({"who": int(getattr(mutation.switch_turn, "who", -1))})
            _MUTATIONS.append(record)


# --------------------------------------------------------------------------- #
# policy wrapper: snapshot the acting player's public view at every decision
# --------------------------------------------------------------------------- #
@dataclass
class TracePolicy:
    inner: Any
    who: int
    seq: list[int]
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", "policy")

    def choose(self, built: Any) -> int:
        view = built.context.player_view
        self.seq[0] += 1
        self.steps.append(
            {
                "seq": self.seq[0],
                "who": int(self.who),
                "request_type": str(built.context.request_type.value),
                "round": int(view.round_number) if view is not None else 0,
                "turn": int(view.current_turn) if view is not None else 0,
                "phase": str(view.phase) if view is not None else "",
                "players": [_player_view(view, who) for who in (0, 1)]
                if view is not None
                else [],
                "legal": [
                    str(getattr(spec, "label", ""))
                    for spec in built.context.legal_low_level_specs
                ],
            }
        )
        return int(self.inner.choose(built))


def _player_view(view: Any, who: int) -> dict[str, Any]:
    player = view.players[int(who)]
    return {
        "characters": [
            {
                "id": int(character.id),
                "definition_id": int(character.definition_id),
                "health": int(character.health),
                "max_health": int(character.max_health),
                "energy": int(character.energy),
                "max_energy": int(character.max_energy),
                "defeated": bool(character.defeated),
                "active": bool(character.is_active),
                "aura": int(character.aura),
                "entities": [int(entity.definition_id) for entity in character.entities],
            }
            for character in player.characters
        ],
        "dice": [int(die) for die in player.dice],
        "hand": [int(card.definition_id) for card in player.hand_cards],
        "pile": [int(card.definition_id) for card in player.pile_cards],
        "declared_end": bool(player.declared_end),
    }


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _assets() -> Any:
    try:
        from agents.scripted.assets import load_assets

        return load_assets()
    except Exception:  # assets not shipped / not built: fall back to raw ids
        return None


@lru_cache(maxsize=1)
def _npc_names() -> dict[str, dict[int, str]]:
    """NPC enemy character/skill names (the 9xxx PvE ids are not in assets)."""
    tables: dict[str, dict[int, str]] = {"character": {}, "skill": {}}
    try:
        from npc.spec import load_enemies

        for enemy in load_enemies().values():
            tables["character"][int(enemy.engine_id)] = enemy.name
            for kind, definition_id in enemy.skill_definition_ids().items():
                skill = enemy.skills.get(kind)
                if skill is not None:
                    tables["skill"][int(definition_id)] = skill.name
            for definition_id, skill in zip(
                enemy.extra_skill_definition_ids(), enemy.extra_skills
            ):
                tables["skill"][int(definition_id)] = skill.name
    except Exception:
        pass
    return tables


def _name(definition_id: int, *, kind: str = "card") -> str:
    assets = _assets()
    if assets is not None:
        catalog = {
            "character": assets.characters,
            "skill": assets.skills,
        }.get(kind, assets.cards)
        info = catalog.get(int(definition_id))
        name = str(getattr(info, "name", "") or "")
        if name:
            return name
    npc_name = _npc_names().get(kind, {}).get(int(definition_id))
    if npc_name:
        return str(npc_name)
    return f"#{int(definition_id)}"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read_rows(raw_path: Path) -> list[dict[str, Any]]:
    """Parse a raw JSONL file, skipping blank/torn trailing lines."""
    rows: list[dict[str, Any]] = []
    if not raw_path.exists():
        return rows
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn trailing line from a hard kill
    return rows


def select_raw_games(
    rows: list[dict[str, Any]], *, min_rounds: int | None
) -> list[dict[str, Any]]:
    """Rows worth replaying: draws always, plus ``rounds >= min_rounds``."""
    selected: list[dict[str, Any]] = []
    for row in rows:
        try:
            rounds = int(row.get("rounds") or 0)
        except (TypeError, ValueError):
            rounds = 0
        if row.get("winner") is None:
            selected.append(row)
        elif min_rounds is not None and rounds >= int(min_rounds):
            selected.append(row)
    return selected


def _owner_map(steps: list[dict[str, Any]], final: Any) -> dict[int, int]:
    """entity id -> owner, accumulated from every decision view (+ final)."""
    owner: dict[int, int] = {}
    for step in steps:
        for who, player in enumerate(step["players"]):
            for character in player["characters"]:
                owner[int(character["id"])] = who
    if final is not None:
        for who, player in enumerate(final.players):
            for character in player.characters:
                owner[int(character.id)] = who
                for entity in character.entities:
                    owner[int(entity.id)] = who
    return owner


def _final_summary(final: Any) -> dict[str, Any]:
    if final is None:
        return {}
    return {
        "phase": final.phase,
        "round": int(final.round_number),
        "turn": int(final.current_turn),
        "winner": final.winner,
        "players": [
            {
                "active_character_id": player.active_character_id,
                "characters": [
                    {
                        "definition_id": int(character.definition_id),
                        "name": _name(int(character.definition_id), kind="character"),
                        "health": int(character.health),
                        "max_health": int(character.max_health),
                        "energy": int(character.energy),
                        "max_energy": int(character.max_energy),
                        "defeated": bool(character.defeated),
                        "active": bool(character.is_active),
                        "aura": int(character.aura),
                        "entities": [
                            {
                                "definition_id": int(entity.definition_id),
                                "name": _name(int(entity.definition_id)),
                                "variables": dict(entity.variables),
                            }
                            for entity in character.entities
                        ],
                    }
                    for character in player.characters
                ],
                "hand_size": len(player.hand_cards),
                "hand": [int(card.definition_id) for card in player.hand_cards],
                "pile_size": len(player.pile_cards),
                "dice": [int(die) for die in player.dice],
                "declared_end": bool(player.declared_end),
                "supports": [int(entity.definition_id) for entity in player.supports],
                "summons": [int(entity.definition_id) for entity in player.summons],
            }
            for player in final.players
        ],
    }


def _round_table(steps: list[dict[str, Any]], final: Any) -> list[dict[str, Any]]:
    """HP / damage / skill use aggregated by round."""
    by_round: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "first": None,
            "last": None,
            # ``attempted`` sums the incoming damage value (a ward/immunity still
            # reports it); ``effective`` is the real HP loss.
            "attempted": {0: 0, 1: 0},
            "effective": {0: 0, 1: 0},
            "absorbed": {0: 0, 1: 0},
            "unattributed": 0,
            "heal": {0: 0, 1: 0},
            "skills": {0: [], 1: []},
            "ends": {0: 0, 1: 0},
        }
    )
    for step in sorted(steps, key=lambda item: item["seq"]):
        bucket = by_round[int(step["round"])]
        if bucket["first"] is None and step["players"]:
            bucket["first"] = step["players"]
        if step["players"]:
            bucket["last"] = step["players"]
    owner = _owner_map(steps, final)
    for mutation in _MUTATIONS:
        bucket = by_round[int(mutation["round"])]
        if mutation["kind"] == "damage":
            target = owner.get(int(mutation["target_id"]))
            value = int(mutation["value"])
            lost = max(0, int(mutation["old_health"]) - int(mutation["new_health"]))
            if mutation.get("heal_kind"):
                if target is not None:
                    bucket["heal"][target] += value
            elif target is not None:
                bucket["attempted"][target] += value
                bucket["effective"][target] += lost
                bucket["absorbed"][target] += max(0, value - lost)
            else:
                bucket["unattributed"] += value
        elif mutation["kind"] == "skill_used":
            bucket["skills"][int(mutation["who"])].append(
                _name(int(mutation["skill_definition_id"]), kind="skill")
            )
        elif mutation["kind"] == "set_player_flag":
            if mutation.get("flag", "").endswith("DECLARED_END") and mutation.get("value"):
                bucket["ends"][int(mutation["who"])] += 1
    table: list[dict[str, Any]] = []
    for round_number in sorted(by_round):
        bucket = by_round[round_number]
        first = bucket["first"] or []
        last = bucket["last"] or []
        table.append(
            {
                "round": int(round_number),
                "skills": {who: bucket["skills"][who] for who in (0, 1)},
                "ends": bucket["ends"],
                "attempted": bucket["attempted"],
                "effective": bucket["effective"],
                "absorbed": bucket["absorbed"],
                "unattributed": bucket["unattributed"],
                "heal": bucket["heal"],
                "hp_start": {
                    who: [character["health"] for character in player["characters"]]
                    for who, player in enumerate(first)
                },
                "hp_last": {
                    who: [character["health"] for character in player["characters"]]
                    for who, player in enumerate(last)
                },
                "hand_last": {who: len(player["hand"]) for who, player in enumerate(last)},
                "pile_last": {who: len(player["pile"]) for who, player in enumerate(last)},
            }
        )
    # A round's last *decision* precedes its end phase (end-phase draws, summon
    # ticks, stepRound), so the authoritative "after round r" board is round
    # r+1's first decision view; the final round is closed by the snapshot.
    for index, entry in enumerate(table):
        next_entry = table[index + 1] if index + 1 < len(table) else None
        if next_entry is not None and next_entry["hp_start"]:
            entry["hp_end"] = next_entry["hp_start"]
            next_first = by_round[next_entry["round"]]["first"] or []
            entry["hand_end"] = {
                who: len(player["hand"]) for who, player in enumerate(next_first)
            }
            entry["pile_end"] = {
                who: len(player["pile"]) for who, player in enumerate(next_first)
            }
        elif final is not None:
            entry["hp_end"] = {
                who: [int(character.health) for character in player.characters]
                for who, player in enumerate(final.players)
            }
            entry["hand_end"] = {
                who: len(player.hand_cards) for who, player in enumerate(final.players)
            }
            entry["pile_end"] = {
                who: len(player.pile_cards) for who, player in enumerate(final.players)
            }
        else:
            entry["hp_end"] = entry["hp_last"]
    return table


def _max_streak(values: list[int]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if int(value) == 0 else 0
        best = max(best, current)
    return best


def _features(table: list[dict[str, Any]]) -> dict[str, Any]:
    """Coarse per-game signals: damage dealt by / absorbed by each player."""
    features: dict[str, Any] = {"rounds": len(table)}
    # A capped game can leave a trailing round with no decisions.
    played = [entry for entry in table if entry["skills"][0] or entry["skills"][1]] or table
    for who in (0, 1):
        dealt = [int(entry["effective"][1 - who]) for entry in played]
        taken = [int(entry["effective"][who]) for entry in table]
        absorbed_by_self = [int(entry["absorbed"][who]) for entry in table]
        features[f"profile_{who}"] = {
            "dealt_total": sum(dealt),
            "taken_total": sum(taken),
            "absorbed_by_self_total": sum(absorbed_by_self),
            "zero_damage_rounds": sum(1 for value in dealt if value == 0),
            "attack_rounds": sum(1 for entry in table if entry["skills"][who]),
            "last_two_rounds_damage_dealt": dealt[-2:],
            "max_zero_streak": _max_streak(dealt),
        }
    return features


def _mutation_lines() -> list[str]:
    """Human-readable lines for damage / skills / ends / winner (all of them)."""
    lines: list[str] = []
    for mutation in _MUTATIONS:
        kind = mutation["kind"]
        prefix = f"r{mutation['round']:>2}/t{mutation['turn']:<3}"
        if kind == "skill_used":
            lines.append(
                f"{prefix} P{mutation['who']} skill "
                f"{_name(mutation['skill_definition_id'], kind='skill')} "
                f"({mutation['skill_type']})"
            )
        elif kind == "damage":
            lines.append(
                f"{prefix} damage {mutation['value']} -> "
                f"{_name(mutation['target_definition_id'], kind='character')}"
                f"[{mutation['target_definition_id']}] "
                f"({mutation['old_health']}->{mutation['new_health']}, "
                f"reaction={mutation['reaction']}, "
                f"defeated={mutation['cause_defeated']})"
            )
        elif kind == "set_player_flag" and "DECLARED_END" in str(mutation.get("flag", "")):
            lines.append(f"{prefix} P{mutation['who']} declared_end={mutation['value']}")
        elif kind == "step_round":
            lines.append(f"{prefix} -- round ended --")
        elif kind == "set_winner":
            lines.append(f"{prefix} winner={mutation['winner']}")
        elif kind == "change_phase":
            lines.append(f"{prefix} phase -> {mutation.get('phase')}")
        elif kind == "switch_active":
            lines.append(
                f"{prefix} P{mutation.get('who')} switch active -> "
                f"{_name(mutation.get('character_definition_id', 0), kind='character')}"
            )
    return lines


def _decision_summary(steps: list[dict[str, Any]], record: Any) -> dict[str, Any]:
    counts: dict[int, Counter] = {0: Counter(), 1: Counter()}
    for step in steps:
        counts[int(step["who"])][step["request_type"]] += 1
    ends = {0: 0, 1: 0}
    for decision in getattr(record, "decisions", ()):
        if decision.chosen_kind == "action_declare_end":
            ends[int(decision.player)] += 1
    return {
        "requests": {who: dict(counter) for who, counter in counts.items()},
        "declare_end_decisions": ends,
    }


# --------------------------------------------------------------------------- #
# one replay
# --------------------------------------------------------------------------- #
def analyse(
    stage_name: str,
    seed: int,
    *,
    policy_spec: str = "heuristic",
    deck_slug: str = DEFAULT_DECK,
    rotation: str | None = None,
    stage_rules: bool = True,
    keep_mutations: bool = True,
) -> dict[str, Any]:
    """Replay one ``(stage, seed)`` and return the forensic trace."""
    stage = get_stage(stage_name)
    contestant = deck_spec(deck_slug)
    npc_deck = DeckSpec(
        name=f"npc_{stage.name}",
        characters=stage.deck_characters,
        cards=contestant.cards,
    )

    del _MUTATIONS[:]
    seq = [0]
    if policy_spec in ("contestant", "contestant_plain"):
        # Same stage-aware contestant used by npc.eval replays.
        from npc.contestant import build_contestant_policy

        policy0 = build_contestant_policy(
            contestant, stage, inject_objectives=(policy_spec == "contestant")
        )
    else:
        policy0 = build_policy(
            policy_spec, seed=derive_seed(seed, "probe_stall_p0"), role="contestant"
        )
    p0 = TracePolicy(policy0, who=0, seq=seq)
    p1 = TracePolicy(npc_policy_for_stage(stage, rotation=rotation), who=1, seq=seq)

    import envs.match as match_mod

    original = match_mod.PolicyPlayer
    match_mod.PolicyPlayer = ProbePlayer
    try:
        record = run_match(
            contestant,
            npc_deck,
            p0,
            p1,
            seed=int(seed),
            record_decisions=True,
            record_final_state=True,
            stage_rules=stage_rules_json(stage.name) if stage_rules else None,
        )
    finally:
        match_mod.PolicyPlayer = original

    final = None
    if record.final_state_json:
        from reps.snapshot import snapshot_state

        try:
            final = snapshot_state(None, record.final_state_json)
        except Exception:
            final = None
    steps = p0.steps + p1.steps
    table = _round_table(steps, final)
    summary = {
        "stage": stage.name,
        "deck": contestant.name,
        "policy": policy_spec,
        "rotation": rotation,
        "seed": int(seed),
        "winner": record.winner,
        "rounds": int(record.rounds),
        "truncated": bool(record.truncated),
        "error": record.error,
        "decisions0": int(record.decisions0),
        "decisions1": int(record.decisions1),
        "fallbacks0": int(record.fallbacks0),
        "fallbacks1": int(record.fallbacks1),
        "final": _final_summary(final),
        "rounds_table": table,
        "features": _features(table),
        "decisions": _decision_summary(steps, record),
        "event_lines": _mutation_lines(),
        "damage_events": sum(1 for m in _MUTATIONS if m["kind"] == "damage"),
        "skill_events": sum(1 for m in _MUTATIONS if m["kind"] == "skill_used"),
        "mutations": list(_MUTATIONS) if keep_mutations else [],
    }
    _print_game(summary)
    return summary


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def _fmt_hp(player: dict[str, Any]) -> str:
    return " ".join(
        f"{character['name']}={character['health']}/{character['max_health']}"
        f"{'†' if character['defeated'] else ('*' if character['active'] else '')}"
        for character in player["characters"]
    )


def _print_game(summary: dict[str, Any]) -> None:
    print("=" * 100)
    print(
        f"{summary['stage']} / {summary['deck']} / {summary['policy']} / "
        f"seed {summary['seed']} -> winner={summary['winner']} "
        f"rounds={summary['rounds']} error={summary['error']} "
        f"truncated={summary['truncated']}"
    )
    final = summary["final"]
    for who, player in enumerate(final.get("players", [])):
        print(
            f"  P{who}: {_fmt_hp(player)}\n"
            f"       hand={player['hand_size']} pile={player['pile_size']} "
            f"dice={player['dice']} declared_end={player['declared_end']} "
            f"supports={player['supports']} summons={player['summons']}"
        )
    print(f"  turn={final.get('turn')} phase={final.get('phase')}")
    print(f"  decisions={summary['decisions']}")
    print(
        "  per-round: round | effective dmg->P0/P1 (absorbed) | skills P0/P1 | "
        "ends P0/P1 | HP at end of round (P0 / P1)"
    )
    for entry in summary["rounds_table"]:
        hp_end = entry.get("hp_end", entry["hp_last"])
        hp0 = "/".join(str(value) for value in hp_end.get(0, []))
        hp1 = "/".join(str(value) for value in hp_end.get(1, []))
        print(
            f"    r{entry['round']:>2} | {entry['effective'][0]:>2}/{entry['effective'][1]:<2} "
            f"({entry['absorbed'][0]}/{entry['absorbed'][1]}) | "
            f"{len(entry['skills'][0])}/{len(entry['skills'][1])} | "
            f"{entry['ends'][0]}/{entry['ends'][1]} | [{hp0}] / [{hp1}]"
        )
    lines = summary["event_lines"]
    print(f"  mutations ({len(lines)}):")
    for line in lines:
        print(f"    {line}")


# --------------------------------------------------------------------------- #
# round-cap report
# --------------------------------------------------------------------------- #
def check_round_cap(
    *, stage_name: str = "chefmao", deck_slug: str = DEFAULT_DECK
) -> dict[str, Any]:
    """Report the engine default ``maxRoundsCount`` and whether it is configurable."""
    info: dict[str, Any] = {
        "attribute": "ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT",
        "run_match_sets_it": "ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT"
        in inspect.getsource(run_match),
    }
    try:
        import gitcg
        from gitcg import low_level

        info["gitcg_version"] = getattr(gitcg, "__version__", "?")
        info["attribute_exists"] = hasattr(low_level, "ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT")
        info["attribute_value"] = (
            int(low_level.ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT)
            if info["attribute_exists"]
            else None
        )
        stage = get_stage(stage_name)
        contestant = deck_spec(deck_slug)
        npc_deck = DeckSpec(
            name=f"npc_{stage.name}",
            characters=stage.deck_characters,
            cards=contestant.cards,
        )
        create_param = _build_create_param(
            contestant, npc_deck, seed=0, stage_rules=stage_rules_json(stage.name)
        )
        from gitcg import Game

        game = Game(create_param=create_param)
        root = json.loads(game.state().json())
        config = root["store"][0] if root.get("store") else {}
        keys = (
            "maxRoundsCount",
            "maxHandsCount",
            "maxDiceCount",
            "initialHandsCount",
        )
        info["default_config"] = {key: config.get(key) for key in keys}
    except Exception as exc:  # engine/assets missing: still report the attribute row
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay NPC stage games with a forensic trace: per-round damage/HP, "
            "skills, declare-end counts, final hands/piles/dice and all mutations."
        )
    )
    parser.add_argument(
        "--stage",
        help=f"NPC stage name (e.g. chefmao); known: {', '.join(stage_names())}",
    )
    parser.add_argument("--seed", type=int, default=1, help="match seed (default: 1)")
    parser.add_argument(
        "--policy", default=None, help="contestant policy spec (default: heuristic)"
    )
    parser.add_argument(
        "--deck",
        default=None,
        help=(
            "contestant deck slug (default: the raw row's deck when loadable, "
            f"else {DEFAULT_DECK})"
        ),
    )
    parser.add_argument("--rotation", default=None, help="NPC rotation name (default: greedy)")
    parser.add_argument(
        "--no-stage-rules",
        dest="stage_rules",
        action="store_false",
        help="do not send configs/npc/special_rules.json to the engine",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="npc eval raw JSONL; replay draws (and rounds >= --min-rounds) from it",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=None,
        help=(
            "with --raw: also replay rows with rounds >= N "
            f"(e.g. {DEFAULT_MIN_ROUNDS}); draws are always replayed"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write the trace(s) as JSONL to this path",
    )
    parser.add_argument(
        "--no-mutations",
        action="store_true",
        help="omit the full mutation list from the JSON output",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="replay at most N selected games"
    )
    parser.add_argument(
        "--check-round-cap",
        action="store_true",
        help="print the engine default maxRoundsCount / attribute id and exit",
    )
    return parser.parse_args(argv)


def _resolve_deck(candidate: str | None, *, explicit: str | None) -> str:
    """``--deck`` wins; otherwise the row's deck if it is a known slug; else default."""
    if explicit:
        return str(explicit)
    if candidate:
        try:
            deck_spec(str(candidate))
            return str(candidate)
        except KeyError:
            print(
                f"[warn] deck {candidate!r} is not an envs.decks slug; "
                f"replaying with {DEFAULT_DECK}"
            )
    return DEFAULT_DECK


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.check_round_cap:
        print(
            json.dumps(
                check_round_cap(
                    stage_name=args.stage or "chefmao",
                    deck_slug=args.deck or DEFAULT_DECK,
                ),
                indent=2,
            )
        )
        return 0

    games: list[dict[str, Any]] = []
    if args.raw is not None:
        rows = read_rows(args.raw)
        if not rows:
            raise SystemExit(f"no readable rows in {args.raw}")
        selected = select_raw_games(rows, min_rounds=args.min_rounds)
        if args.limit is not None:
            selected = selected[: max(0, int(args.limit))]
        if not selected:
            raise SystemExit(
                f"no draws (winner=None)"
                + (f" and no rows with rounds >= {args.min_rounds}" if args.min_rounds else "")
                + f" in {args.raw}"
            )
        print(
            f"replaying {len(selected)} row(s) from {args.raw} "
            f"(draws{'' if args.min_rounds is None else f' + rounds >= {args.min_rounds}'})"
        )
        for row in selected:
            games.append(
                {
                    "stage": str(row["stage"]),
                    "seed": int(row["seed"]),
                    "deck": _resolve_deck(row.get("deck"), explicit=args.deck),
                    "policy": str(args.policy or row.get("policy0") or "heuristic"),
                    "rotation": args.rotation or row.get("rotation"),
                }
            )
    elif args.stage:
        games = [
            {
                "stage": args.stage,
                "seed": int(args.seed),
                "deck": _resolve_deck(None, explicit=args.deck),
                "policy": str(args.policy or "heuristic"),
                "rotation": args.rotation,
            }
        ]
    else:
        raise SystemExit(
            "give --stage <name> (with optional --seed/--policy), --raw <jsonl>, "
            "or --check-round-cap"
        )

    rows = [
        analyse(
            game["stage"],
            game["seed"],
            policy_spec=game["policy"],
            deck_slug=game["deck"],
            rotation=game["rotation"],
            stage_rules=args.stage_rules,
            keep_mutations=not args.no_mutations,
        )
        for game in games
    ]

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(rows)} trace(s) to {args.json_out}")

    print(
        "\nCOMPACT: stage/seed rounds winner | dmg0/dmg1 last2 | "
        "hand0/pile0 hand1/pile1 | ends0/ends1"
    )
    for row in rows:
        features = row["features"]
        final = row["final"]["players"]
        ends = row["decisions"]["declare_end_decisions"]
        print(
            f"  {row['stage']}/{row['seed']} r{row['rounds']} w={row['winner']} | "
            f"{features['profile_0']['dealt_total']}/"
            f"{features['profile_1']['dealt_total']} "
            f"{features['profile_0']['last_two_rounds_damage_dealt']}/"
            f"{features['profile_1']['last_two_rounds_damage_dealt']} | "
            f"{final[0]['hand_size']}/{final[0]['pile_size']} "
            f"{final[1]['hand_size']}/{final[1]['pile_size']} | "
            f"{ends.get(0, 0)}/{ends.get(1, 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
