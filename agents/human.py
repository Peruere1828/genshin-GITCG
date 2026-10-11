"""Human policy: drive a match from the terminal (play against the AI).

A ``Policy`` that renders the current decision (a compact state line + the legal
options) and reads the chosen option index from stdin. It plugs into
``envs.match.run_match`` like any other policy, so a human can play either side
against a scripted / neural / search opponent (see ``scripts/play_vs_ai.py``).

Observation is the same masked ``player_view`` the encoder uses (no hidden
information), so a human sees exactly what the AI is allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_PHASE = {
    "init_hands": "起手换牌",
    "init_actives": "选择出战",
    "roll": "掷骰",
    "action": "行动",
    "end": "结束",
    "game_end": "对局结束",
}


def _option_label(spec: Any, index: int) -> str:
    label = getattr(spec, "label", None)
    if label:
        return str(label)
    return f"选项 {index}"


@dataclass
class HumanPolicy:
    """Reads the human's move for ``player`` from stdin."""

    player: int = 1
    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], None] = print
    name: str = "human"
    _games: int = field(default=0, init=False, repr=False)

    def choose(self, built: Any) -> int:
        codes = [int(c) for c in built.context.legal_low_level_codes]
        if not codes:
            return -1
        specs = tuple(getattr(built.context, "legal_low_level_specs", ()) or ())
        self._render_state(built, specs, len(codes))
        prompt = f"  你的选择 [0..{len(codes) - 1}]（q 退出）: "
        while True:
            raw = str(self.input_fn(prompt)).strip()
            if raw in {"q", "quit", "exit"}:
                raise KeyboardInterrupt("human quit")
            if raw.isdigit() and 0 <= int(raw) < len(codes):
                return codes[int(raw)]
            self.output_fn(f"  非法输入 {raw!r}，请输入 0..{len(codes) - 1}")

    # -- rendering ----------------------------------------------------------- #
    def _render_state(self, built: Any, specs: tuple, n_options: int) -> None:
        ctx = built.context
        view = getattr(ctx, "player_view", None)
        phase = _PHASE.get(str(getattr(view, "phase", "")), str(getattr(view, "phase", "?")))
        req = getattr(ctx.request_type, "value", str(ctx.request_type))
        lines = [f"\n— 第 {getattr(view, 'round_number', '?')} 回合 / {phase} / 请求: {req}"]
        if view is not None:
            me = view.players[self.player]
            active = next((c for c in me.characters if c.is_active), None)
            if active is not None:
                lines.append(
                    f"  出战: id{active.definition_id} HP {active.health}/{active.max_health} "
                    f"能量 {active.energy}/{active.max_energy}"
                )
            lines.append(f"  手牌 {len(me.hand_cards)} 张 · 骰子 {len(me.dice)} 个 · 牌堆 {len(me.pile_cards)}")
        for index in range(n_options):
            spec = specs[index] if index < len(specs) else None
            lines.append(f"  [{index}] {_option_label(spec, index)}")
        self.output_fn("\n".join(lines))
