"""Play a match against the AI from the terminal, and tally a win rate.

Pick the AI's deck and policy and your own deck, then play as the opponent: the
AI answers its turns automatically and you choose your moves from a numbered
menu. ``--games N`` plays N games (fresh seeds) and reports the AI's win rate
with a Wilson 95% CI. Typing ``q`` (or Ctrl-C) quits the current game: it is
recorded as an invalid game (not a loss) and the run ends without scoring it.

This is the interactive counterpart to the automated scoreboards
(``scripts/eval_protocol.py`` / ``eval.arena``).

Usage::

    python -m scripts.play_vs_ai --list-decks
    python -m scripts.play_vs_ai --ai-deck superconduct_aggro --my-deck unyielding_geo
    python -m scripts.play_vs_ai --ai neural:data/checkpoints/x.pt --my-deck natlan_battleship --games 3
"""

from __future__ import annotations

import argparse

from agents.human import HumanPolicy, HumanQuit
from common.seeding import derive_seed
from envs.decks import deck_spec
from envs.match import run_match
from envs.policy import build_policy
from eval.opponents import scripted_opponents
from eval.stats import estimate_rate


def _display_names() -> dict[str, str]:
    return {opp.deck: opp.display for opp in scripted_opponents()}


def _list_decks() -> None:
    names = _display_names()
    print("可用卡组（--ai-deck / --my-deck）:")
    for name, display in names.items():
        print(f"  {name:<32} {display}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-decks", action="store_true", help="list registered decks")
    parser.add_argument("--ai-deck", default="superconduct_aggro", help="AI side deck slug")
    parser.add_argument("--my-deck", default="unyielding_geo", help="your deck slug")
    parser.add_argument("--ai", default=None,
                        help="AI policy spec (default expert:<ai-deck>; e.g. neural:<ckpt>, heuristic)")
    parser.add_argument("--human-player", type=int, default=1, choices=(0, 1),
                        help="which side you play (default 1; AI takes the other)")
    parser.add_argument("--seed", type=int, default=0, help="base seed (game i uses seed+i)")
    parser.add_argument("--games", type=int, default=1, help="number of games to play")
    args = parser.parse_args(argv)

    if args.list_decks:
        _list_decks()
        return 0

    ai_player = 1 - args.human_player
    ai_deck = deck_spec(args.ai_deck)
    my_deck = deck_spec(args.my_deck)
    ai_spec = args.ai or f"expert:{args.ai_deck}"

    names = _display_names()
    print(f"AI: {args.ai_deck}（{names.get(args.ai_deck, '')}） 策略={ai_spec}  座位 P{ai_player}")
    print(f"你: {args.my_deck}（{names.get(args.my_deck, '')}）  座位 P{args.human_player}")

    outcomes: list[float] = []
    for game in range(args.games):
        seed = args.seed + game
        ai_policy = build_policy(ai_spec, seed=derive_seed(seed, "ai"), role=f"p{ai_player}")
        human = HumanPolicy(player=args.human_player)
        print(f"\n===== 第 {game + 1}/{args.games} 局（seed={seed}）=====")
        policies = {ai_player: ai_policy, args.human_player: human}
        try:
            record = run_match(
                ai_deck if ai_player == 0 else my_deck,
                ai_deck if ai_player == 1 else my_deck,
                policies[0],
                policies[1],
                seed=seed,
            )
        except HumanQuit:  # 策略层退出信号（正常路径由 run_match 记进 record.error）
            print("\n已退出。")
            break
        except KeyboardInterrupt:
            print("\n已退出。")
            break
        if record.error or record.truncated:
            # 中途退出 / 引擎错误 / 超步数：都不是有效对局，不该进胜率
            print(f"  结果: 无效对局（error={record.error}, truncated={record.truncated}）")
            if record.error and record.error.startswith("human quit"):
                print("已退出（本局不计入胜率）。")
                break
            continue
        winner = record.winner
        if winner is None:
            print(f"  结果: 平/未分（truncated={record.truncated}）")
            score = 0.5
        else:
            won = winner == ai_player
            print(f"  结果: {'AI 胜' if won else '你 胜'}（winner=P{winner}, 回合 {record.rounds}）")
            score = 1.0 if won else 0.0
        outcomes.append(score)

    if outcomes:
        est = estimate_rate(sum(outcomes), len(outcomes))
        print(f"\nAI 胜率: {est.rate:.3f}  (n={len(outcomes)}, "
              f"Wilson 95% CI [{est.low:.3f}, {est.high:.3f}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
