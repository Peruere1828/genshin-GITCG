"""Chronological record of BOTH players' decisions in one match.

The observation encoder reserves up to ``max_history_tokens`` (32) tokens for a
short action history. This module is the producer of that channel: ``run_match``
appends every resolved decision to a single shared ``MatchHistory`` *before* the
next ``policy.choose`` call, and policies that declare
``uses_match_history = True`` receive its snapshot through
``attach_history`` (training collection and inference use the same object and
therefore the same semantics).

Information constraint (hard requirement, AGENTS.md §0): a recorded step is built
from ``context.player_view`` -- the *masked* view the acting player is allowed to
observe -- never from ``context.full_state`` (the god view). Opponent hand
identities, deck order and dice faces therefore can never enter the history
channel. ``full_state`` is used only as a fallback when a caller built a context
without any masked view at all; production contexts always carry one.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .action_hierarchy import try_low_level_spec_for_code
from .schema import ActionChoice, DecisionContext, LowLevelActionSpec, TrajectoryStep


class MatchHistory:
    """Chronological (oldest first) record of both players' decisions."""

    def __init__(self, max_steps: int = 512) -> None:
        self.max_steps = max(1, int(max_steps))
        self._steps: deque[TrajectoryStep] = deque(maxlen=self.max_steps)

    def record(self, built: Any, code: int) -> None:
        """Append one decision (``built`` is a ``BuiltDecisionContext``-like object)."""
        context: DecisionContext = built.context
        view = context.player_view if context.player_view is not None else context.full_state
        self._steps.append(
            TrajectoryStep(
                acting_player=int(context.acting_player),
                request_type=context.request_type,
                choice=ActionChoice(action_code=int(code)),
                pre_state=view,
                post_state=view,
                reward=0.0,
                done=False,
                legal_low_level_codes=tuple(
                    int(value) for value in context.legal_low_level_codes
                ),
                chosen_low_level_spec=_resolve_spec(context, int(code)),
                player_view=context.player_view,
            )
        )

    def snapshot(self) -> tuple[TrajectoryStep, ...]:
        """All retained steps, oldest first (a plain immutable tuple)."""
        return tuple(self._steps)

    def __len__(self) -> int:
        return len(self._steps)


def _resolve_spec(context: DecisionContext, code: int) -> LowLevelActionSpec | None:
    """Resolve the low-level spec whose ``action_code`` is ``code``, if any."""
    if code < 0:
        return None
    for spec in context.legal_low_level_specs:
        if getattr(spec, "action_code", None) == code:
            return spec
    return try_low_level_spec_for_code(code)


def attach_history(policy: Any, history: MatchHistory | None) -> bool:
    """Give ``history`` to ``policy`` iff it opts in via ``uses_match_history``."""
    if not getattr(policy, "uses_match_history", False):
        return False
    policy.match_history = history
    return True
