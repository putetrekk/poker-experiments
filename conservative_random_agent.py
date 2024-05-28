"""
One custom agent is included in this module:
    - :func:`conservative_random_agent`

"""

from random import random, choice
from typing import Tuple

from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.game.player_state import PlayerState


def conservative_random_agent(game: TexasHoldEm) -> Tuple[ActionType, int]:
    """
    This agent is a conservative random agent.
    """
    moves = game.get_available_moves()
    if ActionType.CHECK in moves.action_types:
        moves.action_types.remove(ActionType.FOLD)

    (chosen_action, raise_amount) = moves.sample()

    # Think twice if considering raise
    if chosen_action == ActionType.RAISE:
        (chosen_action, raise_amount) = moves.sample()

    # If determined to raise, raise conservatively
    if chosen_action == ActionType.RAISE:
        range = moves.raise_range
        min_raise = min(range)
        max_raise = max(range)
        med_raise = min_raise*2 if min_raise*2 < max_raise else max_raise
        raise_options = [min_raise, min_raise, min_raise, med_raise, max_raise]
        raise_amount = choice(raise_options)

    return chosen_action, raise_amount
