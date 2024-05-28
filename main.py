from matplotlib import pyplot as plt
import numpy as np
from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem.agents.basic import random_agent, call_agent
from conservative_random_agent import conservative_random_agent
from multiprocessing import Pool, Process
import seaborn as sns
from time import time

SECONDS_PER_TURN = 20
MINUTES_PER_BLIND = 15
GAMES_TO_SIMULATE = 10
EXPLORATION_FACTOR = 2

blinds = [
    (50, 100),
    (100, 200),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
]


def simulate_game(
    id: int, minutes_per_blind=MINUTES_PER_BLIND, seconds_per_turn=SECONDS_PER_TURN
):
    blind_index = 0
    game_time_seconds = 0
    (small_blind, big_blind) = blinds[blind_index]
    player_count_history = {0: 8}
    game = TexasHoldEm(
        buyin=5000, big_blind=small_blind, small_blind=big_blind, max_players=8
    )
    while game.is_game_running():
        if (
            blind_index < len(blinds) - 1
            and game_time_seconds > (blind_index + 1) * minutes_per_blind * 60
        ):
            blind_index = blind_index + 1
            (small_blind, big_blind) = blinds[blind_index]
            game.small_blind, game.big_blind = small_blind, big_blind
        game.start_hand()
        while game.is_hand_running():
            game_time_seconds = game_time_seconds + seconds_per_turn
            game.take_action(*conservative_random_agent(game))
        player_count_history[game_time_seconds / 60] = sum(
            [1 for player in game.players if player.chips > 0]
        )
    return game_time_seconds, player_count_history


def run_sim_and_parameters(*parameters):
    return simulate_game(*parameters), parameters


def experiment_blind_duration(scatter_axes, seconds_per_turn=SECONDS_PER_TURN):
    blind_durations = list(
        np.arange(
            MINUTES_PER_BLIND / EXPLORATION_FACTOR,
            MINUTES_PER_BLIND * EXPLORATION_FACTOR,
            0.01,
        )
    )
    ids = list(range(len(blind_durations)))
    turn_durations = [seconds_per_turn for _ in blind_durations]

    with Pool(processes=8) as pool:
        game_durations = {}
        for (game_duration, player_counts), (
            id,
            blind_duration,
            turn_duration,
        ) in pool.starmap(
            run_sim_and_parameters, zip(ids, blind_durations, turn_durations)
        ):
            game_durations[blind_duration] = game_duration / 60

    ax = sns.regplot(
        ax=scatter_axes,
        x=list(game_durations.keys()),
        y=list(game_durations.values()),
        order=2,
        scatter_kws={"s": 2},
    )
    ax.set(
        xlabel="Blind duration (minutes)",
        ylabel="Game duration (minutes)",
        ylim=(0, 180),
        title=f"blind duration correlation @ {seconds_per_turn}s turn duration",
    )


def experiment_turn_duration(scatter_axes, minutes_per_blind=MINUTES_PER_BLIND):
    turn_durations = list(
        np.arange(
            SECONDS_PER_TURN / EXPLORATION_FACTOR,
            SECONDS_PER_TURN * EXPLORATION_FACTOR,
            0.01,
        )
    )
    ids = list(range(len(turn_durations)))
    blind_durations = [minutes_per_blind for _ in ids]

    with Pool(processes=8) as pool:
        game_durations = {}
        for (game_duration, player_count_history), (
            id,
            blind_duration,
            turn_duration,
        ) in pool.starmap(
            run_sim_and_parameters, zip(ids, blind_durations, turn_durations)
        ):
            game_durations[turn_duration] = game_duration / 60

    ax = sns.regplot(
        ax=scatter_axes,
        x=list(game_durations.keys()),
        y=list(game_durations.values()),
        order=2,
        scatter_kws={"s": 2},
    )
    ax.set(
        xlabel="Turn duration (seconds)",
        ylabel="Game duration (minutes)",
        ylim=(0, 180),
        title=f"turn duration correlation @ {minutes_per_blind}m blind duration",
    )


def experiment_player_decrease(axes):
    ids = list(range(GAMES_TO_SIMULATE))
    blind_durations = [MINUTES_PER_BLIND for _ in ids]
    turn_durations = [SECONDS_PER_TURN for _ in ids]

    with Pool(processes=8) as pool:
        player_count_series = []
        for (game_duration, player_count_history), _ in pool.starmap(
            run_sim_and_parameters, zip(ids, blind_durations, turn_durations)
        ):
            player_count_series.append(player_count_history)

    for series in player_count_series:
        ax = sns.lineplot(ax=axes, data=series, drawstyle="steps-pre", alpha=0.6)
        ax.set(
            xlabel="Game duration (minutes)",
            ylabel="Player count",
            title=f"player count over {GAMES_TO_SIMULATE} games @ {MINUTES_PER_BLIND}m blind & {SECONDS_PER_TURN}s turn duration",
        )
    axes.set_xlim(0, 180)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    experiment_blind_duration(axes[0, 0], seconds_per_turn=20)
    experiment_turn_duration(axes[0, 1], minutes_per_blind=10)
    experiment_blind_duration(axes[1, 0], seconds_per_turn=40)
    experiment_turn_duration(axes[1, 1], minutes_per_blind=20)
    plt.tight_layout()
    plt.savefig("experiment.png")

    # single plotexperiment.png
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    experiment_player_decrease(axes)
    plt.tight_layout()
    plt.savefig("experiment_single.png")
