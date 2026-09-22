"""
day6_play_games.py -- re-test in games with the Day 6 readouts

Bots:
    encoder  -- 48 encoder inputs only                 [control]
    fly      -- fly reservoir only
    hybrid   -- encoder inputs + fly reservoir
    (raw, random also available)

40 games per bot by default, and prints a rough 95% margin on the score so
you can tell a real result from luck.

FLY_REPEATS=k averages k independent brain runs per move (less noise, k x slower).

Usage:
    python3 day6_play_games.py
    FLY_REPEATS=2 BOTS=hybrid python3 day6_play_games.py
    OPPONENT=greedy python3 day6_play_games.py

Runtime: each fly/hybrid game takes about 1 min per FLY_REPEATS,
so the default (fly + hybrid, 40 games each) is roughly 80 min.
"""

import os
import time

import numpy as np
import chess

from reservoir_core import FlyReservoir
from day5_play_games import (
    ReadoutBot, play_game, make_opponent, make_bot as day5_make_bot,
    MAX_PLIES, ADJ_MARGIN, SEED,
)

N_GAMES = int(os.environ.get("N_GAMES", "40"))
OPPONENT = os.environ.get("OPPONENT", "random")
BOTS = os.environ.get("BOTS", "encoder,fly,hybrid").split(",")
FLY_REPEATS = int(os.environ.get("FLY_REPEATS", "1"))


def make_bot(name, cache):
    if name not in ("fly", "hybrid"):
        return day5_make_bot(name, cache)
    if "reservoir" not in cache:
        print("Building fly reservoir ...")
        cache["reservoir"] = FlyReservoir()
    res = cache["reservoir"]

    def features(cb, ply):
        outs = [res.run(cb, sim_seed=ply * 10 + k) for k in range(FLY_REPEATS)]
        hidden = np.mean([o["hidden_rates"] for o in outs], axis=0)
        if name == "fly":
            return hidden
        return np.concatenate([outs[0]["activation"], hidden])   # same order as training

    readout = "readout_fly_reservoir.npz" if name == "fly" else "readout_fly_plus_input.npz"
    return ReadoutBot(name, readout, features)


def main():
    rng = np.random.default_rng(SEED)
    cache = {}
    opponent = make_opponent(OPPONENT)
    summary = []

    print("=" * 78)
    print(f"DAY 6 -- GAMES vs {OPPONENT}  ({N_GAMES} games/bot, cap {MAX_PLIES} plies, "
          f"adjudicate +/-{ADJ_MARGIN}, fly repeats {FLY_REPEATS})")
    print("=" * 78)

    for name in BOTS:
        bot = make_bot(name.strip(), cache)
        counts = {"win": 0, "draw": 0, "loss": 0}
        mates = {"win": 0, "loss": 0}
        diffs, illegal = [], [0]
        t0 = time.time()
        for g in range(N_GAMES):
            color = chess.WHITE if g % 2 == 0 else chess.BLACK
            result, how, diff = play_game(bot, opponent, color, rng, illegal)
            counts[result] += 1
            if how == "mate":
                mates[result] += 1
            diffs.append(diff)
            print(f"  {bot.name:<8} game {g + 1:>2}/{N_GAMES} "
                  f"({'W' if color == chess.WHITE else 'B'})  {result:<4} ({how})  "
                  f"material {diff:+d}   [{time.time() - t0:5.0f}s]")

        outcomes = np.array([1.0] * counts["win"] + [0.5] * counts["draw"] + [0.0] * counts["loss"])
        score = float(outcomes.mean())
        margin = 1.96 * float(outcomes.std(ddof=1)) / np.sqrt(N_GAMES)
        summary.append((bot.name, counts, mates, float(np.mean(diffs)), score, margin, illegal[0]))

    print("\n" + "=" * 78)
    print(f"RESULTS vs {OPPONENT}   (score = (wins + 0.5*draws) / games)")
    print("=" * 78)
    print(f"{'bot':<10}{'W':>4}{'D':>4}{'L':>4}{'mates W/L':>12}"
          f"{'avg material':>15}{'score':>8}{'+/- 95%':>10}{'illegal':>9}")
    for name, c, m, d, s, mg, ill in summary:
        print(f"{name:<10}{c['win']:>4}{c['draw']:>4}{c['loss']:>4}"
              f"{str(m['win']) + '/' + str(m['loss']):>12}{d:>+15.1f}{s:>8.2f}"
              f"{mg:>10.2f}{ill:>9}")
    print("\nA bot clearly beats random if (score - margin) > 0.50.")


if __name__ == "__main__":
    main()
