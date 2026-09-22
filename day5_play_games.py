"""
day5_play_games.py -- play full games: readout bots vs a random-move opponent

Bots (each uses the SAME trained move readout type from Day 4):
    fly      -- fly connectome reservoir activity (runs the spiking sim every move)
    encoder  -- only the 48 inputs fed into the brain            [control]
    raw      -- raw 768 board features                           [control]
    random   -- random legal moves                               [sanity check]

Games longer than MAX_PLIES are adjudicated by material: >= ADJ_MARGIN
points ahead counts as a win. Bots alternate White/Black.

Also counts illegal move attempts (should be 0 -- success criterion 1).

Usage:
    python3 day5_play_games.py                       # all bots vs random, 20 games each
    OPPONENT=greedy python3 day5_play_games.py       # bonus: vs greedy material bot
    BOTS=fly N_GAMES=10 python3 day5_play_games.py   # just the fly bot, 10 games

Runtime: the fly bot runs a 1 s simulation per move (~0.75 s wall), so
20 fly games take roughly 15-20 min. The other bots are near-instant.
"""

import os
import time

import numpy as np
import chess

from reservoir_core import (
    FlyReservoir, Encoder, canonical, mirror_move, board_to_features, DATA_DIR,
)

N_GAMES = int(os.environ.get("N_GAMES", "20"))
OPPONENT = os.environ.get("OPPONENT", "random")
BOTS = os.environ.get("BOTS", "random,raw,encoder,fly").split(",")
MAX_PLIES = 150
ADJ_MARGIN = 3
SEED = 0

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board, color):
    return sum(len(board.pieces(pt, color)) * v for pt, v in PIECE_VALUES.items())


def usable_moves(board):
    return [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]


# ---------------------------------------------------------------- bots
class RandomBot:
    name = "random"

    def choose(self, board, rng, ply):
        legal = list(board.legal_moves)
        return legal[rng.integers(len(legal))]


class GreedyBot:
    """Takes the biggest immediate material gain; random among ties."""
    name = "greedy"

    def choose(self, board, rng, ply):
        best, best_gain = [], -1
        for m in board.legal_moves:
            gain = 0
            if board.is_en_passant(m):
                gain = 1
            elif board.piece_at(m.to_square):
                gain = PIECE_VALUES[board.piece_at(m.to_square).piece_type]
            if m.promotion:
                gain += PIECE_VALUES[m.promotion] - 1
            if gain > best_gain:
                best, best_gain = [m], gain
            elif gain == best_gain:
                best.append(m)
        return best[rng.integers(len(best))]


class ReadoutBot:
    def __init__(self, name, readout_file, feature_fn):
        self.name = name
        r = np.load(os.path.join(DATA_DIR, readout_file))
        self.mu, self.sd = r["mu"], r["sd"]
        self.Wf, self.bf, self.Wt, self.bt = r["Wf"], r["bf"], r["Wt"], r["bt"]
        self.feature_fn = feature_fn

    def choose(self, board, rng, ply):
        cb = canonical(board)
        x = (self.feature_fn(cb, ply) - self.mu) / self.sd
        F = x @ self.Wf + self.bf
        T = x @ self.Wt + self.bt
        legal = usable_moves(cb)
        scores = [F[m.from_square] + T[m.to_square] for m in legal]
        best = legal[int(np.argmax(scores))]
        return best if board.turn == chess.WHITE else mirror_move(best)


def make_bot(name, cache):
    if name == "random":
        return RandomBot()
    if name == "raw":
        return ReadoutBot("raw", "readout_raw_board_768.npz",
                          lambda cb, ply: board_to_features(cb))
    if name == "encoder":
        enc = cache.setdefault("encoder", Encoder())
        return ReadoutBot("encoder", "readout_encoder_input_48.npz",
                          lambda cb, ply: enc.activation(cb))
    if name == "fly":
        if "reservoir" not in cache:
            print("Building fly reservoir ...")
            cache["reservoir"] = FlyReservoir()
        res = cache["reservoir"]
        return ReadoutBot("fly", "readout_fly_reservoir.npz",
                          lambda cb, ply: res.run(cb, sim_seed=ply)["hidden_rates"])
    raise ValueError(f"unknown bot {name}")


def make_opponent(name):
    return GreedyBot() if name == "greedy" else RandomBot()


# ---------------------------------------------------------------- games
def play_game(bot, opponent, bot_color, rng, illegal):
    board = chess.Board()
    ply = 0
    while not board.is_game_over() and ply < MAX_PLIES:
        mover = bot if board.turn == bot_color else opponent
        move = mover.choose(board, rng, ply)
        if move not in board.legal_moves:
            illegal[0] += 1
            legal = list(board.legal_moves)
            move = legal[rng.integers(len(legal))]
        board.push(move)
        ply += 1

    diff = material(board, bot_color) - material(board, not bot_color)
    if board.is_checkmate():
        return ("win" if board.turn != bot_color else "loss"), "mate", diff
    if board.is_game_over():
        return "draw", "rule", diff
    if diff >= ADJ_MARGIN:
        return "win", "adj", diff
    if diff <= -ADJ_MARGIN:
        return "loss", "adj", diff
    return "draw", "adj", diff


def main():
    rng = np.random.default_rng(SEED)
    cache = {}
    opponent = make_opponent(OPPONENT)
    summary = []

    print("=" * 74)
    print(f"DAY 5 -- GAMES vs {OPPONENT} opponent  ({N_GAMES} games per bot, "
          f"cap {MAX_PLIES} plies, adjudicate at +/-{ADJ_MARGIN})")
    print("=" * 74)

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
        score = (counts["win"] + 0.5 * counts["draw"]) / N_GAMES
        summary.append((bot.name, counts, mates, float(np.mean(diffs)), score, illegal[0]))

    print("\n" + "=" * 74)
    print(f"RESULTS vs {OPPONENT}   (score = (wins + 0.5*draws) / games)")
    print("=" * 74)
    print(f"{'bot':<10}{'W':>4}{'D':>4}{'L':>4}{'mates W/L':>12}"
          f"{'avg material':>15}{'score':>8}{'illegal':>9}")
    for name, c, m, d, s, ill in summary:
        print(f"{name:<10}{c['win']:>4}{c['draw']:>4}{c['loss']:>4}"
              f"{str(m['win']) + '/' + str(m['loss']):>12}{d:>+15.1f}{s:>8.2f}{ill:>9}")
    print("\nscore 0.50 = equal to opponent. Want fly clearly above 0.50 vs random.")


if __name__ == "__main__":
    main()
