"""
day6_build_vs_random.py -- add training positions from games vs a random opponent

Day 5 showed the fly bot plays much worse than its Day 4 accuracy suggested.
Likely cause: it trained on Stockfish-vs-Stockfish positions but played in
chaotic positions created by a random opponent. This adds positions of exactly
that kind: a "hero" side (Stockfish teacher, 25% random moves) vs a purely
random opponent. Only positions where the hero is to move are recorded.

Appends to day4_dataset.npz (resumable, same format).

Usage:
    python3 day6_build_vs_random.py            # grow dataset to 5000 total
    N_TOTAL=6000 python3 day6_build_vs_random.py
"""

import os
import time

import numpy as np
import chess
import chess.engine

from reservoir_core import FlyReservoir, canonical, mirror_move, board_to_features
from day4_build_dataset import (
    find_stockfish, position_key, load_existing, save,
    OUT_PATH, TEACHER_DEPTH, MAX_PLIES, CHECKPOINT_EVERY,
)

N_TOTAL = int(os.environ.get("N_TOTAL", "5000"))
HERO_RANDOM_PROB = 0.25


def main():
    engine_path = find_stockfish()
    print("Building fly reservoir ...")
    reservoir = FlyReservoir()

    data = load_existing()
    seen = {position_key(f) for f in data["fens"]}
    game_id = (max(data["game_ids"]) + 1) if data["game_ids"] else 0
    rng = np.random.default_rng(5000 + game_id)

    if len(data["fens"]) >= N_TOTAL:
        print(f"Already have {len(data['fens'])} >= {N_TOTAL} positions. Nothing to do.")
        return

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    t0 = time.time()
    added = 0
    print(f"Growing dataset to {N_TOTAL} positions (hero vs random opponent). "
          f"Ctrl+C to stop safely.\n")

    try:
        while len(data["fens"]) < N_TOTAL:
            hero = chess.WHITE if game_id % 2 == 0 else chess.BLACK
            board = chess.Board()
            for _ in range(MAX_PLIES):
                if board.is_game_over():
                    break
                legal = list(board.legal_moves)

                if board.turn != hero:                      # random opponent
                    board.push(legal[rng.integers(len(legal))])
                    continue

                teacher = engine.play(board, chess.engine.Limit(depth=TEACHER_DEPTH)).move
                cboard = canonical(board)
                fen = cboard.fen()
                key = position_key(fen)
                if key not in seen:
                    cmove = teacher if board.turn == chess.WHITE else mirror_move(teacher)
                    res = reservoir.run(cboard, sim_seed=len(data["fens"]))
                    data["fens"].append(fen)
                    data["moves"].append(cmove.uci())
                    data["game_ids"].append(game_id)
                    data["reservoir"].append(res["hidden_rates"])
                    data["inputs"].append(res["activation"])
                    data["raw"].append(board_to_features(cboard).astype(np.uint8))
                    seen.add(key)
                    added += 1
                    if added % CHECKPOINT_EVERY == 0:
                        save(data)
                        rate = (time.time() - t0) / added
                        left = (N_TOTAL - len(data["fens"])) * rate
                        print(f"  {len(data['fens']):>5}/{N_TOTAL} positions (game {game_id})"
                              f"  {rate:.2f}s/pos  eta {left / 60:5.1f} min")
                    if len(data["fens"]) >= N_TOTAL:
                        break

                move = legal[rng.integers(len(legal))] if rng.random() < HERO_RANDOM_PROB else teacher
                board.push(move)
            game_id += 1
    except KeyboardInterrupt:
        print("\nStopped by user -- saving progress ...")
    finally:
        engine.quit()
        save(data)
        print(f"Saved {len(data['fens'])} positions from {len(set(data['game_ids']))} games "
              f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
