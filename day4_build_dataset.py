"""
day4_build_dataset.py -- positions + teacher moves + fly-reservoir states

For each position:
    - teacher move from Stockfish (fixed depth)
    - fly reservoir hidden-neuron firing rates (the thing we'll read out from)
    - the 48 encoder inputs and the raw 768 board features (for control baselines)

Games are Stockfish self-play with some random moves mixed in, so positions
are realistic but varied. Positions are stored from the side-to-move's view.

Resumable: Ctrl+C any time; rerunning continues where it left off.
~1.25 s per position with the locked 1000 ms simulation -> 3000 positions ~1 hour.

Usage:
    sudo apt install stockfish          # once
    N_POSITIONS=20 python3 day4_build_dataset.py   # quick test
    python3 day4_build_dataset.py                  # full run (default 3000)
"""

import os
import sys
import time
import shutil

import numpy as np
import chess
import chess.engine

from reservoir_core import (
    FlyReservoir, canonical, mirror_move, board_to_features, DATA_DIR,
)

N_POSITIONS = int(os.environ.get("N_POSITIONS", "3000"))
OUT_PATH = os.path.join(DATA_DIR, "day4_dataset.npz")
TMP_PATH = os.path.join(DATA_DIR, "day4_dataset_tmp.npz")
TEACHER_DEPTH = 8
RANDOM_MOVE_PROB = 0.25
MAX_PLIES = 100
CHECKPOINT_EVERY = 50


def find_stockfish():
    for p in [shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish"]:
        if p and os.path.exists(p):
            return p
    sys.exit("Stockfish not found. Install it with:  sudo apt install stockfish")


def position_key(fen):
    return " ".join(fen.split()[:4])      # ignore move counters for dedupe


def load_existing():
    data = {"fens": [], "moves": [], "game_ids": [], "reservoir": [], "inputs": [], "raw": []}
    if os.path.exists(OUT_PATH):
        d = np.load(OUT_PATH)
        data["fens"] = list(d["fens"])
        data["moves"] = list(d["moves"])
        data["game_ids"] = list(d["game_ids"])
        data["reservoir"] = list(d["reservoir"])
        data["inputs"] = list(d["inputs"])
        data["raw"] = list(d["raw"])
        print(f"Resuming: {len(data['fens'])} positions already in {os.path.basename(OUT_PATH)}")
    return data


def save(data):
    if not data["fens"]:
        return
    np.savez(TMP_PATH,
             fens=np.array(data["fens"]),
             moves=np.array(data["moves"]),
             game_ids=np.array(data["game_ids"], dtype=np.int32),
             reservoir=np.stack(data["reservoir"]).astype(np.float32),
             inputs=np.stack(data["inputs"]).astype(np.float32),
             raw=np.stack(data["raw"]).astype(np.uint8))
    os.replace(TMP_PATH, OUT_PATH)


def main():
    engine_path = find_stockfish()
    print("Building fly reservoir ...")
    reservoir = FlyReservoir()
    print(f"  {reservoir.n_neurons} neurons, {reservoir.n_hidden} hidden (read out)")

    data = load_existing()
    seen = {position_key(f) for f in data["fens"]}
    game_id = (max(data["game_ids"]) + 1) if data["game_ids"] else 0
    rng = np.random.default_rng(1000 + game_id)

    if len(data["fens"]) >= N_POSITIONS:
        print(f"Already have {len(data['fens'])} >= {N_POSITIONS} positions. Nothing to do.")
        return

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    t0 = time.time()
    added = 0
    print(f"Target: {N_POSITIONS} positions. Ctrl+C to stop safely (progress is saved).\n")

    try:
        while len(data["fens"]) < N_POSITIONS:
            board = chess.Board()
            for _ in range(MAX_PLIES):
                if board.is_game_over():
                    break
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
                        left = (N_POSITIONS - len(data["fens"])) * rate
                        print(f"  {len(data['fens']):>5}/{N_POSITIONS} positions "
                              f"(game {game_id})  {rate:.2f}s/pos  eta {left / 60:5.1f} min")
                    if len(data["fens"]) >= N_POSITIONS:
                        break

                if rng.random() < RANDOM_MOVE_PROB:
                    legal = list(board.legal_moves)
                    move = legal[rng.integers(len(legal))]
                else:
                    move = teacher
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
