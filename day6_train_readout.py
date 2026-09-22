"""
day6_train_readout.py -- retrain readouts on the bigger dataset, + hybrid

Same readout and method as Day 4, plus one new representation:
    fly_plus_input -- the 48 encoder inputs + 386 fly reservoir neurons

The key comparison is fly_plus_input vs encoder_input_48: the hybrid sees
everything the encoder control sees, PLUS the fly brain. If it scores higher,
the connectome is genuinely adding information.

Overwrites readout_*.npz (so Day 5/6 game bots use the new readouts) and
saves readout_fly_plus_input.npz.
"""

import os
import sys

import numpy as np

from day4_train_readout import (
    build_tables, train, topk_accuracy, L2_GRID, DATASET, DATA_DIR, SPLIT_SEED,
)


def main():
    if not os.path.exists(DATASET):
        sys.exit("day4_dataset.npz not found.")
    d = np.load(DATASET)
    fens, moves, games = d["fens"], d["moves"], d["game_ids"]
    reservoir = d["reservoir"].astype(np.float64)
    inputs = d["inputs"].astype(np.float64)
    reps = {
        "raw_board_768": d["raw"].astype(np.float64),
        "encoder_input_48": inputs,
        "fly_reservoir": reservoir,
        "fly_plus_input": np.hstack([inputs, reservoir]),   # order matters for play
    }

    ugames = np.unique(games)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(ugames)
    n_g = len(ugames)
    g_tr = ugames[: int(0.70 * n_g)]
    g_va = ugames[int(0.70 * n_g): int(0.85 * n_g)]
    g_te = ugames[int(0.85 * n_g):]

    tab_tr, keep_tr = build_tables(fens, moves, np.where(np.isin(games, g_tr))[0])
    tab_va, keep_va = build_tables(fens, moves, np.where(np.isin(games, g_va))[0])
    tab_te, keep_te = build_tables(fens, moves, np.where(np.isin(games, g_te))[0])

    rand_top1 = float(np.mean(1.0 / tab_te["n_legal"]))
    rand_top3 = float(np.mean(np.minimum(3, tab_te["n_legal"]) / tab_te["n_legal"]))

    print("=" * 72)
    print(f"DAY 6 -- READOUT TRAINING   ({len(fens)} positions, {n_g} games)")
    print(f"train {tab_tr['n']} / val {tab_va['n']} / test {tab_te['n']} positions")
    print("=" * 72)

    rows = [("random legal move", "-", float("nan"), rand_top1, rand_top3)]
    for name, X in reps.items():
        mu = X[keep_tr].mean(axis=0)
        sd = X[keep_tr].std(axis=0) + 1e-6
        Z = (X - mu) / sd
        best = None
        for l2 in L2_GRID:
            P = train(Z[keep_tr], tab_tr, l2)
            va = topk_accuracy(Z[keep_va], P, tab_va, 1)
            print(f"  {name:<18} l2={l2:<7g} val top-1 = {va:.3f}")
            if best is None or va > best[0]:
                best = (va, l2, P)
        va, l2, P = best
        te1 = topk_accuracy(Z[keep_te], P, tab_te, 1)
        te3 = topk_accuracy(Z[keep_te], P, tab_te, 3)
        rows.append((name, f"{l2:g}", va, te1, te3))
        np.savez(os.path.join(DATA_DIR, f"readout_{name}.npz"), mu=mu, sd=sd, l2=l2, **P)

    print("\n" + "=" * 72)
    print("How often the readout picks Stockfish's move")
    print("=" * 72)
    print(f"{'representation':<20}{'l2':>8}{'val top-1':>11}{'test top-1':>12}{'test top-3':>12}")
    for name, l2, va, t1, t3 in rows:
        va_s = "-" if np.isnan(va) else f"{va:.3f}"
        print(f"{name:<20}{l2:>8}{va_s:>11}{t1:>12.3f}{t3:>12.3f}")
    print("\nKey test: fly_plus_input vs encoder_input_48 on BOTH val and test.")


if __name__ == "__main__":
    main()
