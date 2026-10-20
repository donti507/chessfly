"""
day4_train_readout.py -- train the move readout (legal moves only)

Readout: for each position, score every LEGAL move as
    score(move) = from_logit[from_square] + to_logit[to_square]
where from_logits / to_logits are linear functions of the representation.
Softmax over legal moves only, trained to match Stockfish's move.
Illegal moves are impossible by construction.

Three representations, same readout, same data:
    raw_board_768     -- the board itself (no fly brain)         [control]
    encoder_input_48  -- only the 48 numbers fed INTO the brain  [control]
    fly_reservoir     -- fly connectome hidden-neuron activity

If fly_reservoir beats encoder_input_48, the connectome is adding something.

Train/val/test split is BY GAME (no leakage between near-identical positions).
Saves readout_<name>.npz for each representation.
"""

import os
import sys

import numpy as np
import chess

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(DATA_DIR, "day4_dataset.npz")
L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
EPOCHS = 400
LR = 0.01
SPLIT_SEED = 0


def usable_moves(board):
    # drop underpromotions: they share from/to squares with the queen promotion
    return [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]


def build_tables(fens, moves, idx):
    pos, frm, to, target, offsets, n_legal, keep = [], [], [], [], [], [], []
    flat = row = 0
    for i in idx:
        board = chess.Board(str(fens[i]))
        t = chess.Move.from_uci(str(moves[i]))
        legal = usable_moves(board)
        tgt = next((k for k, m in enumerate(legal)
                    if m.from_square == t.from_square and m.to_square == t.to_square), None)
        if tgt is None:
            continue
        offsets.append(flat)
        target.append(flat + tgt)
        for m in legal:
            pos.append(row)
            frm.append(m.from_square)
            to.append(m.to_square)
        flat += len(legal)
        row += 1
        keep.append(i)
        n_legal.append(len(legal))
    tab = {"pos": np.array(pos), "frm": np.array(frm), "to": np.array(to),
           "target": np.array(target), "offsets": np.array(offsets),
           "n": row, "n_legal": np.array(n_legal)}
    return tab, np.array(keep)


def move_scores(X, P, tab):
    F = X @ P["Wf"] + P["bf"]
    T = X @ P["Wt"] + P["bt"]
    return F, T, F[tab["pos"], tab["frm"]] + T[tab["pos"], tab["to"]]


def loss_and_grads(X, P, tab, l2):
    F, T, s = move_scores(X, P, tab)
    off, pos, n = tab["offsets"], tab["pos"], tab["n"]
    m = np.maximum.reduceat(s, off)
    e = np.exp(s - m[pos])
    Z = np.add.reduceat(e, off)
    loss = -np.mean(s[tab["target"]] - m - np.log(Z))
    loss += 0.5 * l2 * (np.sum(P["Wf"] ** 2) + np.sum(P["Wt"] ** 2))

    g = e / Z[pos]
    g[tab["target"]] -= 1.0
    g /= n
    dF = np.zeros_like(F)
    dT = np.zeros_like(T)
    np.add.at(dF, (pos, tab["frm"]), g)
    np.add.at(dT, (pos, tab["to"]), g)
    grads = {"Wf": X.T @ dF + l2 * P["Wf"], "bf": dF.sum(axis=0),
             "Wt": X.T @ dT + l2 * P["Wt"], "bt": dT.sum(axis=0)}
    return loss, grads


def train(X, tab, l2):
    D = X.shape[1]
    P = {"Wf": np.zeros((D, 64)), "bf": np.zeros(64),
         "Wt": np.zeros((D, 64)), "bt": np.zeros(64)}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, EPOCHS + 1):
        _, g = loss_and_grads(X, P, tab, l2)
        for k in P:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1 ** step)
            vh = v[k] / (1 - b2 ** step)
            P[k] -= LR * mh / (np.sqrt(vh) + eps)
    return P


def topk_accuracy(X, P, tab, k):
    _, _, s = move_scores(X, P, tab)
    bounds = np.append(tab["offsets"], len(s))
    correct = 0
    for r in range(tab["n"]):
        seg = s[bounds[r]:bounds[r + 1]]
        t = tab["target"][r] - bounds[r]
        correct += int((seg > seg[t]).sum() < k)
    return correct / tab["n"]


def main():
    if not os.path.exists(DATASET):
        sys.exit("day4_dataset.npz not found -- run day4_build_dataset.py first.")
    d = np.load(DATASET)
    fens, moves, games = d["fens"], d["moves"], d["game_ids"]
    reps = {
        "raw_board_768": d["raw"].astype(np.float64),
        "encoder_input_48": d["inputs"].astype(np.float64),
        "fly_reservoir": d["reservoir"].astype(np.float64),
    }

    ugames = np.unique(games)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(ugames)
    n_g = len(ugames)
    g_tr = ugames[: int(0.70 * n_g)]
    g_va = ugames[int(0.70 * n_g): int(0.85 * n_g)]
    g_te = ugames[int(0.85 * n_g):]
    if min(len(g_tr), len(g_va), len(g_te)) == 0:
        sys.exit(f"Only {n_g} games in the dataset -- build more positions first.")

    tab_tr, keep_tr = build_tables(fens, moves, np.where(np.isin(games, g_tr))[0])
    tab_va, keep_va = build_tables(fens, moves, np.where(np.isin(games, g_va))[0])
    tab_te, keep_te = build_tables(fens, moves, np.where(np.isin(games, g_te))[0])

    rand_top1 = float(np.mean(1.0 / tab_te["n_legal"]))
    rand_top3 = float(np.mean(np.minimum(3, tab_te["n_legal"]) / tab_te["n_legal"]))

    print("=" * 72)
    print(f"DAY 4 -- READOUT TRAINING   ({len(fens)} positions, {n_g} games)")
    print(f"train {tab_tr['n']} / val {tab_va['n']} / test {tab_te['n']} positions "
          f"(split by game)")
    print(f"avg legal moves per test position: {tab_te['n_legal'].mean():.1f}")
    print("=" * 72)

    rows = [("random legal move", "-", rand_top1, rand_top3)]
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
        _, l2, P = best

        te1 = topk_accuracy(Z[keep_te], P, tab_te, 1)
        te3 = topk_accuracy(Z[keep_te], P, tab_te, 3)
        rows.append((name, f"{l2:g}", te1, te3))
        np.savez(os.path.join(DATA_DIR, f"readout_{name}.npz"),
                 mu=mu, sd=sd, l2=l2, **P)

    print("\n" + "=" * 72)
    print("TEST SET -- how often the readout picks Stockfish's move")
    print("=" * 72)
    print(f"{'representation':<20}{'l2':>8}{'top-1':>10}{'top-3':>10}")
    for name, l2, t1, t3 in rows:
        print(f"{name:<20}{l2:>8}{t1:>10.3f}{t3:>10.3f}")
    print("\nSaved readout_raw_board_768.npz, readout_encoder_input_48.npz, "
          "readout_fly_reservoir.npz")


if __name__ == "__main__":
    main()
