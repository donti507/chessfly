"""
kaggle_phase3.py -- Phase 3: let the fly connectome LEARN chess

Instead of a frozen reservoir, the network's synapse STRENGTHS are trained,
while the fly's WIRING (who connects to whom) and each neuron's excitatory /
inhibitory type stay fixed -- like learning in a real brain.

Conditions (same data, same readout, same training):
  linear       board -> move readout, no brain                   [reference]
  feedforward  board -> 434 neurons -> readout, no recurrence    [control]
  real         434-neuron rate network with the REAL fly wiring
  rewired      same neurons/weights per neuron, targets shuffled [key control]
  dense        434 neurons, all-to-all trainable (standard RNN)  [control]
Each condition runs for several seeds; rewired uses a different random
wiring per seed.

Reuses kaggle_phase2.py (dataset, network, legal-move tables), so run it in
the same folder. Needs the 100k dataset cache or regenerates it.

Usage:  python kaggle_phase3.py
Options (env): SEEDS (default 0,1,2), EPOCHS3 (15), T_STEPS (10),
               CONDITIONS (linear,feedforward,real,rewired,dense)
"""

import os
import time

import numpy as np
import pandas as pd
import chess
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

os.environ.setdefault("N_POSITIONS", "100000")
import kaggle_phase2 as p2
from reservoir_core import board_to_features

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("EPOCHS3", "15"))
T_STEPS = int(os.environ.get("T_STEPS", "10"))
CONDITIONS = os.environ.get("CONDITIONS", "linear,feedforward,real,rewired,dense").split(",")
ALPHA = 0.5          # leak: h <- (1-a) h + a (input)
BATCH = 512
LR = 1e-3
RHO = 0.9            # initial spectral radius of the recurrent weights
OUT = p2.OUT_DIR
log = p2.log


class ChessBrain(nn.Module):
    """Rate network: h <- (1-a) h + a (relu(h) @ W + U x), T steps, then readout."""

    def __init__(self, kind, n=434, d_in=768, mask=None, signs=None, init_mag=None):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.readout = nn.Linear(d_in, 128)
            return
        self.inp = nn.Linear(d_in, n)
        self.readout = nn.Linear(n, 128)
        if kind == "feedforward":
            return
        if kind == "dense":
            self.W = nn.Parameter(torch.randn(n, n) * (RHO / np.sqrt(n)))
            return
        # connectome-constrained: fixed mask + fixed signs (Dale's law), learned magnitudes
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.register_buffer("sign", torch.tensor(signs, dtype=torch.float32)[:, None])
        self.log_mag = nn.Parameter(torch.log(torch.tensor(init_mag, dtype=torch.float32) + 1e-6))

    def weights(self):
        if self.kind == "dense":
            return self.W
        return self.sign * torch.exp(self.log_mag) * self.mask

    def effective_params(self):
        n = sum(p.numel() for p in self.parameters())
        if self.kind in ("real", "rewired"):
            n = n - self.log_mag.numel() + int(self.mask.sum().item())
        return n

    def forward(self, x):
        if self.kind == "linear":
            return self.readout(x)
        u = self.inp(x)
        if self.kind == "feedforward":
            return self.readout(torch.relu(u))
        W = self.weights()
        h = torch.zeros_like(u)
        for _ in range(T_STEPS):
            h = (1 - ALPHA) * h + ALPHA * (torch.relu(h) @ W + u)
        return self.readout(torch.relu(h))


def build_model(kind, seed, C, signs):
    if kind not in ("real", "rewired"):
        return ChessBrain(kind)
    Cc = C if kind == "real" else p2.rewire(C, seed=100 + seed)
    W0 = Cc * signs[:, None]
    rho = float(np.max(np.abs(np.linalg.eigvals(W0))))
    return ChessBrain(kind, mask=(Cc > 0).astype(np.float32), signs=signs,
                      init_mag=Cc * (RHO / rho))


def run(kind, seed, X, tabs, splits, device, C, signs):
    LF, LT, MASK, TG = tabs
    tr, va, te = splits
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(kind, seed, C, signs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def scores(idx):
        out = model(X[idx])
        s = out[:, :64].gather(1, LF[idx]) + out[:, 64:].gather(1, LT[idx])
        return s.masked_fill(~MASK[idx], -1e9)

    @torch.no_grad()
    def accuracy(idx_np, k):
        model.eval()
        correct = 0
        for s in range(0, len(idx_np), 4096):
            idx = torch.from_numpy(idx_np[s:s + 4096]).to(device)
            top = scores(idx).topk(k, dim=1).indices
            correct += int((top == TG[idx][:, None]).any(dim=1).sum())
        model.train()
        return correct / len(idx_np)

    tr_t = torch.from_numpy(tr).to(device)
    best_va, best_state = -1.0, None
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            loss = Fnn.cross_entropy(scores(b), TG[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        va_acc = accuracy(va, 1)
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    te1, te3 = accuracy(te, 1), accuracy(te, 3)
    if kind == "real":
        torch.save(best_state, os.path.join(OUT, f"phase3_real_seed{seed}.pt"))
    log(f"  {kind:<12} seed {seed}  val {best_va:.3f}  test {te1:.3f}  top-3 {te3:.3f}  "
        f"params {model.effective_params():>7}  [{time.time() - t0:5.0f}s]")
    return {"condition": kind, "seed": seed, "val_top1": best_va, "test_top1": te1,
            "test_top3": te3, "params": model.effective_params()}


def main():
    log("=" * 78)
    log(f"CHESSFLY PHASE 3 -- trainable connectome   seeds={SEEDS}  epochs={EPOCHS}  "
        f"T={T_STEPS}")
    log("=" * 78)
    sf = p2.ensure_stockfish()
    p2.ensure_network_files()
    fens, moves, cps, gids = p2.build_dataset(sf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    adj, meta, signs = p2.load_network()
    C = np.zeros(adj.shape, dtype=np.float32)
    C[adj.row, adj.col] = adj.data
    log(f"Network: {adj.shape[0]} neurons, {adj.nnz} connections, "
        f"{int((signs < 0).sum())} inhibitory")

    log("Preparing features and legal-move tables ...")
    raw = np.stack([board_to_features(chess.Board(str(f))) for f in fens]).astype(np.float32)
    LF, LT, MASK, TG, valid = p2.legal_tables(fens, moves)
    tr, va, te = p2.split_by_game(gids, valid)
    X = torch.from_numpy(raw).to(device)
    tabs = tuple(torch.from_numpy(a).to(device) for a in (LF, LT, MASK, TG))
    n_legal = MASK[te].sum(axis=1)
    log(f"Split by game: train {len(tr)} / val {len(va)} / test {len(te)}")
    log(f"Random legal move: test top-1 {np.mean(1.0 / n_legal):.3f}")

    rows = []
    for kind in CONDITIONS:
        log(f"Training {kind} ...")
        for seed in SEEDS:
            rows.append(run(kind.strip(), seed, X, tabs, (tr, va, te), device, C, signs))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase3_results.csv"), index=False)
    ci = 1.96 * np.sqrt(0.2 * 0.8 / len(te))

    log("\n" + "=" * 78)
    log(f"PHASE 3 RESULTS  (mean +/- std over {len(SEEDS)} seeds; "
        f"single-run test CI ~ +/-{ci:.3f})")
    log("=" * 78)
    log(f"{'condition':<13}{'params':>9}{'val top-1':>16}{'test top-1':>16}{'test top-3':>16}")
    for kind in CONDITIONS:
        g = df[df["condition"] == kind.strip()]
        if g.empty:
            continue
        f = lambda c: f"{g[c].mean():.3f}+/-{g[c].std(ddof=0):.3f}"
        log(f"{kind:<13}{int(g['params'].iloc[0]):>9}{f('val_top1'):>16}"
            f"{f('test_top1'):>16}{f('test_top3'):>16}")

    if {"real", "rewired"} <= set(df["condition"]):
        r = df[df["condition"] == "real"]
        w = df[df["condition"] == "rewired"]
        log("\nKey comparison: does the REAL fly wiring learn chess better than random wiring?")
        log(f"  mean test diff (real - rewired): {r['test_top1'].mean() - w['test_top1'].mean():+.3f}")
        log(f"  mean val  diff (real - rewired): {r['val_top1'].mean() - w['val_top1'].mean():+.3f}")
        beats = r["test_top1"].min() > w["test_top1"].max()
        log(f"  worst real seed beats best rewired seed on test: {'YES' if beats else 'no'}")
    log(f"\nSaved {os.path.join(OUT, 'phase3_results.csv')} and phase3_real_seed*.pt")


if __name__ == "__main__":
    main()
