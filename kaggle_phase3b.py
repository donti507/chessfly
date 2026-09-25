"""
kaggle_phase3b.py -- Does real fly wiring help a network LEARN when information
must flow THROUGH it?

Design (fixes the Phase 3 shortcut):
  * The board enters ONLY N_SENSORY sensory neurons; moves are read ONLY from the
    remaining (output) neurons. Every bit of chess information must travel
    through the fly's synapses.
  * Wiring (mask) and excitatory/inhibitory signs are fixed; synapse strengths,
    input and readout are trained (rate network, T_STEPS steps).
  * Learning curves over training-set sizes (data efficiency), several seeds.

Conditions:
  real        exact fly wiring
  rewired     each neuron keeps its outgoing weights, targets shuffled
  degshuffle  degree-preserving edge swaps: every neuron keeps in- AND out-degree
  no_wiring   no synapses -> output neurons receive nothing (floor / sanity check)
  linear      board -> readout directly (reference)

Decision rule (set in advance): real "wins" at a data size only if its WORST
seed beats the control's BEST seed on test.

Usage:  python kaggle_phase3b.py
Env:    SEEDS (0,1,2)  SIZES (1000,3000,10000,30000,69000)  STEPS (3000)
        N_SENSORY (150)  T_STEPS (10)  CONDITIONS
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
from reservoir_core import board_to_features, select_input_neurons

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
SIZES = [int(s) for s in os.environ.get("SIZES", "1000,3000,10000,30000,69000").split(",")]
CONDITIONS = os.environ.get("CONDITIONS", "linear,no_wiring,real,rewired,degshuffle").split(",")
N_SENSORY = int(os.environ.get("N_SENSORY", "150"))
STEPS = int(os.environ.get("STEPS", "3000"))
T_STEPS = int(os.environ.get("T_STEPS", "10"))
EVAL_EVERY, BATCH, LR, ALPHA, RHO = 250, 512, 1e-3, 0.5, 0.9
OUT = p2.OUT_DIR
log = p2.log


def degree_preserving_shuffle(C, seed, swaps_per_edge=10):
    """Directed double-edge swaps (a->b, c->d) => (a->d, c->b). Keeps every neuron's
    in-degree and out-degree; each edge keeps its source's weight (and sign)."""
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(C)
    w = C[pre, post].copy()
    post = post.copy()
    E = len(pre)
    edges = set(zip(pre.tolist(), post.tolist()))
    n_try = swaps_per_edge * E
    I, J = rng.integers(E, size=n_try), rng.integers(E, size=n_try)
    for i, j in zip(I.tolist(), J.tolist()):
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == c or b == d or (a, d) in edges or (c, b) in edges:
            continue
        edges.discard((a, b)); edges.discard((c, d))
        edges.add((a, d)); edges.add((c, b))
        post[i], post[j] = d, b
    out = np.zeros_like(C)
    out[pre, post] = w
    return out


class FlowBrain(nn.Module):
    def __init__(self, kind, n, sensory, output, C=None, signs=None, d_in=768):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.readout = nn.Linear(d_in, 128)
            return
        P = torch.zeros(len(sensory), n)
        P[torch.arange(len(sensory)), torch.as_tensor(sensory)] = 1.0
        self.register_buffer("P", P)
        self.register_buffer("outp", torch.as_tensor(output, dtype=torch.long))
        self.inp = nn.Linear(d_in, len(sensory))
        self.readout = nn.Linear(len(output), 128)
        self.has_W = C is not None
        if self.has_W:
            W0 = C * signs[:, None]
            rho = float(np.max(np.abs(np.linalg.eigvals(W0))))
            self.register_buffer("mask", torch.tensor((C > 0).astype(np.float32)))
            self.register_buffer("sign", torch.tensor(signs, dtype=torch.float32)[:, None])
            self.log_mag = nn.Parameter(torch.log(torch.tensor(C * (RHO / rho),
                                                               dtype=torch.float32) + 1e-6))

    def effective_params(self):
        n = sum(p.numel() for p in self.parameters())
        if self.kind != "linear" and self.has_W:
            n = n - self.log_mag.numel() + int(self.mask.sum().item())
        return n

    def forward(self, x):
        if self.kind == "linear":
            return self.readout(x)
        u = self.inp(x) @ self.P
        h = torch.zeros_like(u)
        W = self.sign * torch.exp(self.log_mag) * self.mask if self.has_W else None
        for _ in range(T_STEPS):
            rec = torch.relu(h) @ W if W is not None else 0.0
            h = (1 - ALPHA) * h + ALPHA * (rec + u)
        return self.readout(torch.relu(h[:, self.outp]))


def main():
    log("=" * 78)
    log(f"CHESSFLY PHASE 3b -- information must flow through the wiring")
    log(f"seeds={SEEDS} sizes={SIZES} steps={STEPS} sensory={N_SENSORY} T={T_STEPS}")
    log("=" * 78)
    sf = p2.ensure_stockfish()
    p2.ensure_network_files()
    fens, moves, cps, gids = p2.build_dataset(sf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adj, meta, signs = p2.load_network()
    n = adj.shape[0]
    C = np.zeros(adj.shape, dtype=np.float32)
    C[adj.row, adj.col] = adj.data
    sensory = np.array(select_input_neurons(adj, meta, N_SENSORY))
    output = np.setdiff1d(np.arange(n), sensory)
    log(f"Network: {n} neurons, {adj.nnz} connections | sensory {len(sensory)}, output {len(output)}")

    log("Building null networks ...")
    nets = {("real", s): C for s in SEEDS}
    for s in SEEDS:
        nets[("rewired", s)] = p2.rewire(C, seed=100 + s)
        nets[("degshuffle", s)] = degree_preserving_shuffle(C, seed=200 + s)
    for kind in ["real", "rewired", "degshuffle"]:
        M = nets[(kind, SEEDS[0])]
        one_hop = float((M[np.ix_(sensory, output)] > 0).any(axis=0).mean())
        log(f"  {kind:<11} edges {int((M > 0).sum()):>6}  output neurons 1 hop from sensory: "
            f"{100 * one_hop:5.1f}%")

    log("Preparing features and legal-move tables ...")
    raw = np.stack([board_to_features(chess.Board(str(f))) for f in fens]).astype(np.float32)
    LF, LT, MASK, TG, valid = p2.legal_tables(fens, moves)
    tr, va, te = p2.split_by_game(gids, valid)
    X = torch.from_numpy(raw).to(device)
    LF, LT, MASK, TG = (torch.from_numpy(a).to(device) for a in (LF, LT, MASK, TG))
    va_sub = np.random.default_rng(0).permutation(va)[:5000]
    log(f"train {len(tr)} / val {len(va)} / test {len(te)} | random legal move test "
        f"top-1 {np.mean(1.0 / MASK[te].sum(dim=1).cpu().numpy()):.3f}")

    def run(kind, seed, size):
        torch.manual_seed(seed)
        tr_sub = np.random.default_rng(seed).choice(tr, size=min(size, len(tr)), replace=False)
        Cn = nets.get((kind, seed)) if kind in ("real", "rewired", "degshuffle") else None
        model = FlowBrain(kind, n, sensory, output, C=Cn, signs=signs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        def scores(idx):
            out = model(X[idx])
            s = out[:, :64].gather(1, LF[idx]) + out[:, 64:].gather(1, LT[idx])
            return s.masked_fill(~MASK[idx], -1e9)

        @torch.no_grad()
        def acc(idx_np, k):
            model.eval()
            c = 0
            for s in range(0, len(idx_np), 4096):
                idx = torch.from_numpy(idx_np[s:s + 4096]).to(device)
                c += int((scores(idx).topk(k, dim=1).indices == TG[idx][:, None]).any(1).sum())
            model.train()
            return c / len(idx_np)

        tr_t = torch.from_numpy(tr_sub).to(device)
        best, state = -1.0, None
        for step in range(1, STEPS + 1):
            b = tr_t[torch.randint(len(tr_t), (BATCH,), device=device)]
            loss = Fnn.cross_entropy(scores(b), TG[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % EVAL_EVERY == 0:
                v = acc(va_sub, 1)
                if v > best:
                    best, state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
        model.load_state_dict(state)
        return {"condition": kind, "seed": seed, "size": len(tr_sub), "val_top1": acc(va, 1),
                "test_top1": acc(te, 1), "test_top3": acc(te, 3),
                "params": model.effective_params()}

    rows = []
    t0 = time.time()
    total = len(CONDITIONS) * len(SIZES) * len(SEEDS)
    for kind in [c.strip() for c in CONDITIONS]:
        for size in SIZES:
            for seed in SEEDS:
                r = run(kind, seed, size)
                rows.append(r)
                el = time.time() - t0
                log(f"  [{len(rows):>3}/{total}] {kind:<11} n={r['size']:>6} seed {seed}  "
                    f"test {r['test_top1']:.3f}  top-3 {r['test_top3']:.3f}  "
                    f"[{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase3b_results.csv"), index=False)

    log("\n" + "=" * 78)
    log("LEARNING CURVES -- test top-1, mean +/- std over seeds")
    log("=" * 78)
    head = f"{'condition':<12}" + "".join(f"{('n=' + str(min(s, len(tr)))):>15}" for s in SIZES)
    log(head)
    for kind in [c.strip() for c in CONDITIONS]:
        line = f"{kind:<12}"
        for size in SIZES:
            g = df[(df.condition == kind) & (df["size"] == min(size, len(tr)))]["test_top1"]
            line += f"{g.mean():>9.3f}+/-{g.std(ddof=0):.3f}"
        log(line)

    for ctrl in ["rewired", "degshuffle"]:
        if not {"real", ctrl} <= set(df.condition):
            continue
        log(f"\nREAL vs {ctrl.upper()}  (decision rule: worst real seed > best {ctrl} seed)")
        wins = 0
        for size in SIZES:
            sz = min(size, len(tr))
            r = df[(df.condition == "real") & (df["size"] == sz)]["test_top1"]
            c = df[(df.condition == ctrl) & (df["size"] == sz)]["test_top1"]
            win = r.min() > c.max()
            lose = r.max() < c.min()
            wins += int(win)
            verdict = "REAL WINS" if win else ("real loses" if lose else "no clear difference")
            log(f"  n={sz:>6}: diff {r.mean() - c.mean():+.4f}   {verdict}")
        log(f"  -> real wins at {wins}/{len(SIZES)} data sizes")
    log(f"\nSaved {os.path.join(OUT, 'phase3b_results.csv')}")


if __name__ == "__main__":
    main()
