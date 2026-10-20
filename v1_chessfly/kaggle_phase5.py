"""
kaggle_phase5.py -- Phase 5: navigation, the task the compass circuit evolved for

Angular-velocity integration: a simulated fly turns randomly. For the first
CUE_STEPS steps it sees a landmark (its true heading); afterwards it only gets
turning signals and must keep track of its heading by integrating them.

Same design as Phase 3b: input enters only 150 sensory neurons, heading is read
only from the other 284, wiring mask and E/I signs fixed, synapse strengths trained.

Conditions: no_wiring (floor), real, rewired, degshuffle, dense (unconstrained RNN).
Metrics: mean heading error (degrees, lower = better) on test runs of the training
length, and on runs TWICE as long (long-horizon generalization).
Decision rule (set in advance): real "wins" at a data size only if its WORST seed
has lower error than the control's BEST seed.

Prediction for a double dissociation: real wiring wins here (it lost at chess).

Usage:  python kaggle_phase5.py     (needs the network files from pull_subnetwork_v2.py)
Env:    SEEDS (0,1,2)  SIZES (100,300,1000,3000)  STEPS5 (2000)  T_TRAIN (40)  T_TEST (80)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import kaggle_phase2 as p2
import kaggle_phase3b as p3b
from reservoir_core import select_input_neurons

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
SIZES = [int(s) for s in os.environ.get("SIZES", "100,300,1000,3000").split(",")]
CONDITIONS = os.environ.get("CONDITIONS", "no_wiring,real,rewired,degshuffle,dense").split(",")
T_TRAIN = int(os.environ.get("T_TRAIN", "40"))
T_TEST = int(os.environ.get("T_TEST", "80"))
STEPS5 = int(os.environ.get("STEPS5", "2000"))
CUE_STEPS, MICRO, BATCH, LR, ALPHA, RHO = 5, 2, 128, 1e-3, 0.5, 0.9
OMEGA_MAX, N_VAL, N_TEST, EVAL_EVERY = 0.35, 500, 2000, 250
OUT = p2.OUT_DIR
log = p2.log


def make_runs(n, T, seed):
    """Random turning runs. x: (n,T,4) = [turn rate, cue cos, cue sin, cue on]; y: (n,T,2)."""
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-np.pi, np.pi, n)
    omega = np.zeros((n, T))
    w = rng.normal(0, 0.1, n)
    for t in range(T):
        w = 0.85 * w + rng.normal(0, 0.08, n)
        omega[:, t] = np.clip(w, -OMEGA_MAX, OMEGA_MAX)
    theta = theta0[:, None] + np.cumsum(omega, axis=1)
    x = np.zeros((n, T, 4), dtype=np.float32)
    x[:, :, 0] = omega / OMEGA_MAX
    x[:, :CUE_STEPS, 1] = np.cos(theta[:, :CUE_STEPS])
    x[:, :CUE_STEPS, 2] = np.sin(theta[:, :CUE_STEPS])
    x[:, :CUE_STEPS, 3] = 1.0
    y = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    return x, y


class NavBrain(nn.Module):
    def __init__(self, kind, n, sensory, output, C=None, signs=None):
        super().__init__()
        self.kind, self.n = kind, n
        P = torch.zeros(len(sensory), n)
        P[torch.arange(len(sensory)), torch.as_tensor(sensory)] = 1.0
        self.register_buffer("P", P)
        self.register_buffer("outp", torch.as_tensor(output, dtype=torch.long))
        self.inp = nn.Linear(4, len(sensory))
        self.readout = nn.Linear(len(output), 2)
        if kind in ("real", "rewired", "degshuffle"):
            W0 = C * signs[:, None]
            rho = float(np.max(np.abs(np.linalg.eigvals(W0))))
            self.register_buffer("mask", torch.tensor((C > 0).astype(np.float32)))
            self.register_buffer("sign", torch.tensor(signs, dtype=torch.float32)[:, None])
            self.log_mag = nn.Parameter(torch.log(torch.tensor(C * (RHO / rho), dtype=torch.float32) + 1e-6))
        elif kind == "dense":
            self.W = nn.Parameter(torch.randn(n, n) * (RHO / np.sqrt(n)))

    def weights(self):
        if self.kind in ("real", "rewired", "degshuffle"):
            return self.sign * torch.exp(self.log_mag) * self.mask
        return self.W if self.kind == "dense" else None

    def effective_params(self):
        k = sum(p.numel() for p in self.parameters())
        if self.kind in ("real", "rewired", "degshuffle"):
            k = k - self.log_mag.numel() + int(self.mask.sum().item())
        return k

    def forward(self, x):
        B, T, _ = x.shape
        W = self.weights()
        U = self.inp(x) @ self.P
        h = torch.zeros(B, self.n, device=x.device)
        outs = []
        for t in range(T):
            for _ in range(MICRO):
                rec = torch.relu(h) @ W if W is not None else 0.0
                h = (1 - ALPHA) * h + ALPHA * (rec + U[:, t])
            outs.append(self.readout(torch.relu(h[:, self.outp])))
        return torch.stack(outs, dim=1)


def heading_error(pred, y):
    """Mean absolute angle error in degrees, after the cue period."""
    a = torch.atan2(pred[..., 1], pred[..., 0]) - torch.atan2(y[..., 1], y[..., 0])
    a = torch.remainder(a + np.pi, 2 * np.pi) - np.pi
    return float(torch.rad2deg(a[:, CUE_STEPS:].abs()).mean())


def main():
    log("=" * 78)
    log("CHESSFLY PHASE 5 -- navigation (angular-velocity integration)")
    log(f"seeds={SEEDS} sizes={SIZES} steps={STEPS5} T_train={T_TRAIN} T_test={T_TEST}")
    log("=" * 78)
    p2.ensure_network_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj, meta, signs = p2.load_network()
    n = adj.shape[0]
    C = np.zeros(adj.shape, dtype=np.float32)
    C[adj.row, adj.col] = adj.data
    sensory = np.array(select_input_neurons(adj, meta, p3b.N_SENSORY))
    output = np.setdiff1d(np.arange(n), sensory)
    log(f"Network: {n} neurons, {adj.nnz} connections | sensory {len(sensory)}, output {len(output)} | {device}")

    nets = {}
    for s in SEEDS:
        nets[("real", s)] = C
        nets[("rewired", s)] = p2.rewire(C, seed=100 + s)
        nets[("degshuffle", s)] = p3b.degree_preserving_shuffle(C, seed=200 + s)

    to = lambda a: torch.from_numpy(a).to(device)
    xv, yv = map(to, make_runs(N_VAL, T_TRAIN, seed=999))
    xt, yt = map(to, make_runs(N_TEST, T_TRAIN, seed=1001))
    xl, yl = map(to, make_runs(N_TEST, T_TEST, seed=1002))

    @torch.no_grad()
    def evaluate(model, x, y):
        model.eval()
        preds = torch.cat([model(x[i:i + 500]) for i in range(0, len(x), 500)])
        model.train()
        return heading_error(preds, y)

    def run(kind, seed, size):
        torch.manual_seed(seed)
        xtr, ytr = map(to, make_runs(size, T_TRAIN, seed=10_000 + 100 * seed + size))
        model = NavBrain(kind, n, sensory, output, C=nets.get((kind, seed)), signs=signs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best, state = 1e9, None
        for step in range(1, STEPS5 + 1):
            b = torch.randint(size, (min(BATCH, size),), device=device)
            loss = ((model(xtr[b]) - ytr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % EVAL_EVERY == 0:
                e = evaluate(model, xv, yv)
                if e < best:
                    best, state = e, {k: t.detach().clone() for k, t in model.state_dict().items()}
        model.load_state_dict(state)
        return {"condition": kind, "seed": seed, "size": size, "val_err": best,
                "test_err": evaluate(model, xt, yt), "long_err": evaluate(model, xl, yl),
                "params": model.effective_params()}

    rows, t0 = [], time.time()
    conds = [c.strip() for c in CONDITIONS]
    total = len(conds) * len(SIZES) * len(SEEDS)
    for kind in conds:
        for size in SIZES:
            for seed in SEEDS:
                r = run(kind, seed, size)
                rows.append(r)
                el = time.time() - t0
                log(f"  [{len(rows):>2}/{total}] {kind:<11} runs={size:>5} seed {seed}  "
                    f"error {r['test_err']:5.1f} deg   2x longer {r['long_err']:5.1f} deg   "
                    f"[{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase5_navigation.csv"), index=False)

    for metric, name in [("test_err", f"HEADING ERROR, runs of {T_TRAIN} steps"),
                         ("long_err", f"HEADING ERROR, runs of {T_TEST} steps (2x longer than training)")]:
        log("\n" + "=" * 78)
        log(f"{name} -- degrees, mean +/- std (lower = better; chance = 90)")
        log("=" * 78)
        log(f"{'condition':<12}" + "".join(f"{('runs=' + str(s)):>15}" for s in SIZES))
        for kind in conds:
            line = f"{kind:<12}"
            for size in SIZES:
                g = df[(df.condition == kind) & (df["size"] == size)][metric]
                line += f"{g.mean():>9.1f}+/-{g.std(ddof=0):4.1f}"
            log(line)

    wins_total = {}
    for ctrl in ["rewired", "degshuffle"]:
        if not {"real", ctrl} <= set(df.condition):
            continue
        log(f"\nREAL vs {ctrl.upper()}  (test error; rule: worst real seed < best {ctrl} seed)")
        wins = 0
        for size in SIZES:
            r = df[(df.condition == "real") & (df["size"] == size)]["test_err"]
            c = df[(df.condition == ctrl) & (df["size"] == size)]["test_err"]
            win, lose = r.max() < c.min(), r.min() > c.max()
            wins += int(win)
            log(f"  runs={size:>5}: real {r.mean():5.1f} vs {c.mean():5.1f} deg   "
                f"{'REAL WINS' if win else ('real loses' if lose else 'no clear difference')}")
        wins_total[ctrl] = wins
        log(f"  -> real wins at {wins}/{len(SIZES)} data sizes")

    if wins_total:
        log("\nDOUBLE DISSOCIATION CHECK")
        log("  Chess (Phase 3b): real wiring LOST to both controls at 4/5 data sizes.")
        best = min(wins_total.values())
        verdict = ("SUPPORTED: real wiring wins at navigation and loses at chess"
                   if best >= len(SIZES) // 2 + 1 else
                   "NOT SUPPORTED: real wiring does not clearly win at navigation either")
        log(f"  Navigation: real wins at {wins_total} data sizes -> {verdict}")
    log(f"\nSaved {os.path.join(OUT, 'phase5_navigation.csv')}")


if __name__ == "__main__":
    main()
