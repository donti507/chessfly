"""
kaggle_phase5b.py -- Phase 5b: navigation with anatomically correct ports

Phase 5 plugged the circuit in at arbitrary "ports" (inputs into 150 neurons chosen
by connection count, heading read from all the others) and the real wiring lost.
In the fly, signals enter and leave at specific cell types:
    turning (angular velocity) -> PEN neurons
    landmark (visual cue)      -> ER ring neurons
    heading readout            <- EPG compass neurons
This script runs BOTH designs so they can be compared directly:
    original    omega + cue -> 150 sensory neurons, heading <- other 284 (= Phase 5)
    anatomical  omega -> PEN, cue -> ER, heading <- EPG
Ports are defined by neuron identity, so shuffled networks get the same ports
(only who-connects-to-whom changes): a fair comparison.

Metric: heading error in degrees (lower = better; chance = 90), on test runs of the
training length and on runs twice as long. Decision rule as before: real wins at a
data size only if its WORST seed beats the control's BEST seed.

Usage:  python kaggle_phase5b.py        (run with Save Version -> Save & Run All)
Env:    SEEDS (0,1,2)  SIZES5B (300,1000,3000)  DESIGNS (original,anatomical)
        KINDS (real,rewired,degshuffle)  STEPS5 (2000)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import kaggle_phase2 as p2
import kaggle_phase3b as p3b
import kaggle_phase5 as p5
from reservoir_core import select_input_neurons

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
SIZES = [int(s) for s in os.environ.get("SIZES5B", "300,1000,3000").split(",")]
DESIGNS = os.environ.get("DESIGNS", "original,anatomical").split(",")
KINDS = os.environ.get("KINDS", "real,rewired,degshuffle").split(",")
OUT = p2.OUT_DIR
log = p2.log


def one_hot(idx, n):
    P = torch.zeros(len(idx), n)
    P[torch.arange(len(idx)), torch.as_tensor(idx)] = 1.0
    return P


class PortBrain(nn.Module):
    """Rate network with separate ports for turning input, cue input and readout."""

    def __init__(self, n, omega_idx, cue_idx, out_idx, C, signs):
        super().__init__()
        self.n = n
        self.register_buffer("P_om", one_hot(omega_idx, n))
        self.register_buffer("P_cue", one_hot(cue_idx, n))
        self.register_buffer("outp", torch.as_tensor(out_idx, dtype=torch.long))
        self.in_om = nn.Linear(1, len(omega_idx))
        self.in_cue = nn.Linear(3, len(cue_idx))
        self.readout = nn.Linear(len(out_idx), 2)
        W0 = C * signs[:, None]
        rho = float(np.max(np.abs(np.linalg.eigvals(W0))))
        self.register_buffer("mask", torch.tensor((C > 0).astype(np.float32)))
        self.register_buffer("sign", torch.tensor(signs, dtype=torch.float32)[:, None])
        self.log_mag = nn.Parameter(torch.log(torch.tensor(C * (p5.RHO / rho), dtype=torch.float32) + 1e-6))

    def forward(self, x):
        B, T, _ = x.shape
        W = self.sign * torch.exp(self.log_mag) * self.mask
        U = self.in_om(x[..., :1]) @ self.P_om + self.in_cue(x[..., 1:]) @ self.P_cue
        h = torch.zeros(B, self.n, device=x.device)
        outs = []
        for t in range(T):
            for _ in range(p5.MICRO):
                h = (1 - p5.ALPHA) * h + p5.ALPHA * (torch.relu(h) @ W + U[:, t])
            outs.append(self.readout(torch.relu(h[:, self.outp])))
        return torch.stack(outs, dim=1)


def main():
    log("=" * 78)
    log("CHESSFLY PHASE 5b -- navigation: original vs anatomically correct ports")
    log(f"designs={DESIGNS} kinds={KINDS} seeds={SEEDS} sizes={SIZES} steps={p5.STEPS5}")
    log("=" * 78)
    p2.ensure_network_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj, meta, signs = p2.load_network()
    n = adj.shape[0]
    types = meta["type"].astype(str).to_numpy()
    C = np.zeros(adj.shape, dtype=np.float32)
    C[adj.row, adj.col] = adj.data

    sensory = np.array(select_input_neurons(adj, meta, p3b.N_SENSORY))
    others = np.setdiff1d(np.arange(n), sensory)
    pick = lambda prefix: np.array([i for i, t in enumerate(types) if t.startswith(prefix)])
    pen, er, epg = pick("PEN"), pick("ER"), pick("EPG")
    ports = {"original": (sensory, sensory, others), "anatomical": (pen, er, epg)}
    log(f"Network: {n} neurons | original ports: in {len(sensory)}, out {len(others)} | "
        f"anatomical ports: turning->PEN {len(pen)}, landmark->ER {len(er)}, heading<-EPG {len(epg)}")

    nets = {}
    for s in SEEDS:
        nets[("real", s)] = C
        nets[("rewired", s)] = p2.rewire(C, seed=100 + s)
        nets[("degshuffle", s)] = p3b.degree_preserving_shuffle(C, seed=200 + s)

    to = lambda a: torch.from_numpy(a).to(device)
    xv, yv = map(to, p5.make_runs(p5.N_VAL, p5.T_TRAIN, seed=999))
    xt, yt = map(to, p5.make_runs(p5.N_TEST, p5.T_TRAIN, seed=1001))
    xl, yl = map(to, p5.make_runs(p5.N_TEST, p5.T_TEST, seed=1002))

    @torch.no_grad()
    def evaluate(model, x, y):
        model.eval()
        preds = torch.cat([model(x[i:i + 500]) for i in range(0, len(x), 500)])
        model.train()
        return p5.heading_error(preds, y)

    def run(design, kind, seed, size):
        torch.manual_seed(seed)
        xtr, ytr = map(to, p5.make_runs(size, p5.T_TRAIN, seed=10_000 + 100 * seed + size))
        om, cue, out = ports[design]
        model = PortBrain(n, om, cue, out, nets[(kind, seed)], signs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=p5.LR)
        best, state = 1e9, None
        for step in range(1, p5.STEPS5 + 1):
            b = torch.randint(size, (min(p5.BATCH, size),), device=device)
            loss = ((model(xtr[b]) - ytr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % p5.EVAL_EVERY == 0:
                e = evaluate(model, xv, yv)
                if e < best:
                    best, state = e, {k: t.detach().clone() for k, t in model.state_dict().items()}
        model.load_state_dict(state)
        return {"design": design, "condition": kind, "seed": seed, "size": size, "val_err": best,
                "test_err": evaluate(model, xt, yt), "long_err": evaluate(model, xl, yl)}

    rows, t0 = [], time.time()
    designs, kinds = [d.strip() for d in DESIGNS], [k.strip() for k in KINDS]
    total = len(designs) * len(kinds) * len(SIZES) * len(SEEDS)
    path = os.path.join(OUT, "phase5b_navigation.csv")
    for design in designs:
        for kind in kinds:
            for size in SIZES:
                for seed in SEEDS:
                    r = run(design, kind, seed, size)
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(path, index=False)    # save as we go
                    el = time.time() - t0
                    log(f"  [{len(rows):>2}/{total}] {design:<10} {kind:<10} runs={size:>5} seed {seed}  "
                        f"error {r['test_err']:5.1f}  2x longer {r['long_err']:5.1f} deg  "
                        f"[{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)

    summary = {}
    for design in designs:
        for metric, name in [("test_err", "runs of training length"), ("long_err", "runs 2x longer")]:
            log("\n" + "=" * 78)
            log(f"{design.upper()} PORTS -- heading error, {name} (degrees, mean +/- std; lower = better)")
            log("=" * 78)
            log(f"{'condition':<12}" + "".join(f"{('runs=' + str(s)):>15}" for s in SIZES))
            for kind in kinds:
                line = f"{kind:<12}"
                for size in SIZES:
                    g = df[(df.design == design) & (df.condition == kind) & (df["size"] == size)][metric]
                    line += f"{g.mean():>9.1f}+/-{g.std(ddof=0):4.1f}"
                log(line)
        for ctrl in [k for k in kinds if k != "real"]:
            wins = 0
            log(f"\n{design.upper()}: REAL vs {ctrl.upper()} (rule: worst real seed < best {ctrl} seed)")
            for size in SIZES:
                sel = lambda k: df[(df.design == design) & (df.condition == k) & (df["size"] == size)]["test_err"]
                r, c = sel("real"), sel(ctrl)
                win, lose = r.max() < c.min(), r.min() > c.max()
                wins += int(win)
                log(f"  runs={size:>5}: real {r.mean():5.1f} vs {c.mean():5.1f} deg   "
                    f"{'REAL WINS' if win else ('real loses' if lose else 'no clear difference')}")
            summary[(design, ctrl)] = wins
            log(f"  -> real wins at {wins}/{len(SIZES)} data sizes")

    log("\n" + "=" * 78)
    log("VERDICT")
    log("=" * 78)
    for design in designs:
        w = [summary.get((design, c), 0) for c in kinds if c != "real"]
        log(f"  {design:<10}: real wins at {w} data sizes vs {[c for c in kinds if c != 'real']}")
    if "anatomical" in designs:
        wa = min(summary.get(("anatomical", c), 0) for c in kinds if c != "real")
        if wa >= len(SIZES) // 2 + 1:
            log("  -> With anatomical ports the real wiring WINS at navigation: the circuit is")
            log("     specialized, but only when signals enter and leave where biology puts them.")
        else:
            log("  -> Even with anatomical ports the real wiring does not clearly win: in this")
            log("     trained rate model the real connectome is a general handicap.")
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
