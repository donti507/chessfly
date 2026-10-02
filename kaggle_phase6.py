"""
kaggle_phase6.py -- Phase 6: the flexibility sweep

Thesis: a connectome is an inductive bias that helps only when learning is
constrained. Prediction: a CROSSOVER -- real wiring wins when little is learned
inside the circuit (L0/L1) and loses when every synapse is free (L3).

Learning-freedom levels (input and readout are always trained):
  L0  recurrent weights frozen (synapse counts, scaled)       ~0 circuit params
  L1  one learnable strength per (pre cell type, post cell type) pair
  L2  one learnable gain per presynaptic neuron                434
  L3  every synapse learnable (Phases 3b-5b)                   ~34,000
Networks: real, rewired, degshuffle (neuron identities and cell types fixed,
only edges shuffled -> shuffled nets touch MORE type pairs, i.e. get MORE L1
parameters, which favors the controls).
Tasks (anatomical ports: turning -> PEN, landmark -> ER, heading <- EPG):
  hold       landmark for 5 steps, then darkness and no turning: pure memory
  integrate  landmark for 5 steps, then random turning: angular integration
Metric: heading error in degrees (lower = better, chance 90). Decision rule:
real wins only if its WORST seed beats the control's BEST seed.

Usage:  python kaggle_phase6.py      (~2.5 h on a T4; saves CSV after every run)
Env:    SEEDS (0,1,2)  SIZES6 (100,1000)  LEVELS (L0,L1,L2,L3)  TASKS (hold,integrate)
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

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
SIZES = [int(s) for s in os.environ.get("SIZES6", "100,1000").split(",")]
LEVELS = os.environ.get("LEVELS", "L0,L1,L2,L3").split(",")
TASKS = os.environ.get("TASKS", "hold,integrate").split(",")
KINDS = os.environ.get("KINDS", "real,rewired,degshuffle").split(",")
OUT = p2.OUT_DIR
log = p2.log


def make_task(task, n, T, seed):
    x, y = p5.make_runs(n, T, seed)
    if task == "hold":                       # no turning: heading stays at its cue value
        x[:, :, 0] = 0.0
        th0 = np.arctan2(y[:, 0, 1], y[:, 0, 0])
        y = np.repeat(np.stack([np.cos(th0), np.sin(th0)], -1)[:, None, :], T, axis=1).astype(np.float32)
        x[:, :p5.CUE_STEPS, 1] = np.cos(th0)[:, None]
        x[:, :p5.CUE_STEPS, 2] = np.sin(th0)[:, None]
    return x, y


def one_hot(idx, n):
    P = torch.zeros(len(idx), n)
    P[torch.arange(len(idx)), torch.as_tensor(idx)] = 1.0
    return P


class FlexBrain(nn.Module):
    def __init__(self, level, n, ports, C, signs, type_id):
        super().__init__()
        self.level, self.n = level, n
        om, cue, out = ports
        self.register_buffer("P_om", one_hot(om, n))
        self.register_buffer("P_cue", one_hot(cue, n))
        self.register_buffer("outp", torch.as_tensor(out, dtype=torch.long))
        self.in_om, self.in_cue = nn.Linear(1, len(om)), nn.Linear(3, len(cue))
        self.readout = nn.Linear(len(out), 2)
        rho = float(np.max(np.abs(np.linalg.eigvals(C * signs[:, None]))))
        base = C * (p5.RHO / rho)
        self.register_buffer("base", torch.tensor(base, dtype=torch.float32))
        self.register_buffer("sign", torch.tensor(signs, dtype=torch.float32)[:, None])
        self.register_buffer("mask", torch.tensor((C > 0).astype(np.float32)))
        K = int(type_id.max()) + 1
        if level == "L1":
            tp = type_id[:, None] * K + type_id[None, :]
            self.register_buffer("tp", torch.tensor(tp, dtype=torch.long))
            self.g = nn.Parameter(torch.zeros(K * K))
            self.n_circuit = int(len(np.unique(tp[C > 0])))
        elif level == "L2":
            self.g = nn.Parameter(torch.zeros(n))
            self.n_circuit = n
        elif level == "L3":
            self.log_mag = nn.Parameter(torch.log(self.base + 1e-6))
            self.n_circuit = int((C > 0).sum())
        else:
            self.n_circuit = 0

    def weights(self):
        if self.level == "L1":
            return self.sign * self.base * torch.exp(self.g[self.tp]) * self.mask
        if self.level == "L2":
            return self.sign * self.base * torch.exp(self.g)[:, None]
        if self.level == "L3":
            return self.sign * torch.exp(self.log_mag) * self.mask
        return self.sign * self.base

    def forward(self, x):
        B, T, _ = x.shape
        W = self.weights()
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
    log("CHESSFLY PHASE 6 -- flexibility sweep (does structure help under constrained learning?)")
    log(f"levels={LEVELS} tasks={TASKS} kinds={KINDS} seeds={SEEDS} sizes={SIZES} steps={p5.STEPS5}")
    log("=" * 78)
    p2.ensure_network_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj, meta, signs = p2.load_network()
    n = adj.shape[0]
    types = meta["type"].astype(str).tolist()
    type_id = np.unique(types, return_inverse=True)[1]
    C = np.zeros(adj.shape, dtype=np.float32)
    C[adj.row, adj.col] = adj.data
    pick = lambda prefix: np.array([i for i, t in enumerate(types) if t.startswith(prefix)])
    ports = (pick("PEN"), pick("ER"), pick("EPG"))
    log(f"Network: {n} neurons, {int(type_id.max()) + 1} cell types | ports: PEN {len(ports[0])}, "
        f"ER {len(ports[1])}, EPG {len(ports[2])} | {device}")

    nets = {}
    for s in SEEDS:
        nets[("real", s)] = C
        nets[("rewired", s)] = p2.rewire(C, seed=100 + s)
        nets[("degshuffle", s)] = p3b.degree_preserving_shuffle(C, seed=200 + s)

    to = lambda a: torch.from_numpy(a).to(device)
    evalsets = {task: tuple(map(to, make_task(task, p5.N_VAL, p5.T_TRAIN, 999))) +
                tuple(map(to, make_task(task, p5.N_TEST, p5.T_TRAIN, 1001))) +
                tuple(map(to, make_task(task, p5.N_TEST, p5.T_TEST, 1002))) for task in TASKS}

    @torch.no_grad()
    def evaluate(model, x, y):
        model.eval()
        preds = torch.cat([model(x[i:i + 500]) for i in range(0, len(x), 500)])
        model.train()
        return p5.heading_error(preds, y)

    def run(task, level, kind, seed, size):
        torch.manual_seed(seed)
        xtr, ytr = map(to, make_task(task, size, p5.T_TRAIN, 10_000 + 100 * seed + size))
        xv, yv, xt, yt, xl, yl = evalsets[task]
        model = FlexBrain(level, n, ports, nets[(kind, seed)], signs, type_id).to(device)
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
        return {"task": task, "level": level, "condition": kind, "seed": seed, "size": size,
                "circuit_params": model.n_circuit, "val_err": best,
                "test_err": evaluate(model, xt, yt), "long_err": evaluate(model, xl, yl)}

    tasks, levels, kinds = [t.strip() for t in TASKS], [l.strip() for l in LEVELS], [k.strip() for k in KINDS]
    total = len(tasks) * len(levels) * len(kinds) * len(SIZES) * len(SEEDS)
    rows, t0 = [], time.time()
    path = os.path.join(OUT, "phase6_flexibility.csv")
    for task in tasks:
        for level in levels:
            for kind in kinds:
                for size in SIZES:
                    for seed in SEEDS:
                        r = run(task, level, kind, seed, size)
                        rows.append(r)
                        pd.DataFrame(rows).to_csv(path, index=False)
                        el = time.time() - t0
                        log(f"  [{len(rows):>3}/{total}] {task:<9} {level} {kind:<10} runs={size:>5} seed {seed} "
                            f"params {r['circuit_params']:>6}  error {r['test_err']:5.1f}  2x {r['long_err']:5.1f} deg  "
                            f"[{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)

    verdicts = {}
    for task in tasks:
        log("\n" + "=" * 78)
        log(f"TASK: {task.upper()} -- heading error (deg, mean +/- std; lower = better)")
        log("=" * 78)
        for size in SIZES:
            log(f"  training runs = {size}")
            log(f"  {'level':<6}" + "".join(f"{k:>18}" for k in kinds) + "    verdict (real vs controls)")
            for level in levels:
                sel = lambda k: df[(df.task == task) & (df.level == level) & (df.condition == k) & (df["size"] == size)]["test_err"]
                cells = "".join(f"{sel(k).mean():>11.1f}+/-{sel(k).std(ddof=0):4.1f}" for k in kinds)
                r = sel("real")
                ctrl = [sel(k) for k in kinds if k != "real"]
                if all(r.max() < c.min() for c in ctrl):
                    v = "REAL WINS"
                elif all(r.min() > c.max() for c in ctrl):
                    v = "real loses"
                else:
                    v = "mixed / no clear difference"
                verdicts[(task, size, level)] = v
                log(f"  {level:<6}{cells}    {v}")

    log("\n" + "=" * 78)
    log("CROSSOVER CHECK")
    log("=" * 78)
    for task in tasks:
        for size in SIZES:
            v = [verdicts[(task, size, l)] for l in levels]
            low = any(x == "REAL WINS" for l, x in zip(levels, v) if l in ("L0", "L1"))
            high = verdicts.get((task, size, "L3")) == "real loses"
            tag = ("CROSSOVER: real wins under constrained learning, loses when free" if low and high else
                   "real wins at every level" if all(x == "REAL WINS" for x in v) else
                   "no crossover")
            log(f"  {task:<9} runs={size:>5}: " + ", ".join(f"{l}={x}" for l, x in zip(levels, v)))
            log(f"  {'':<9} -> {tag}")
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
