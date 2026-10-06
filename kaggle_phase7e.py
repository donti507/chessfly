"""
kaggle_phase7e.py -- Phase 7e: replication + dose of the structural dissociation

Phase 7d (5 networks per condition, 20% rewired) found, contrary to its prediction:
  rewiring RECIPROCAL edges  (symmetry down) -> hold intact (1.9 deg), integrate better (32)
  rewiring ONE-WAY edges     (symmetry kept) -> hold broken (26.2 deg), integrate 41.7
This replicates it with NEW random networks at three doses (5%, 10%, 20% of edges).

New pre-registered predictions (one-sided Mann-Whitney, p < 0.05), written before running:
  P1 (hold):      control error > target error at 10% AND at 20%
  P2 (integrate): target error < control error at 20%
  P3 (dose):      across real + target networks, integrate error falls as more reciprocal
                  edges are rewired (Spearman rho < 0, p < 0.05)
REPLICATED only if P1, P2 and P3 all hold.

Usage:  python kaggle_phase7e.py    (~95 min on a T4; saves CSV after every run)
Env:    FRACS (0.05,0.1,0.2)  NETS (5)  REAL_SEEDS (5)  TASKS (hold,integrate)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu, spearmanr

import kaggle_phase2 as p2
import kaggle_phase5 as p5
import kaggle_phase6 as p6
from kaggle_phase7d import targeted_shuffle, symmetry

FRACS = [float(f) for f in os.environ.get("FRACS", "0.05,0.1,0.2").split(",")]
NETS = int(os.environ.get("NETS", "5"))
REAL_SEEDS = int(os.environ.get("REAL_SEEDS", "5"))
TASKS = os.environ.get("TASKS", "hold,integrate").split(",")
LEVEL, SIZE = "L2", 1000
OUT = p2.OUT_DIR
log = p2.log


def main():
    log("=" * 78)
    log("CHESSFLY PHASE 7e -- replication + dose: reciprocal vs one-way rewiring (fresh networks)")
    log(f"fracs={FRACS} nets={NETS} real seeds={REAL_SEEDS} tasks={TASKS} level={LEVEL}")
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

    nets = [("real", 0.0, 0.0, s, C, symmetry(C, signs)) for s in range(REAL_SEEDS)]
    for li, f in enumerate(FRACS):
        for mode, base in [("target", 5000), ("control", 6000)]:      # seeds never used before
            for k in range(NETS):
                M, got = targeted_shuffle(C, f, seed=base + 100 * li + k, mode=mode)
                nets.append((mode, f, got, k, M, symmetry(M, signs)))
            log(f"  {mode:<7} {f:4.2f}: moved {np.mean([x[2] for x in nets[-NETS:]]):.3f}, "
                f"symmetry {np.mean([x[5] for x in nets[-NETS:]]):.3f}")

    to = lambda a: torch.from_numpy(a).to(device)
    evalsets = {t: tuple(map(to, p6.make_task(t, p5.N_VAL, p5.T_TRAIN, 999))) +
                tuple(map(to, p6.make_task(t, p5.N_TEST, p5.T_TRAIN, 1001))) +
                tuple(map(to, p6.make_task(t, p5.N_TEST, p5.T_TEST, 1002))) for t in TASKS}

    @torch.no_grad()
    def evaluate(model, x, y):
        model.eval()
        preds = torch.cat([model(x[i:i + 500]) for i in range(0, len(x), 500)])
        model.train()
        return p5.heading_error(preds, y)

    def run(task, train_seed, Cnet):
        torch.manual_seed(train_seed)
        xtr, ytr = map(to, p6.make_task(task, SIZE, p5.T_TRAIN, 10_000 + 100 * train_seed + SIZE))
        xv, yv, xt, yt, xl, yl = evalsets[task]
        model = p6.FlexBrain(LEVEL, n, ports, Cnet, signs, type_id).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=p5.LR)
        best, state = 1e9, None
        for step in range(1, p5.STEPS5 + 1):
            b = torch.randint(SIZE, (min(p5.BATCH, SIZE),), device=device)
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
        return evaluate(model, xt, yt), evaluate(model, xl, yl)

    rows, t0 = [], time.time()
    total = len(TASKS) * len(nets)
    path = os.path.join(OUT, "phase7e_replication.csv")
    for task in [t.strip() for t in TASKS]:
        for name, f, got, k, Cnet, sym in nets:
            te, le = run(task, k if name == "real" else 0, Cnet)
            rows.append({"task": task, "condition": name, "frac": f, "moved": got, "net": k,
                         "symmetry": sym, "test_err": te, "long_err": le})
            pd.DataFrame(rows).to_csv(path, index=False)
            el = time.time() - t0
            log(f"  [{len(rows):>2}/{total}] {task:<9} {name:<7} {f:4.2f} net {k} sym {sym:.3f}  error {te:5.1f}  "
                f"2x {le:5.1f} deg  [{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)

    log("\n" + "=" * 78)
    log("RESULTS (median test error, degrees; lower = better)")
    log("=" * 78)
    log(f"{'condition':<10}{'frac':>6}{'symmetry':>10}" + "".join(f"{t:>12}" for t in TASKS))
    for name, f in [("real", 0.0)] + [(m, f) for f in FRACS for m in ["target", "control"]]:
        d = df[(df.condition == name) & (df.frac == f)]
        log(f"{name:<10}{f:>6.2f}{d.symmetry.mean():>10.3f}" +
            "".join(f"{d[d.task == t].test_err.median():>12.1f}" for t in TASKS))

    def mw(task, f, alt):
        d = df[(df.task == task) & (df.frac == f)]
        return mannwhitneyu(d[d.condition == "control"].test_err, d[d.condition == "target"].test_err,
                            alternative=alt).pvalue

    checks = []
    if "hold" in TASKS:
        for f in [x for x in FRACS if x in (0.1, 0.2)]:
            p = mw("hold", f, "greater")
            checks.append(p < 0.05)
            log(f"\nP1 hold, {f:.2f}: control worse than target? p = {p:.4f} -> {'YES' if p < 0.05 else 'no'}")
    if "integrate" in TASKS:
        if 0.2 in FRACS:
            p = mw("integrate", 0.2, "greater")
            checks.append(p < 0.05)
            log(f"P2 integrate, 0.20: target better than control? p = {p:.4f} -> {'YES' if p < 0.05 else 'no'}")
        d = df[(df.task == "integrate") & (df.condition.isin(["real", "target"]))]
        rho = spearmanr(d.frac, d.test_err)
        checks.append(rho.correlation < 0 and rho.pvalue < 0.05)
        log(f"P3 dose: integrate error vs share of reciprocal edges rewired: rho = {rho.correlation:+.2f}, "
            f"p = {rho.pvalue:.4f} -> {'YES' if checks[-1] else 'no'}")
    log("VERDICT: " + ("REPLICATED -- one-way edges carry maintenance; reciprocal edges impede updating"
                       if checks and all(checks) else "NOT replicated by the pre-registered rule"))
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
