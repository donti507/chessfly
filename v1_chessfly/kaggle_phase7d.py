"""
kaggle_phase7d.py -- Phase 7d: causal test -- does SYMMETRY drive the hold/update trade-off?

Phase 7c: connection symmetry predicts integration error (rho = +0.90; +0.74 within
partial shuffles). Here symmetry is manipulated directly. Two degree-preserving
rewirings move the SAME fraction of the real wiring's edges:
    target   edges taken only from reciprocal pairs -> symmetry 0.81 -> ~0.6
    control  edges taken only from non-reciprocal edges -> symmetry stays 0.81
Neither creates new reciprocal pairs; every neuron keeps its in- and out-degree.

Pre-registered predictions (one-sided Mann-Whitney, p < 0.05), CAUSAL SUPPORT if both:
  (1) integrate: target networks have LOWER error than control networks
  (2) hold:      target networks have HIGHER error than control networks

Conditions: real (5 training seeds), target and control (5 networks each) at FRAC.
Level L2, anatomical ports, 1000 training runs.

Usage:  python kaggle_phase7d.py    (~45 min on a T4; saves CSV after every run)
Env:    FRAC (0.2)  NETS (5)  REAL_SEEDS (5)  TASKS (hold,integrate)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu

import kaggle_phase2 as p2
import kaggle_phase5 as p5
import kaggle_phase6 as p6

FRAC = float(os.environ.get("FRAC", "0.2"))
NETS = int(os.environ.get("NETS", "5"))
REAL_SEEDS = int(os.environ.get("REAL_SEEDS", "5"))
TASKS = os.environ.get("TASKS", "hold,integrate").split(",")
LEVEL, SIZE = "L2", 1000
OUT = p2.OUT_DIR
log = p2.log


def targeted_shuffle(C, frac, seed, mode, max_tries_per_edge=60):
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(C)
    w = C[pre, post].copy()
    post = post.copy()
    E = len(pre)
    edges = set(zip(pre.tolist(), post.tolist()))
    original = set(edges)
    recip0 = np.array([(b, a) in edges for a, b in zip(pre, post)])
    pool = np.where(recip0 if mode == "target" else ~recip0)[0]
    moved, tries, target, max_tries = 0, 0, frac * E, max_tries_per_edge * E
    while moved < target and tries < max_tries:
        i, j = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
        tries += 1
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == c or b == d or (a, d) in edges or (c, b) in edges:
            continue
        if (d, a) in edges or (b, c) in edges:
            continue
        moved -= ((a, b) not in original) + ((c, d) not in original)
        edges.discard((a, b)); edges.discard((c, d)); edges.add((a, d)); edges.add((c, b))
        post[i], post[j] = d, b
        moved += ((a, d) not in original) + ((c, b) not in original)
    out = np.zeros_like(C)
    out[pre, post] = w
    return out, moved / E


def symmetry(Cnet, signs):
    W = Cnet * signs[:, None]
    return float((((W + W.T) / 2) ** 2).sum() / (W ** 2).sum())


def main():
    log("=" * 78)
    log("CHESSFLY PHASE 7d -- causal test: symmetry-breaking vs symmetry-preserving rewiring")
    log(f"frac={FRAC} nets={NETS} real seeds={REAL_SEEDS} tasks={TASKS} level={LEVEL}")
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

    nets = [("real", 0.0, s, C, symmetry(C, signs)) for s in range(REAL_SEEDS)]
    for mode in ["target", "control"]:
        for k in range(NETS):
            M, got = targeted_shuffle(C, FRAC, seed=(700 if mode == "target" else 800) + k, mode=mode)
            nets.append((mode, got, k, M, symmetry(M, signs)))
            log(f"  {mode:<7} network {k}: moved {got:.2f} of edges, symmetry {nets[-1][-1]:.3f}")

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
    path = os.path.join(OUT, "phase7d_symmetry_causal.csv")
    for task in [t.strip() for t in TASKS]:
        for name, got, k, Cnet, sym in nets:
            te, le = run(task, k if name == "real" else 0, Cnet)
            rows.append({"task": task, "condition": name, "net": k, "moved": got, "symmetry": sym,
                         "test_err": te, "long_err": le})
            pd.DataFrame(rows).to_csv(path, index=False)
            el = time.time() - t0
            log(f"  [{len(rows):>2}/{total}] {task:<9} {name:<7} net {k} symmetry {sym:.3f}  error {te:5.1f}  "
                f"2x {le:5.1f} deg  [{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)

    log("\n" + "=" * 78)
    log("RESULTS (median test error, degrees; lower = better)")
    log("=" * 78)
    log(f"{'condition':<10}{'moved':>7}{'symmetry':>10}" + "".join(f"{t:>12}" for t in TASKS))
    for name in ["real", "target", "control"]:
        d = df[df.condition == name]
        log(f"{name:<10}{d.moved.mean():>7.2f}{d.symmetry.mean():>10.3f}" +
            "".join(f"{d[d.task == t].test_err.median():>12.1f}" for t in TASKS))

    ok = []
    if "integrate" in TASKS:
        d = df[df.task == "integrate"]
        p = mannwhitneyu(d[d.condition == "target"].test_err, d[d.condition == "control"].test_err, alternative="less").pvalue
        ok.append(p < 0.05)
        log(f"\n(1) integrate: symmetry-broken better than symmetry-preserved? p = {p:.4f} -> {'YES' if p < 0.05 else 'no'}")
    if "hold" in TASKS:
        d = df[df.task == "hold"]
        p = mannwhitneyu(d[d.condition == "target"].test_err, d[d.condition == "control"].test_err, alternative="greater").pvalue
        ok.append(p < 0.05)
        log(f"(2) hold: symmetry-broken worse than symmetry-preserved?    p = {p:.4f} -> {'YES' if p < 0.05 else 'no'}")
    log("VERDICT: " + ("CAUSAL SUPPORT -- symmetry drives the maintenance/updating trade-off"
                       if ok and all(ok) else "causal role of symmetry NOT supported by the pre-registered rule"))
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
