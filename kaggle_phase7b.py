"""
kaggle_phase7b.py -- Phase 7b: dose-response test of the slow-mode mechanism

Phase 7 found that the number of slow modes (eigenvalues above 2/3 of the largest)
predicts how well a network learns to integrate turns (Spearman -0.83, 61 networks).
Partial degree-preserving shuffling of the real wiring changes slow modes
NON-monotonically: real 3 -> about 1 at 50-64% shuffled -> 20-58 when fully shuffled.
That gives opposite predictions:
  slow-mode mechanism : 50-60% shuffled networks integrate NO BETTER than real
  "more random = better": 50-60% shuffled networks integrate better than real

Conditions: shuffle fraction 0 (real, 5 training seeds) and 0.25, 0.5, 0.6, 1.0
(5 different networks each). Level L2, anatomical ports, 1000 training runs.

Pre-registered verdict -- MECHANISM SUPPORTED only if all three hold:
  (a) across networks, slow modes vs integrate error: Spearman rho < 0, p < 0.05
  (b) 50% and 60% shuffled networks do NOT integrate better than real
      (one-sided Mann-Whitney p > 0.05 for "shuffled better")
  (c) fully shuffled networks DO integrate better than real (p < 0.05)

Usage:  python kaggle_phase7b.py     (~70 min on a T4; saves CSV after every run)
Env:    FRACS (0.25,0.5,0.6,1.0)  NETS (5)  REAL_SEEDS (5)  TASKS (hold,integrate)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, mannwhitneyu

import kaggle_phase2 as p2
import kaggle_phase5 as p5
import kaggle_phase6 as p6

FRACS = [float(f) for f in os.environ.get("FRACS", "0.25,0.5,0.6,1.0").split(",")]
NETS = int(os.environ.get("NETS", "5"))
REAL_SEEDS = int(os.environ.get("REAL_SEEDS", "5"))
TASKS = os.environ.get("TASKS", "hold,integrate").split(",")
LEVEL, SIZE, FAIL = "L2", 1000, 80.0
OUT = p2.OUT_DIR
log = p2.log


def partial_degree_shuffle(C, frac, seed, max_tries_per_edge=40):
    """Degree-preserving swaps until a fraction `frac` of the original edges has moved."""
    if frac <= 0:
        return C.copy(), 0.0
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(C)
    w = C[pre, post].copy()
    post = post.copy()
    E = len(pre)
    original = set(zip(pre.tolist(), post.tolist()))
    edges = set(original)
    moved, tries, target, max_tries = 0, 0, frac * E, max_tries_per_edge * E
    while moved < target and tries < max_tries:
        i, j = rng.integers(E, size=2)
        tries += 1
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == c or b == d or (a, d) in edges or (c, b) in edges:
            continue
        moved -= ((a, b) not in original) + ((c, d) not in original)
        edges.discard((a, b)); edges.discard((c, d)); edges.add((a, d)); edges.add((c, b))
        post[i], post[j] = d, b
        moved += ((a, d) not in original) + ((c, b) not in original)
    out = np.zeros_like(C)
    out[pre, post] = w
    return out, moved / E


def slow_modes(Cnet, signs):
    lam = np.abs(np.linalg.eigvals(Cnet * signs[:, None]))
    return int((lam / lam.max() > 2 / 3).sum())


def main():
    log("=" * 78)
    log("CHESSFLY PHASE 7b -- dose-response test of the slow-mode mechanism")
    log(f"fracs={FRACS} nets/level={NETS} real seeds={REAL_SEEDS} tasks={TASKS} level={LEVEL}")
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

    log("Building networks ...")
    nets = [("real", 0.0, 0.0, s, C, slow_modes(C, signs)) for s in range(REAL_SEEDS)]
    for f in FRACS:
        for k in range(NETS):
            M, got = partial_degree_shuffle(C, f, seed=int(1000 * f) + k)
            nets.append((f"shuffle{f:g}", f, got, k, M, slow_modes(M, signs)))
            log(f"  shuffle {f:4.2f}: network {k} moved {got:.2f} of edges, slow modes {nets[-1][-1]}")

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
    path = os.path.join(OUT, "phase7b_dose_response.csv")
    for task in [t.strip() for t in TASKS]:
        for name, frac, got, k, Cnet, sm in nets:
            train_seed = k if name == "real" else 0
            te, le = run(task, train_seed, Cnet)
            rows.append({"task": task, "condition": name, "frac": frac, "moved": got, "net": k,
                         "slow_modes": sm, "test_err": te, "long_err": le})
            pd.DataFrame(rows).to_csv(path, index=False)
            el = time.time() - t0
            log(f"  [{len(rows):>3}/{total}] {task:<9} {name:<11} net {k} slow {sm:>3}  error {te:5.1f}  "
                f"2x {le:5.1f} deg  [{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)

    log("\n" + "=" * 78)
    log("DOSE-RESPONSE (median test error, degrees; lower = better)")
    log("=" * 78)
    log(f"{'condition':<12}{'moved':>7}{'slow modes':>12}" + "".join(f"{t:>14}" for t in TASKS) + "   hold failures")
    for name in ["real"] + [f"shuffle{f:g}" for f in FRACS]:
        d = df[df.condition == name]
        line = f"{name:<12}{d.moved.mean():>7.2f}{d.slow_modes.median():>12.0f}"
        line += "".join(f"{d[d.task == t].test_err.median():>14.1f}" for t in TASKS)
        if "hold" in TASKS:
            line += f"   {int((d[d.task == 'hold'].test_err > FAIL).sum())}/{len(d[d.task == 'hold'])}"
        log(line)

    if "integrate" in TASKS:
        d = df[df.task == "integrate"]
        per_net = d.groupby(["condition", "net"]).agg(slow=("slow_modes", "first"), err=("test_err", "median")).reset_index()
        per_net = pd.concat([per_net[per_net.condition != "real"],
                             per_net[per_net.condition == "real"].assign(err=d[d.condition == "real"].test_err.median()).head(1)])
        rho = spearmanr(per_net.slow, per_net.err)
        real = d[d.condition == "real"].test_err.to_numpy()
        a = rho.correlation < 0 and rho.pvalue < 0.05
        mid = d[d.frac.isin([0.5, 0.6])].test_err.to_numpy()
        p_mid = mannwhitneyu(mid, real, alternative="less").pvalue if len(mid) else np.nan
        b = p_mid > 0.05
        full = d[d.frac == 1.0].test_err.to_numpy()
        p_full = mannwhitneyu(full, real, alternative="less").pvalue if len(full) else np.nan
        c = p_full < 0.05
        log("\nMECHANISM TEST (integrate)")
        log(f"  (a) slow modes vs error across {len(per_net)} networks: rho = {rho.correlation:+.2f}, "
            f"p = {rho.pvalue:.3g} -> {'met' if a else 'not met'}")
        log(f"  (b) 50-60% shuffled better than real? p = {p_mid:.3g} -> {'met (not better)' if b else 'not met (they ARE better)'}")
        log(f"  (c) fully shuffled better than real?   p = {p_full:.3g} -> {'met' if c else 'not met'}")
        log("  VERDICT: " + ("MECHANISM SUPPORTED -- performance follows slow modes, not randomness"
                             if a and b and c else "mechanism NOT supported by the pre-registered rule"))
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
