"""
kaggle_phase7f.py -- Phase 7f: is holding about the LANDMARK MAP or the compass's INTERNAL loops?

Phase 7e (replicated, dose-dependent): rewiring one-way connections breaks holding a heading;
rewiring reciprocal connections does not. 58% of one-way synapses are in ER -> EPG, the route
by which the landmark reaches the compass. This separates two explanations by scrambling the
MAP of single pathways: target neurons are relabeled within that pathway only, so every
connection keeps its strength and every source neuron keeps its targets' count, but which
target each connection reaches is randomized.

  scramble_ER_EPG  landmark input map (ring neurons -> compass)
  scramble_EPG_D7  internal one-way loop (compass -> inhibitory D7)
  scramble_D7_PEN  internal one-way loop (D7 -> PEN)
  scramble_ER_ER   mostly reciprocal pathway (reference; symmetry 0.81 -> ~0.56)

Pre-registered decision (one-sided Mann-Whitney vs real, hold task, p < 0.05):
  ROUTING  : ER->EPG scramble is worse than real AND worse than both internal scrambles
  INTERNAL : at least one internal scramble is worse than real AND ER->EPG scramble is not
  otherwise: BOTH or NEITHER, reported as such.

Usage:  python kaggle_phase7f.py    (~70 min on a T4; saves CSV after every run)
Env:    NETS (5)  REAL_SEEDS (5)  TASKS (hold,integrate)
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

NETS = int(os.environ.get("NETS", "5"))
REAL_SEEDS = int(os.environ.get("REAL_SEEDS", "5"))
TASKS = os.environ.get("TASKS", "hold,integrate").split(",")
LEVEL, SIZE = "L2", 1000
OUT = p2.OUT_DIR
log = p2.log


def scramble(C, pre_idx, post_idx, seed):
    M = C.copy()
    perm = np.random.default_rng(seed).permutation(len(post_idx))
    M[np.ix_(pre_idx, post_idx)] = C[np.ix_(pre_idx, post_idx[perm])]
    return M


def symmetry(Cnet, signs):
    W = Cnet * signs[:, None]
    return float((((W + W.T) / 2) ** 2).sum() / (W ** 2).sum())


def main():
    log("=" * 78)
    log("CHESSFLY PHASE 7f -- landmark map vs internal loops: pathway-map scrambles")
    log(f"nets={NETS} real seeds={REAL_SEEDS} tasks={TASKS} level={LEVEL}")
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
    PEN, ER, EPG, D7 = pick("PEN"), pick("ER"), pick("EPG"), pick("Delta7")
    ports = (PEN, ER, EPG)
    pathways = {"scramble_ER_EPG": (ER, EPG), "scramble_EPG_D7": (EPG, D7),
                "scramble_D7_PEN": (D7, PEN), "scramble_ER_ER": (ER, ER)}

    nets = [("real", s, C, symmetry(C, signs), 0.0) for s in range(REAL_SEEDS)]
    for pi, (name, (a, b)) in enumerate(pathways.items()):
        blk = C[np.ix_(a, b)]
        for k in range(NETS):
            M = scramble(C, a, b, seed=11000 + 100 * pi + k)
            relocated = float(np.abs(M[np.ix_(a, b)] - blk).sum() / (2 * blk.sum()))
            nets.append((name, k, M, symmetry(M, signs), relocated))
        log(f"  {name:<16} pathway synapses relocated {np.mean([x[4] for x in nets[-NETS:]]):.2f}, "
            f"symmetry {np.mean([x[3] for x in nets[-NETS:]]):.3f}")

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
    path = os.path.join(OUT, "phase7f_pathways.csv")
    for task in [t.strip() for t in TASKS]:
        for name, k, Cnet, sym, rel in nets:
            te, le = run(task, k if name == "real" else 0, Cnet)
            rows.append({"task": task, "condition": name, "net": k, "symmetry": sym, "relocated": rel,
                         "test_err": te, "long_err": le})
            pd.DataFrame(rows).to_csv(path, index=False)
            el = time.time() - t0
            log(f"  [{len(rows):>2}/{total}] {task:<9} {name:<16} net {k}  error {te:5.1f}  2x {le:5.1f} deg  "
                f"[{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)

    log("\n" + "=" * 78)
    log("RESULTS (median test error, degrees; lower = better)")
    log("=" * 78)
    names = ["real"] + list(pathways)
    log(f"{'condition':<18}{'relocated':>10}{'symmetry':>10}" + "".join(f"{t:>12}" for t in TASKS))
    for name in names:
        d = df[df.condition == name]
        log(f"{name:<18}{d.relocated.mean():>10.2f}{d.symmetry.mean():>10.3f}" +
            "".join(f"{d[d.task == t].test_err.median():>12.1f}" for t in TASKS))

    if "hold" in TASKS:
        h = df[df.task == "hold"]
        err = {nm: h[h.condition == nm].test_err.to_numpy() for nm in names}
        worse = lambda x, y: mannwhitneyu(err[x], err[y], alternative="greater").pvalue
        log("\nHold: worse than real? (one-sided Mann-Whitney)")
        pv = {nm: worse(nm, "real") for nm in pathways}
        for nm, p in pv.items():
            log(f"  {nm:<16} p = {p:.4f} -> {'WORSE' if p < 0.05 else 'not worse'}")
        route = pv["scramble_ER_EPG"] < 0.05 and all(worse("scramble_ER_EPG", x) < 0.05
                                                       for x in ["scramble_EPG_D7", "scramble_D7_PEN"])
        internal = (pv["scramble_EPG_D7"] < 0.05 or pv["scramble_D7_PEN"] < 0.05) and pv["scramble_ER_EPG"] >= 0.05
        verdict = ("ROUTING -- holding depends on the precise landmark-to-compass map" if route else
                   "INTERNAL -- holding depends on the compass's internal one-way loops" if internal else
                   "BOTH" if pv["scramble_ER_EPG"] < 0.05 else "NEITHER pathway alone breaks holding")
        log(f"VERDICT: {verdict}")
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
