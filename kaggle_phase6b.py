"""
kaggle_phase6b.py -- Phase 6b: null distribution (is the real wiring's hold advantage luck?)

Phase 6 suggested the real wiring is a reliable substrate for HOLDING a heading
(never failed), but each null type had only 3 random networks. Here we draw many:
    real        5 training seeds
    rewired     N_NULL different random rewirings   (1 training seed each)
    degshuffle  N_NULL different degree-preserving shuffles
Level L2 (one learnable gain per neuron), anatomical ports, 1000 training runs.

Pre-registered decision rule (fixed before running):
  real has a GENUINE advantage on a task only if
    (a) its median error beats >= 95% of the networks of BOTH null types, and
    (b) a one-sided Mann-Whitney test (real < null) gives p < 0.05 for both.
Also reported: failure rate (error > 80 deg = network never learned the task).

Usage:  python kaggle_phase6b.py    (~2.5 h on a T4; saves CSV after every run)
Env:    N_NULL (30)  REAL_SEEDS (5)  TASKS (hold,integrate)  LEVEL (L2)  SIZE6B (1000)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu

import kaggle_phase2 as p2
import kaggle_phase3b as p3b
import kaggle_phase5 as p5
import kaggle_phase6 as p6

N_NULL = int(os.environ.get("N_NULL", "30"))
REAL_SEEDS = int(os.environ.get("REAL_SEEDS", "5"))
TASKS = os.environ.get("TASKS", "hold,integrate").split(",")
LEVEL = os.environ.get("LEVEL", "L2")
SIZE = int(os.environ.get("SIZE6B", "1000"))
FAIL = 80.0
OUT = p2.OUT_DIR
log = p2.log


def main():
    log("=" * 78)
    log(f"CHESSFLY PHASE 6b -- null distribution: real ({REAL_SEEDS} seeds) vs {N_NULL} rewired "
        f"+ {N_NULL} degree-preserving networks")
    log(f"tasks={TASKS} level={LEVEL} training runs={SIZE} steps={p5.STEPS5}")
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

    jobs = [("real", s, 0) for s in range(REAL_SEEDS)]
    jobs += [("rewired", 0, 300 + k) for k in range(N_NULL)]
    jobs += [("degshuffle", 0, 400 + k) for k in range(N_NULL)]

    def network(kind, net_seed):
        if kind == "real":
            return C
        if kind == "rewired":
            return p2.rewire(C, seed=net_seed)
        return p3b.degree_preserving_shuffle(C, seed=net_seed)

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

    def run(task, kind, train_seed, net_seed, Cnet):
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
        return {"task": task, "condition": kind, "train_seed": train_seed, "net_seed": net_seed,
                "val_err": best, "test_err": evaluate(model, xt, yt), "long_err": evaluate(model, xl, yl)}

    rows, t0 = [], time.time()
    total = len(TASKS) * len(jobs)
    path = os.path.join(OUT, "phase6b_null_distribution.csv")
    for task in [t.strip() for t in TASKS]:
        for kind, train_seed, net_seed in jobs:
            r = run(task, kind, train_seed, net_seed, network(kind, net_seed))
            rows.append(r)
            pd.DataFrame(rows).to_csv(path, index=False)
            el = time.time() - t0
            log(f"  [{len(rows):>3}/{total}] {task:<9} {kind:<10} net {net_seed:>3} seed {train_seed}  "
                f"error {r['test_err']:5.1f}  2x {r['long_err']:5.1f} deg  "
                f"[{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)

    log("\n" + "=" * 78)
    log("NULL-DISTRIBUTION RESULTS (test heading error, degrees; lower = better)")
    log("=" * 78)
    for task in [t.strip() for t in TASKS]:
        d = df[df.task == task]
        real = d[d.condition == "real"]["test_err"].to_numpy()
        med = float(np.median(real))
        log(f"\n{task.upper()}: real median {med:.1f} deg (seeds: {', '.join(f'{x:.1f}' for x in real)}), "
            f"real failures {int((real > FAIL).sum())}/{len(real)}")
        genuine = True
        for ctrl in ["rewired", "degshuffle"]:
            null = d[d.condition == ctrl]["test_err"].to_numpy()
            beats = float((null > med).mean())
            p = float(mannwhitneyu(real, null, alternative="less").pvalue)
            ok = beats >= 0.95 and p < 0.05
            genuine &= ok
            log(f"  vs {ctrl:<10}: null median {np.median(null):5.1f}, best {null.min():5.1f}, "
                f"failures {int((null > FAIL).sum())}/{len(null)} | real beats {100 * beats:5.1f}% "
                f"of networks | Mann-Whitney p = {p:.4f} -> {'passes' if ok else 'does not pass'}")
        log(f"  VERDICT ({task}): " + ("GENUINE ADVANTAGE for the real wiring (pre-registered rule met)"
                                       if genuine else "no genuine advantage by the pre-registered rule"))
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
