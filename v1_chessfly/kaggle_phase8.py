"""
kaggle_phase8.py -- Phase 8: do the findings generalize to an INDEPENDENT connectome?

All results so far come from one fly (male CNS). This pulls the same head-direction cell
types from the hemibrain (hemibrain:v1.2.1, a different fly and reconstruction) and
reruns the key experiments with the same model (L2, anatomical ports, 1000 runs).
Signs are assigned by cell type, matching the male CNS predictions:
EPG / PEN / PEG excitatory; Delta7 and ER ring neurons inhibitory.

Conditions:
  real          hemibrain wiring (5 training seeds)
  degshuffle    10 degree-preserving shuffles
  recip20       20% of edges rewired, reciprocal edges only     (symmetry falls)
  oneway20      20% of edges rewired, one-way edges only        (symmetry kept)
  scr_ER_EPG    landmark map scrambled within its pathway
  scr_ER_ER     ring-neuron pathway scrambled (mostly reciprocal)

Pre-registered replication tests (male CNS findings; one-sided Mann-Whitney, p < 0.05):
  R1 hold:      oneway20 worse than recip20
  R2 hold:      scr_ER_EPG NOT worse than real (p >= 0.05)
  R3 integrate: degshuffle better than real
  R4 hold:      within oneway20 + recip20, lower ER->EPG strength goes with higher error
                (Spearman rho < 0, p < 0.05)

Needs the NEUPRINT_TOKEN Kaggle secret (the hemibrain is public). The pulled network is
saved to chessfly_out/ so later runs skip the download.

Usage:  python kaggle_phase8.py    (~100 min on a T4; saves CSV after every run)
Env:    NULLS (10)  NETS (5)  REAL_SEEDS (5)  DATASET (hemibrain:v1.2.1)
"""

import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu, spearmanr

import kaggle_phase2 as p2
import kaggle_phase3b as p3b
import kaggle_phase5 as p5
import kaggle_phase6 as p6
from kaggle_phase7d import targeted_shuffle, symmetry
from kaggle_phase7f import scramble

DATASET = os.environ.get("DATASET", "hemibrain:v1.2.1")
NULLS = int(os.environ.get("NULLS", "10"))
NETS = int(os.environ.get("NETS", "5"))
REAL_SEEDS = int(os.environ.get("REAL_SEEDS", "5"))
TYPE_REGEX = "(EPG.*|PEG.*|PEN.*|Delta7.*|ER[1-6].*)"
LEVEL, SIZE, TASKS = "L2", 1000, ["hold", "integrate"]
OUT = p2.OUT_DIR
log = p2.log
TAG = DATASET.split(":")[0]


def pull_network():
    adj_path, meta_path = os.path.join(OUT, f"{TAG}_adjacency.npz"), os.path.join(OUT, f"{TAG}_neurons.csv")
    if os.path.exists(adj_path) and os.path.exists(meta_path):
        log(f"Using saved {DATASET} network")
        return sp.load_npz(adj_path).tocoo(), pd.read_csv(meta_path)
    from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit("NEUPRINT_TOKEN missing: add it under Add-ons -> Secrets and run the token cell.")
    client = Client("neuprint.janelia.org", dataset=DATASET, token=token)
    crit = NC(type=TYPE_REGEX, regex=True)
    neurons, _ = fetch_neurons(crit, client=client)
    neurons = neurons.sort_values(["type", "bodyId"]).reset_index(drop=True)
    _, conn = fetch_adjacencies(crit, crit, min_total_weight=3, client=client)
    conn = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    idx = {b: i for i, b in enumerate(neurons.bodyId)}
    conn = conn[conn.bodyId_pre.isin(idx) & conn.bodyId_post.isin(idx)]
    adj = sp.coo_matrix((conn.weight.to_numpy(float), (conn.bodyId_pre.map(idx), conn.bodyId_post.map(idx))),
                        shape=(len(neurons), len(neurons)))
    sp.save_npz(adj_path, adj.tocsr())
    neurons[["bodyId", "type"]].to_csv(meta_path, index=False)
    return adj, neurons[["bodyId", "type"]]


def main():
    log("=" * 78)
    log(f"CHESSFLY PHASE 8 -- generality: core experiments in an independent connectome ({DATASET})")
    log("=" * 78)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj, meta = pull_network()
    types = meta["type"].astype(str).tolist()
    n = len(types)
    signs = np.array([-1.0 if (t.startswith("Delta7") or t.startswith("ER")) else 1.0 for t in types])
    type_id = np.unique(types, return_inverse=True)[1]
    C = np.zeros((n, n), dtype=np.float32)
    C[adj.row, adj.col] = adj.data
    pick = lambda prefix: np.array([i for i, t in enumerate(types) if t.startswith(prefix)])
    PEN, ER, EPG = pick("PEN"), pick("ER"), pick("EPG")
    E = C > 0
    log(f"Network: {n} neurons, {int(E.sum())} connections, {int((signs < 0).sum())} inhibitory, "
        f"{len(np.unique(types))} cell types | PEN {len(PEN)}, ER {len(ER)}, EPG {len(EPG)}")
    log(f"Reciprocated edges {100 * (E & E.T).sum() / E.sum():.0f}% | symmetry {symmetry(C, signs):.3f} "
        f"(male CNS: 60%, 0.813)")
    if min(len(PEN), len(ER), len(EPG)) == 0:
        raise SystemExit("A port cell type is missing in this dataset; cannot run the anatomical design.")
    ports = (PEN, ER, EPG)
    er_epg0 = C[np.ix_(ER, EPG)].sum()

    nets = [("real", s, C) for s in range(REAL_SEEDS)]
    nets += [("degshuffle", k, p3b.degree_preserving_shuffle(C, seed=8000 + k)) for k in range(NULLS)]
    nets += [("recip20", k, targeted_shuffle(C, 0.2, 8100 + k, "target")[0]) for k in range(NETS)]
    nets += [("oneway20", k, targeted_shuffle(C, 0.2, 8200 + k, "control")[0]) for k in range(NETS)]
    nets += [("scr_ER_EPG", k, scramble(C, ER, EPG, 8300 + k)) for k in range(NETS)]
    nets += [("scr_ER_ER", k, scramble(C, ER, ER, 8400 + k)) for k in range(NETS)]
    log(f"Built {len(nets)} networks")

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
    path = os.path.join(OUT, f"phase8_{TAG}.csv")
    for task in TASKS:
        for name, k, Cnet in nets:
            te, le = run(task, k if name == "real" else 0, Cnet)
            rows.append({"task": task, "condition": name, "net": k, "symmetry": symmetry(Cnet, signs),
                         "er_epg": float(Cnet[np.ix_(ER, EPG)].sum() / er_epg0), "test_err": te, "long_err": le})
            pd.DataFrame(rows).to_csv(path, index=False)
            el = time.time() - t0
            log(f"  [{len(rows):>2}/{total}] {task:<9} {name:<11} net {k}  error {te:5.1f}  2x {le:5.1f} deg  "
                f"[{el / 60:5.1f} min, eta {el / len(rows) * (total - len(rows)) / 60:5.1f}]")
    df = pd.DataFrame(rows)

    log("\n" + "=" * 78)
    log(f"RESULTS in {DATASET} (median test error, degrees; lower = better)")
    log("=" * 78)
    log(f"{'condition':<12}{'symmetry':>10}{'ER->EPG':>9}{'hold':>9}{'integrate':>11}   male CNS (hold / integrate)")
    ref = {"real": "1.7 / 55.2", "degshuffle": "5.4 / 17.7", "recip20": "1.5 / 38.6", "oneway20": "18.6 / 41.1",
           "scr_ER_EPG": "2.2 / 56.4", "scr_ER_ER": "1.7 / 34.4"}
    for name in ref:
        d = df[df.condition == name]
        log(f"{name:<12}{d.symmetry.mean():>10.3f}{d.er_epg.mean():>9.2f}{d[d.task == 'hold'].test_err.median():>9.1f}"
            f"{d[d.task == 'integrate'].test_err.median():>11.1f}   {ref[name]}")

    err = lambda task, c: df[(df.task == task) & (df.condition == c)].test_err.to_numpy()
    tests = {}
    tests["R1 hold: one-way rewiring worse than reciprocal"] = \
        mannwhitneyu(err("hold", "oneway20"), err("hold", "recip20"), alternative="greater").pvalue < 0.05
    tests["R2 hold: landmark-map scramble NOT worse than real"] = \
        mannwhitneyu(err("hold", "scr_ER_EPG"), err("hold", "real"), alternative="greater").pvalue >= 0.05
    tests["R3 integrate: degree-preserving shuffles better than real"] = \
        mannwhitneyu(err("integrate", "degshuffle"), err("integrate", "real"), alternative="less").pvalue < 0.05
    d = df[(df.task == "hold") & df.condition.isin(["oneway20", "recip20"])]
    rho = spearmanr(d.er_epg, d.test_err)
    tests["R4 hold: less ER->EPG strength, more error"] = rho.correlation < 0 and rho.pvalue < 0.05
    log("\nREPLICATION TESTS")
    for k, v in tests.items():
        log(f"  {k:<58} {'REPLICATED' if v else 'not replicated'}")
    log(f"  (R4 rho = {rho.correlation:+.2f}, p = {rho.pvalue:.3g})")
    log(f"VERDICT: {sum(tests.values())}/{len(tests)} core findings replicated in {DATASET}")
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
