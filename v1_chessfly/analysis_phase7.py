"""
analysis_phase7.py -- mechanism analyses that need no GPU (Phases 7, 7c, 7g)

Rebuilds the exact networks of earlier Kaggle phases from their random seeds and measures
structural / dynamical properties, then relates them to the trained-network errors.

  phase 7   slow modes, non-normality, silent neurons, intrinsic memory  vs  Phase 6b errors
  phase 7c  connection symmetry, reciprocity, real eigenvalues           vs  Phase 7b errors
  phase 7g  cell-type pathway strengths (ER->EPG, EPG->D7)               vs  Phase 7e errors

Inputs (repo folder): subnetwork_adjacency.npz, subnetwork_neurons.csv, subnetwork_nt.csv,
                      phase6b_null_distribution.csv, phase7b_dose_response.csv, phase7e_replication.csv
Outputs:              phase7_mechanism.csv, phase7c_symmetry.csv, phase7g_landmark_drive.csv

Usage:  python analysis_phase7.py [7|7c|7g|all]      (CPU only, ~2 minutes)
"""

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

C = sp.load_npz("subnetwork_adjacency.npz").toarray().astype(np.float64)
TYPES = pd.read_csv("subnetwork_neurons.csv")["type"].astype(str).tolist()
SIGN = pd.read_csv("subnetwork_nt.csv").sort_values("index")["sign"].to_numpy().astype(float)
N = len(TYPES)
pick = lambda p: np.array([i for i, t in enumerate(TYPES) if t.startswith(p)])
PEN, ER, EPG, D7 = pick("PEN"), pick("ER"), pick("EPG"), pick("Delta7")


# ---------------------------------------------------------------- network builders (same code as Kaggle phases)
def rewire(W, seed):                                   # kaggle_phase2.rewire
    rng = np.random.default_rng(seed)
    Wr = np.empty_like(W)
    for i in range(W.shape[0]):
        Wr[i] = W[i, rng.permutation(W.shape[1])]
    return Wr


def degshuffle(C, seed, swaps_per_edge=10):            # kaggle_phase3b.degree_preserving_shuffle
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(C); w = C[pre, post].copy(); post = post.copy(); E = len(pre)
    edges = set(zip(pre.tolist(), post.tolist()))
    I, J = rng.integers(E, size=swaps_per_edge * E), rng.integers(E, size=swaps_per_edge * E)
    for i, j in zip(I.tolist(), J.tolist()):
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == c or b == d or (a, d) in edges or (c, b) in edges:
            continue
        edges.discard((a, b)); edges.discard((c, d)); edges.add((a, d)); edges.add((c, b)); post[i], post[j] = d, b
    out = np.zeros_like(C); out[pre, post] = w
    return out


def partial_shuffle(C, frac, seed, max_tries_per_edge=40):   # kaggle_phase7b.partial_degree_shuffle
    if frac <= 0:
        return C.copy()
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(C); w = C[pre, post].copy(); post = post.copy(); E = len(pre)
    original = set(zip(pre.tolist(), post.tolist())); edges = set(original)
    moved, tries = 0, 0
    while moved < frac * E and tries < max_tries_per_edge * E:
        i, j = rng.integers(E, size=2); tries += 1
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == c or b == d or (a, d) in edges or (c, b) in edges:
            continue
        moved -= ((a, b) not in original) + ((c, d) not in original)
        edges.discard((a, b)); edges.discard((c, d)); edges.add((a, d)); edges.add((c, b)); post[i], post[j] = d, b
        moved += ((a, d) not in original) + ((c, b) not in original)
    out = np.zeros_like(C); out[pre, post] = w
    return out


def targeted_shuffle(C, frac, seed, mode, max_tries_per_edge=60):   # kaggle_phase7d.targeted_shuffle
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(C); w = C[pre, post].copy(); post = post.copy(); E = len(pre)
    edges = set(zip(pre.tolist(), post.tolist())); original = set(edges)
    recip0 = np.array([(b, a) in edges for a, b in zip(pre, post)])
    pool = np.where(recip0 if mode == "target" else ~recip0)[0]
    moved, tries = 0, 0
    while moved < frac * E and tries < max_tries_per_edge * E:
        i, j = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]; tries += 1
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == c or b == d or (a, d) in edges or (c, b) in edges or (d, a) in edges or (b, c) in edges:
            continue
        moved -= ((a, b) not in original) + ((c, d) not in original)
        edges.discard((a, b)); edges.discard((c, d)); edges.add((a, d)); edges.add((c, b)); post[i], post[j] = d, b
        moved += ((a, d) not in original) + ((c, b) not in original)
    out = np.zeros_like(C); out[pre, post] = w
    return out


# ---------------------------------------------------------------- measures
def slow_modes(M):
    lam = np.abs(np.linalg.eigvals(M * SIGN[:, None]))
    return int((lam / lam.max() > 2 / 3).sum())


def symmetry_measures(M):
    W = M * SIGN[:, None]; E = M > 0
    lam = np.linalg.eigvals(W); top = lam[np.argsort(-np.abs(lam))][:20]
    return {"reciprocity": float((E & E.T).sum() / E.sum()),
            "symmetry": float((((W + W.T) / 2) ** 2).sum() / (W ** 2).sum()),
            "real_top20": float((np.abs(top.imag) < 1e-9).mean())}


def dynamics_measures(M, n_init=3, trials=400, T=40, cue=5, alpha=0.5, micro=2):
    """Untrained L2 network (all gains 1) on the hold task: silent neurons and intrinsic memory."""
    W0 = M * SIGN[:, None]; W = W0 * (0.9 / np.abs(np.linalg.eigvals(W0)).max())
    th = np.random.default_rng(0).uniform(-np.pi, np.pi, trials)
    cue_x = np.stack([np.cos(th), np.sin(th), np.ones(trials)], 1)
    out = {"silent_all": [], "silent_epg": [], "mem_late": []}
    for k in range(n_init):
        r = np.random.default_rng(100 + k)
        Wc, bc, bo = r.uniform(-.577, .577, (3, len(ER))), r.uniform(-.577, .577, len(ER)), r.uniform(-1, 1, len(PEN))
        h = np.zeros((trials, N))
        for t in range(T):
            U = np.zeros((trials, N)); U[:, PEN] = bo; U[:, ER] = cue_x @ Wc + bc if t < cue else bc
            for _ in range(micro):
                h = (1 - alpha) * h + alpha * (np.maximum(h, 0) @ W + U)
        act = np.maximum(h, 0)
        out["silent_all"].append(float((act.max(0) < 1e-6).mean()))
        out["silent_epg"].append(float((act[:, EPG].max(0) < 1e-6).mean()))
        Z = np.c_[act[:, EPG], np.ones(trials)]; y = np.c_[np.cos(th), np.sin(th)]; half = trials // 2
        Wd = np.linalg.solve(Z[:half].T @ Z[:half] + 1e-2 * np.eye(Z.shape[1]), Z[:half].T @ y[:half])
        p = Z[half:] @ Wd
        out["mem_late"].append(float(np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(p[:, 1], p[:, 0]) - th[half:]))))).mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


# ---------------------------------------------------------------- analyses
def phase7():
    res = pd.read_csv("phase6b_null_distribution.csv")
    err = res.groupby(["task", "condition", "net_seed"])["test_err"].median().unstack("task").reset_index()
    rows = []
    for _, r in err.iterrows():
        kind, s = r["condition"], int(r["net_seed"])
        M = C if kind == "real" else (rewire(C, s) if kind == "rewired" else degshuffle(C, s))
        m = {"condition": kind, "net_seed": s, "slow_modes": slow_modes(M), **dynamics_measures(M),
             "hold_err": r["hold"], "integrate_err": r["integrate"]}
        rows.append(m)
    df = pd.DataFrame(rows); df.to_csv("phase7_mechanism.csv", index=False)
    for m in ["slow_modes", "silent_epg", "mem_late"]:
        print(f"  {m:<11} vs integrate: rho = {spearmanr(df[m], df.integrate_err).correlation:+.2f}")


def phase7c():
    res = pd.read_csv("phase7b_dose_response.csv")
    rows = []
    for (cond, net), g in res.groupby(["condition", "net"]):
        f = float(g.frac.iloc[0])
        if cond == "real" and net > 0:
            continue
        M = C if f == 0 else partial_shuffle(C, f, int(1000 * f) + int(net))
        m = {"condition": cond, "frac": f, "net": net, "slow_modes": slow_modes(M), **symmetry_measures(M)}
        for task in ["hold", "integrate"]:
            d = res[(res.condition == cond) & (res.task == task)]
            m[task] = float(d.test_err.median()) if cond == "real" else float(d[d.net == net].test_err.iloc[0])
        rows.append(m)
    df = pd.DataFrame(rows); df.to_csv("phase7c_symmetry.csv", index=False)
    print(f"  symmetry vs integrate: rho = {spearmanr(df.symmetry, df.integrate).correlation:+.2f}")


def phase7g():
    res = pd.read_csv("phase7e_replication.csv")
    er0, d70 = C[np.ix_(ER, EPG)].sum(), C[np.ix_(EPG, D7)].sum()
    rows = []
    for li, f in enumerate([0.05, 0.1, 0.2]):
        for mode, base in [("target", 5000), ("control", 6000)]:
            for k in range(5):
                M = targeted_shuffle(C, f, base + 100 * li + k, mode)
                r = res[(res.condition == mode) & np.isclose(res.frac, f) & (res.net == k)]
                rows.append({"mode": mode, "frac": f, "net": k, "er_epg": M[np.ix_(ER, EPG)].sum() / er0,
                             "epg_d7": M[np.ix_(EPG, D7)].sum() / d70,
                             "hold": float(r[r.task == "hold"].test_err.iloc[0]),
                             "integrate": float(r[r.task == "integrate"].test_err.iloc[0])})
    real = res[res.condition == "real"]
    rows.append({"mode": "real", "frac": 0, "net": 0, "er_epg": 1.0, "epg_d7": 1.0,
                 "hold": real[real.task == "hold"].test_err.median(), "integrate": real[real.task == "integrate"].test_err.median()})
    df = pd.DataFrame(rows); df.to_csv("phase7g_landmark_drive.csv", index=False)
    print(f"  ER->EPG strength vs hold: rho = {spearmanr(df.er_epg, df.hold).correlation:+.2f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("7", phase7), ("7c", phase7c), ("7g", phase7g)]:
        if which in (name, "all"):
            print(f"Phase {name}:"); fn()
