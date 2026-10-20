"""Mechanism analysis of trained pilot networks (CPU re-simulation), per preregistration H4."""
import numpy as np, pandas as pd
from ripser import ripser
z = np.load("pilot_params.npz", allow_pickle=True)
conds = [c.split("|") for c in z["conds"]]; tid = z["type_id"]; ER, PEN, EPG = z["ER"], z["PEN"], z["EPG"]
N = z["W0"].shape[1]; mER = np.zeros(N); mER[ER] = 1; mPEN = np.zeros(N); mPEN[PEN] = 1; mEPG = np.zeros(N); mEPG[EPG] = 1
CUE, MICRO = 5, 2
types = pd.read_csv("../conn/networks/mcns/subnetwork_neurons.csv").type.values
D7 = np.flatnonzero(pd.Series(types).str.startswith("Delta7").values)

def weights(k, silence=None):
    g = z["g_type"][k][tid] if conds[k][1] == "celltype" else z["g_neuron"][k]
    W = z["W0"][k] * np.exp(g)[:, None]
    if silence is not None: W = W.copy(); W[silence, :] = 0
    return W

def simulate(k, th, T, W, record=False):
    cue = np.c_[np.cos(th), np.sin(th), np.ones(len(th))]
    u_cue = (cue @ z["Wc"][k] + z["bc"][k]) * mER + z["bv"][k] * mPEN
    u_base = z["bc"][k] * mER + z["bv"][k] * mPEN
    h = np.zeros((len(th), N)); hs, ys = [], []
    for t in range(T):
        u = u_cue if t < CUE else u_base
        for _ in range(MICRO): h = 0.5 * h + 0.5 * (np.maximum(h, 0) @ W + u)
        y = (np.maximum(h, 0) * mEPG) @ z["R"][k] + z["r"][k]; ys.append(np.arctan2(y[:, 1], y[:, 0]))
        if record: hs.append(h.copy())
    return np.array(ys).T, (np.stack(hs, 1) if record else h), u_base

def err(ang, th):
    return np.degrees(np.abs(np.angle(np.exp(1j * (ang[:, CUE:] - th[:, None]))))).mean()

def jac_multipliers(h, W):
    """Eigenvalue moduli of the 2-microstep map's Jacobian at state h (column convention)."""
    J1 = 0.5 * np.eye(N) + 0.5 * (W.T * (h > 0)[None, :])
    return np.sort(np.abs(np.linalg.eigvals(J1 @ J1)))[::-1]

def analyse(k, silence=None):
    W = weights(k, silence); th = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    ang, H, ub = simulate(k, th, 200, W, record=True)
    e200 = err(ang, th)
    X = np.maximum(H[:, 100, EPG], 0)                          # EPG rates 95 steps after the cue
    Xc = X - X.mean(0); s = np.linalg.svd(Xc, compute_uv=False); pc2 = (s[:2] ** 2).sum() / max((s ** 2).sum(), 1e-12)
    if pc2 > 0 and np.abs(Xc).sum() > 1e-9:
        dg = ripser(Xc / (np.linalg.norm(Xc, axis=1).mean() + 1e-12), maxdim=1)["dgms"][1]
        pers = np.sort(dg[:, 1] - dg[:, 0])[::-1] if len(dg) else np.array([0.0])
        h1_ratio = pers[0] / (pers[1] if len(pers) > 1 and pers[1] > 0 else 1e-3)
        h1_top = pers[0]
    else:
        h1_ratio, h1_top = 0.0, 0.0
    # linear stability at settled states: slowest multiplier vs next (return vs drift)
    m = np.array([jac_multipliers(H[i, -1], W)[:2] for i in range(0, 64, 8)])
    lam1, lam2 = m[:, 0].mean(), m[:, 1].mean()
    ratio = np.log(max(lam2, 1e-12)) / np.log(min(lam1, 1 - 1e-9))   # return rate / drift rate
    ring = pc2 >= 0.70 and h1_top > 0.3 and h1_ratio >= 3
    return dict(err200=e200, pc2=pc2, h1_top=h1_top, h1_ratio=h1_ratio, lam1=lam1, lam2=lam2,
                return_over_drift=ratio, ring=ring, fast_return=ratio >= 5)
