import numpy as np, pandas as pd, sys
from ripser import ripser
sys.path.insert(0, "/home/claude/s4")
from sim_np import z, conds, fly, params
NET = "/home/claude/conn/networks"
def simulate(t, p, th, T=200):
    W = t["W"] * np.exp(p["g"])[:, None]; n = len(th)
    cue = np.c_[np.cos(th), np.sin(th), np.ones(n)]
    uc = (cue @ p["Wc"] + p["bc"]) * t["mER"] + p["bv"] * t["mPEN"]; ub = p["bc"] * t["mER"] + p["bv"] * t["mPEN"]
    h = np.zeros((n, t["n"])); H = []
    for s in range(T):
        u = uc if s < 5 else ub
        for _ in range(2): h = 0.5 * h + 0.5 * (np.maximum(h, 0) @ W + u)
        H.append(h.copy())
    return np.stack(H, 1), W
def decode(t, p, H):
    y = (np.maximum(H, 0) * t["mEPG"]) @ p["R"] + p["r"]; return np.arctan2(y[..., 1], y[..., 0])
def analyse(t, p, epg):
    th = np.linspace(-np.pi, np.pi, 64, endpoint=False); H, W = simulate(t, p, th)
    ang = decode(t, p, H); err200 = np.degrees(np.abs(np.angle(np.exp(1j * (ang[:, 5:] - th[:, None]))))).mean()
    X = np.maximum(H[:, 100, epg], 0); Xc = X - X.mean(0); s = np.linalg.svd(Xc, compute_uv=False)
    pc2 = (s[:2] ** 2).sum() / max((s ** 2).sum(), 1e-12)
    U = np.linalg.svd(Xc, full_matrices=False)[0][:, :2] * s[:2]
    phi = np.arctan2(U[:, 1], U[:, 0]); wind = np.sum(np.angle(np.exp(1j * np.diff(np.r_[phi, phi[0]])))) / (2 * np.pi)
    rad = np.linalg.norm(U, axis=1); rcv = rad.std() / max(rad.mean(), 1e-12)
    if np.abs(Xc).sum() > 1e-9:
        dg = ripser(Xc / (np.linalg.norm(Xc, axis=1).mean() + 1e-12), maxdim=1)["dgms"][1]
        pers = np.sort(dg[:, 1] - dg[:, 0])[::-1] if len(dg) else np.array([0.0])
        h1_top, h1_ratio = pers[0], pers[0] / (pers[1] if len(pers) > 1 and pers[1] > 0 else 1e-3)
    else: h1_top, h1_ratio = 0.0, 0.0
    N = t["n"]; mults = []
    for i in range(0, 64, 8):
        J1 = 0.5 * np.eye(N) + 0.5 * (W.T * (H[i, -1] > 0)[None, :]); mults.append(np.sort(np.abs(np.linalg.eigvals(J1 @ J1)))[::-1][:2])
    l1, l2 = np.mean(mults, 0); ratio = np.log(max(l2, 1e-12)) / np.log(min(l1, 1 - 1e-9))
    ring = pc2 >= 0.70 and h1_top > 0.3 and h1_ratio >= 3
    return dict(err200=err200, pc2=pc2, h1_top=h1_top, h1_ratio=h1_ratio, winding=wind, radius_cv=rcv, lam1=l1, lam2=l2,
                return_over_drift=ratio, ring=ring, fast_return=ratio >= 5, H4=ring and ratio >= 5)
