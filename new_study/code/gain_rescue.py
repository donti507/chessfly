import numpy as np, pandas as pd, scipy.sparse as sp, sys
sys.path.insert(0, "/home/claude/conn")
z = np.load("/home/claude/s4c/stage4c_params.npz", allow_pickle=True)
conds = [c.split("|") for c in z["conds"]]                  # fly|gains|seed|level
MAPS = np.load("/home/claude/conn/networks/transplant_maps.npz", allow_pickle=True)
NET = "/home/claude/conn/networks"
def fly(f):
    meta = pd.read_csv(f"{NET}/{f}/subnetwork_neurons.csv"); C = sp.load_npz(f"{NET}/{f}/subnetwork_adjacency.npz").toarray().astype(float)
    s = pd.read_csv(f"{NET}/{f}/subnetwork_nt.csv").sort_values("index").sign.values.astype(float)
    S = C * s[:, None]; S *= 0.9 / np.abs(np.linalg.eigvals(S)).max()
    g = meta.type.astype(str).str.extract(r"^(EPG|PEN|PEG|Delta7|ER)")[0].values
    return dict(W=S, n=len(meta), mER=(g == "ER") * 1., mPEN=(g == "PEN") * 1., mEPG=(g == "EPG") * 1.)
def params(k, n_src, m):
    return dict(g=z["g_neuron"][k][:n_src][m], Wc=z["Wc"][k][:, m], bc=z["bc"][k][m], bv=z["bv"][k][m], R=z["R"][k][m], r=z["r"][k])
def hold_err(t, p, c=1.0, n=150, seed=999, T=40):
    th = np.random.default_rng(seed).uniform(-np.pi, np.pi, n); W = c * t["W"] * np.exp(p["g"])[:, None]
    cue = np.c_[np.cos(th), np.sin(th), np.ones(n)]
    uc = (cue @ p["Wc"] + p["bc"]) * t["mER"] + p["bv"] * t["mPEN"]; ub = p["bc"] * t["mER"] + p["bv"] * t["mPEN"]
    h = np.zeros((n, t["n"])); e = []
    for s in range(T):
        u = uc if s < 5 else ub
        for _ in range(2): h = 0.5 * h + 0.5 * (np.maximum(h, 0) @ W + u)
        if s >= 5:
            y = (np.maximum(h, 0) * t["mEPG"]) @ p["R"] + p["r"]; e.append(np.abs(np.angle(np.exp(1j * (np.arctan2(y[:, 1], y[:, 0]) - th)))))
    return np.degrees(np.mean(e))
