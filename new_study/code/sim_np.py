import numpy as np, pandas as pd, scipy.sparse as sp
z = np.load("/home/claude/s4/stage4_params.npz", allow_pickle=True)
conds = [c.split("|") for c in z["conds"]]
NET = "/home/claude/conn/networks"
def fly(f, C=None):
    meta = pd.read_csv(f"{NET}/{f}/subnetwork_neurons.csv")
    if C is None: C = sp.load_npz(f"{NET}/{f}/subnetwork_adjacency.npz").toarray().astype(float)
    sign = pd.read_csv(f"{NET}/{f}/subnetwork_nt.csv").sort_values("index").sign.values.astype(float)
    S = C * sign[:, None]; S = S * (0.9 / np.abs(np.linalg.eigvals(S)).max())
    g = meta.type.astype(str).str.extract(r"^(EPG|PEN|PEG|Delta7|ER)")[0].values
    return dict(W=S, n=len(meta), tid=pd.factorize(meta.type.astype(str))[0], mER=(g == "ER") * 1.0, mPEN=(g == "PEN") * 1.0, mEPG=(g == "EPG") * 1.0)
def params(k, src, m):
    """per-neuron parameters of network k (trained on src) carried to a target through map m (target -> source index)."""
    g = z["g_type"][k][src["tid"]] if conds[k][1] == "celltype" else z["g_neuron"][k][:src["n"]]
    return dict(g=g[m], Wc=z["Wc"][k][:, m], bc=z["bc"][k][m], bv=z["bv"][k][m], R=z["R"][k][m], r=z["r"][k])
def hold_error(tgt, p, n=500, T=40, seed=1001):
    th = np.random.default_rng(seed).uniform(-np.pi, np.pi, n)
    W = tgt["W"] * np.exp(p["g"])[:, None]
    cue = np.c_[np.cos(th), np.sin(th), np.ones(n)]
    u_cue = (cue @ p["Wc"] + p["bc"]) * tgt["mER"] + p["bv"] * tgt["mPEN"]; u_b = p["bc"] * tgt["mER"] + p["bv"] * tgt["mPEN"]
    h = np.zeros((n, tgt["n"])); errs = []
    for t in range(T):
        u = u_cue if t < 5 else u_b
        for _ in range(2): h = 0.5 * h + 0.5 * (np.maximum(h, 0) @ W + u)
        if t >= 5:
            y = (np.maximum(h, 0) * tgt["mEPG"]) @ p["R"] + p["r"]
            errs.append(np.abs(np.angle(np.exp(1j * (np.arctan2(y[:, 1], y[:, 0]) - th)))))
    return np.degrees(np.mean(errs))
