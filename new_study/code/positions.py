"""Infer each neuron's compass position from wiring (classical MDS on connectivity profiles)."""
import pandas as pd, numpy as np, scipy.sparse as sp, re
def load(name):
    n = pd.read_csv(f"networks/{name}/subnetwork_neurons.csv"); A = sp.load_npz(f"networks/{name}/subnetwork_adjacency.npz").toarray()
    return n, A
def infer_angles(A, rows):
    X = np.c_[A[rows], A[:, rows].T].astype(float)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T; Sc = S - S.mean(0) - S.mean(1)[:, None] + S.mean()
    w, V = np.linalg.eigh(Sc); V = V[:, ::-1][:, :2]
    return np.arctan2(V[:, 1], V[:, 0])
def label_glom(instance):
    m = re.search(r"_([LR])(\d)", str(instance)); return (m.group(1), int(m.group(2))) if m else (None, None)
def agreement(ang, side, glom):
    out = {}
    for s in ("L", "R"):
        m = side == s
        if m.sum() < 4: continue
        la = (glom[m] - 1) * 2 * np.pi / 8
        out[s] = max(abs(np.mean(np.exp(1j * (ang[m] - k * la)))) for k in (1, -1))
    return out

def anchored_angles(A, rows, epg_rows, epg_ang):
    """Position of other neurons = weighted circular mean of the EPG positions they connect with."""
    W = A[np.ix_(rows, epg_rows)] + A[np.ix_(epg_rows, rows)].T
    z = (W * np.exp(1j * epg_ang)[None, :]).sum(1)
    return np.angle(z), np.abs(z) / np.maximum(W.sum(1), 1e-12)
