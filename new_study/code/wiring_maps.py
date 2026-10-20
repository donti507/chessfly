"""Neuron-level wiring maps: connection strength vs angular offset between pre and post neurons,
per pathway, with positions inferred from wiring (positions.py). ER ring neurons are not
direction-tuned, so pathways involving ER are compared at the type level only."""
import numpy as np, pandas as pd
from positions import load, infer_angles, anchored_angles
GROUPS = ["EPG", "PEN_a", "PEN_b", "PEG", "Delta7"]
NB = 8                                              # 45-degree bins of angular offset

def group_of(types):
    return types.str.extract(r"^(EPG|PEN_a|PEN_b|PEG|Delta7|ER)")[0].values

def positions(n, A):
    g = group_of(n.type); ang = np.full(len(n), np.nan)
    epg = np.flatnonzero(g == "EPG"); ang[epg] = infer_angles(A, epg)
    for t in GROUPS[1:]:
        rows = np.flatnonzero(g == t); ang[rows] = anchored_angles(A, rows, epg, ang[epg])[0]
    return g, ang

def maps(A, g, ang, reflect=False):
    a = -ang if reflect else ang
    out = {}
    for p in GROUPS:
        for q in GROUPS:
            i, j = np.flatnonzero(g == p), np.flatnonzero(g == q)
            W = A[np.ix_(i, j)]
            if W.sum() == 0: continue
            d = np.angle(np.exp(1j * (a[j][None, :] - a[i][:, None])))      # post minus pre angle
            b = np.clip(((d + np.pi) / (2 * np.pi) * NB).astype(int), 0, NB - 1)
            prof = np.bincount(b.ravel(), weights=W.ravel(), minlength=NB)
            out[(p, q)] = prof / prof.sum()
    return out

def type_shares(A, g):
    G = GROUPS + ["ER"]
    M = np.array([[A[np.ix_(g == x, g == y)].sum() for y in G] for x in G])
    return M / np.maximum(M.sum(0, keepdims=True), 1e-12)

def map_distance(m1, m2, s1, s2):
    """Total-variation distance between angular profiles, weighted by pathway importance."""
    G = GROUPS + ["ER"]; d, w = 0.0, 0.0
    for (p, q), v in m1.items():
        if (p, q) not in m2: continue
        wt = 0.5 * (s1[G.index(p), G.index(q)] + s2[G.index(p), G.index(q)])
        d += wt * 0.5 * np.abs(v - m2[(p, q)]).sum(); w += wt
    return d / w

def share_distance(s1, s2):
    return 0.5 * np.abs(s1 - s2).sum(0).mean()
