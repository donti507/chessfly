"""Neuron correspondences between flies for the wiring transplant (preregistration, Section 4).
Each target neuron is matched to the source neuron of the same family with the nearest inferred
compass angle, after aligning the two flies' angle frames (rotation + mirror) by maximizing the
correlation of their connectivity under the map. No task performance is used for alignment."""
import numpy as np, pandas as pd, re
from positions import load, infer_angles, anchored_angles
FLIES = ["mcns", "hemibrain", "fafb", "banc"]

def family(t):
    t = str(t)
    for pat, fam in [(r"^PEN_a", "PEN_a"), (r"^PEN_b", "PEN_b"), (r"^EPGt", "EPGt"), (r"^EPG", "EPG"),
                     (r"^PEG", "PEG"), (r"^Delta7", "Delta7")]:
        if re.match(pat, t): return fam
    m = re.match(r"^(ER\d[a-z]?)", t)
    return m.group(1) if m else t

def fly_data(name):
    n, A = load(name); fam = n.type.map(family).values
    ang = np.full(len(n), np.nan); epg = np.flatnonzero(np.isin(fam, ["EPG", "EPGt"]))
    ang[epg] = infer_angles(A, epg)
    others = np.flatnonzero(~np.isin(fam, ["EPG", "EPGt"]))
    ang[others], conc = anchored_angles(A, others, epg, ang[epg])
    sharp = np.full(len(n), 1.0); sharp[others] = conc
    return dict(n=n, A=A, fam=fam, ang=ang, sharp=sharp)

def nearest_map(src, tgt, rot, refl):
    a_t = (-tgt["ang"] if refl else tgt["ang"]) + rot
    m = np.empty(len(tgt["fam"]), int)
    for f in np.unique(tgt["fam"]):
        it, is_ = np.flatnonzero(tgt["fam"] == f), np.flatnonzero(src["fam"] == f)
        d = np.abs(np.angle(np.exp(1j * (a_t[it][:, None] - src["ang"][is_][None, :]))))
        m[it] = is_[d.argmin(1)]
    return m

def align(src, tgt, n_rot=72):
    best = (-np.inf, None)
    Lt = np.log1p(tgt["A"])
    tuned = np.flatnonzero(~pd.Series(tgt["fam"]).str.startswith("ER").values)   # align on direction-tuned neurons
    for refl in (False, True):
        for rot in np.linspace(0, 2 * np.pi, n_rot, endpoint=False):
            m = nearest_map(src, tgt, rot, refl)
            Ls = np.log1p(src["A"][np.ix_(m[tuned], m[tuned])]); r = np.corrcoef(Ls.ravel(), Lt[np.ix_(tuned, tuned)].ravel())[0, 1]
            if r > best[0]: best = (r, (rot, refl, m))
    return best[0], best[1]

def infer_sides(d):
    """Left/right from wiring: PEN shift direction (output vs input angle), then partners' majority."""
    A, fam, ang = d["A"], d["fam"], d["ang"]
    epg = np.flatnonzero(np.isin(fam, ["EPG", "EPGt"])); side = np.zeros(len(fam))
    pen = np.flatnonzero(np.isin(fam, ["PEN_a", "PEN_b"]))
    z_in = (A[np.ix_(epg, pen)] * np.exp(1j * ang[epg])[:, None]).sum(0)     # EPG -> PEN (input side, PB)
    z_out = (A[np.ix_(pen, epg)] * np.exp(1j * ang[epg])[None, :]).sum(1)    # PEN -> EPG (output side, EB)
    side[pen] = np.sign(np.angle(z_out / z_in))
    # everyone else: weighted majority of PEN partners (both directions)
    rest = np.setdiff1d(np.arange(len(fam)), pen)
    W = A[np.ix_(rest, pen)] + A[np.ix_(pen, rest)].T
    side[rest] = np.sign(W @ side[pen])
    return side

def nearest_map_sided(src, tgt, rot, refl):
    a_t = (-tgt["ang"] if refl else tgt["ang"]) + rot
    s_t = -tgt["side"] if refl else tgt["side"]          # mirroring swaps left and right
    m = np.empty(len(tgt["fam"]), int)
    for f in np.unique(tgt["fam"]):
        for s in (-1, 0, 1):
            it = np.flatnonzero((tgt["fam"] == f) & (s_t == s))
            if not len(it): continue
            is_ = np.flatnonzero((src["fam"] == f) & (src["side"] == s))
            if not len(is_): is_ = np.flatnonzero(src["fam"] == f)
            d = np.abs(np.angle(np.exp(1j * (a_t[it][:, None] - src["ang"][is_][None, :]))))
            m[it] = is_[d.argmin(1)]
    return m

def align_sided(src, tgt, n_rot=72):
    best = (-np.inf, None); Lt = np.log1p(tgt["A"])
    tuned = np.flatnonzero(~pd.Series(tgt["fam"]).str.startswith("ER").values)
    for refl in (False, True):
        for rot in np.linspace(0, 2 * np.pi, n_rot, endpoint=False):
            m = nearest_map_sided(src, tgt, rot, refl)
            r = np.corrcoef(np.log1p(src["A"][np.ix_(m[tuned], m[tuned])]).ravel(), Lt[np.ix_(tuned, tuned)].ravel())[0, 1]
            if r > best[0]: best = (r, (rot, refl, m))
    return best

def rank_map(src, tgt, rot, refl):
    """One-to-one (as far as counts allow): within each family, match by rank of angle around the
    ring, choosing the circular offset that minimizes angular mismatch."""
    a_t = (-tgt["ang"] if refl else tgt["ang"]) + rot
    m = np.empty(len(tgt["fam"]), int)
    for f in np.unique(tgt["fam"]):
        it, is_ = np.flatnonzero(tgt["fam"] == f), np.flatnonzero(src["fam"] == f)
        ot, os_ = it[np.argsort(np.mod(a_t[it], 2 * np.pi))], is_[np.argsort(np.mod(src["ang"][is_], 2 * np.pi))]
        nt, ns = len(ot), len(os_)
        best = (np.inf, None)
        for off in range(ns):
            j = (np.floor(np.arange(nt) * ns / nt).astype(int) + off) % ns
            cost = np.abs(np.angle(np.exp(1j * (a_t[ot] - src["ang"][os_[j]])))).sum()
            if cost < best[0]: best = (cost, j)
        m[ot] = os_[best[1]]
    return m

def align_rank(src, tgt, n_rot=72):
    best = (-np.inf, None); Lt = np.log1p(tgt["A"])
    tuned = np.flatnonzero(~pd.Series(tgt["fam"]).str.startswith("ER").values)
    for refl in (False, True):
        for rot in np.linspace(0, 2 * np.pi, n_rot, endpoint=False):
            m = rank_map(src, tgt, rot, refl)
            r = np.corrcoef(np.log1p(src["A"][np.ix_(m[tuned], m[tuned])]).ravel(), Lt[np.ix_(tuned, tuned)].ravel())[0, 1]
            if r > best[0]: best = (r, (rot, refl, m))
    return best
