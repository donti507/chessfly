"""alt_metrics.py -- robustness (final version: direction-tuned neurons, ID matching within flies) of the difference decomposition to the choice of distance measure.
For each comparison, neurons are matched one-to-one (identity for same-order variants, otherwise the
same rank maps used for transplants), then three standard measures are computed:
  weight_dissim : 1 - Spearman correlation of input fractions (A_ij / sum_i A_ij) over the union of edges
  jaccard       : Jaccard distance between the two edge sets
  share_dist    : distance between cell-type pathway input shares (wiring_maps.share_distance)"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from positions import load
from transplant_maps import fly_data, align_rank
from wiring_maps import type_shares, share_distance, positions

COMPS = [("filtering", "mcns", "mcns_neuprint"), ("filtering", "fafb", "fafb_unfiltered"),
         ("proofreading", "mcns_v09", "mcns"), ("proofreading", "banc_v626", "banc"),
         ("synapse detector", "fafb", "fafb_buhmann"), ("synapse detector", "fafb_unfiltered", "fafb_buhmann"),
         ("between flies", "mcns", "hemibrain"), ("between flies", "hemibrain", "fafb"), ("between flies", "fafb", "banc"),
         ("between flies", "mcns", "fafb"), ("between flies", "hemibrain", "banc"), ("between flies", "mcns", "banc")]
BASE = {"mcns_neuprint": "mcns", "mcns_v09": "mcns", "fafb_unfiltered": "fafb", "fafb_buhmann": "fafb"}
MAPS = np.load("networks/transplant_maps.npz", allow_pickle=True)
cache = {}
def get(v):
    if v not in cache:
        n, A = load(v); cache[v] = (n, A)
    return cache[v]
def neuron_map(src, tgt):
    ns, _ = get(src); nt, _ = get(tgt)
    if len(ns) == len(nt) and (ns.bodyId.values == nt.bodyId.values).all(): return np.arange(len(nt))
    bs, bt = BASE.get(src, src), BASE.get(tgt, tgt)
    if bs != bt and f"{bs}__{bt}" in MAPS.files: return MAPS[f"{bs}__{bt}"]
    return align_rank(fly_data(src), fly_data(tgt))[1][2]          # e.g. banc_v626 -> banc
def infrac(A):
    s = A.sum(0, keepdims=True); return np.divide(A, s, out=np.zeros_like(A), where=s > 0)

def id_map(src, tgt):
    ns, _ = get(src); nt, _ = get(tgt); ids = {b: i for i, b in enumerate(ns.bodyId)}
    keep = np.array([i for i, b in enumerate(nt.bodyId) if b in ids]); return keep, np.array([ids[b] for b in nt.bodyId.values[keep]])
rows = []
for kind, a, b in COMPS:
    (na, Aa), (nb, Ab) = get(a), get(b)
    same_fly = BASE.get(a, a) == BASE.get(b, b) or (a, b) == ("banc_v626", "banc")
    if same_fly: ti, si = id_map(a, b)                     # within one fly: match by neuron ID
    else: ti, si = np.arange(len(nb)), neuron_map(a, b)    # between flies: validated rank maps
    tuned = ~nb.type.astype(str).str.startswith("ER").values[ti]; ti, si = ti[tuned], si[tuned]   # direction-tuned only
    A1, A2 = Aa[np.ix_(si, si)], Ab[np.ix_(ti, ti)]; F1, F2 = infrac(A1), infrac(A2); E = (A1 > 0) | (A2 > 0)
    ga, _ = positions(na, Aa); gb, _ = positions(nb, Ab)
    rows.append(dict(source=kind, comparison=f"{a} vs {b}", matching="neuron ID" if same_fly else "rank map", tuned_neurons=len(ti),
                     weight_dissim_tuned=1 - spearmanr(F1[E], F2[E]).correlation, jaccard_tuned=1 - ((A1 > 0) & (A2 > 0)).sum() / E.sum(),
                     share_dist=share_distance(type_shares(Aa, ga), type_shares(Ab, gb))))
R = pd.DataFrame(rows); R.to_csv("alt_metrics_robustness.csv", index=False); print(R.round(3).to_string(index=False))
