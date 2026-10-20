"""distance_decomposition.py -- reproduces difference_decomposition.csv (Fig 2A).

Each wiring version: positions inferred from wiring (positions.py), angular wiring maps per pathway
(wiring_maps.py; ER pathways excluded because ER neurons are not direction-tuned). Each version's
maps are oriented once, choosing the mirror (reflect or not) that best matches the male CNS v1.0
reference maps; pairwise distances are then computed between these oriented maps. Distances are
total-variation distances between angular profiles, weighted by the mean of the two networks'
pathway shares (shares include ER rows/columns, but ER pathways contribute no angular profile).
Run from the folder that contains networks/."""
import numpy as np, pandas as pd
from wiring_maps import positions, maps, type_shares, map_distance
from positions import load

VERSIONS = ["mcns", "mcns_neuprint", "mcns_v09", "hemibrain", "fafb", "fafb_unfiltered", "fafb_buhmann", "banc", "banc_v626"]
F = {}
for v in VERSIONS:
    n, A = load(v); g, ang = positions(n, A); F[v] = dict(A=A, g=g, ang=ang, shares=type_shares(A, g))
ref = maps(F["mcns"]["A"], F["mcns"]["g"], F["mcns"]["ang"])
for f in F.values():
    cands = [maps(f["A"], f["g"], f["ang"], r) for r in (False, True)]
    f["maps"] = min(cands, key=lambda m: map_distance(ref, m, F["mcns"]["shares"], f["shares"]))
D = lambda a, b: map_distance(F[a]["maps"], F[b]["maps"], F[a]["shares"], F[b]["shares"])
rows = [("filtering", "same fly", "male CNS Codex vs neuPrint", D("mcns", "mcns_neuprint")),
        ("filtering", "same fly", "FlyWire filtered vs unfiltered", D("fafb", "fafb_unfiltered")),
        ("proofreading", "same fly", "male CNS v0.9 vs v1.0", D("mcns_v09", "mcns")),
        ("proofreading", "same fly", "BANC v626 vs v888", D("banc_v626", "banc")),
        ("synapse detector", "same fly", "FlyWire Princeton vs Buhmann", D("fafb", "fafb_buhmann")),
        ("synapse detector", "same fly", "FlyWire unfiltered vs Buhmann", D("fafb_unfiltered", "fafb_buhmann")),
        ("between flies", "same pipeline family", "male CNS vs hemibrain", D("mcns", "hemibrain")),
        ("between flies", "different pipelines", "hemibrain vs FlyWire", D("hemibrain", "fafb")),
        ("between flies", "different pipelines", "FlyWire vs BANC", D("fafb", "banc")),
        ("between flies", "different pipelines", "male CNS vs FlyWire", D("mcns", "fafb")),
        ("between flies", "different pipelines", "hemibrain vs BANC", D("hemibrain", "banc")),
        ("between flies", "different pipelines", "male CNS vs BANC", D("mcns", "banc"))]
out = pd.DataFrame(rows, columns=["source", "kind", "comparison", "map_distance"])
out.to_csv("difference_decomposition_recomputed.csv", index=False)
print(out.round(3).to_string(index=False))
