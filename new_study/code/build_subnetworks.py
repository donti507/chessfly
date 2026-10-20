"""Build head-direction subnetworks (EPG, PEG, PEN, Delta7, ER1-6) from FlyWire Codex downloads,
in the same format as the original ChessFly network files. Signs by cell type (Delta7, ER inhibitory)."""
import pandas as pd, numpy as np, scipy.sparse as sp, re, os, sys
R = "raw/"
PAT = re.compile(r"^(EPG|PEG|PEN|Delta7|ER[1-6])")
THR = int(sys.argv[1]) if len(sys.argv) > 1 else 3

def types_for(name):
    if name == "fafb":
        t = pd.read_csv(R + "fafb_cell_types.csv.gz").rename(columns={"root_id": "id", "primary_type": "type"})
    else:
        t = pd.read_csv(R + f"{name}_neurons.csv.gz", usecols=["Root ID", "Primary Cell Type", "Soma side"])
        t = t.rename(columns={"Root ID": "id", "Primary Cell Type": "type", "Soma side": "side"})
    t = t[t.type.astype(str).str.match(PAT)].copy()
    return t.sort_values(["type", "id"]).reset_index(drop=True)

def build(name, conn_file):
    t = types_for(name); ids = set(t.id)
    rows = []
    for chunk in pd.read_csv(R + conn_file, usecols=["pre_root_id", "post_root_id", "syn_count"], chunksize=2_000_000):
        rows.append(chunk[chunk.pre_root_id.isin(ids) & chunk.post_root_id.isin(ids)])
    C = pd.concat(rows).groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum()
    C = C[(C.syn_count >= THR) & (C.pre_root_id != C.post_root_id)]
    idx = {b: i for i, b in enumerate(t.id)}
    n = len(t)
    A = sp.coo_matrix((C.syn_count.values.astype(float), (C.pre_root_id.map(idx), C.post_root_id.map(idx))), shape=(n, n)).tocsr()
    out = f"networks/{name}"; os.makedirs(out, exist_ok=True)
    sp.save_npz(f"{out}/subnetwork_adjacency.npz", A)
    t.rename(columns={"id": "bodyId"}).to_csv(f"{out}/subnetwork_neurons.csv", index=False)
    sign = np.where(t.type.str.startswith(("Delta7", "ER")), -1, 1)
    pd.DataFrame({"index": range(n), "sign": sign}).to_csv(f"{out}/subnetwork_nt.csv", index=False)
    E = A > 0
    rec = (E.multiply(E.T)).sum() / E.sum()
    grp = t.type.str.extract(r"^(EPG|PEG|PEN|Delta7|ER)")[0].value_counts().to_dict()
    print(f"{name:5s}: {n} neurons {grp} | {E.sum()} connections (>= {THR} syn) | {int((sign<0).sum())} inhibitory | reciprocal {rec:.1%} | synapses {int(A.sum())}")
    return t, A

if __name__ == "__main__":
    for name, f in [("mcns", "mcns_connections.csv.gz"), ("fafb", "fafb_connections_filtered.csv.gz"), ("banc", "banc_connections.csv.gz")]:
        build(name, f)
