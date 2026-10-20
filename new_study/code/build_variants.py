"""build_variants.py -- builds the five same-fly wiring variants (same rules as build_subnetworks.py):
>=3 synapses per neuron pair after summing over neuropils, no self-loops, signs by cell type.

  fafb_unfiltered : FlyWire FAFB v783, Codex 'Connections (Unfiltered)' file (Princeton synapses,
                    no Codex threshold), same neurons and order as fafb
  fafb_buhmann    : FlyWire FAFB v783, Codex 'Connections Predicted With Buhmann et al.' file,
                    same neurons and order as fafb
  mcns_neuprint   : male CNS v1.0 as pulled from neuPrint for paper v1 (subnetwork_adjacency.npz)
  mcns_v09        : male CNS v0.9 from Codex (neuron attributes + filtered connections)
  banc_v626       : BANC v626 from Codex (neuron attributes + filtered connections)
Inputs are the Codex downloads in raw/; run after build_subnetworks.py."""
import os, shutil, re
import numpy as np, pandas as pd, scipy.sparse as sp

PAT = re.compile(r"^(EPG|PEG|PEN|Delta7|ER[1-6])")

def pairs(conn_file, ids):
    rows = [ch[ch.pre_root_id.isin(ids) & ch.post_root_id.isin(ids)]
            for ch in pd.read_csv("raw/" + conn_file, usecols=["pre_root_id", "post_root_id", "syn_count"], chunksize=2_000_000)]
    C = pd.concat(rows).groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum()
    return C[(C.syn_count >= 3) & (C.pre_root_id != C.post_root_id)]

def save(name, neurons, C):
    idx = {b: i for i, b in enumerate(neurons.bodyId)}
    A = sp.coo_matrix((C.syn_count.astype(float), (C.pre_root_id.map(idx), C.post_root_id.map(idx))), shape=(len(neurons),) * 2).tocsr()
    os.makedirs(f"networks/{name}", exist_ok=True); sp.save_npz(f"networks/{name}/subnetwork_adjacency.npz", A)
    neurons.to_csv(f"networks/{name}/subnetwork_neurons.csv", index=False)
    sign = np.where(neurons.type.str.startswith(("Delta7", "ER")), -1, 1)
    pd.DataFrame({"index": range(len(neurons)), "sign": sign}).to_csv(f"networks/{name}/subnetwork_nt.csv", index=False)
    print(f"{name:16s} {len(neurons)} neurons, {(A > 0).sum()} connections, {int(A.sum())} synapses")

def codex_neurons(neuron_file):
    t = pd.read_csv("raw/" + neuron_file, usecols=["Root ID", "Primary Cell Type"]).rename(columns={"Root ID": "bodyId", "Primary Cell Type": "type"})
    return t[t.type.astype(str).str.match(PAT)].sort_values(["type", "bodyId"]).reset_index(drop=True)

fafb = pd.read_csv("networks/fafb/subnetwork_neurons.csv"); ids = set(fafb.bodyId)
save("fafb_unfiltered", fafb, pairs("fafb_connections_unfiltered.csv.gz", ids))
save("fafb_buhmann", fafb, pairs("fafb_connections_buhmann.csv.gz", ids))
for name, nf, cf in [("mcns_v09", "mcns09_neurons.csv.gz", "mcns09_connections.csv.gz"), ("banc_v626", "banc626_neurons.csv.gz", "banc626_connections.csv.gz")]:
    t = codex_neurons(nf); save(name, t, pairs(cf, set(t.bodyId)))
if os.path.exists("v1_network/subnetwork_adjacency.npz"):            # paper v1 neuPrint network
    os.makedirs("networks/mcns_neuprint", exist_ok=True)
    shutil.copy("v1_network/subnetwork_adjacency.npz", "networks/mcns_neuprint/")
    for f in ["subnetwork_neurons.csv", "subnetwork_nt.csv"]: shutil.copy(f"networks/mcns/{f}", "networks/mcns_neuprint/")
