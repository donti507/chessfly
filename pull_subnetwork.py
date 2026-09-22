import numpy as np
import scipy.sparse as sp
from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

import os
TOKEN = os.environ["NEUPRINT_TOKEN"]
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=TOKEN)

criteria = NC(type=[
    "EPG", "PEG", "PEN", "FB", "FC", "PFN", "PFL", "hDelta", "vDelta",
])

print("Fetching neurons matching central-complex criteria...")
neurons_df, roi_counts = fetch_neurons(criteria)
print(f"Found {len(neurons_df)} neurons")

if len(neurons_df) == 0:
    raise SystemExit(
        "No neurons matched -- cell type names vary by dataset version. "
        "Run neuprint.fetch_meta(client) and check the NeuronCriteria "
        "docs / type list for male-cns:v1.0 before re-running."
    )

body_ids = neurons_df["bodyId"].tolist()

print("Fetching adjacencies (this can take a few minutes)...")
_, conn_df = fetch_adjacencies(sources=body_ids, targets=body_ids)
print(f"Found {len(conn_df)} directed connections")

id_to_idx = {bid: i for i, bid in enumerate(body_ids)}
n = len(body_ids)

rows = conn_df["bodyId_pre"].map(id_to_idx).values
cols = conn_df["bodyId_post"].map(id_to_idx).values
weights = conn_df["weight"].values.astype(np.float32)

adj = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()

sp.save_npz("subnetwork_adjacency.npz", adj)
neurons_df.to_csv("subnetwork_neurons.csv", index=False)
np.save("subnetwork_bodyids.npy", np.array(body_ids))

print(f"Saved graph: {n} neurons, {adj.nnz} nonzero connections")
print("Files: subnetwork_adjacency.npz, subnetwork_neurons.csv, subnetwork_bodyids.npy")
