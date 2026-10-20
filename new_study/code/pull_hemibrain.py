"""
pull_hemibrain.py -- re-create the hemibrain head-direction subnetwork (4th fly).

Colab (CPU, about 2 minutes). First add your neuPrint token under the key icon (Secrets)
with the name NEUPRINT_TOKEN and switch on notebook access, then run:

    !pip install -q neuprint-python
    import os
    from google.colab import userdata
    os.environ["NEUPRINT_TOKEN"] = userdata.get("NEUPRINT_TOKEN")
    !python pull_hemibrain.py

Output: hemibrain_hd.zip containing the per-pair synapse counts (no threshold applied, so
the same >=3 rule can be applied later) and the neuron table with types and instances.
"""

import os
import zipfile

import pandas as pd
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies, fetch_neurons

TYPES = "(EPG.*|PEG.*|PEN.*|Delta7.*|ER[1-6].*)"
client = Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1", token=os.environ["NEUPRINT_TOKEN"])

crit = NC(type=TYPES, regex=True)
neurons, _ = fetch_neurons(crit, client=client)
neurons = neurons[["bodyId", "type", "instance"]].sort_values(["type", "bodyId"])
_, conn = fetch_adjacencies(crit, crit, min_total_weight=1, client=client)
conn = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()

neurons.to_csv("hemibrain_hd_neurons.csv", index=False)
conn.to_csv("hemibrain_hd_edges.csv", index=False)
with zipfile.ZipFile("hemibrain_hd.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("hemibrain_hd_neurons.csv")
    z.write("hemibrain_hd_edges.csv")
print(f"{len(neurons)} neurons, {len(conn)} connected pairs -> hemibrain_hd.zip")
print(neurons.type.str.extract(r"^(EPG|PEG|PEN|Delta7|ER)")[0].value_counts().to_string())
