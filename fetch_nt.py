"""
fetch_nt.py -- pull neurotransmitter predictions for the 68 subnetwork neurons

Why: the Day 3.5 sweep showed the network is either silent or runaway.
That's what a purely excitatory recurrent network does. Real central-complex
neurons include GABAergic / glutamatergic (inhibitory) cells, so we sign
each neuron's outgoing synapses by its predicted transmitter.

Sign convention (standard assumption for Drosophila fast synapses):
    acetylcholine        -> +1 (excitatory)
    GABA, glutamate      -> -1 (inhibitory; fly glutamate mostly acts via GluCl)
    anything else/unknown -> +1 (counted and reported so you can see how many)

Output: subnetwork_nt.csv  (row order == adjacency matrix index order)

Usage:
    export NEUPRINT_TOKEN="..."
    python3 fetch_nt.py
    # if it can't find the NT column, it prints all columns; then:
    NT_COLUMN=someColumnName python3 fetch_nt.py
"""

import os
import sys

import numpy as np
import pandas as pd
from neuprint import Client, fetch_neurons, NeuronCriteria as NC

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = "male-cns:v1.0"

CANDIDATE_COLS = [
    "predictedNt", "consensusNt", "celltypePredictedNt",
    "predicted_nt", "predictedNeurotransmitter",
]
EXCITATORY = {"acetylcholine", "ach"}
INHIBITORY = {"gaba", "glutamate", "glu"}


def find_nt_column(columns):
    override = os.environ.get("NT_COLUMN")
    if override:
        if override not in columns:
            sys.exit(f"NT_COLUMN={override} not in returned columns.")
        return override
    for c in CANDIDATE_COLS:
        if c in columns:
            return c
    # heuristic fallback
    for c in columns:
        cl = c.lower()
        if ("nt" in cl and ("pred" in cl or "consensus" in cl)) or "transmitter" in cl:
            return c
    return None


def main():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit('Set your token first:  export NEUPRINT_TOKEN="..."')

    client = Client("neuprint.janelia.org", dataset=DATASET, token=token)
    bodyids = np.load(os.path.join(DATA_DIR, "subnetwork_bodyids.npy"))
    print(f"Fetching {len(bodyids)} neurons from {DATASET} ...")

    # NOTE: fetch_neurons returns (neuron_df, roi_counts_df) -- neuron table is FIRST
    neuron_df, _ = fetch_neurons(NC(bodyId=bodyids.tolist()), client=client)

    print("\nColumns returned by neuPrint:")
    print(", ".join(neuron_df.columns))

    col = find_nt_column(list(neuron_df.columns))
    if col is None:
        sys.exit("\nCould not find a neurotransmitter column. Pick one from the list "
                 "above and rerun with NT_COLUMN=<name> python3 fetch_nt.py")
    print(f"\nUsing neurotransmitter column: '{col}'")

    # reorder to match adjacency matrix index order
    df = neuron_df.set_index("bodyId").reindex(bodyids)
    n_missing_rows = int(df["type"].isna().sum()) if "type" in df else 0
    if n_missing_rows:
        print(f"WARNING: {n_missing_rows} bodyIds were not returned by neuPrint")

    nt = df[col].fillna("unknown").astype(str).str.strip().str.lower()
    sign = nt.map(lambda x: -1 if x in INHIBITORY else 1).astype(int)

    out = pd.DataFrame({
        "index": np.arange(len(bodyids)),
        "bodyId": bodyids,
        "type": df["type"].values if "type" in df else "",
        "nt": nt.values,
        "sign": sign.values,
    })
    out_path = os.path.join(DATA_DIR, "subnetwork_nt.csv")
    out.to_csv(out_path, index=False)

    print("\nTransmitter counts:")
    print(nt.value_counts().to_string())
    unknown = (~nt.isin(EXCITATORY | INHIBITORY)).sum()
    print(f"\nExcitatory: {(sign > 0).sum() - unknown}   "
          f"Inhibitory: {(sign < 0).sum()}   "
          f"Other/unknown (treated as +1): {unknown}")

    print("\nTransmitter by cell type:")
    print(pd.crosstab(out["type"], out["nt"]).to_string())
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
