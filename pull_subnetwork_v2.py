"""
pull_subnetwork_v2.py -- head-direction ring circuit WITH inhibition

Why: the v1 subnetwork (68 neurons) turned out to be 100% cholinergic
(EPG, PEG, FC, 1 FB) -- no inhibitory cells at all, so the reservoir could
only be silent or runaway. This pulls the fly head-direction ring circuit,
which has real E/I balance:

    EPG, PEG, PEN   -> cholinergic (excitatory)
    Delta7          -> glutamatergic (inhibitory in fly)
    ER1..ER6        -> GABAergic ring neurons (inhibitory)

Uses regex matching so subtype names (PEN_a, ER4d, ...) are included.

Writes the SAME filenames the calibration script expects:
    subnetwork_adjacency.npz, subnetwork_neurons.csv,
    subnetwork_bodyids.npy, subnetwork_nt.csv
Old v1 files are backed up first as cx68_subnetwork_*.

Usage:
    export NEUPRINT_TOKEN=$(grep -oP 'TOKEN = "\\K[^"]+' pull_subnetwork.py)
    python3 pull_subnetwork_v2.py

    # to try a different set of types without editing the file:
    TYPE_REGEX="(EPG.*|PEN.*|Delta7.*)" python3 pull_subnetwork_v2.py
"""

import os
import sys
import shutil

import numpy as np
import pandas as pd
import scipy.sparse as sp
from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = "male-cns:v1.0"
TYPE_REGEX = os.environ.get("TYPE_REGEX", "(EPG.*|PEG.*|PEN.*|Delta7.*|ER[1-6].*)")
MIN_TOTAL_WEIGHT = 3        # drop connections with < 3 synapses (likely noise)
LARGE_NETWORK_WARN = 800

EXCITATORY = {"acetylcholine", "ach"}
INHIBITORY = {"gaba", "glutamate", "glu"}
OUT_FILES = ["subnetwork_adjacency.npz", "subnetwork_neurons.csv",
             "subnetwork_bodyids.npy", "subnetwork_nt.csv"]


def backup_old_files():
    for f in OUT_FILES:
        src = os.path.join(DATA_DIR, f)
        dst = os.path.join(DATA_DIR, "cx68_" + f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  backed up {f} -> cx68_{f}")


def main():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit("Set NEUPRINT_TOKEN first (see Usage at top of file).")

    client = Client("neuprint.janelia.org", dataset=DATASET, token=token)
    criteria = NC(type=TYPE_REGEX, regex=True)

    print(f"Querying {DATASET} for types matching {TYPE_REGEX} ...")
    neuron_df, _ = fetch_neurons(criteria, client=client)   # neuron table is FIRST
    if neuron_df.empty:
        sys.exit("No neurons matched. Check TYPE_REGEX.")

    neuron_df = neuron_df.sort_values(["type", "bodyId"]).reset_index(drop=True)
    bodyids = neuron_df["bodyId"].to_numpy()
    n = len(bodyids)

    # ---- neurotransmitter signs (cell-type-level prediction is more robust) ----
    nt_col = "celltypePredictedNt" if "celltypePredictedNt" in neuron_df else "predictedNt"
    nt = neuron_df[nt_col]
    if nt_col != "predictedNt" and "predictedNt" in neuron_df:
        nt = nt.fillna(neuron_df["predictedNt"])
    nt = nt.fillna("unknown").astype(str).str.strip().str.lower()
    sign = nt.map(lambda x: -1 if x in INHIBITORY else 1).astype(int)

    print(f"\nUsing '{nt_col}' for signs.")
    print("\nNeurons by type and transmitter:")
    print(pd.crosstab(neuron_df["type"], nt).to_string())

    n_inh = int((sign < 0).sum())
    if n_inh == 0:
        print("\nWARNING: still zero inhibitory neurons. Check the type table above "
              "-- Delta7 / ER types may be named differently in this dataset.")

    # ---- connectivity ----
    print("\nFetching connections (this can take a minute) ...")
    # NOTE: fetch_adjacencies returns (neuron_df, conn_df) -- connections are SECOND
    _, conn_df = fetch_adjacencies(criteria, criteria,
                                   min_total_weight=MIN_TOTAL_WEIGHT, client=client)
    # conn_df has one row per (pre, post, ROI) -> sum across ROIs
    conn = conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()

    index_of = {b: i for i, b in enumerate(bodyids)}
    conn = conn[conn["bodyId_pre"].isin(index_of) & conn["bodyId_post"].isin(index_of)]
    rows = conn["bodyId_pre"].map(index_of).to_numpy()
    cols = conn["bodyId_post"].map(index_of).to_numpy()
    adj = sp.csr_matrix((conn["weight"].to_numpy(dtype=float), (rows, cols)), shape=(n, n))

    # ---- save ----
    print("\nBacking up v1 files:")
    backup_old_files()

    sp.save_npz(os.path.join(DATA_DIR, "subnetwork_adjacency.npz"), adj)
    np.save(os.path.join(DATA_DIR, "subnetwork_bodyids.npy"), bodyids)

    meta_cols = [c for c in ["bodyId", "type", "instance", "inputRois", "outputRois"]
                 if c in neuron_df]
    neuron_df[meta_cols].to_csv(os.path.join(DATA_DIR, "subnetwork_neurons.csv"), index=False)

    pd.DataFrame({
        "index": np.arange(n), "bodyId": bodyids, "type": neuron_df["type"].values,
        "nt": nt.values, "sign": sign.values,
    }).to_csv(os.path.join(DATA_DIR, "subnetwork_nt.csv"), index=False)

    # ---- summary ----
    coo = adj.tocoo()
    inh_weight_frac = coo.data[sign.values[coo.row] < 0].sum() / max(coo.data.sum(), 1)
    in_degree = np.asarray((adj > 0).sum(axis=0)).flatten()

    print("\n" + "=" * 60)
    print(f"Neurons:              {n}")
    print(f"Connections (>= {MIN_TOTAL_WEIGHT} syn): {adj.nnz}")
    print(f"Inhibitory neurons:   {n_inh}  ({100 * n_inh / n:.1f}%)")
    print(f"Inhibitory syn weight:{100 * inh_weight_frac:6.1f}%")
    print(f"Mean in-degree:       {in_degree.mean():.1f}")
    print(f"Neurons w/ no inputs: {int((in_degree == 0).sum())}")
    print("=" * 60)
    if n > LARGE_NETWORK_WARN:
        print(f"\nNOTE: {n} neurons is big for numpy-mode Brian2. If the sweep is too "
              "slow, drop ring neurons:\n  TYPE_REGEX=\"(EPG.*|PEG.*|PEN.*|Delta7.*)\" "
              "python3 pull_subnetwork_v2.py")
    print("\nSaved. Next: python3 day3_6_calibration.py")


if __name__ == "__main__":
    main()
