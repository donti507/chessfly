"""
build_lif_simulator.py

Day 2 of the Fly Connectome Chess Engine project.

Goal (per project plan): load the real connectome subnetwork pulled in Day 1
(subnetwork_adjacency.npz / subnetwork_neurons.csv / subnetwork_bodyids.npy),
wire it up as a Brian2 LIF spiking network, inject some input current into a
handful of "input" neurons, and confirm that (a) neurons actually spike and
(b) activity propagates through the real recurrent wiring to neurons that
were NOT directly stimulated. This is a sanity check only -- no chess yet.

Run:
    python3 build_lif_simulator.py

Expects, in the same directory (or pass --data-dir):
    subnetwork_adjacency.npz   scipy sparse CSR, weighted directed graph
    subnetwork_neurons.csv     bodyId, type, instance, roi columns
    subnetwork_bodyids.npy     matrix index -> real neuPrint bodyId
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, StateMonitor, PoissonGroup,
    run, ms, mV, nA, Hz, prefs, seed as brian_seed,
)

# Run in pure-numpy codegen mode: no C++ compiler needed. Slower than the
# default 'cython' target but far more portable across machines/containers,
# and at 68 neurons / a few hundred ms of simulated time it's plenty fast.
prefs.codegen.target = "numpy"


# ---------------------------------------------------------------------------
# Config -- tune these; they're the main knobs for the Day 2 sanity check
#
# Gotcha found while tuning against test data (recheck against the REAL
# subnetwork -- a random test graph is not the real connectome, exact
# numbers will differ, but the two failure modes below are structural and
# will likely show up again):
#   - WEIGHT_SCALE too low + INPUT_WEIGHT too low -> nothing crosses
#     threshold, 0 spikes total, even the driven neurons stay silent.
#   - WEIGHT_SCALE too high -> the whole 68-neuron network fires in
#     synchronized lockstep bursts every ~15ms. It "passes" the propagation
#     check but is a degenerate regime: a synchronized global burst carries
#     almost no information about which input neurons were driven, which
#     is bad news for Day 3+ (need different board states to produce
#     *distinguishable* output patterns, not the same burst every time).
#   The values below sit in between: driven neurons fire irregularly and
#   a bit of activity reaches downstream neurons without everything
#   collapsing into one synchronized blob. Re-tune after swapping in the
#   real adjacency matrix -- real connectome structure (not random) may
#   shift where this sweet spot is.
# ---------------------------------------------------------------------------
N_INPUT_NEURONS = 6          # how many nodes get direct external drive
SIM_DURATION = 300 * ms
WEIGHT_SCALE = 0.08 * mV     # mV of EPSP-equivalent per "synapse count" unit
INPUT_RATE = 200 * Hz        # Poisson drive rate onto input neurons
INPUT_WEIGHT = 6 * mV
RANDOM_SEED = 0


def load_subnetwork(data_dir):
    adj = sp.load_npz(os.path.join(data_dir, "subnetwork_adjacency.npz")).tocoo()
    neurons = pd.read_csv(os.path.join(data_dir, "subnetwork_neurons.csv"))
    bodyids = np.load(os.path.join(data_dir, "subnetwork_bodyids.npy"))
    assert adj.shape[0] == adj.shape[1] == len(neurons) == len(bodyids), (
        "adjacency / neurons / bodyids sizes don't line up -- check Day 1 "
        "output before debugging Brian2, this is a much easier bug to catch here."
    )
    return adj, neurons, bodyids


def build_network(adj, n_input_neurons, seed):
    brian_seed(seed)
    rng = np.random.default_rng(seed)
    N = adj.shape[0]

    # Standard leaky integrate-and-fire neuron.
    eqs = """
    dv/dt = (v_rest - v) / tau : volt (unless refractory)
    v_rest : volt
    tau : second
    """
    neurons = NeuronGroup(
        N, eqs,
        threshold="v > -50*mV",
        reset="v = -65*mV",
        refractory=3 * ms,
        method="euler",
    )
    neurons.v = -65 * mV
    neurons.v_rest = -65 * mV
    neurons.tau = 10 * ms

    # Recurrent synapses straight from the real connectome adjacency matrix.
    # NOTE (known simplification, flag for later): neuPrint synapse counts
    # carry no excitatory/inhibitory sign by default -- everything here is
    # treated as excitatory. Revisit once/if predicted neurotransmitter type
    # is pulled in (fetch_adjacencies / custom cypher can return it).
    syn = Synapses(neurons, neurons, "w : volt", on_pre="v_post += w")
    syn.connect(i=adj.row, j=adj.col)
    # Scale raw synapse counts down into a sane EPSP range -- using raw
    # counts (can be 1-50+) directly as mV jumps would either never cross
    # threshold or blow every neuron up on the first spike. This is the
    # single most important knob in this script; if nothing spikes, or
    # everything spikes in the first 5ms, adjust WEIGHT_SCALE first.
    syn.w = adj.data * WEIGHT_SCALE

    # Pick input neurons: highest out-degree nodes, so the sanity check
    # actually has a chance of propagating rather than dead-ending.
    out_degree = np.asarray(adj.tocsr().sum(axis=1)).flatten()
    input_idx = np.argsort(-out_degree)[:n_input_neurons]

    drive = PoissonGroup(n_input_neurons, rates=INPUT_RATE)
    drive_syn = Synapses(drive, neurons, on_pre="v_post += INPUT_WEIGHT")
    drive_syn.connect(i=np.arange(n_input_neurons), j=input_idx)

    return neurons, syn, drive, drive_syn, input_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    adj, neuron_meta, bodyids = load_subnetwork(args.data_dir)
    print(f"Loaded subnetwork: {adj.shape[0]} neurons, {adj.nnz} directed connections")

    neurons, syn, drive, drive_syn, input_idx = build_network(
        adj, N_INPUT_NEURONS, RANDOM_SEED
    )
    print(f"Input neurons (by index): {sorted(input_idx.tolist())}")
    print(f"Input neuron types: {neuron_meta.iloc[input_idx]['type'].tolist()}")

    spikes = SpikeMonitor(neurons)
    voltage = StateMonitor(neurons, "v", record=True)

    run(SIM_DURATION)

    # ---- Sanity check ----
    total_spikes = spikes.num_spikes
    spiking_neuron_idx = set(np.unique(spikes.i))
    input_set = set(input_idx.tolist())
    downstream_spiking = spiking_neuron_idx - input_set

    print("\n--- Day 2 sanity check ---")
    print(f"Total spikes over {SIM_DURATION}: {total_spikes}")
    print(f"Neurons that spiked at least once: {len(spiking_neuron_idx)} / {adj.shape[0]}")
    print(f"Of those, NOT directly driven (i.e. activity propagated through "
          f"real connectome wiring): {len(downstream_spiking)}")

    if total_spikes == 0:
        print("FAIL: nothing spiked. Try raising WEIGHT_SCALE or INPUT_RATE/INPUT_WEIGHT.")
    elif len(downstream_spiking) == 0:
        print("PARTIAL: only directly-driven neurons spiked, no propagation through "
              "the recurrent graph. Try raising WEIGHT_SCALE.")
    else:
        print("PASS: network spikes and activity propagates through the real "
              "connectome wiring beyond the directly-stimulated neurons.")

    # Save a raster plot for a quick visual check.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(spikes.t / ms, spikes.i, ".k", markersize=3)
        for idx in input_idx:
            ax.axhline(idx, color="red", alpha=0.15, linewidth=6)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("neuron index")
        ax.set_title("Day 2 sanity check: spike raster (red bands = driven inputs)")
        out_path = os.path.join(args.data_dir, "day2_spike_raster.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved raster plot: {out_path}")
    except ImportError:
        print("\n(matplotlib not installed -- skipping raster plot; "
              "pip install matplotlib if you want it)")


if __name__ == "__main__":
    main()
