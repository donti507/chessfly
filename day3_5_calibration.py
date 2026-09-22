"""
DAY 3.5 -- Calibration sweep

Goal: find a parameter regime where the 68-neuron connectome reservoir
produces (a) broad activity (not just 6/68 neurons firing) and
(b) meaningfully different states for different chess positions
(mean pairwise cosine similarity noticeably below the ~0.855 baseline
from Day 3, ideally without being so chaotic that ALL positions
look equally different -- that's just noise, not signal).

This does NOT train anything. It's diagnostic only.

Sweeps:
    - WEIGHT_SCALE (how strongly connectome synapses drive voltage)
    - INPUT_WEIGHT (how strongly the 6 chess-input neurons drive the network)
    - input neuron selection strategy (highest out-degree vs. spread across types)
    - simulation duration

Uses a reduced set of 4 representative positions (not all 6) to keep the
sweep fast; once you pick a config, rerun day3_chess_reservoir.py with the
same WEIGHT_SCALE / INPUT_WEIGHT values for the full 6-position analysis.

Output: day3_5_calibration_results.csv
"""

import os
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
import chess

from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, PoissonGroup,
    run, ms, mV, Hz, prefs, seed as brian_seed,
)

prefs.codegen.target = "numpy"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
N_INPUT_NEURONS = 6
N_FEATURES = 64 * 12
MIN_RATE_HZ = 5.0
MAX_RATE_HZ = 200.0
RANDOM_SEED = 0

# ---- Sweep grid (kept small on purpose -- this is diagnostic, not exhaustive) ----
WEIGHT_SCALES_MV = [0.08, 0.20, 0.40]     # Day 3 used 0.08
INPUT_WEIGHTS_MV = [6.0, 15.0, 30.0]      # Day 3 used 6.0
DURATIONS_MS = [300.0]                     # keep fixed for now
INPUT_STRATEGIES = ["out_degree", "diverse_types"]

PIECE_TYPES = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]


def board_to_features(board):
    features = np.zeros(N_FEATURES, dtype=np.float32)
    for plane, (piece_type, color) in enumerate(PIECE_TYPES):
        for square in board.pieces(piece_type, color):
            features[plane * 64 + square] = 1.0
    return features


def make_test_positions():
    positions = {}
    positions["start"] = chess.Board()

    b = chess.Board()
    b.push_san("e4")
    positions["after_e4"] = b.copy()

    b2 = chess.Board()
    b2.push_san("d4")
    positions["after_d4"] = b2.copy()

    b3 = chess.Board()
    for mv in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6"]:
        b3.push_san(mv)
    positions["ruy_lopez_exchange"] = b3.copy()

    return positions


def make_projection(seed):
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0, size=(N_INPUT_NEURONS, N_FEATURES)).astype(np.float32)
    proj /= (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)
    return proj


def features_to_rates(features, projection):
    raw = projection @ features
    activated = 1.0 / (1.0 + np.exp(-raw))
    return (MIN_RATE_HZ + activated * (MAX_RATE_HZ - MIN_RATE_HZ)).astype(np.float32)


def select_input_neurons(adj, neuron_meta, strategy):
    if strategy == "out_degree":
        out_degree = np.asarray(adj.tocsr().sum(axis=1)).flatten()
        return np.argsort(-out_degree)[:N_INPUT_NEURONS]

    if strategy == "diverse_types":
        # pick highest out-degree neuron from as many distinct cell types as possible
        out_degree = np.asarray(adj.tocsr().sum(axis=1)).flatten()
        order = np.argsort(-out_degree)
        seen_types = set()
        chosen = []
        for idx in order:
            t = neuron_meta.iloc[idx]["type"]
            if t not in seen_types:
                seen_types.add(t)
                chosen.append(idx)
            if len(chosen) == N_INPUT_NEURONS:
                break
        if len(chosen) < N_INPUT_NEURONS:
            for idx in order:
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) == N_INPUT_NEURONS:
                    break
        return np.array(chosen)

    raise ValueError(f"unknown strategy {strategy}")


def build_reservoir(adj, input_indices, weight_scale_mv, input_weight_mv):
    N = adj.shape[0]
    eqs = """
    dv/dt = (v_rest - v) / tau : volt (unless refractory)
    v_rest : volt
    tau : second
    """
    neurons = NeuronGroup(
        N, eqs, threshold="v > -50*mV", reset="v = -65*mV",
        refractory=3 * ms, method="euler",
    )
    neurons.v = -65 * mV
    neurons.v_rest = -65 * mV
    neurons.tau = 10 * ms

    syn = Synapses(neurons, neurons, model="w : volt", on_pre="v_post += w")
    syn.connect(i=adj.row, j=adj.col)
    syn.w = adj.data * (weight_scale_mv * mV)

    drive = PoissonGroup(len(input_indices), rates=0 * Hz)
    drive_syn = Synapses(drive, neurons, model="", on_pre=f"v_post += {input_weight_mv}*mV")
    drive_syn.connect(i=np.arange(len(input_indices)), j=input_indices)

    return neurons, syn, drive, drive_syn


def run_position(board, projection, adj, input_indices, seed, duration_ms,
                  weight_scale_mv, input_weight_mv):
    features = board_to_features(board)
    rates = features_to_rates(features, projection)

    brian_seed(seed)
    neurons, syn, drive, drive_syn = build_reservoir(
        adj, input_indices, weight_scale_mv, input_weight_mv
    )
    drive.rates = rates * Hz
    spike_monitor = SpikeMonitor(neurons)
    run(duration_ms * ms)

    spike_counts = np.bincount(
        np.asarray(spike_monitor.i, dtype=int), minlength=adj.shape[0]
    ).astype(np.float32)
    spike_rates = spike_counts / (duration_ms / 1000.0)

    return {
        "spike_rates": spike_rates,
        "total_spikes": int(spike_monitor.num_spikes),
        "active_neurons": int(np.count_nonzero(spike_counts)),
    }


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def mean_pairwise_similarity(states_dict):
    names = list(states_dict.keys())
    vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vals.append(cosine_similarity(states_dict[names[i]], states_dict[names[j]]))
    return float(np.mean(vals)) if vals else 0.0


def main():
    adj = sp.load_npz(os.path.join(DATA_DIR, "subnetwork_adjacency.npz")).tocoo()
    neuron_meta = pd.read_csv(os.path.join(DATA_DIR, "subnetwork_neurons.csv"))
    N = adj.shape[0]

    positions = make_test_positions()
    projection = make_projection(RANDOM_SEED)

    print("=" * 72)
    print(f"DAY 3.5 -- CALIBRATION SWEEP  ({N} neurons, {adj.nnz} connections)")
    print("=" * 72)

    results = []
    combos = list(itertools.product(
        WEIGHT_SCALES_MV, INPUT_WEIGHTS_MV, DURATIONS_MS, INPUT_STRATEGIES
    ))
    print(f"\nRunning {len(combos)} configs x {len(positions)} positions "
          f"= {len(combos) * len(positions)} simulations...\n")

    for weight_scale, input_weight, duration, strategy in combos:
        input_indices = select_input_neurons(adj, neuron_meta, strategy)

        states = {}
        active_counts = []
        total_spike_counts = []

        for name, board in positions.items():
            res = run_position(
                board, projection, adj, input_indices, RANDOM_SEED, duration,
                weight_scale, input_weight,
            )
            states[name] = res["spike_rates"]
            active_counts.append(res["active_neurons"])
            total_spike_counts.append(res["total_spikes"])

        mean_sim = mean_pairwise_similarity(states)
        mean_active = float(np.mean(active_counts))
        mean_spikes = float(np.mean(total_spike_counts))

        row = {
            "weight_scale_mV": weight_scale,
            "input_weight_mV": input_weight,
            "duration_ms": duration,
            "input_strategy": strategy,
            "mean_active_neurons": mean_active,
            "pct_active": round(100 * mean_active / N, 1),
            "mean_total_spikes": mean_spikes,
            "mean_pairwise_similarity": round(mean_sim, 4),
        }
        results.append(row)

        print(f"scale={weight_scale:>5.2f}mV  input={input_weight:>5.1f}mV  "
              f"strategy={strategy:<14}  active={mean_active:>5.1f}/{N} "
              f"({row['pct_active']:>5.1f}%)  spikes={mean_spikes:>7.1f}  "
              f"similarity={mean_sim:.3f}")

    df = pd.DataFrame(results)
    out_path = os.path.join(DATA_DIR, "day3_5_calibration_results.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 72)
    print("SWEEP COMPLETE -- saved day3_5_calibration_results.csv")
    print("=" * 72)

    # Heuristic pick: want high activity (broad participation) AND
    # low-ish similarity (positions are distinguishable), not just noise.
    # Filter to configs with reasonable activity first, then sort by
    # lowest similarity among those.
    reasonable = df[df["pct_active"] >= 30]
    if len(reasonable) == 0:
        print("\nNo config reached >=30% active neurons -- consider widening "
              "WEIGHT_SCALES_MV / INPUT_WEIGHTS_MV ranges further, or the "
              "network itself needs more neurons.")
        best = df.sort_values("pct_active", ascending=False).iloc[0]
    else:
        best = reasonable.sort_values("mean_pairwise_similarity").iloc[0]

    print("\nSuggested config to carry into Day 4:")
    print(best.to_string())


if __name__ == "__main__":
    main()
