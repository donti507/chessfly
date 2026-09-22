"""
DAY 3.7 -- Calibration sweep v3: fix the INPUT, not the reservoir

Diagnosis from Day 3.6: sep_ratio < 1 and decode at chance everywhere.
Input rates were already ~0.986 similar across positions, and Poisson spike
noise (~18% at 100 Hz / 300 ms) swamped the few-% differences between
positions before the reservoir even saw them.

Changes:
  1. Deterministic current injection (I_ext) into input neurons -- no Poisson.
  2. Features centered on the mean of 200 random positions, and per-neuron
     gain normalized, so between-position differences use the sigmoid's range.
  3. Small membrane noise (NOISE_SIGMA_MV) on every neuron so sep_ratio still
     measures robustness rather than being trivially infinite.
  4. 8 test positions (chance decode = 0.125).
  5. Network built once per (sign, n_input, scale) and reused across I_MAX.

Needs: subnetwork_adjacency.npz, subnetwork_neurons.csv, subnetwork_nt.csv
Output: day3_7_calibration_results.csv
"""

import os
import time
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
import chess

from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network,
    ms, mV, prefs, seed as brian_seed,
)

prefs.codegen.target = "numpy"
from brian2 import BrianLogger
BrianLogger.suppress_hierarchy("brian2.codegen.generators.base")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
N_FEATURES = 64 * 12
PROJ_SEED = 0
REF_SEED = 123
TEST_SEED = 7
N_REF_POSITIONS = 200
SIM_SEEDS = [0, 1, 2]
DURATION_MS = 1000.0
NOISE_SIGMA_MV = 1.0
INPUT_GAIN = 1.5                  # std of pre-sigmoid input across random positions
INPUT_STRATEGY = "diverse_types"

# ---- Sweep grid ----
SIGN_MODES = ["signed"]
N_INPUTS = [48, 72]
WEIGHT_SCALES_MV = [0.14, 0.18, 0.22, 0.28, 0.35]
I_MAX_MV = [20.0, 30.0]           # max steady drive; threshold is 15 mV above rest

PIECE_TYPES = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]

LIF_EQS = """
dv/dt = (v_rest - v + I_ext) / tau + sigma * sqrt(2 / tau) * xi : volt (unless refractory)
v_rest : volt
I_ext : volt
tau : second
sigma : volt
"""


# ---------------------------------------------------------------- positions
def board_to_features(board):
    features = np.zeros(N_FEATURES, dtype=np.float32)
    for plane, (piece_type, color) in enumerate(PIECE_TYPES):
        for square in board.pieces(piece_type, color):
            features[plane * 64 + square] = 1.0
    return features


def random_position(rng, n_plies):
    b = chess.Board()
    for _ in range(n_plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(moves[rng.integers(len(moves))])
    return b


def make_reference_positions():
    rng = np.random.default_rng(REF_SEED)
    return [random_position(rng, int(rng.integers(0, 61))) for _ in range(N_REF_POSITIONS)]


def make_test_positions():
    positions = {"start": chess.Board()}
    b = chess.Board(); b.push_san("e4"); positions["after_e4"] = b
    b = chess.Board(); b.push_san("d4"); positions["after_d4"] = b
    b = chess.Board()
    for mv in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6"]:
        b.push_san(mv)
    positions["ruy_lopez_exchange"] = b
    rng = np.random.default_rng(TEST_SEED)
    positions["random_mid_1"] = random_position(rng, 16)
    positions["random_mid_2"] = random_position(rng, 24)
    positions["random_late_1"] = random_position(rng, 40)
    positions["random_late_2"] = random_position(rng, 60)
    return positions


class Encoder:
    """Board -> activation in [0, 1] for each input neuron."""

    def __init__(self, n_input, ref_boards, seed):
        feats = np.stack([board_to_features(b) for b in ref_boards])
        self.mean = feats.mean(axis=0)
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0.0, 1.0, size=(n_input, N_FEATURES)).astype(np.float32)
        raw = (feats - self.mean) @ self.proj.T
        self.scale = INPUT_GAIN / (raw.std(axis=0) + 1e-8)

    def activation(self, board):
        raw = (self.proj @ (board_to_features(board) - self.mean)) * self.scale
        return 1.0 / (1.0 + np.exp(-raw))


# ---------------------------------------------------------------- network
def select_input_neurons(adj, neuron_meta, n_input, strategy):
    out_degree = np.asarray(adj.tocsr().sum(axis=1)).flatten()
    order = np.argsort(-out_degree)
    if strategy == "out_degree":
        return order[:n_input]
    seen, chosen = set(), []
    for idx in order:
        t = neuron_meta.iloc[idx]["type"]
        if t not in seen:
            seen.add(t)
            chosen.append(idx)
        if len(chosen) == n_input:
            break
    for idx in order:
        if len(chosen) == n_input:
            break
        if idx not in chosen:
            chosen.append(idx)
    return np.array(chosen)


def load_signs(n_neurons):
    df = pd.read_csv(os.path.join(DATA_DIR, "subnetwork_nt.csv")).sort_values("index")
    if len(df) != n_neurons:
        raise ValueError(f"subnetwork_nt.csv has {len(df)} rows, expected {n_neurons}")
    return df["sign"].to_numpy(dtype=float)


def synapse_weights(adj, weight_scale_mv, sign_mode, signs):
    w = adj.data.astype(float) * weight_scale_mv
    inh_gain = 2.0 if sign_mode == "signed_inh2x" else 1.0
    return w * np.where(signs[adj.row] < 0, -inh_gain, 1.0)


def build_network(adj, weights_mv):
    n = adj.shape[0]
    neurons = NeuronGroup(n, LIF_EQS, threshold="v > -50*mV", reset="v = -65*mV",
                          refractory=3 * ms, method="euler")
    neurons.v = -65 * mV
    neurons.v_rest = -65 * mV
    neurons.tau = 10 * ms
    neurons.sigma = NOISE_SIGMA_MV * mV
    neurons.I_ext = 0 * mV
    neurons.run_regularly("v = clip(v, -80*mV, 0*mV)", when="after_synapses")

    syn = Synapses(neurons, neurons, model="w : volt",
                   on_pre="v_post += w")
    syn.connect(i=adj.row, j=adj.col)
    syn.w = weights_mv * mV

    monitor = SpikeMonitor(neurons)
    net = Network(neurons, syn, monitor)
    net.store()
    return net, neurons, monitor


def simulate(net, neurons, monitor, input_idx, activation, i_max_mv, sim_seed, n_neurons):
    net.restore()
    drive = np.zeros(n_neurons)
    drive[input_idx] = activation * i_max_mv
    neurons.I_ext = drive * mV
    brian_seed(sim_seed)
    net.run(DURATION_MS * ms, namespace={})
    counts = np.bincount(np.asarray(monitor.i, dtype=int), minlength=n_neurons)
    return counts.astype(np.float32) / (DURATION_MS / 1000.0)


# ---------------------------------------------------------------- metrics
def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return np.nan if d == 0 else float(a @ b / d)


def mean_pairwise_cosine(vectors):
    vals = [cosine(a, b) for a, b in itertools.combinations(vectors, 2)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def evaluate(states):
    """states: (P positions, S seeds, H hidden neurons)."""
    P, S, _ = states.shape
    within = [np.linalg.norm(states[p, a] - states[p, b])
              for p in range(P) for a, b in itertools.combinations(range(S), 2)]
    between = [np.linalg.norm(states[p, a] - states[q, b])
               for p, q in itertools.combinations(range(P), 2)
               for a in range(S) for b in range(S)]
    within_d, between_d = float(np.mean(within)), float(np.mean(between))
    sep_ratio = between_d / within_d if within_d > 1e-9 else np.nan
    res_sim = np.nanmean([mean_pairwise_cosine([states[p, s] for p in range(P)])
                          for s in range(S)])
    correct = 0
    for t in range(S):
        train = [s for s in range(S) if s != t]
        centroids = states[:, train, :].mean(axis=1)
        for p in range(P):
            dists = np.linalg.norm(centroids - states[p, t], axis=1)
            if np.allclose(dists, dists[0]):
                continue
            correct += int(np.argmin(dists) == p)
    return within_d, between_d, sep_ratio, float(res_sim), correct / (P * S)


def classify_regime(pct_active_hidden, res_sim):
    if pct_active_hidden < 10:
        return "silent"
    if not np.isnan(res_sim) and res_sim > 0.99:
        return "saturated"
    return "useful"


# ---------------------------------------------------------------- main
def main():
    adj = sp.load_npz(os.path.join(DATA_DIR, "subnetwork_adjacency.npz")).tocoo()
    neuron_meta = pd.read_csv(os.path.join(DATA_DIR, "subnetwork_neurons.csv"))
    n_neurons = adj.shape[0]
    signs = load_signs(n_neurons)
    print(f"Network: {n_neurons} neurons, {adj.nnz} connections, "
          f"{int((signs < 0).sum())} inhibitory")

    positions = make_test_positions()
    pos_names = list(positions.keys())
    ref_boards = make_reference_positions()

    encoders, activations = {}, {}
    for n_in in N_INPUTS:
        encoders[n_in] = Encoder(n_in, ref_boards, PROJ_SEED)
        activations[n_in] = {k: encoders[n_in].activation(b) for k, b in positions.items()}
        acts = list(activations[n_in].values())
        min_diff = min(np.linalg.norm(a - b) for a, b in itertools.combinations(acts, 2))
        print(f"{n_in:>2} inputs: input cosine sim={mean_pairwise_cosine(acts):.3f}, "
              f"closest pair activation distance={min_diff:.3f}")

    builds = list(itertools.product(SIGN_MODES, N_INPUTS, WEIGHT_SCALES_MV))
    n_configs = len(builds) * len(I_MAX_MV)
    print("=" * 78)
    print(f"DAY 3.7 -- CALIBRATION v3: {n_configs} configs x {len(pos_names)} positions "
          f"x {len(SIM_SEEDS)} seeds = {n_configs * len(pos_names) * len(SIM_SEEDS)} sims")
    print(f"Chance decode accuracy = {1 / len(pos_names):.3f}")
    print("=" * 78)

    results = []
    t0 = time.time()
    done = 0
    for sign_mode, n_in, w_scale in builds:
        input_idx = select_input_neurons(adj, neuron_meta, n_in, INPUT_STRATEGY)
        hidden_mask = np.ones(n_neurons, dtype=bool)
        hidden_mask[input_idx] = False
        net, neurons, monitor = build_network(
            adj, synapse_weights(adj, w_scale, sign_mode, signs))

        for i_max in I_MAX_MV:
            states = np.zeros((len(pos_names), len(SIM_SEEDS), hidden_mask.sum()), np.float32)
            active_fracs = []
            for p, name in enumerate(pos_names):
                for s, sim_seed in enumerate(SIM_SEEDS):
                    rates = simulate(net, neurons, monitor, input_idx,
                                     activations[n_in][name], i_max, sim_seed, n_neurons)
                    states[p, s] = rates[hidden_mask]
                    active_fracs.append(np.count_nonzero(rates[hidden_mask]) / hidden_mask.sum())

            within_d, between_d, sep, res_sim, dec = evaluate(states)
            pct_active = 100 * float(np.mean(active_fracs))
            regime = classify_regime(pct_active, res_sim)
            results.append({
                "sign_mode": sign_mode, "n_input": n_in,
                "weight_scale_mV": w_scale, "i_max_mV": i_max,
                "pct_hidden_active": round(pct_active, 1),
                "mean_hidden_rate_hz": round(float(states.mean()), 2),
                "reservoir_sim": round(res_sim, 4),
                "within_dist": round(within_d, 2),
                "between_dist": round(between_d, 2),
                "sep_ratio": round(sep, 3) if not np.isnan(sep) else np.nan,
                "decode_acc": round(dec, 3),
                "regime": regime,
            })

            done += 1
            eta = (time.time() - t0) / done * (n_configs - done)
            print(f"[{done:>3}/{n_configs}] {sign_mode:<13} in={n_in:>2} scale={w_scale:.2f} "
                  f"Imax={i_max:>4.1f} | active={pct_active:5.1f}% res_sim={res_sim:.3f} "
                  f"sep={sep:5.2f} decode={dec:.2f} [{regime}]  eta {eta / 60:4.1f}m")

    df = pd.DataFrame(results)
    out_path = os.path.join(DATA_DIR, "day3_7_calibration_results.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print(f"Saved {out_path}")
    print("\nRegime counts by sign mode:")
    print(pd.crosstab(df["sign_mode"], df["regime"]).to_string())

    useful = df[df["regime"] == "useful"]
    if useful.empty:
        print("\nNo 'useful' configs. Paste this output back for the next adjustment.")
        return
    top = useful.sort_values(["decode_acc", "sep_ratio"], ascending=False).head(10)
    print("\nTop 10 useful configs (by decode accuracy, then separation ratio):")
    print(top.to_string(index=False))
    print(f"\nChance = {1 / len(pos_names):.3f}. Want decode_acc near 1.0 and sep_ratio >> 1.")


if __name__ == "__main__":
    main()
