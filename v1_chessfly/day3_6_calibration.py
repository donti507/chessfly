"""
DAY 3.6 -- Calibration sweep v2

Fixes from Day 3.5:
  1. Inhibition: synapses signed by presynaptic neurotransmitter
     (needs subnetwork_nt.csv from fetch_nt.py; unsigned runs as a control).
  2. Finer weight grid through the 0.08 -> 0.20 phase transition.
  3. More input neurons (6 vs 24) and a projection scaled so the sigmoid
     isn't saturated (Day 3.5 inputs were pinned near 5 / 200 Hz).
  4. Metrics that mean something:
       input_sim       - cosine similarity of the INPUT rate vectors (reference)
       reservoir_sim   - cosine similarity of reservoir states (hidden neurons only)
       sep_ratio       - between-position distance / within-position (seed) distance
                         > 1 means position signal beats Poisson noise
       decode_acc      - leave-one-seed-out nearest-centroid accuracy on the
                         4 positions using hidden neurons only (chance = 0.25)
     Input neurons are EXCLUDED from the state vector so we measure what the
     connectome adds, not the input passing straight through.
  5. Network built once per config and store()/restore()d -> much faster.

Output: day3_6_calibration_results.csv
"""

import os
import time
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
import chess

from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, PoissonGroup, Network,
    ms, mV, Hz, prefs, seed as brian_seed,
)

prefs.codegen.target = "numpy"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
N_FEATURES = 64 * 12
MIN_RATE_HZ = 5.0
MAX_RATE_HZ = 200.0
PROJ_SEED = 0
SIM_SEEDS = [0, 1, 2]            # different Poisson noise -> noise floor
DURATION_MS = 300.0
TYPICAL_PIECES = 32              # used to scale the projection
INPUT_STRATEGY = "diverse_types"

# ---- Sweep grid ----
WEIGHT_SCALES_MV = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20]
INPUT_WEIGHTS_MV = [6.0, 10.0]
SIGN_MODES = ["signed", "signed_inh2x"]
N_INPUTS = [6, 24]

PIECE_TYPES = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]

LIF_EQS = """
dv/dt = (v_rest - v) / tau : volt (unless refractory)
v_rest : volt
tau : second
"""


# ---------------------------------------------------------------- encoding
def board_to_features(board):
    features = np.zeros(N_FEATURES, dtype=np.float32)
    for plane, (piece_type, color) in enumerate(PIECE_TYPES):
        for square in board.pieces(piece_type, color):
            features[plane * 64 + square] = 1.0
    return features


def make_test_positions():
    positions = {"start": chess.Board()}
    b = chess.Board(); b.push_san("e4"); positions["after_e4"] = b
    b = chess.Board(); b.push_san("d4"); positions["after_d4"] = b
    b = chess.Board()
    for mv in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6"]:
        b.push_san(mv)
    positions["ruy_lopez_exchange"] = b
    return positions


def make_projection(n_input, seed):
    # Scaled so projection @ features has std ~1 for a ~32-piece board,
    # keeping the sigmoid in its sensitive range instead of pinned at the ends.
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 1.0, size=(n_input, N_FEATURES))
            / np.sqrt(TYPICAL_PIECES)).astype(np.float32)


def features_to_rates(features, projection):
    raw = projection @ features
    activated = 1.0 / (1.0 + np.exp(-raw))
    return (MIN_RATE_HZ + activated * (MAX_RATE_HZ - MIN_RATE_HZ)).astype(np.float32)


# ---------------------------------------------------------------- network
def select_input_neurons(adj, neuron_meta, n_input, strategy):
    out_degree = np.asarray(adj.tocsr().sum(axis=1)).flatten()
    order = np.argsort(-out_degree)
    if strategy == "out_degree":
        return order[:n_input]
    if strategy == "diverse_types":
        seen, chosen = set(), []
        for idx in order:
            t = neuron_meta.iloc[idx]["type"]
            if t not in seen:
                seen.add(t)
                chosen.append(idx)
            if len(chosen) == n_input:
                break
        for idx in order:                      # top up if too few distinct types
            if len(chosen) == n_input:
                break
            if idx not in chosen:
                chosen.append(idx)
        return np.array(chosen)
    raise ValueError(f"unknown strategy {strategy}")


def load_signs(n_neurons):
    path = os.path.join(DATA_DIR, "subnetwork_nt.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).sort_values("index")
    if len(df) != n_neurons:
        raise ValueError(f"subnetwork_nt.csv has {len(df)} rows, expected {n_neurons}")
    return df["sign"].to_numpy(dtype=float)


def synapse_weights(adj, weight_scale_mv, sign_mode, signs):
    w = adj.data.astype(float) * weight_scale_mv
    if sign_mode == "unsigned":
        return w
    inh_gain = 2.0 if sign_mode == "signed_inh2x" else 1.0
    pre_sign = signs[adj.row]
    pre_sign = np.where(pre_sign < 0, -inh_gain, 1.0)
    return w * pre_sign


def build_network(adj, input_indices, weights_mv, input_weight_mv):
    n = adj.shape[0]
    neurons = NeuronGroup(n, LIF_EQS, threshold="v > -50*mV", reset="v = -65*mV",
                          refractory=3 * ms, method="euler")
    neurons.v = -65 * mV
    neurons.v_rest = -65 * mV
    neurons.tau = 10 * ms

    # clip keeps inhibition from driving v to absurd negative values
    syn = Synapses(neurons, neurons, model="w : volt",
                   on_pre="v_post = clip(v_post + w, -80*mV, 0*mV)")
    syn.connect(i=adj.row, j=adj.col)
    syn.w = weights_mv * mV

    drive = PoissonGroup(len(input_indices), rates=0 * Hz)
    drive_syn = Synapses(drive, neurons, on_pre=f"v_post += {input_weight_mv}*mV")
    drive_syn.connect(i=np.arange(len(input_indices)), j=input_indices)

    monitor = SpikeMonitor(neurons)
    net = Network(neurons, syn, drive, drive_syn, monitor)
    net.store()
    return net, drive, monitor


def simulate(net, drive, monitor, input_rates_hz, sim_seed, n_neurons):
    net.restore()
    drive.rates = input_rates_hz * Hz
    brian_seed(sim_seed)
    net.run(DURATION_MS * ms, namespace={})   # empty namespace: no local-variable clashes
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
    """states: array (P positions, S seeds, H hidden neurons) of firing rates."""
    P, S, _ = states.shape

    within = [np.linalg.norm(states[p, a] - states[p, b])
              for p in range(P) for a, b in itertools.combinations(range(S), 2)]
    between = [np.linalg.norm(states[p, a] - states[q, b])
               for p, q in itertools.combinations(range(P), 2)
               for a in range(S) for b in range(S)]
    within_d, between_d = float(np.mean(within)), float(np.mean(between))
    sep_ratio = between_d / within_d if within_d > 1e-9 else np.nan

    # reservoir cosine similarity between positions (same seed), averaged over seeds
    res_sim = np.nanmean([mean_pairwise_cosine([states[p, s] for p in range(P)])
                          for s in range(S)])

    # leave-one-seed-out nearest-centroid decoding
    correct = 0
    for t in range(S):
        train = [s for s in range(S) if s != t]
        centroids = states[:, train, :].mean(axis=1)
        for p in range(P):
            dists = np.linalg.norm(centroids - states[p, t], axis=1)
            if np.allclose(dists, dists[0]):   # no information at all
                continue
            correct += int(np.argmin(dists) == p)
    decode_acc = correct / (P * S)

    return within_d, between_d, sep_ratio, float(res_sim), decode_acc


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
    sign_modes = SIGN_MODES
    if signs is None:
        print("NOTE: subnetwork_nt.csv not found -> running UNSIGNED only. "
              "Run fetch_nt.py first to test inhibition.")
        sign_modes = ["unsigned"]
    else:
        inh_neurons = int((signs < 0).sum())
        inh_frac = float(adj.data[signs[adj.row] < 0].sum() / adj.data.sum())
        print(f"Signs loaded: {inh_neurons}/{n_neurons} inhibitory neurons, "
              f"{100 * inh_frac:.1f}% of synapse weight is inhibitory")
        if inh_neurons == 0:
            print("WARNING: zero inhibitory neurons -- signed runs will equal unsigned.")

    positions = make_test_positions()
    pos_names = list(positions.keys())
    features = {k: board_to_features(b) for k, b in positions.items()}

    # input-level reference similarity per input size
    projections, input_rates, input_sims = {}, {}, {}
    for n_in in N_INPUTS:
        projections[n_in] = make_projection(n_in, PROJ_SEED)
        input_rates[n_in] = {k: features_to_rates(features[k], projections[n_in])
                             for k in pos_names}
        input_sims[n_in] = mean_pairwise_cosine(list(input_rates[n_in].values()))
        print(f"Input-level similarity with {n_in:>2} input neurons: {input_sims[n_in]:.3f}")

    combos = list(itertools.product(sign_modes, N_INPUTS, WEIGHT_SCALES_MV, INPUT_WEIGHTS_MV))
    n_sims = len(combos) * len(pos_names) * len(SIM_SEEDS)
    print("=" * 78)
    print(f"DAY 3.6 -- CALIBRATION SWEEP v2  ({n_neurons} neurons, {adj.nnz} connections)")
    print(f"{len(combos)} configs x {len(pos_names)} positions x {len(SIM_SEEDS)} seeds "
          f"= {n_sims} simulations")
    print("=" * 78)

    results = []
    t0 = time.time()
    for ci, (sign_mode, n_in, w_scale, in_w) in enumerate(combos, 1):
        input_idx = select_input_neurons(adj, neuron_meta, n_in, INPUT_STRATEGY)
        hidden_mask = np.ones(n_neurons, dtype=bool)
        hidden_mask[input_idx] = False

        weights = synapse_weights(adj, w_scale, sign_mode, signs)
        net, drive, monitor = build_network(adj, input_idx, weights, in_w)

        states = np.zeros((len(pos_names), len(SIM_SEEDS), hidden_mask.sum()), np.float32)
        active_fracs = []
        for p, name in enumerate(pos_names):
            for s, sim_seed in enumerate(SIM_SEEDS):
                rates = simulate(net, drive, monitor, input_rates[n_in][name],
                                 sim_seed, n_neurons)
                states[p, s] = rates[hidden_mask]
                active_fracs.append(np.count_nonzero(rates[hidden_mask]) / hidden_mask.sum())

        within_d, between_d, sep, res_sim, dec = evaluate(states)
        pct_active = 100 * float(np.mean(active_fracs))
        regime = classify_regime(pct_active, res_sim)

        results.append({
            "sign_mode": sign_mode,
            "n_input": n_in,
            "weight_scale_mV": w_scale,
            "input_weight_mV": in_w,
            "pct_hidden_active": round(pct_active, 1),
            "mean_hidden_rate_hz": round(float(states.mean()), 2),
            "input_sim": round(input_sims[n_in], 4),
            "reservoir_sim": round(res_sim, 4),
            "within_dist": round(within_d, 2),
            "between_dist": round(between_d, 2),
            "sep_ratio": round(sep, 3) if not np.isnan(sep) else np.nan,
            "decode_acc": round(dec, 3),
            "regime": regime,
        })

        elapsed = time.time() - t0
        eta = elapsed / ci * (len(combos) - ci)
        print(f"[{ci:>3}/{len(combos)}] {sign_mode:<13} in={n_in:>2} scale={w_scale:.2f} "
              f"inW={in_w:>4.1f} | active={pct_active:5.1f}% res_sim={res_sim:.3f} "
              f"sep={sep:5.2f} decode={dec:.2f} [{regime}]  eta {eta/60:4.1f}m")

    df = pd.DataFrame(results)
    out_path = os.path.join(DATA_DIR, "day3_6_calibration_results.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print(f"Saved {out_path}")
    print("\nRegime counts by sign mode:")
    print(pd.crosstab(df["sign_mode"], df["regime"]).to_string())

    useful = df[df["regime"] == "useful"]
    if useful.empty:
        print("\nNo 'useful' configs (all silent or saturated). Paste this output back "
              "for the next adjustment.")
        return

    top = useful.sort_values(["decode_acc", "sep_ratio"], ascending=False).head(10)
    print("\nTop 10 useful configs (by decode accuracy, then separation ratio):")
    print(top.to_string(index=False))
    print("\nChance decode accuracy = 0.25. Look for decode_acc near 1.0 AND sep_ratio "
          "well above 1 -- that's position signal clearly beating noise.")


if __name__ == "__main__":
    main()
