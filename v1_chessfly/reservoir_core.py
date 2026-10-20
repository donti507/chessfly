"""
reservoir_core.py -- the LOCKED fly-connectome reservoir (Day 3.7 winner)

Shared by every Day 4+ script (dataset building, training, playing, and the
Day 7 visualization) so boards are encoded and the network is run identically.

Locked config (Day 3.7: decode_acc 1.000, sep_ratio 11.6):
    434-neuron head-direction circuit, synapses signed by predicted NT,
    48 input neurons, weight scale 0.22 mV, I_max 30 mV, 1000 ms, 1 mV noise.

Boards are always shown to the network from the side-to-move's perspective:
if Black is to move, the board is mirrored (colors swapped, ranks flipped),
so the network only ever sees "White to move" positions.
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import chess

from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network,
    ms, mV, prefs, seed as brian_seed, BrianLogger,
)

prefs.codegen.target = "numpy"
BrianLogger.suppress_hierarchy("brian2.codegen.generators.base")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- locked config ----
N_INPUT = 48
WEIGHT_SCALE_MV = 0.22
I_MAX_MV = 30.0
DURATION_MS = 1000.0
NOISE_SIGMA_MV = 1.0
INPUT_GAIN = 1.5
PROJ_SEED = 0
REF_SEED = 123
N_REF_POSITIONS = 200
N_FEATURES = 64 * 12

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


# ---------------------------------------------------------------- chess helpers
def canonical(board):
    """Board from the side-to-move's perspective (always White to move)."""
    return board if board.turn == chess.WHITE else board.mirror()


def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      promotion=move.promotion)


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


class Encoder:
    """Canonical board -> activation in [0, 1] for each input neuron."""

    def __init__(self, n_input=N_INPUT, seed=PROJ_SEED):
        rng_ref = np.random.default_rng(REF_SEED)
        ref = [canonical(random_position(rng_ref, int(rng_ref.integers(0, 61))))
               for _ in range(N_REF_POSITIONS)]
        feats = np.stack([board_to_features(b) for b in ref])
        self.mean = feats.mean(axis=0)
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0.0, 1.0, size=(n_input, N_FEATURES)).astype(np.float32)
        raw = (feats - self.mean) @ self.proj.T
        self.scale = INPUT_GAIN / (raw.std(axis=0) + 1e-8)

    def activation(self, board):
        raw = (self.proj @ (board_to_features(board) - self.mean)) * self.scale
        return 1.0 / (1.0 + np.exp(-raw))


# ---------------------------------------------------------------- network
def select_input_neurons(adj, neuron_meta, n_input):
    """Highest out-degree neuron from each cell type first, then top up."""
    out_degree = np.asarray(adj.tocsr().sum(axis=1)).flatten()
    order = np.argsort(-out_degree)
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


class FlyReservoir:
    def __init__(self):
        adj = sp.load_npz(os.path.join(DATA_DIR, "subnetwork_adjacency.npz")).tocoo()
        self.neuron_meta = pd.read_csv(os.path.join(DATA_DIR, "subnetwork_neurons.csv"))
        nt = pd.read_csv(os.path.join(DATA_DIR, "subnetwork_nt.csv")).sort_values("index")
        self.n_neurons = adj.shape[0]
        signs = nt["sign"].to_numpy(dtype=float)

        self.input_idx = select_input_neurons(adj, self.neuron_meta, N_INPUT)
        self.hidden_mask = np.ones(self.n_neurons, dtype=bool)
        self.hidden_mask[self.input_idx] = False
        self.encoder = Encoder()

        weights = adj.data.astype(float) * WEIGHT_SCALE_MV * np.where(signs[adj.row] < 0, -1.0, 1.0)

        self.neurons = NeuronGroup(self.n_neurons, LIF_EQS, threshold="v > -50*mV",
                                   reset="v = -65*mV", refractory=3 * ms, method="euler")
        self.neurons.v = -65 * mV
        self.neurons.v_rest = -65 * mV
        self.neurons.tau = 10 * ms
        self.neurons.sigma = NOISE_SIGMA_MV * mV
        self.neurons.I_ext = 0 * mV
        self.neurons.run_regularly("v = clip(v, -80*mV, 0*mV)", when="after_synapses")

        self.syn = Synapses(self.neurons, self.neurons, model="w : volt", on_pre="v_post += w")
        self.syn.connect(i=adj.row, j=adj.col)
        self.syn.w = weights * mV

        self.monitor = SpikeMonitor(self.neurons)
        self.net = Network(self.neurons, self.syn, self.monitor)
        self.net.store()

    @property
    def n_hidden(self):
        return int(self.hidden_mask.sum())

    def run(self, board, sim_seed=0, record_spikes=False):
        """board must already be canonical (White to move)."""
        act = self.encoder.activation(board)
        self.net.restore()
        drive = np.zeros(self.n_neurons)
        drive[self.input_idx] = act * I_MAX_MV
        self.neurons.I_ext = drive * mV
        brian_seed(sim_seed)
        self.net.run(DURATION_MS * ms, namespace={})

        spike_i = np.asarray(self.monitor.i, dtype=int)
        counts = np.bincount(spike_i, minlength=self.n_neurons)
        rates = counts.astype(np.float32) / (DURATION_MS / 1000.0)
        out = {"activation": act.astype(np.float32),
               "hidden_rates": rates[self.hidden_mask],
               "all_rates": rates}
        if record_spikes:
            out["spike_i"] = spike_i
            out["spike_t_ms"] = np.asarray(self.monitor.t / ms)
        return out
