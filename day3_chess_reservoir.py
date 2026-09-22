"""
DAY 3 — Chess -> Real Drosophila Connectome Reservoir

Research goal:
    Test whether different chess positions produce distinguishable
    activity states in the fixed 68-neuron Drosophila connectome reservoir.

IMPORTANT:
    - Connectome weights remain fixed.
    - Input projection remains fixed.
    - No readout training yet.
    - No chess move prediction yet.
    - Each position starts from the same resting state.

Pipeline:

    chess position
          |
          v
    768 board features
          |
          v
    fixed random projection
          |
          v
    6 input spike channels
          |
          v
    REAL 68-neuron connectome
          |
          v
    68-dimensional spike-rate state

Outputs:
    day3_reservoir_states.npz
    day3_position_summary.csv
    day3_state_similarity.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import chess

from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    PoissonGroup,
    run,
    ms,
    mV,
    Hz,
    prefs,
    seed as brian_seed,
)

# Use NumPy code generation.
prefs.codegen.target = "numpy"


# ======================================================================
# CONFIGURATION
# ======================================================================

N_INPUT_NEURONS = 6
N_FEATURES = 64 * 12

SIM_DURATION_MS = 300.0

RANDOM_SEED = 0

# Same connectome scaling used in the successful Day-2 experiment.
WEIGHT_SCALE = 0.08 * mV

# Chess -> reservoir input.
MIN_RATE_HZ = 5.0
MAX_RATE_HZ = 200.0
INPUT_WEIGHT = 6.0 * mV

# LIF parameters.
TAU = 10 * ms
V_REST = -65 * mV
V_THRESHOLD = -50 * mV
V_RESET = -65 * mV
REFRACTORY = 3 * ms


# ======================================================================
# LOAD CONNECTOME
# ======================================================================

def load_subnetwork(data_dir):

    adjacency_path = os.path.join(
        data_dir,
        "subnetwork_adjacency.npz",
    )

    neurons_path = os.path.join(
        data_dir,
        "subnetwork_neurons.csv",
    )

    bodyids_path = os.path.join(
        data_dir,
        "subnetwork_bodyids.npy",
    )

    adj = sp.load_npz(adjacency_path).tocoo()

    neuron_meta = pd.read_csv(
        neurons_path
    )

    bodyids = np.load(
        bodyids_path
    )

    assert adj.shape[0] == adj.shape[1], (
        "Adjacency matrix is not square."
    )

    assert adj.shape[0] == len(neuron_meta), (
        "Adjacency and neuron metadata have different sizes."
    )

    assert adj.shape[0] == len(bodyids), (
        "Adjacency and bodyId array have different sizes."
    )

    return adj, neuron_meta, bodyids


# ======================================================================
# CHESS ENCODER
# ======================================================================

# Six white piece types followed by six black piece types.
PIECE_TYPES = [
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),

    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
]


def board_to_features(board):

    features = np.zeros(
        N_FEATURES,
        dtype=np.float32,
    )

    for plane, (piece_type, color) in enumerate(
        PIECE_TYPES
    ):

        for square in board.pieces(
            piece_type,
            color,
        ):

            index = plane * 64 + square

            features[index] = 1.0

    return features


# ======================================================================
# FIXED INPUT PROJECTION
# ======================================================================

def make_projection(seed):

    rng = np.random.default_rng(seed)

    projection = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            N_INPUT_NEURONS,
            N_FEATURES,
        ),
    ).astype(np.float32)

    # Normalize every chess feature's projection vector.
    projection /= (
        np.linalg.norm(
            projection,
            axis=0,
            keepdims=True,
        )
        + 1e-8
    )

    return projection


def features_to_rates(
    features,
    projection,
):

    raw = projection @ features

    # Fixed nonlinear activation.
    activated = 1.0 / (
        1.0 + np.exp(-raw)
    )

    rates = (
        MIN_RATE_HZ
        + activated
        * (MAX_RATE_HZ - MIN_RATE_HZ)
    )

    return rates.astype(
        np.float32
    )


# ======================================================================
# BUILD REAL CONNECTOME RESERVOIR
# ======================================================================

def build_reservoir(
    adj,
    input_indices,
):

    N = adj.shape[0]

    # --------------------------------------------------------------
    # LIF neurons
    #
    # IMPORTANT:
    # Brian2 threshold/reset expressions are deliberately written
    # as literal Brian2 strings. Do NOT replace them with an f-string
    # containing "-50*mV", because Python formatting can produce
    # "-50. mV", which Brian2 cannot parse.
    # --------------------------------------------------------------

    eqs = """
    dv/dt = (v_rest - v) / tau : volt (unless refractory)
    v_rest : volt
    tau : second
    """

    neurons = NeuronGroup(
        N,
        eqs,
        threshold="v > -50*mV",
        reset="v = -65*mV",
        refractory=3 * ms,
        method="euler",
    )

    neurons.v = -65 * mV
    neurons.v_rest = -65 * mV
    neurons.tau = 10 * ms

    # --------------------------------------------------------------
    # REAL CONNECTOME CONNECTIONS
    # --------------------------------------------------------------

    syn = Synapses(
        neurons,
        neurons,
        model="w : volt",
        on_pre="v_post += w",
    )

    syn.connect(
        i=adj.row,
        j=adj.col,
    )

    syn.w = adj.data * WEIGHT_SCALE

    # --------------------------------------------------------------
    # EXTERNAL CHESS INPUT
    # --------------------------------------------------------------

    drive = PoissonGroup(
        len(input_indices),
        rates=0 * Hz,
    )

    drive_syn = Synapses(
        drive,
        neurons,
        model="",
        on_pre="v_post += INPUT_WEIGHT",
    )

    drive_syn.connect(
        i=np.arange(
            len(input_indices)
        ),
        j=input_indices,
    )

    return (
        neurons,
        syn,
        drive,
        drive_syn,
    )


# ======================================================================
# RUN ONE POSITION
# ======================================================================

def run_position(
    board,
    projection,
    adj,
    input_indices,
    seed,
    duration_ms,
):

    # --------------------------------------------------------------
    # Board -> 768 features
    # --------------------------------------------------------------

    features = board_to_features(
        board
    )

    # --------------------------------------------------------------
    # 768 features -> six input rates
    # --------------------------------------------------------------

    rates = features_to_rates(
        features,
        projection,
    )

    # --------------------------------------------------------------
    # Make a fresh reservoir.
    #
    # Every chess position therefore begins from:
    #
    #     v = -65 mV
    #
    # This prevents one position from contaminating the next.
    # --------------------------------------------------------------

    brian_seed(seed)

    (
        neurons,
        syn,
        drive,
        drive_syn,
    ) = build_reservoir(
        adj,
        input_indices,
    )

    # Set the six Poisson input rates.
    drive.rates = rates * Hz

    # Monitor every reservoir neuron.
    spike_monitor = SpikeMonitor(
        neurons
    )

    # --------------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------------

    run(
        duration_ms * ms
    )

    # --------------------------------------------------------------
    # Convert spikes -> 68-dimensional state
    # --------------------------------------------------------------

    spike_counts = np.bincount(
        np.asarray(
            spike_monitor.i,
            dtype=int,
        ),
        minlength=adj.shape[0],
    ).astype(
        np.float32
    )

    spike_rates = (
        spike_counts
        / (duration_ms / 1000.0)
    )

    return {
        "features": features,
        "input_rates": rates,
        "spike_counts": spike_counts,
        "spike_rates": spike_rates,
        "total_spikes": int(
            spike_monitor.num_spikes
        ),
        "active_neurons": int(
            np.count_nonzero(
                spike_counts
            )
        ),
    }


# ======================================================================
# TEST CHESS POSITIONS
# ======================================================================

def make_test_positions():

    positions = {}

    # --------------------------------------------------------------
    # Starting position
    # --------------------------------------------------------------

    positions["start"] = chess.Board()

    # --------------------------------------------------------------
    # 1.e4
    # --------------------------------------------------------------

    board = chess.Board()

    board.push_san("e4")

    positions["after_e4"] = board.copy()

    # --------------------------------------------------------------
    # 1.e4 e5 2.Nf3
    # --------------------------------------------------------------

    board.push_san("e5")
    board.push_san("Nf3")

    positions[
        "after_e4_e5_Nf3"
    ] = board.copy()

    # --------------------------------------------------------------
    # 1.d4
    # --------------------------------------------------------------

    board = chess.Board()

    board.push_san("d4")

    positions["after_d4"] = board.copy()

    # --------------------------------------------------------------
    # 1.d4 Nf6 2.c4
    # --------------------------------------------------------------

    board.push_san("Nf6")
    board.push_san("c4")

    positions[
        "after_d4_Nf6_c4"
    ] = board.copy()

    # --------------------------------------------------------------
    # Ruy Lopez exchange structure
    # --------------------------------------------------------------

    board = chess.Board()

    moves = [
        "e4",
        "e5",
        "Nf3",
        "Nc6",
        "Bb5",
        "a6",
        "Bxc6",
        "dxc6",
    ]

    for move in moves:
        board.push_san(move)

    positions[
        "ruy_lopez_exchange"
    ] = board.copy()

    return positions


# ======================================================================
# COSINE SIMILARITY
# ======================================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def pairwise_similarity(states):

    names = list(
        states.keys()
    )

    matrix = np.zeros(
        (
            len(names),
            len(names),
        ),
        dtype=np.float32,
    )

    for i, name_a in enumerate(
        names
    ):

        for j, name_b in enumerate(
            names
        ):

            matrix[i, j] = (
                cosine_similarity(
                    states[name_a],
                    states[name_b],
                )
            )

    return names, matrix


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default=os.path.dirname(
            os.path.abspath(__file__)
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=SIM_DURATION_MS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "DAY 3 — CHESS → REAL CONNECTOME RESERVOIR"
    )
    print("=" * 72)

    # --------------------------------------------------------------
    # Load connectome
    # --------------------------------------------------------------

    adj, neuron_meta, bodyids = (
        load_subnetwork(
            args.data_dir
        )
    )

    print()
    print(
        f"Connectome: "
        f"{adj.shape[0]} neurons, "
        f"{adj.nnz} directed connections"
    )

    # --------------------------------------------------------------
    # Select current six input neurons.
    #
    # This preserves the successful Day-2 input selection.
    # We will later test alternative input populations.
    # --------------------------------------------------------------

    out_degree = np.asarray(
        adj.tocsr().sum(axis=1)
    ).flatten()

    input_indices = np.argsort(
        -out_degree
    )[:N_INPUT_NEURONS]

    print()
    print(
        "Input neurons:",
        sorted(
            input_indices.tolist()
        ),
    )

    print(
        "Input cell types:",
        neuron_meta.iloc[
            input_indices
        ]["type"].tolist(),
    )

    # --------------------------------------------------------------
    # Fixed input projection
    # --------------------------------------------------------------

    projection = make_projection(
        args.seed
    )

    # --------------------------------------------------------------
    # Chess test set
    # --------------------------------------------------------------

    positions = (
        make_test_positions()
    )

    print()
    print(
        f"Testing {len(positions)} "
        "chess positions..."
    )

    states = {}

    summary_rows = []

    # --------------------------------------------------------------
    # Run each position
    # --------------------------------------------------------------

    for name, board in (
        positions.items()
    ):

        print()
        print("-" * 72)
        print(name)

        print(board)

        result = run_position(
            board=board,
            projection=projection,
            adj=adj,
            input_indices=input_indices,
            seed=args.seed,
            duration_ms=args.duration,
        )

        states[name] = (
            result["spike_rates"]
        )

        summary_rows.append(
            {
                "position": name,
                "fen": board.fen(),
                "active_board_features":
                    int(
                        result[
                            "features"
                        ].sum()
                    ),
                "total_spikes":
                    result[
                        "total_spikes"
                    ],
                "active_reservoir_neurons":
                    result[
                        "active_neurons"
                    ],
                "mean_input_rate_hz":
                    float(
                        result[
                            "input_rates"
                        ].mean()
                    ),
                "max_input_rate_hz":
                    float(
                        result[
                            "input_rates"
                        ].max()
                    ),
            }
        )

        print(
            "Total spikes:",
            result["total_spikes"],
        )

        print(
            "Active reservoir neurons:",
            result["active_neurons"],
            "/",
            adj.shape[0],
        )

        print(
            "Input rates:",
            np.round(
                result[
                    "input_rates"
                ],
                1,
            ),
        )

    # --------------------------------------------------------------
    # Similarity matrix
    # --------------------------------------------------------------

    names, similarity = (
        pairwise_similarity(
            states
        )
    )

    print()
    print("=" * 72)
    print(
        "RESERVOIR STATE SIMILARITY"
    )
    print("=" * 72)

    print()
    print(
        "Cosine similarity of "
        "68-dimensional spike-rate vectors:"
    )

    header = (
        " " * 28
    )

    for name in names:

        header += (
            f"{name[:12]:>14}"
        )

    print(header)

    for i, name in enumerate(
        names
    ):

        row = (
            f"{name[:26]:26}"
        )

        for j in range(
            len(names)
        ):

            row += (
                f"{similarity[i, j]:14.3f}"
            )

        print(row)

    # --------------------------------------------------------------
    # Mean pairwise similarity
    # --------------------------------------------------------------

    pair_values = []

    for i in range(
        len(names)
    ):

        for j in range(
            i + 1,
            len(names),
        ):

            pair_values.append(
                similarity[i, j]
            )

    mean_pair_similarity = float(
        np.mean(pair_values)
    )

    print()
    print(
        f"Mean pairwise similarity: "
        f"{mean_pair_similarity:.3f}"
    )

    # --------------------------------------------------------------
    # Save numerical state
    # --------------------------------------------------------------

    state_matrix = np.vstack(
        [
            states[name]
            for name in names
        ]
    )

    np.savez(
        os.path.join(
            args.data_dir,
            "day3_reservoir_states.npz",
        ),
        states=state_matrix,
        position_names=np.array(
            names
        ),
        similarity=similarity,
        projection=projection,
        input_indices=input_indices,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        os.path.join(
            args.data_dir,
            "day3_position_summary.csv",
        ),
        index=False,
    )

    print()
    print("Saved:")
    print(
        "  day3_reservoir_states.npz"
    )
    print(
        "  day3_position_summary.csv"
    )

    # --------------------------------------------------------------
    # Similarity plot
    # --------------------------------------------------------------

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        image = ax.imshow(
            similarity,
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        ax.set_xticks(
            range(len(names))
        )

        ax.set_yticks(
            range(len(names))
        )

        ax.set_xticklabels(
            names,
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            names
        )

        ax.set_title(
            "Day 3: Reservoir-state similarity"
        )

        ax.set_xlabel(
            "Chess position"
        )

        ax.set_ylabel(
            "Chess position"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="cosine similarity",
        )

        fig.tight_layout()

        output_path = os.path.join(
            args.data_dir,
            "day3_state_similarity.png",
        )

        fig.savefig(
            output_path,
            dpi=150,
        )

        plt.close(fig)

        print(
            "  day3_state_similarity.png"
        )

    except ImportError:

        print()
        print(
            "matplotlib not installed; "
            "skipping similarity plot."
        )

    # --------------------------------------------------------------
    # Final status
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("DAY 3 COMPLETE")
    print("=" * 72)

    print()
    print(
        "Different chess positions have now been "
        "run through the fixed connectome reservoir."
    )

    print()
    print(
        "This does NOT yet demonstrate chess ability."
    )

    print(
        "It tests whether the reservoir produces "
        "different internal states for different positions."
    )

    print()
    print(
        "Next: train a readout to predict chess moves."
    )


if __name__ == "__main__":
    main()

