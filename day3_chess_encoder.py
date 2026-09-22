"""
day3_chess_encoder.py

Day 3: Chess board -> spike pattern.

Goal:
    Convert a chess position into deterministic spike trains that can be
    injected into the existing 6-neuron input interface of the Day 2
    connectome reservoir.

This is intentionally a simple v1 encoder.

Encoding:
    64 squares x 12 piece planes = 768 binary features.

    Each feature is assigned a deterministic spike probability/rate.
    Features are projected onto the 6 physical input neurons using a fixed
    random projection matrix.

    The resulting 6 input-neuron rates are converted into Poisson spike
    trains.

The connectome itself is NOT modified.

Run:
    python3 day3_chess_encoder.py

Dependencies:
    python-chess
    numpy
"""

import argparse
import os

import chess
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_INPUT_NEURONS = 6

# 768 = 64 board squares x 12 piece types
N_FEATURES = 64 * 12

RANDOM_SEED = 0

# Input rates sent to the reservoir.
MIN_RATE_HZ = 5.0
MAX_RATE_HZ = 200.0

# Number of milliseconds represented by one generated spike pattern.
ENCODING_WINDOW_MS = 300


# ---------------------------------------------------------------------------
# Piece encoding
# ---------------------------------------------------------------------------

# python-chess piece types:
#
#   1 = pawn
#   2 = knight
#   3 = bishop
#   4 = rook
#   5 = queen
#   6 = king
#
# We use separate planes for White and Black.
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
    """
    Convert a chess.Board into a 768-dimensional binary vector.

    Feature layout:

        plane 0  = white pawns
        plane 1  = white knights
        ...
        plane 5  = white kings
        plane 6  = black pawns
        ...
        plane 11 = black kings

    Within each plane, the 64 entries correspond to chess squares
    0..63 as defined by python-chess.
    """

    features = np.zeros(N_FEATURES, dtype=np.float32)

    for plane, (piece_type, color) in enumerate(PIECE_TYPES):

        for square in board.pieces(piece_type, color):

            index = plane * 64 + square
            features[index] = 1.0

    return features


# ---------------------------------------------------------------------------
# Fixed projection into the six physical input neurons
# ---------------------------------------------------------------------------

def make_projection(seed=RANDOM_SEED):
    """
    Create a deterministic random projection.

    IMPORTANT:
        This matrix is fixed.

    It is NOT trained and does NOT modify the connectome.

    Its only purpose is to compress the 768 chess features into six
    reservoir input channels.
    """

    rng = np.random.default_rng(seed)

    projection = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(N_INPUT_NEURONS, N_FEATURES),
    ).astype(np.float32)

    # Normalize each feature column so no feature gets an arbitrary
    # advantage merely because of projection magnitude.
    projection /= (
        np.linalg.norm(projection, axis=0, keepdims=True) + 1e-8
    )

    return projection


def features_to_rates(features, projection):
    """
    Convert board features into six Poisson input rates.

    We use the fixed projection followed by a sigmoid.

    Empty board positions and occupied positions therefore produce
    different reservoir input patterns.
    """

    raw = projection @ features

    # Sigmoid -> [0, 1]
    activated = 1.0 / (1.0 + np.exp(-raw))

    # Map to usable Poisson rates.
    rates = MIN_RATE_HZ + activated * (
        MAX_RATE_HZ - MIN_RATE_HZ
    )

    return rates.astype(np.float32)


# ---------------------------------------------------------------------------
# Poisson spike generation
# ---------------------------------------------------------------------------

def rates_to_spikes(rates_hz, duration_ms, seed=RANDOM_SEED):
    """
    Generate deterministic Poisson spike trains.

    Returns:
        list of arrays

    Each array contains spike times in milliseconds for one reservoir
    input neuron.

    The seed makes this reproducible for debugging.
    """

    rng = np.random.default_rng(seed)

    duration_s = duration_ms / 1000.0

    spike_trains = []

    for rate in rates_hz:

        expected_spikes = rate * duration_s

        n_spikes = rng.poisson(expected_spikes)

        if n_spikes == 0:
            spike_trains.append(np.array([], dtype=np.float32))
            continue

        times = rng.uniform(
            0.0,
            duration_ms,
            size=n_spikes,
        )

        times.sort()

        spike_trains.append(times.astype(np.float32))

    return spike_trains


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_board_encoding(board, features, rates, spike_trains):

    print()
    print("=" * 70)
    print("DAY 3 CHESS ENCODER")
    print("=" * 70)

    print()
    print("Position:")
    print(board)

    print()
    print("FEN:")
    print(board.fen())

    print()
    print("Active chess features:")
    print(int(features.sum()), "/", N_FEATURES)

    print()
    print("Reservoir input rates:")

    for i, rate in enumerate(rates):

        print(
            f"  input neuron {i:2d}: "
            f"{rate:7.2f} Hz   "
            f"{len(spike_trains[i]):3d} spikes"
        )

    print()
    print("Spike times (ms):")

    for i, spikes in enumerate(spike_trains):

        preview = spikes[:10]

        if len(spikes) > 10:
            suffix = " ..."
        else:
            suffix = ""

        print(
            f"  neuron {i}: "
            f"{np.array2string(preview, precision=1)}"
            f"{suffix}"
        )

    print()
    print("Encoder sanity checks:")

    checks = {
        "768-dimensional feature vector":
            features.shape == (N_FEATURES,),

        "binary features":
            np.all((features == 0) | (features == 1)),

        "six input rates":
            rates.shape == (N_INPUT_NEURONS,),

        "rates finite":
            np.all(np.isfinite(rates)),

        "rates within configured range":
            np.all((rates >= MIN_RATE_HZ) &
                   (rates <= MAX_RATE_HZ)),

        "six spike trains":
            len(spike_trains) == N_INPUT_NEURONS,
    }

    all_pass = True

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {name}")

        all_pass &= passed

    print()

    if all_pass:
        print("PASS: chess position successfully encoded into")
        print("      six deterministic reservoir input channels.")
    else:
        print("FAIL: encoder sanity check failed.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fen",
        default=None,
        help="Chess FEN to encode. Defaults to the starting position.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for the fixed projection/spike generation.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=ENCODING_WINDOW_MS,
        help="Encoding window in milliseconds.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Board
    # ---------------------------------------------------------------

    if args.fen is None:

        board = chess.Board()

    else:

        board = chess.Board(args.fen)

    # ---------------------------------------------------------------
    # Encode
    # ---------------------------------------------------------------

    features = board_to_features(board)

    projection = make_projection(args.seed)

    rates = features_to_rates(
        features,
        projection,
    )

    spike_trains = rates_to_spikes(
        rates,
        args.duration,
        args.seed,
    )

    # ---------------------------------------------------------------
    # Print result
    # ---------------------------------------------------------------

    print_board_encoding(
        board,
        features,
        rates,
        spike_trains,
    )


if __name__ == "__main__":
    main()

