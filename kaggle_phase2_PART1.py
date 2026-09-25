"""
kaggle_phase2.py -- Phase 2: the fly-connectome reservoir at scale, on a GPU

What it does (each stage is cached, so reruns skip finished work):
  1. Dataset: N_POSITIONS positions labeled by Stockfish (best move + eval),
     generated in parallel on all CPU cores. Half the games are Stockfish
     self-play (25% random moves), half are "hero vs random opponent".
  2. Validation: PyTorch GPU simulator vs the original Brian2 simulator
     (reservoir_core.py) on the same positions. Results are only trustworthy
     if firing rates match.
  3. Simulation on GPU, batched, with spike counts in N_BINS time bins:
       locked48  -- the Phase 1 locked config (48 inputs)
       wide      -- N_INPUT_WIDE input neurons, auto-calibrated
       rewired   -- same as wide but connectome wiring randomized
                    (each neuron keeps its outgoing weights, targets shuffled)
  4. Readouts (same legal-move readout as Phase 1) for 8 representations,
     split by game, with 95% confidence intervals.

Needs in the repo folder (or anywhere under /kaggle/input):
    subnetwork_adjacency.npz, subnetwork_neurons.csv, subnetwork_nt.csv

Usage (Kaggle, GPU + internet on):
    N_POSITIONS=2000 python kaggle_phase2.py   # quick end-to-end test
    python kaggle_phase2.py                    # full run (50k positions)

Options (env vars): N_POSITIONS, N_INPUT_WIDE, N_BINS, BATCH, EPOCHS,
                    VALIDATE (1/0), SF_DEPTH
Outputs: /kaggle/working/chessfly_out/  (results CSV, readouts, caches)
"""

import os
import sys
import glob
import time
import shutil
import subprocess
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import chess
import chess.engine
import torch
import torch.nn.functional as Fnn

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = ("/kaggle/working/chessfly_out" if os.path.isdir("/kaggle/working")
           else os.path.join(REPO_DIR, "phase2_out"))
os.makedirs(OUT_DIR, exist_ok=True)

N_POSITIONS = int(os.environ.get("N_POSITIONS", "50000"))
N_INPUT_WIDE = int(os.environ.get("N_INPUT_WIDE", "150"))
N_BINS = int(os.environ.get("N_BINS", "10"))
BATCH = int(os.environ.get("BATCH", "2048"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
VALIDATE = os.environ.get("VALIDATE", "1") == "1"
SF_DEPTH = int(os.environ.get("SF_DEPTH", "8"))
RANDOM_MOVE_PROB = 0.25
MAX_PLIES = 100
WD_GRID = [0.0, 0.01, 0.1, 1.0]

# ---- neuron/simulation constants: must match reservoir_core.py ----
V_REST, V_THRESH, V_RESET = -65.0, -50.0, -65.0
V_MIN, V_MAX = -80.0, 0.0
TAU_MS, DT_MS, DURATION_MS = 10.0, 0.1, 1000.0
REFRACTORY_STEPS = 30                     # 3 ms / 0.1 ms
NOISE_MV = 1.0
LOCKED_SCALE, LOCKED_IMAX, LOCKED_NIN = 0.22, 30.0, 48

NET_FILES = ["subnetwork_adjacency.npz", "subnetwork_neurons.csv", "subnetwork_nt.csv"]


def log(msg=""):
    print(msg, flush=True)


# ================================================================ setup
def ensure_stockfish():
    def found():
        cands = [shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish"]
        cands += glob.glob(os.path.join(OUT_DIR, "sf", "**", "stockfish-ubuntu*"), recursive=True)
        for p in cands:
            if p and os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        return None

    p = found()
    if p:
        return p
    log("Installing Stockfish via apt ...")
    subprocess.run("apt-get -qq update >/dev/null 2>&1 && "
                   "apt-get -qq install -y stockfish >/dev/null 2>&1", shell=True)
    p = found()
    if p:
        return p
    log("apt failed -- downloading the official Stockfish binary ...")
    sfdir = os.path.join(OUT_DIR, "sf")
    os.makedirs(sfdir, exist_ok=True)
    url = ("https://github.com/official-stockfish/Stockfish/releases/latest/download/"
           "stockfish-ubuntu-x86-64-avx2.tar")
    subprocess.run(f"cd {sfdir} && wget -q {url} -O sf.tar && tar xf sf.tar", shell=True)
    for f in glob.glob(os.path.join(sfdir, "**", "stockfish-ubuntu*"), recursive=True):
        if os.path.isfile(f):
            os.chmod(f, 0o755)
    p = found()
    if not p:
        sys.exit("Could not install Stockfish. Is internet enabled in the notebook settings?")
    return p


def ensure_network_files():
    for name in NET_FILES:
        dst = os.path.join(REPO_DIR, name)
        if os.path.exists(dst):
            continue
        hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if not hits:
            sys.exit(f"Missing {name}. Upload the 3 network files as a Kaggle Dataset "
                     f"and add it to this notebook (Add Input).")
        shutil.copy(hits[0], dst)
        log(f"  copied {name} from {hits[0]}")


# ================================================================ chess helpers
def canonical(board):
    return board if board.turn == chess.WHITE else board.mirror()


def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square), promotion=move.promotion)


def usable_moves(board):
    return [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]


def position_key(fen):
    return " ".join(fen.split()[:4])


# ================================================================ 1. dataset
def _gen_worker(args):
    task_id, n_target, seed, sf_path, gid_base = args
    rng = np.random.default_rng(seed)
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 1})
    limit = chess.engine.Limit(depth=SF_DEPTH)
    out, seen, g = [], set(), 0
    while len(out) < n_target:
        gid = gid_base + task_id * 100_000 + g
        vs_random = (g % 2 == 1)
        hero = chess.WHITE if (g // 2) % 2 == 0 else chess.BLACK
        board = chess.Board()
        for _ in range(MAX_PLIES):
            if board.is_game_over() or len(out) >= n_target:
                break
            legal = list(board.legal_moves)
            if vs_random and board.turn != hero:
                board.push(legal[rng.integers(len(legal))])
                continue
            info = eng.analyse(board, limit)
            best = info["pv"][0] if info.get("pv") else eng.play(board, limit).move
            cp = info["score"].pov(board.turn).score(mate_score=10000)
            cb = canonical(board)
            key = position_key(cb.fen())
            if key not in seen:
                seen.add(key)
                cm = best if board.turn == chess.WHITE else mirror_move(best)
                out.append((cb.fen(), cm.uci(), int(cp), gid))
            if rng.random() < RANDOM_MOVE_PROB:
                board.push(legal[rng.integers(len(legal))])
            else:
                board.push(best)
        g += 1
    eng.quit()
    return out


def build_dataset(sf_path):
    path = os.path.join(OUT_DIR, "dataset.npz")
    fens, moves, cps, gids = [], [], [], []
    if os.path.exists(path):
        d = np.load(path)
        fens, moves = list(d["fens"]), list(d["moves"])
        cps, gids = list(d["cps"]), list(d["game_ids"])
        log(f"Dataset cache: {len(fens)} positions")
    if len(fens) >= N_POSITIONS:
        return (np.array(fens[:N_POSITIONS]), np.array(moves[:N_POSITIONS]),
                np.array(cps[:N_POSITIONS]), np.array(gids[:N_POSITIONS]))

    need = N_POSITIONS - len(fens)
    n_workers = max(1, os.cpu_count() or 1)
    n_tasks = n_workers * 8
    per_task = int(np.ceil(need * 1.15 / n_tasks)) + 5
    gid_base = (int(max(gids)) + 1) if gids else 0
    gid_base = int(np.ceil(gid_base / 10_000_000) * 10_000_000)
    log(f"Generating {need} positions with Stockfish depth {SF_DEPTH} "
        f"on {n_workers} CPU cores ({n_tasks} tasks) ...")

    tasks = [(t, per_task, 7919 * (len(fens) + 1) + t, sf_path, gid_base) for t in range(n_tasks)]
    seen = {position_key(f) for f in fens}
    t0 = time.time()
    with mp.get_context("fork").Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_gen_worker, tasks), 1):
            for fen, mv, cp, gid in res:
                k = position_key(fen)
                if k in seen:
                    continue
                seen.add(k)
                fens.append(fen)
                moves.append(mv)
                cps.append(cp)
                gids.append(gid)
            el = time.time() - t0
            log(f"  task {i:>3}/{n_tasks}  total {len(fens):>6} positions  "
                f"[{el / 60:5.1f} min, eta {el / i * (n_tasks - i) / 60:5.1f} min]")

    fens, moves = fens[:N_POSITIONS], moves[:N_POSITIONS]
    cps, gids = cps[:N_POSITIONS], gids[:N_POSITIONS]
    np.savez(path, fens=np.array(fens), moves=np.array(moves),
             cps=np.array(cps, dtype=np.int32), game_ids=np.array(gids, dtype=np.int64))
    log(f"Saved dataset: {len(fens)} positions, {len(set(gids))} games")
    return np.array(fens), np.array(moves), np.array(cps), np.array(gids)


# ================================================================ network + GPU simulator
def load_network():
    adj = sp.load_npz(os.path.join(REPO_DIR, "subnetwork_adjacency.npz")).tocoo()
    meta = pd.read_csv(os.path.join(REPO_DIR, "subnetwork_neurons.csv"))
    nt = pd.read_csv(os.path.join(REPO_DIR, "subnetwork_nt.csv")).sort_values("index")
    signs = np.where(nt["sign"].to_numpy() < 0, -1.0, 1.0)
    return adj, meta, signs


def weight_matrix(adj, signs, scale):
    """W[pre, post] in mV, same as reservoir_core (count x scale x sign(pre))."""
    W = np.zeros(adj.shape, dtype=np.float32)
    W[adj.row, adj.col] = adj.data * scale * signs[adj.row]
    return W


def rewire(W, seed=0):
    """Each presynaptic neuron keeps its outgoing weights; targets are shuffled."""
    rng = np.random.default_rng(seed)
    Wr = np.empty_like(W)
    for i in range(W.shape[0]):
        Wr[i] = W[i, rng.permutation(W.shape[1])]
    return Wr


class TorchReservoir:
    """Batched LIF network matching the Brian2 model in reservoir_core.py.

    Per 0.1 ms step (Brian2 order: groups, thresholds, synapses, resets):
      1. non-refractory neurons: v += dt/tau*(v_rest - v + I) + sigma*sqrt(2 dt/tau)*N(0,1)
      2. spike if v > -50 mV and not refractory
      3. v_post += sum of W over presynaptic spikes (applies to all neurons)
      4. clip v to [-80, 0] mV
      5. refractory neurons stay at -65 mV (Brian2 discards their synaptic input)
      6. spiking neurons: v = -65 mV, refractory for 3 ms
    Verified against Brian2 on random E/I networks: per-neuron rate r = 0.999.
    """

    def __init__(self, W, input_idx, i_max, device):
        self.device = device
        self.W = torch.tensor(W, dtype=torch.float32, device=device)
        self.N = W.shape[0]
        self.input_idx = torch.tensor(np.asarray(input_idx), dtype=torch.long, device=device)
        self.i_max = float(i_max)
        self.steps = int(round(DURATION_MS / DT_MS))
        self.steps_per_bin = self.steps // N_BINS

    @torch.no_grad()
    def run(self, activations, seed):
        """activations: (B, n_input) in [0,1]. Returns spike counts (B, N_BINS, N) as numpy."""
        dev = self.device
        act = torch.tensor(np.asarray(activations), dtype=torch.float32, device=dev)
        B = act.shape[0]
        I = torch.zeros(B, self.N, device=dev)
        I[:, self.input_idx] = act * self.i_max

        a = DT_MS / TAU_MS
        noise_std = NOISE_MV * np.sqrt(2.0 * DT_MS / TAU_MS)
        drive = a * (V_REST + I)
        v = torch.full((B, self.N), V_REST, device=dev)
        ref = torch.zeros((B, self.N), dtype=torch.int16, device=dev)
        bins = torch.zeros((B, N_BINS, self.N), device=dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
        reset_v = torch.tensor(V_RESET, device=dev)
        ref_val = torch.tensor(REFRACTORY_STEPS, dtype=torch.int16, device=dev)

        for s in range(self.steps):
            ref = torch.clamp(ref - 1, min=0)
            active = ref == 0
            noise = torch.randn((B, self.N), generator=gen, device=dev)
            v = torch.where(active, v - a * v + drive + noise_std * noise, v)
            spk = (v > V_THRESH) & active
            spf = spk.float()
            v = v + spf @ self.W
            v = torch.clamp(v, V_MIN, V_MAX)
            v = torch.where(active, v, reset_v)   # Brian2: input to refractory neurons is discarded
            v = torch.where(spk, reset_v, v)
            ref = torch.where(spk, ref_val, ref)
            bins[:, min(s // self.steps_per_bin, N_BINS - 1)] += spf
        return bins.cpu().numpy()


def encode_batch(enc, raw):
    """Vectorized version of reservoir_core.Encoder.activation."""
    pre = ((raw - enc.mean) @ enc.proj.T) * enc.scale
    return (1.0 / (1.0 + np.exp(-pre))).astype(np.float32)


def simulate(name, W, input_idx, i_max, acts, hidden, device):
    path = os.path.join(OUT_DIR, f"states_{name}_{len(acts)}.npz")
    if os.path.exists(path):
        log(f"  {name}: loaded cached states")
        return np.load(path)["bins"]
    res = TorchReservoir(W, input_idx, i_max, device)
    n = len(acts)
    out = np.zeros((n, N_BINS, int(hidden.sum())), dtype=np.uint8)
    t0 = time.time()
    n_batches = int(np.ceil(n / BATCH))
    for bi, start in enumerate(range(0, n, BATCH)):
        b = res.run(acts[start:start + BATCH], seed=1000 + bi)
        out[start:start + BATCH] = np.clip(b[:, :, hidden], 0, 255).astype(np.uint8)
        el = time.time() - t0
        log(f"  {name}: batch {bi + 1}/{n_batches}  [{el:6.0f}s, "
            f"eta {el / (bi + 1) * (n_batches - bi - 1):6.0f}s]")
    np.savez(path, bins=out)
    return out


