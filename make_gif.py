"""
make_gif.py -- animated README GIF of the chessfly fly-compass circuit.

Everything that is *data* comes from the repo:
  * subnetwork_adjacency.npz  -> synapse counts (434 x 434, rows = presynaptic)
  * subnetwork_neurons.csv    -> cell "type" of each neuron
  * subnetwork_nt.csv         -> "sign" (-1 = inhibitory), sorted by "index"
  * reservoir_core.FlyReservoir().run(board, record_spikes=True)
                              -> real spike times for 1 s of simulated activity,
                                 chess starting position as input

Everything that is *layout* (3D coordinates) is a schematic and is labelled as such
on the GIF. Layout jitter uses a fixed seed and carries no information.

Scenes (15 fps, 12 s, loops seamlessly):
  0-8 s    camera orbits 360 deg; the 1 s spike train is replayed 8x slower.
           Glow: 1 at the spike, linearly-squared fade to 0 over 150 ms of
           *simulated* time (= 1.2 s on screen). White sparks run along each
           spiking neuron's 2 strongest outgoing connections.
  8-12 s   lines crossfade real wiring -> degree-preserving shuffled wiring
           ("Real wiring" / "Shuffled wiring"), then back to real wiring so the
           GIF loops without a jump. Remaining glow decays naturally (no new
           spikes are invented).

Usage (from the repo root, with the four data files present):
    pip install matplotlib pillow
    python make_gif.py                      # writes docs/fly_circuit.gif
    python make_gif.py --preview 5          # also dumps a few PNG frames to docs/_preview/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# ----------------------------------------------------------------------------- settings
BG = "#0D1422"
FPS = 15                                               # target; lowered automatically only if 8 MB is impossible
SCENE1_S, SCENE2_S = 8.0, 4.0
SIM_MS = 1000.0                                        # reservoir_core.DURATION_MS
SLOWDOWN = SCENE1_S / (SIM_MS / 1000.0)                # 8x slower than simulated time
FADE_MS = 150.0                                        # glow fade, simulated ms
SPARK_MS = 40.0                                        # visual travel time of a spark, simulated ms
SPARKS_PER_NEURON = 2                                  # strongest outgoing connections used
MAX_SPARKS = 450                                       # legibility cap on simultaneous sparks
N_LINES = 1500
LINE_DIM, LINE_WIDTH, LINE_AA = 0.24, 0.5, False       # lines: solid, pre-blended with BG (compresses ~2x better than alpha)
FIG_W_IN, FIG_H_IN = 8.0, 6.0                          # x DPI/100 -> 800x600 px at the top of the ladder
ELEV, AZIM0 = 22.0, -90.0                              # azim -90: camera looks toward +y, so +y is "behind"

COLORS = {
    "ER": "#FF35B0",       # magenta
    "EPG": "#FFB524",      # amber
    "PEN": "#18D6BE",      # teal (PEN + PEG)
    "D7": "#9A7CFF",       # violet
}
LABELS = {"ER": "ER ring neurons", "EPG": "EPG compass neurons",
          "PEN": "PEN / PEG (bridge)", "D7": "Delta7"}


# ----------------------------------------------------------------------------- data
def load_data():
    adj = sp.load_npz(REPO / "subnetwork_adjacency.npz").tocoo()
    meta = pd.read_csv(REPO / "subnetwork_neurons.csv")
    nt = pd.read_csv(REPO / "subnetwork_nt.csv").sort_values("index")
    n = adj.shape[0]
    assert adj.shape == (n, n) and len(meta) == n and len(nt) == n, "file sizes disagree"
    types = meta["type"].astype(str).to_numpy()
    sign = nt["sign"].to_numpy(dtype=float)

    group = np.full(n, "", dtype=object)
    for g, prefixes in (("ER", ("ER",)), ("EPG", ("EPG",)),
                        ("PEN", ("PEN", "PEG")), ("D7", ("Delta7",))):
        for i, t in enumerate(types):
            if t.startswith(prefixes):
                group[i] = g
    unassigned = np.flatnonzero(group == "")
    if len(unassigned):
        raise SystemExit("Types not covered by the layout: "
                         + ", ".join(sorted(set(types[unassigned]))))
    counts = {g: int((group == g).sum()) for g in COLORS}
    print(f"[data] {n} neurons, {adj.nnz} connections, {int((sign < 0).sum())} inhibitory | {counts}")
    return adj, types, sign, group, counts


# ----------------------------------------------------------------------------- layout (schematic)
def schematic_layout(types, group, seed=7):
    rng = np.random.default_rng(seed)
    n = len(types)
    pos = np.zeros((n, 3))
    order = lambda idx: idx[np.lexsort((idx, types[idx]))]      # by type, then index

    # ER ring neurons on a torus (R = 1.0, r = 0.27)
    idx = order(np.flatnonzero(group == "ER"))
    m = len(idx)
    theta = 2 * np.pi * np.arange(m) / m
    phi = (np.arange(m) * 2.399963) % (2 * np.pi)                # golden angle around the tube
    theta = theta + rng.normal(0, 0.01, m)
    R, r = 1.0, 0.27
    pos[idx, 0] = (R + r * np.cos(phi)) * np.cos(theta)
    pos[idx, 1] = (R + r * np.cos(phi)) * np.sin(theta)
    pos[idx, 2] = r * np.sin(phi)

    # EPG compass neurons on an inner ring (radius 0.62)
    idx = order(np.flatnonzero(group == "EPG"))
    m = len(idx)
    theta = 2 * np.pi * np.arange(m) / m
    rad = 0.62 + rng.normal(0, 0.012, m)
    pos[idx, 0], pos[idx, 1] = rad * np.cos(theta), rad * np.sin(theta)
    pos[idx, 2] = rng.normal(0, 0.02, m)

    # protocerebral bridge: arched bars above the ring
    def arch(idx, y0, dz):
        idx = order(idx)
        m = len(idx)
        x = np.linspace(-0.85, 0.85, m)
        z = 0.78 + 0.30 * (1 - (x / 0.85) ** 2) + dz
        pos[idx, 0] = x
        pos[idx, 1] = y0 + rng.normal(0, 0.012, m)
        pos[idx, 2] = z + 0.035 * ((np.arange(m) % 3) - 1)
    arch(np.flatnonzero(group == "PEN"), y0=0.0, dz=0.0)
    arch(np.flatnonzero(group == "D7"), y0=0.17, dz=-0.02)      # just behind
    return pos


# ----------------------------------------------------------------------------- wiring
def top_edges(rows, cols, w, k):
    """k strongest connections (self-connections excluded), stable tie-break."""
    keep = np.flatnonzero(rows != cols)
    order = keep[np.argsort(-w[keep], kind="stable")][:k]
    return rows[order], cols[order], w[order]


def degree_preserving_shuffle(rows, cols, w, seed=11, swaps_per_edge=10):
    """Directed double-edge swaps: (a->b),(c->d) => (a->d),(c->b).
    Every neuron keeps its in-degree and out-degree; weights stay with their source slot."""
    rng = np.random.default_rng(seed)
    src, dst = rows.copy(), cols.copy()
    m = len(src)
    present = set(zip(src.tolist(), dst.tolist()))
    done = tries = 0
    target = swaps_per_edge * m
    while done < target and tries < 30 * target:
        tries += 1
        i, j = rng.integers(m, size=2)
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if i == j or a == c or b == d or a == d or c == b:
            continue
        if (a, d) in present or (c, b) in present:
            continue
        present.remove((a, b)); present.remove((c, d))
        present.add((a, d)); present.add((c, b))
        dst[i], dst[j] = d, b
        done += 1
    assert (np.bincount(src, minlength=1) == np.bincount(rows, minlength=1)).all()
    assert (np.bincount(dst, minlength=1) == np.bincount(cols, minlength=1)).all()
    print(f"[shuffle] {done} double-edge swaps ({tries} tries); in/out degrees preserved")
    return src, dst, w.copy()


# ----------------------------------------------------------------------------- real spikes
def real_spikes():
    import chess
    from reservoir_core import FlyReservoir, canonical
    print("[sim] running FlyReservoir on the chess starting position (a minute or so)...")
    t0 = time.time()
    res = FlyReservoir()
    out = res.run(canonical(chess.Board()), sim_seed=0, record_spikes=True)
    si = np.asarray(out["spike_i"], dtype=int)
    st = np.asarray(out["spike_t_ms"], dtype=float)
    o = np.argsort(st, kind="stable")
    si, st = si[o], st[o]
    print(f"[sim] {len(si)} real spikes from {len(np.unique(si))} neurons in {time.time()-t0:.0f}s "
          f"(mean {len(si)/res.n_neurons:.1f} Hz/neuron)")
    return si, st


# ----------------------------------------------------------------------------- animation helpers
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def timeline(f, fps):
    """Per-frame state: camera azimuth, simulated time (ms), shuffle mix, text alphas."""
    t = f / fps
    if t < SCENE1_S:
        u = t / SCENE1_S
        az = AZIM0 + 360.0 * (0.5 - 0.5 * np.cos(np.pi * u))     # eased: starts/stops gently
    else:
        az = AZIM0
    t_sim = t / SLOWDOWN * 1000.0                                # ms; keeps running past 1000 ms (no new spikes)
    s = t - SCENE1_S                                             # seconds into scene 2
    if s < 0:
        mix, a_real, a_shuf = 0.0, 1.0 - smooth(t / 0.7), 0.0    # tail of last scene's "Real wiring"
    else:
        mix = smooth(s - 1.0) * (1 - smooth(s - 3.0))            # real -> shuffled (1-2 s) -> real (3-4 s)
        a_real = smooth(s / 0.3) * (1 - smooth((s - 1.0) / 0.5)) + smooth((s - 3.4) / 0.6)
        a_shuf = smooth((s - 1.5) / 0.5) * (1 - smooth((s - 2.9) / 0.4))
    return az, t_sim, mix, float(np.clip(a_real, 0, 1)), float(np.clip(a_shuf, 0, 1))


def segments(pos, r, c):
    return np.stack([pos[r], pos[c]], axis=1)


def render(fps, width_px, args, pos, group, edges_real, edges_shuf, spk_i, spk_t, spark_src, spark_dst, spark_t,
           spark_keep, counts):
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=width_px / FIG_W_IN, facecolor=BG)
    try:
        ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    except TypeError:
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-0.3, 1.2)
    ax.set_box_aspect((2.6, 2.6, 1.6), zoom=1.55)

    base_rgb = np.array([to_rgb(COLORS[g]) for g in group])
    bg_rgb = np.array(to_rgb(BG))
    n = len(group)

    def line_collection(edges):
        r, c, _ = edges
        col = bg_rgb + LINE_DIM * (base_rgb[r] - bg_rgb)
        lc = Line3DCollection(segments(pos, r, c), colors=col, linewidths=LINE_WIDTH, zorder=1)
        lc.set_alpha(1.0)
        lc.set_antialiased(LINE_AA)
        ax.add_collection3d(lc)
        return lc
    lc_real, lc_shuf = line_collection(edges_real), line_collection(edges_shuf)

    halo = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, c=[(0, 0, 0, 0)], depthshade=False,
                      edgecolors="none", zorder=2)
    nodes = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=6, c=base_rgb, depthshade=False,
                       edgecolors="none", zorder=3)
    spark_art = [None]

    # ---- 2D overlay text
    fig.text(0.018, 0.972, "Fly compass circuit \u2014 real spiking activity (434 neurons)",
             color="#DCE6F5", fontsize=9.5, ha="left", va="top")
    fig.text(0.018, 0.022, "schematic layout", color="#7F8DA6", fontsize=7, ha="left", va="bottom",
             style="italic")
    for k, g in enumerate(["ER", "EPG", "PEN", "D7"]):
        fig.text(0.985, 0.022 + 0.034 * (3 - k), f"\u25CF {LABELS[g]} ({counts[g]})",
                 color=COLORS[g], fontsize=7, ha="right", va="bottom")
    t_real = fig.text(0.5, 0.075, "Real wiring", color="#F2F6FF", fontsize=15, ha="center", va="center",
                      fontweight="bold", alpha=0)
    t_shuf = fig.text(0.5, 0.075, "Shuffled wiring", color="#F2F6FF", fontsize=15, ha="center",
                      va="center", fontweight="bold", alpha=0)

    n_frames = int(round((SCENE1_S + SCENE2_S) * fps))
    frames = []
    for f in range(n_frames):
        az, t_sim, mix, a_real, a_shuf = timeline(f, fps)
        ax.view_init(elev=ELEV, azim=az)

        # glow from the most recent real spike of each neuron
        hi = np.searchsorted(spk_t, t_sim, side="right")
        lo = np.searchsorted(spk_t, t_sim - FADE_MS, side="left")
        last = np.full(n, -1e9)
        np.maximum.at(last, spk_i[lo:hi], spk_t[lo:hi])
        g = np.clip(1.0 - (t_sim - last) / FADE_MS, 0.0, 1.0) ** 2

        core = base_rgb * (1 - 0.6 * g[:, None]) + 0.6 * g[:, None]          # towards white
        rgba = np.concatenate([core, (0.60 + 0.40 * g)[:, None]], axis=1)
        nodes.set_facecolors(rgba)
        nodes.set_sizes(7 + 30 * g)
        hal = np.concatenate([base_rgb, (0.30 * g)[:, None]], axis=1)
        halo.set_facecolors(hal)
        halo.set_sizes(10 + 170 * g)

        lc_real.set_alpha(1 - mix)
        lc_shuf.set_alpha(mix)

        # sparks along the strongest outgoing connections of neurons that just spiked
        if spark_art[0] is not None:
            spark_art[0].remove(); spark_art[0] = None
        a = np.searchsorted(spark_t, t_sim - SPARK_MS, side="right")
        b = np.searchsorted(spark_t, t_sim, side="right")
        sel = np.flatnonzero(spark_keep[a:b]) + a
        if len(sel):
            frac = (t_sim - spark_t[sel]) / SPARK_MS
            p = pos[spark_src[sel]] * (1 - frac[:, None]) + pos[spark_dst[sel]] * frac[:, None]
            col = np.ones((len(sel), 4)); col[:, 3] = 0.95 - 0.5 * frac
            spark_art[0] = ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=5, c=col, depthshade=False,
                                      edgecolors="none", zorder=5)

        t_real.set_alpha(a_real); t_shuf.set_alpha(a_shuf)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        if args.preview and f in args.preview_frames:
            d = REPO / "docs" / "_preview"; d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(buf).save(d / f"frame_{f:03d}.png")
        if f % 20 == 0:
            print(f"[render] {fps} fps, {width_px}px: frame {f}/{n_frames}")
    plt.close(fig)
    return frames


# ----------------------------------------------------------------------------- GIF encoding
def encode_gif(frames, path, fps, ncol):
    """Shared palette for all frames (no palette flicker), no dithering. Returns the file size."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # GIF delays are whole centiseconds: spread the rounding so the average is exactly 1/fps
    dur = [10 * (round((k + 1) * 100 / fps) - round(k * 100 / fps)) for k in range(len(frames))]
    imgs = [Image.fromarray(f) for f in frames]
    sample = imgs[:: max(1, len(imgs) // 14)]
    sheet = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        sheet.paste(im, (0, k * im.height))
    pal = sheet.quantize(colors=ncol, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False, disposal=1)
    size = path.stat().st_size
    print(f"[gif] {fps} fps, {imgs[0].width}x{imgs[0].height}px, {ncol} colours, {len(q)} frames "
          f"-> {size/1e6:.2f} MB")
    return size


# (fps, width, colours): tried in this order until the file fits. Quality drops left to right; the GIF
# is re-rendered natively at each new size/fps (resampling would blur the crisp lines and *grow* the file).
LADDER = [(15, 800, 64), (15, 800, 32), (15, 720, 32), (15, 640, 32),
          (12, 800, 32), (12, 720, 32), (10, 800, 32), (10, 720, 32)]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs" / "fly_circuit.gif"))
    ap.add_argument("--max-mb", type=float, default=7.6, help="size budget (limit is 8 MB)")
    ap.add_argument("--preview", type=int, default=0, help="also save N evenly spaced PNG frames")
    args = ap.parse_args()
    args.preview_frames = set(np.linspace(0, int(round((SCENE1_S + SCENE2_S) * FPS)) - 1, args.preview).astype(int)) if args.preview else set()

    adj, types, sign, group, counts = load_data()
    n = adj.shape[0]
    rows, cols, w = adj.row.astype(int), adj.col.astype(int), adj.data.astype(float)

    pos = schematic_layout(types, group)
    edges_real = top_edges(rows, cols, w, N_LINES)
    sr, sc, sw = degree_preserving_shuffle(rows, cols, w)
    edges_shuf = top_edges(sr, sc, sw, N_LINES)
    print(f"[edges] drawing the {N_LINES} strongest of {len(w)} connections "
          f"(weights {edges_real[2].min():.0f}..{edges_real[2].max():.0f} synapses)")

    spk_i, spk_t = real_spikes()
    assert spk_t.max() <= SIM_MS + 1e-6

    # strongest outgoing connections per neuron (from all real connections)
    csr = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    top_targets = {}
    for i in range(n):
        s, e = csr.indptr[i], csr.indptr[i + 1]
        j, ww = csr.indices[s:e], csr.data[s:e]
        ok = j != i
        j, ww = j[ok], ww[ok]
        top_targets[i] = j[np.argsort(-ww, kind="stable")[:SPARKS_PER_NEURON]]
    ev_t, ev_src, ev_dst = [], [], []
    for i, t in zip(spk_i, spk_t):
        for j in top_targets[i]:
            ev_t.append(t); ev_src.append(i); ev_dst.append(j)
    spark_t, spark_src, spark_dst = np.array(ev_t), np.array(ev_src, dtype=int), np.array(ev_dst, dtype=int)
    # legibility: keep a fixed random fraction of sparks so the peak count stays <= MAX_SPARKS
    edges_t = np.arange(0, SIM_MS + 1, 5.0)
    conc = (np.searchsorted(spark_t, edges_t, side="right") -
            np.searchsorted(spark_t, edges_t - SPARK_MS, side="right"))
    p_keep = min(1.0, MAX_SPARKS / max(1, conc.max()))
    spark_keep = np.random.default_rng(3).random(len(spark_t)) < p_keep
    print(f"[sparks] {len(spark_t)} spark events, peak {conc.max()} simultaneous, "
          f"showing {100*p_keep:.0f}% of them")

    frames, key, ok = None, None, False
    for fps, width, ncol in LADDER:
        if key != (fps, width):
            frames = render(fps, width, args, pos, group, edges_real, edges_shuf, spk_i, spk_t,
                            spark_src, spark_dst, spark_t, spark_keep, counts)
            key = (fps, width)
        if encode_gif(frames, args.out, fps, ncol) <= args.max_mb * 1e6:
            ok = True
            break
    if not ok:
        print("[gif] WARNING: still over the limit after the whole ladder; try a smaller N_LINES")
    out = Path(args.out)
    print(f"[done] {out}  ({out.stat().st_size/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
