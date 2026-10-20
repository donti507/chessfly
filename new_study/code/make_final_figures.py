"""Final publication figures (Stage 5). Reads the result files of Stages 1-4 and writes figures_final/."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
OUT = "/mnt/user-data/outputs/figures_final/"
INK, MUT, GRID = "#1C2830", "#5B6970", "#9AA6AC"
C = dict(filter="#3E9E93", proof="#7C6CE0", detector="#D29A2E", flies="#C8397F", art="#2E6F8E", real="#D29A2E", shuf="#3E9E93")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 10.5, "axes.edgecolor": "#B9C3BD", "font.family": "DejaVu Sans"})
def note(fig, s, y=-0.03): fig.text(0.01, y, s, fontsize=7.6, color=MUT)

# ---------------- Figure 1: design
def fig1():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.05, 1, 1.15]))
    a = ax[0]; a.set_xlim(0, 10); a.set_ylim(0, 10); a.set_aspect("equal"); a.axis("off"); a.set_title("A   Compass circuit and model ports")
    nodes = {"ER": (2.2, 5.5, "#C8397F", "ER ring\n(inhib.)"), "EPG": (5.4, 5.5, "#D29A2E", "EPG\ncompass"), "D7": (8.2, 8.0, "#7C6CE0", "\u03947\n(inhib.)"),
             "PEN": (8.2, 3.0, "#3E9E93", "PEN"), "PEG": (5.4, 2.0, "#3E9E93", "PEG")}
    for k, (x, y, c, l) in nodes.items():
        a.add_patch(Circle((x, y), 0.95, fc=c, alpha=0.18, ec=c, lw=1.8)); a.text(x, y, l, ha="center", va="center", fontsize=8)
    def arr(p, q, c=GRID, both=False):
        (x1, y1), (x2, y2) = nodes[p][:2], nodes[q][:2]; v = np.array([x2 - x1, y2 - y1]); u = v / np.linalg.norm(v)
        a.add_patch(FancyArrowPatch((x1, y1) + u, (x2, y2) - u, arrowstyle="<|-|>" if both else "-|>", mutation_scale=11, color=c, lw=1.3))
    arr("ER", "EPG", "#D29A2E"); arr("EPG", "D7"); arr("D7", "PEN"); arr("EPG", "PEN", both=True); arr("EPG", "PEG", both=True)
    a.annotate("landmark cue", (2.2, 6.5), (2.2, 8.9), ha="center", fontsize=8, arrowprops=dict(arrowstyle="-|>", color=INK))
    a.annotate("angular velocity", (8.9, 2.3), (8.4, 0.5), ha="center", fontsize=8, arrowprops=dict(arrowstyle="-|>", color=INK))
    a.annotate("heading readout", (5.9, 6.4), (6.3, 9.4), ha="center", fontsize=8, arrowprops=dict(arrowstyle="<|-", color=INK))
    b = ax[1]; b.axis("off"); b.set_xlim(0, 10); b.set_ylim(0, 10); b.set_title("B   Nine wiring versions of one circuit")
    items = [("male CNS v1.0", "Janelia", 9.0, "#D29A2E"), ("  + neuPrint filtering", "same fly", 8.2, "#3E9E93"), ("  + v0.9 proofreading", "same fly", 7.4, "#7C6CE0"),
             ("hemibrain v1.2.1", "Janelia", 6.4, "#D29A2E"), ("FlyWire FAFB v783", "Princeton", 5.4, "#D29A2E"),
             ("  + unfiltered", "same fly", 4.6, "#3E9E93"), ("  + Buhmann synapses", "same fly", 3.8, "#D29A2E"),
             ("BANC v888", "Harvard", 2.8, "#D29A2E"), ("  + v626 proofreading", "same fly", 2.0, "#7C6CE0")]
    for txt, tag, y, c in items:
        b.add_patch(FancyBboxPatch((0.3, y - 0.32), 9.2, 0.64, boxstyle="round,pad=0.02,rounding_size=0.15", fc=c, alpha=0.12, ec="none"))
        b.text(0.5, y, txt, va="center", fontsize=8.6, fontweight="bold" if not txt.startswith(" ") else "normal"); b.text(9.3, y, tag, va="center", ha="right", fontsize=7.8, color=MUT)
    b.text(5, 0.8, "4 flies \u00b7 3 pipelines \u00b7 5 same-fly variants\n(filtering, proofreading, synapse detector)", ha="center", fontsize=8, color=MUT)
    c = ax[2]; c.axis("off"); c.set_xlim(0, 10); c.set_ylim(0, 10); c.set_title("C   Experiments")
    steps = [("1  Measure", "wiring distance between every pair\n(compass position inferred from wiring)"),
             ("2  Train", "rate network on each wiring to hold\na cued heading (clean or with noise)"),
             ("3  Transplant", "test on another wiring without\nretraining (one-to-one neuron maps)"),
             ("4  Refit", "re-tune only ~30 cell-type gains\non the new wiring, test again")]
    for i, (h, d) in enumerate(steps):
        y = 8.6 - i * 2.25
        c.add_patch(FancyBboxPatch((0.3, y - 0.85), 9.4, 1.7, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#EEF2F0", ec="#B9C3BD"))
        c.text(0.6, y + 0.35, h, fontweight="bold", fontsize=9); c.text(0.6, y - 0.35, d, fontsize=8, va="center")
        if i < 3: c.add_patch(FancyArrowPatch((5, y - 0.9), (5, y - 1.35), arrowstyle="-|>", mutation_scale=10, color=GRID))
    fig.tight_layout(); fig.savefig(OUT + "fig1_design.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: one scale for all differences
def fig2():
    D = pd.read_csv("/mnt/user-data/outputs/difference_decomposition.csv"); R = pd.read_csv("/mnt/user-data/outputs/shuffle_curve_distances.csv")
    hold = {("degree-preserving", 0.25): 19.7, ("degree-preserving", 0.50): 18.2, ("one-way edges", 0.05): 3.9, ("one-way edges", 0.10): 13.4, ("one-way edges", 0.20): 18.5,
            ("reciprocal edges", 0.05): 1.4, ("reciprocal edges", 0.10): 2.0, ("reciprocal edges", 0.20): 1.5}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = ax[0]; cmap = {"filtering": C["filter"], "proofreading": C["proof"], "synapse detector": C["detector"], "between flies": C["flies"]}
    order = ["filtering", "proofreading", "synapse detector", "between flies"]
    for i, src in enumerate(order):
        x = D[D.source == src]
        a.scatter(x.map_distance, np.full(len(x), i) + np.linspace(-0.12, 0.12, len(x)), s=46, color=cmap[src], edgecolors="white", zorder=3)
    a.axvspan(0, 0.032, color="#E8ECEF", zorder=0); a.text(0.016, 3.55, "noise\nfloor", ha="center", fontsize=7.5, color=MUT)
    a.set_yticks(range(4), ["same fly:\nfiltering", "same fly:\nproofreading", "same fly:\nsynapse detector", "different\nflies"]); a.set_ylim(-0.6, 3.9)
    a.set_xlabel("Wiring-map distance"); a.set_title("A   Measurement alone rivals differences between flies")
    a.annotate("BANC v626 vs v888", (0.104, 1.1), (0.135, 1.5), fontsize=7.5, arrowprops=dict(arrowstyle="-", color=GRID))
    a.annotate("male CNS v0.9 = v1.0", (0.0, 0.9), (0.03, 0.55), fontsize=7.5, arrowprops=dict(arrowstyle="-", color=GRID))
    b = ax[1]; colc = {"degree-preserving": C["shuf"], "one-way edges": C["real"], "reciprocal edges": GRID}
    b.axvspan(0.084, 0.104, color=C["detector"], alpha=0.12); b.text(0.094, 22.6, "same fly,\nmeasurement", ha="center", fontsize=7.5, color=C["detector"])
    b.axvspan(0.108, 0.212, color=C["flies"], alpha=0.08); b.text(0.16, 22.6, "different flies", ha="center", fontsize=7.5, color=C["flies"])
    for k, c_ in colc.items():
        s = R[R.kind == k]; pts = [(0, 1.7)] + [(r.map_dist, hold[(k, r.frac)]) for r in s.itertuples() if (k, r.frac) in hold]
        x, y = zip(*sorted(pts)); b.plot(x, y, "-o", color=c_, lw=2, ms=5, label=f"{k} rewired")
    b.set_xlim(-0.005, 0.24); b.set_ylim(0, 25); b.set_xlabel("Wiring-map distance from the male CNS circuit"); b.set_ylabel("Model hold error (deg)")
    b.set_title("B   Both lie in the range where rewiring breaks heading memory"); b.legend(frameon=False, fontsize=8, loc="center right")
    note(fig, "Distances: per-pathway angular wiring maps (positions inferred from wiring), total-variation distance weighted by pathway share. Hold errors: male CNS models, five networks per point.")
    fig.tight_layout(); fig.savefig(OUT + "fig2_differences_one_scale.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 3: tuned ring attractor, knife-edge fragility
def fig3():
    E = np.load("fig_ring_examples.npz"); H = pd.read_csv("h4_stage4.csv"); S = pd.read_csv("fig_scale_curve.csv")
    own = pd.read_csv("/home/claude/s4/stage4_results.csv"); own = own[own.own][["source", "gains", "seed", "test40"]].rename(columns={"source": "fly"})
    H = H.merge(own, on=["fly", "gains", "seed"]); H = H[(H.test40 <= 80) & (H.condition == "intact")]
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.7), gridspec_kw=dict(width_ratios=[1, 1, 1.1, 1.3]))
    for a, key, col, ttl in [(ax[0], "real", C["real"], "A   Real wiring: round ring"), (ax[1], "shuffle_max", C["shuf"], "B   Shuffled: distorted loop")]:
        P = E[key] / np.abs(E[key]).max(); sc = a.scatter(P[:, 0], P[:, 1], c=E["th"], cmap="twilight", s=22, edgecolors="none")
        a.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color=col, lw=0.8, alpha=0.6); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_xlabel("PC1 of compass activity"); a.set_ylabel("PC2"); a.set_title(ttl)
    a = ax[2]; flies = ["mcns", "hemibrain", "fafb", "banc"]; lab = ["male CNS", "hemibrain", "FlyWire", "BANC"]
    for i, f in enumerate(flies):
        x = H[H.fly == f]; a.scatter(np.full(len(x), i) + np.linspace(-0.15, 0.15, len(x)), x.lam1, color=C["real"], s=22, alpha=0.8)
    a.axhline(1, color=INK, ls="--", lw=0.8); a.set_xticks(range(4), lab, fontsize=8); a.set_ylim(0.98, 1.05); a.set_ylabel("Slowest Jacobian multiplier")
    a.set_title(f"C   Near-perfect attractor in every fly\n     (H4 met in {int(H.H4.sum())}/{len(H)} networks)")
    a = ax[3]; cols = dict(zip(flies, [C["real"], "#2E6F8E", C["flies"], C["shuf"]]))
    for f, l in zip(flies, lab):
        g = S[S.fly == f].groupby("scale").err.median(); a.plot((g.index - 1) * 100, g.values, "-o", ms=4, color=cols[f], label=l)
    a.set_xlabel("Change of all weights (%)"); a.set_ylabel("Hold error (deg)"); a.set_title("D   Knife-edge: 1% breaks memory"); a.legend(frameon=False, fontsize=8)
    note(fig, "A-B: pilot networks (male CNS), EPG activity 95 steps after the cue, colour = cued heading. C-D: Stage 4 clean-trained networks, per-neuron gains (D: seeds 0-1).")
    fig.tight_layout(); fig.savefig(OUT + "fig3_ring_and_fragility.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 4: transplants and the precision-robustness trade-off
def fig4():
    d = pd.read_csv("/home/claude/s4/stage4_results.csv"); own = d[d.own][["source", "gains", "seed", "test40"]].rename(columns={"test40": "own40"})
    d = d.merge(own, on=["source", "gains", "seed"]); d = d[d.own40 <= 80]
    names = {"mcns": "male CNS", "hemibrain": "hemibrain", "fafb": "FlyWire", "banc": "BANC"}; fl = list(names)
    M = d[d.gains == "neuron"].pivot_table(index="source", columns="target", values="test40", aggfunc="median").loc[fl, fl]
    c = pd.read_csv("/home/claude/s4c/stage4c_results.csv"); cc = c[c.condition == "clean"]; o = cc[cc.own][["source", "seed", "level", "test40"]].rename(columns={"test40": "own40"})
    o = o[o.own40 <= 80]; sc = c[c.condition.str.startswith("scale")].merge(o, on=["source", "seed", "level"])
    tr = cc[~cc.own].merge(o, on=["source", "seed", "level"]); tr["jan"] = (tr.source + tr.target).isin(["mcnshemibrain", "hemibrainmcns"])
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1.1, 1.1]))
    a = ax[0]; im = a.imshow(M.values, cmap="magma_r", vmin=0, vmax=90)
    for i in range(4):
        for j in range(4): a.text(j, i, f"{M.values[i, j]:.0f}", ha="center", va="center", fontsize=8.5, color="white" if M.values[i, j] > 50 else INK)
    a.set_xticks(range(4), [names[f] for f in fl], rotation=30, fontsize=8); a.set_yticks(range(4), [names[f] for f in fl], fontsize=8)
    a.set_xlabel("tested on"); a.set_ylabel("trained on"); a.set_title("A   Clean models: every transplant fails"); plt.colorbar(im, ax=a, fraction=0.046, label="hold error (deg)")
    L = [0.0, 0.25, 0.5, 1.0]; x = np.arange(4)
    def med(s): return np.median(s) if len(s) else np.nan
    a = ax[1]
    a.plot(x, [med(o[o.level == l].own40) for l in L], "-s", color=INK, lw=2, label="precision (own wiring)")
    a.plot(x, [med(sc[(sc.level == l) & (sc.condition == "scale0.95")].test40) for l in L], "-o", color=C["detector"], lw=2, label="weights x0.95")
    a.plot(x, [med(sc[(sc.level == l) & (sc.condition == "scale1.05")].test40) for l in L], "-o", color=C["flies"], lw=2, label="weights x1.05")
    a.set_xticks(x, ["clean", "0.25", "0.5", "1.0"]); a.set_xlabel("Training-noise level"); a.set_ylabel("Hold error (deg)"); a.legend(frameon=False, fontsize=8)
    a.set_title("B   Noise buys robustness, costs precision")
    a = ax[2]
    a.plot(x, [med(tr[tr.jan & (tr.level == l)].test40) for l in L], "-o", color="#2E6F8E", lw=2, label="male CNS \u2194 hemibrain")
    a.plot(x, [med(tr[~tr.jan & (tr.level == l)].test40) for l in L], "-o", color=C["flies"], lw=2, label="pairs with FlyWire or BANC")
    a.plot(x, [med(o[o.level == l].own40) for l in L], "--s", color=INK, lw=1.2, label="own wiring")
    a.axhline(90, color=GRID, ls=":"); a.set_xticks(x, ["clean", "0.25", "0.5", "1.0"]); a.set_xlabel("Training-noise level"); a.set_ylabel("Hold error after transplant (deg)")
    a.set_title("C   Robustness gives only partial transfer"); a.legend(frameon=False, fontsize=8, loc="center right")
    note(fig, "A: Stage 4, per-neuron gains, median of trained seeds (preregistered H1: transplant ratio 87-162, 'fragile'). B-C: Stage 4c, 48 networks, per-neuron gains, 40-step tests.")
    fig.tight_layout(); fig.savefig(OUT + "fig4_transplant_tradeoff.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 5: transfer depends on distance, not its source; refit
def fig5():
    t = pd.read_csv("/home/claude/s4d/stage4d_transplants.csv"); t = t[t.level == 0.5].dropna(subset=["refit40"])
    dist = {frozenset(k): v for k, v in {("mcns","mcns_neuprint"): .022, ("fafb","fafb_unfiltered"): .038, ("fafb","fafb_buhmann"): .090, ("fafb_unfiltered","fafb_buhmann"): .084,
            ("mcns","hemibrain"): .117, ("hemibrain","fafb"): .108, ("mcns","fafb"): .137, ("fafb","banc"): .125, ("hemibrain","banc"): .185, ("mcns","banc"): .212}.items()}
    g = t.groupby(["source", "target", "category"]).agg(zs=("test40", "median"), rf=("refit40", "median"), own=("own40", "median")).reset_index()
    g["d"] = [dist.get(frozenset((a, b)), np.nan) for a, b in zip(g.source, g.target)]; g = g.dropna()
    col = {"2 same fly: filtering": C["filter"], "3 same fly: synapse detector": C["detector"], "4 different flies": C["flies"]}
    lab = {"2 same fly: filtering": "same fly, filtering", "3 same fly: synapse detector": "same fly, synapse detector", "4 different flies": "different flies"}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    for c_, sub in g.groupby("category"):
        for _, r in sub.iterrows(): a.plot([r.d, r.d], [r.zs, r.rf], color=col[c_], lw=0.7, alpha=0.45)
        a.scatter(sub.d, sub.zs, color=col[c_], marker="x", s=34, alpha=0.75); a.scatter(sub.d, sub.rf, color=col[c_], s=48, edgecolors="white", label=lab[c_], zorder=3)
    a.axhline(g.own.median(), color=INK, ls="--", lw=0.9); a.text(0.215, g.own.median() + 2.5, "own wiring", fontsize=7.5, ha="right")
    a.axhline(90, color=GRID, ls=":"); a.text(0.215, 92.5, "chance", fontsize=7.5, ha="right", color=MUT)
    a.set_xlabel("Wiring-map distance between source and target"); a.set_ylabel("Hold error after transplant (deg)"); a.set_ylim(0, 100)
    a.set_title("A   Transfer fails with distance, whatever its source"); a.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.18))
    a.text(0.01, 0.03, "x zero-shot   \u25cf after refitting ~30 cell-type gains   (\u03c1 = +0.67, p = 0.001)", transform=a.transAxes, fontsize=7.8, color=MUT)
    a = ax[1]
    for i, c_ in enumerate(col):
        s = t[t.category == c_]; zs, rf = s.test40.median(), s.refit40.median()
        a.bar(i - 0.2, zs, 0.38, color=col[c_], alpha=0.35); a.bar(i + 0.2, rf, 0.38, color=col[c_])
        a.text(i + 0.2, rf + 2, f"{(s.refit40 < 30).mean():.0%}", ha="center", fontsize=8.5)
    a.set_xticks(range(3), ["same fly,\nfiltering", "same fly,\nsynapse detector", "different\nflies"]); a.set_ylabel("Median hold error (deg)"); a.set_ylim(0, 100)
    a.axhline(90, color=GRID, ls=":"); a.set_title("B   Cell-type refit repairs only the smallest differences")
    a.text(-0.45, 94, "light: zero-shot   dark: after refit   %: rescued (<30 deg)", fontsize=7.8, color=MUT)
    note(fig, "Stage 4d, noise level 0.5: 21 networks on 7 wiring versions (3 seeds each), per-neuron gains. Detector vs different flies after refit: p = 0.20 (Mann-Whitney); categories differ, p = 1e-6 (Kruskal-Wallis).")
    fig.tight_layout(); fig.savefig(OUT + "fig5_transfer_by_source.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 6: real fly data
def fig6():
    M = np.load("fig_offset_msd.npz"); d = pd.read_csv("/mnt/user-data/outputs/drift_final_numbers.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = ax[0]; L = M["lags"]
    for k, c_, l in [("dark", INK, "darkness"), ("low", C["detector"], "low-contrast cue"), ("bright", C["shuf"], "bright cue")]:
        v = M[k]; a.plot(L, v[0], "-o", color=c_, lw=2, ms=4, label=l); a.fill_between(L, v[1], v[2], color=c_, alpha=0.15)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Time lag (s)"); a.set_ylabel("Mean squared change of bump-heading offset (rad\u00b2)")
    a.set_title("A   Darkness destabilizes the fly compass (15 flies)"); a.legend(frameon=False, fontsize=8.5)
    a = ax[1]; lab = ["darkness", "bright cue\n(jitter floor)", "darkness\nminus bright"]
    for i, r in d.iterrows():
        a.errorbar(i, r.D, yerr=[[r.D - r.lo], [r.hi - r.D]], fmt="o", color=[INK, C["shuf"], C["flies"]][i], capsize=4, ms=7)
    a.axhline(0, color=GRID, lw=0.8)
    a.set_xticks(range(3), lab, fontsize=8.5); a.set_ylabel("Diffusion coefficient D (rad\u00b2/s)"); a.set_title("B   Short-timescale drift (9 flies)")
    note(fig, "DANDI 000289 (Basnak et al. 2025), contrast experiment. A: all frames incl. turning, bootstrap over flies. B: no-turning runs >= 5 s, MSD = 2Dt + 2s\u00b2 fit; 95% CIs from fly bootstrap.")
    fig.tight_layout(); fig.savefig(OUT + "fig6_fly_drift.png", bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f(); print("made", f.__name__)
