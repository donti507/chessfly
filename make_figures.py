"""
make_figures.py -- figures 9-12 for docs/ from the phase result CSVs

  fig9_flexibility.png         Phase 6   learning-freedom levels L0-L3, hold & integrate
  fig10_null_distribution.png  Phase 6b  real wiring vs 30 rewired + 30 degree-preserving networks
  fig11_dose_response.png      Phase 7b + 7e  gradual shuffling; one-way vs reciprocal rewiring
  fig12_pathway_scrambles.png  Phase 7f  neuron-level map scrambles within single pathways

Usage:  python make_figures.py      (reads CSVs from the repo folder, writes to docs/)
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AMBER, MAG, TEAL, INK, MUTED, GREY, BLUE = "#D29A2E", "#C8397F", "#3E9E93", "#1C2830", "#5B6970", "#9AA6AC", "#3E6F8E"
COL = {"real": AMBER, "rewired": MAG, "degshuffle": TEAL}
NAME = {"real": "Real fly wiring", "rewired": "Random rewiring", "degshuffle": "Degree-preserving shuffle"}
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": "#B9C3BD"})
OUT = "docs"
os.makedirs(OUT, exist_ok=True)
jitter = lambda n, s=0.08, seed=0: np.random.default_rng(seed).uniform(-s, s, n)


def note(fig, text):
    fig.text(0.01, -0.04, text, fontsize=8.4, color=MUTED)


def fig9():
    df = pd.read_csv("phase6_flexibility.csv"); df = df[df["size"] == 1000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    levels = ["L0", "L1", "L2", "L3"]
    for ax, task, title in [(axes[0], "hold", "Hold heading"), (axes[1], "integrate", "Integrate turns")]:
        for j, kind in enumerate(["real", "rewired", "degshuffle"]):
            d = df[(df.task == task) & (df.condition == kind)]
            med = [d[d.level == l].test_err.median() for l in levels]
            x = np.arange(4) + (j - 1) * 0.22
            ax.plot(x, med, "-o", color=COL[kind], lw=2, label=NAME[kind], zorder=3)
            for i, l in enumerate(levels):
                v = d[d.level == l].test_err.to_numpy()
                ax.scatter(np.full(len(v), x[i]) + jitter(len(v), 0.04, i + j), v, s=12, color=COL[kind], alpha=.45, zorder=2)
        ax.set_xticks(range(4), ["L0\nfrozen", "L1\ncell-type pairs", "L2\nper-neuron gain", "L3\nevery synapse"], fontsize=8.8)
        ax.set_title(title); ax.axhline(90, color=GREY, ls=":", lw=1)
    axes[0].set_ylabel("Heading error (deg)"); axes[0].legend(frameon=False, fontsize=8.8)
    axes[0].text(3.3, 92, "chance", fontsize=8, color=MUTED, ha="right")
    fig.suptitle("Phase 6: how much learning freedom the circuit gets", x=0.01, ha="left", fontweight="bold")
    note(fig, "1,000 training runs, anatomical ports. Lines: medians of 3 seeds; dots: individual runs. Errors near 88 deg are networks that failed to train.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig9_flexibility.png", bbox_inches="tight"); plt.close(fig)


def fig10():
    df = pd.read_csv("phase6b_null_distribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, task, title in [(axes[0], "hold", "Hold heading"), (axes[1], "integrate", "Integrate turns")]:
        d = df[df.task == task]
        for j, kind in enumerate(["rewired", "degshuffle"]):
            v = d[d.condition == kind].test_err.to_numpy()
            ax.scatter(np.full(len(v), j) + jitter(len(v), 0.18, j), v, s=22, color=COL[kind], alpha=.75, edgecolors="white", linewidths=.4)
            ax.hlines(np.median(v), j - .28, j + .28, color=INK, lw=1.4)
        real = d[d.condition == "real"].test_err
        ax.axhspan(real.min(), real.max(), color=AMBER, alpha=.25, lw=0)
        ax.axhline(real.median(), color=AMBER, lw=2.2, label=f"real wiring (median {real.median():.1f} deg)")
        ax.set_xticks([0, 1], ["30 random\nrewirings", "30 degree-preserving\nshuffles"]); ax.set_xlim(-.6, 1.6)
        ax.set_title(title); ax.legend(frameon=False, fontsize=8.8, loc="upper center")
    axes[0].set_ylabel("Heading error (deg)")
    fig.suptitle("Phase 6b: the real wiring against a null distribution", x=0.01, ha="left", fontweight="bold")
    note(fig, "Each dot is one random network (L2, 1,000 runs); black bars are medians; the amber band spans the 5 real-wiring seeds.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig10_null_distribution.png", bbox_inches="tight"); plt.close(fig)


def fig11():
    b = pd.read_csv("phase7b_dose_response.csv"); e = pd.read_csv("phase7e_replication.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]; order = [0.0, 0.25, 0.5, 0.6, 1.0]; lab = ["Real", "25%", "50%", "60%", "Full"]
    for task, c, mk in [("hold", AMBER, "s"), ("integrate", BLUE, "o")]:
        med = [b[(b.task == task) & np.isclose(b.frac, f)].test_err.median() for f in order]
        ax.plot(range(5), med, "-" + mk, color=c, lw=2.2, label=task)
        for i, f in enumerate(order):
            v = b[(b.task == task) & np.isclose(b.frac, f)].test_err.to_numpy(); v = v[v < 80]
            ax.scatter(np.full(len(v), i) + jitter(len(v), .06, i), v, s=10, color=c, alpha=.4)
    ax.set_xticks(range(5), lab); ax.set_xlabel("Connections shuffled (degree-preserving)"); ax.set_ylabel("Heading error (deg)")
    ax.set_title("7b: gradual shuffling of the real wiring"); ax.legend(frameon=False, fontsize=9)
    ax = axes[1]; fr = [0.0, 0.05, 0.1, 0.2]
    for mode, c, lab2 in [("control", AMBER, "one-way edges rewired"), ("target", GREY, "reciprocal edges rewired")]:
        med = [e[(e.task == "hold") & (((e.condition == "real") & (f == 0)) | ((e.condition == mode) & np.isclose(e.frac, f)))].test_err.median() for f in fr]
        ax.plot([100 * f for f in fr], med, "-o", color=c, lw=2.4, label=lab2)
        for f in fr[1:]:
            v = e[(e.task == "hold") & (e.condition == mode) & np.isclose(e.frac, f)].test_err.to_numpy()
            ax.scatter(np.full(len(v), 100 * f) + jitter(len(v), .4, int(100 * f)), v, s=12, color=c, alpha=.45)
    ax.set_xlabel("Connections rewired (%)"); ax.set_ylabel("Hold error (deg)")
    ax.set_title("7e: one-way vs reciprocal (fresh networks)"); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Phases 7b and 7e: dose-response", x=0.01, ha="left", fontweight="bold")
    note(fig, "L2, anatomical ports, 1,000 runs. Lines: medians of 5 networks; dots: individual networks (7b dots omit failed runs above 80 deg).")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig11_dose_response.png", bbox_inches="tight"); plt.close(fig)


def fig12():
    df = pd.read_csv("phase7f_pathways.csv")
    conds = ["real", "scramble_ER_EPG", "scramble_EPG_D7", "scramble_D7_PEN", "scramble_ER_ER"]
    lab = ["Real", "ER→EPG\n(landmark map)", "EPG→Δ7", "Δ7→PEN", "ER→ER\n(reciprocal)"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, task, title in [(axes[0], "hold", "Hold heading"), (axes[1], "integrate", "Integrate turns")]:
        for i, c in enumerate(conds):
            v = df[(df.task == task) & (df.condition == c)].test_err.to_numpy()
            ax.bar(i, np.median(v), 0.62, color=AMBER if c == "real" else BLUE, alpha=.85)
            ax.scatter(np.full(len(v), i) + jitter(len(v), .12, i), v, s=12, color=INK, alpha=.6, zorder=3)
        ax.set_xticks(range(5), lab, fontsize=8.6); ax.set_title(title)
    axes[0].set_ylabel("Heading error (deg)")
    fig.suptitle("Phase 7f: scrambling the neuron-level map inside single pathways", x=0.01, ha="left", fontweight="bold")
    note(fig, "Target neurons relabeled within one pathway: strengths and source degrees kept, pathway totals unchanged. Bars: medians; dots: networks.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig12_pathway_scrambles.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig9, fig10, fig11, fig12):
        f(); print("made", f.__name__)
