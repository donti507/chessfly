"""
pilot_stage3.py -- Stage 3 pilot (preregistration doi:10.5281/zenodo.23382486, Section 6)

Male CNS head-direction wiring (Codex v1.0 network from hd_networks_4flies.zip):
  wiring    : real, 25% degree-preserving shuffle, maximal shuffle
  gains     : one per presynaptic neuron, or one per presynaptic cell type
  seeds     : 3 per condition                      -> 18 networks, trained IN PARALLEL on one GPU
Task: hold (cued heading, no turning). Tests at 40, 80 and 200 steps.
Model and training follow ChessFly level L2: h <- 0.5 h + 0.5 (ReLU(h) W + u), 2 updates/step,
W = sign * gain * kappa * counts (spectral radius 0.9), Adam 1e-3, batch 128, 2000 steps,
clip 1.0 PER NETWORK, best-validation checkpoint every 250 steps.

Colab (GPU T4):
    !unzip -oq hd_networks_4flies.zip            # gives networks/mcns/...
    !python pilot_stage3.py --smoke              # 1-minute check, tiny settings
    !python pilot_stage3.py                      # full pilot, 2000 steps (~4 min on T4)
    !python pilot_stage3.py --steps 8000         # longer training until converged (~17 min)
Outputs (send back): pilot_results.csv, pilot_params.npz
"""

import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--net", default="networks/mcns")
ap.add_argument("--steps", type=int, default=2000, help="training updates (pilot: 2000; use 8000 to converge)")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS, EVAL_EVERY, BATCH, N_TRAIN = (20, 10, 32, 200) if args.smoke else (args.steps, 250, 128, 1000)
N_VAL, N_TEST = (100, 100) if args.smoke else (500, 2000)
T_TRAIN, CUE, MICRO, ALPHA, LR = 40, 5, 2, 0.5, 1e-3
print(f"device {dev} | smoke={args.smoke}")

# ------------------------------------------------------------------ network
C0 = sp.load_npz(f"{args.net}/subnetwork_adjacency.npz").toarray().astype(np.float64)
meta = pd.read_csv(f"{args.net}/subnetwork_neurons.csv")
sign = pd.read_csv(f"{args.net}/subnetwork_nt.csv").sort_values("index")["sign"].to_numpy().astype(np.float64)
types = meta["type"].astype(str).to_numpy()
N = len(types)
grp = pd.Series(types).str.extract(r"^(EPG|PEN|PEG|Delta7|ER)")[0].to_numpy()
ER, PEN, EPG = (np.flatnonzero(grp == g) for g in ("ER", "PEN", "EPG"))
type_id = pd.factorize(types)[0]; NT = type_id.max() + 1
print(f"network: {N} neurons, {(C0 > 0).sum()} connections, {NT} cell types")


def partial_shuffle(C, frac, seed, max_tries_per_edge=40):   # same as ChessFly analysis_phase7.py
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(C); w = C[pre, post].copy(); post = post.copy(); E = len(pre)
    original = set(zip(pre.tolist(), post.tolist())); edges = set(original)
    moved, tries = 0, 0
    while moved < frac * E and tries < max_tries_per_edge * E:
        i, j = rng.integers(E, size=2); tries += 1
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == c or b == d or (a, d) in edges or (c, b) in edges:
            continue
        moved -= ((a, b) not in original) + ((c, d) not in original)
        edges.discard((a, b)); edges.discard((c, d)); edges.add((a, d)); edges.add((c, b)); post[i], post[j] = d, b
        moved += ((a, d) not in original) + ((c, b) not in original)
    out = np.zeros_like(C); out[pre, post] = w
    return out, moved / E


wirings = {"real": (C0, 0.0)}
wirings["shuffle25"] = partial_shuffle(C0, 0.25, 9025)
wirings["shuffle_max"] = partial_shuffle(C0, 1.0, 9100)
for k, (_, m) in wirings.items():
    print(f"  {k:12s} edges moved {m:.3f}")

conds = [(w, g, s) for w in wirings for g in ("neuron", "celltype") for s in range(3)]
K = len(conds)


def scaled(C):
    S = C * sign[:, None]
    rho = np.abs(np.linalg.eigvals(S)).max()
    return S * (0.9 / rho)                     # signed, spectral radius 0.9


W0 = torch.tensor(np.stack([scaled(wirings[w][0]) for w, _, _ in conds]), dtype=torch.float32, device=dev)
mask_er = torch.zeros(N, device=dev); mask_er[ER] = 1
mask_pen = torch.zeros(N, device=dev); mask_pen[PEN] = 1
mask_epg = torch.zeros(N, device=dev); mask_epg[EPG] = 1
tid = torch.tensor(type_id, device=dev)
is_ct = torch.tensor([g == "celltype" for _, g, _ in conds], device=dev)

# ------------------------------------------------------------------ tasks (hold)
def hold_set(n, T, seed):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, n)
    tgt = np.repeat(th[:, None], T, 1)
    return torch.tensor(th, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

val = hold_set(N_VAL, T_TRAIN, 999)
tests = {T: hold_set(N_TEST, T, 1001 + i) for i, T in enumerate((40, 80, 200))}

# ------------------------------------------------------------------ parameters (one slice per network)
def init_params():
    P = {k: [] for k in ("g_neuron", "g_type", "Wc", "bc", "wv", "bv", "R", "r")}
    for _, _, s in conds:
        g = torch.Generator().manual_seed(s)
        P["g_neuron"].append(torch.zeros(N)); P["g_type"].append(torch.zeros(NT))
        P["Wc"].append((torch.rand(3, N, generator=g) * 2 - 1) * 0.577)
        P["bc"].append((torch.rand(N, generator=g) * 2 - 1) * 0.577)
        P["wv"].append((torch.rand(N, generator=g) * 2 - 1))
        P["bv"].append((torch.rand(N, generator=g) * 2 - 1))
        P["R"].append((torch.rand(N, 2, generator=g) * 2 - 1) * (1 / np.sqrt(len(EPG))))
        P["r"].append(torch.zeros(2))
    return {k: torch.stack(v).to(dev).requires_grad_(True) for k, v in P.items()}

P = init_params()


def run(P, theta, T, W_override=None):
    """theta: (K, B) cued headings. Returns predictions (K, B, T, 2)."""
    Kb, B = theta.shape
    gain = torch.where(is_ct[:, None], P["g_type"][:, tid], P["g_neuron"])          # (K, N)
    W = (W_override if W_override is not None else W0) * torch.exp(gain)[:, :, None]
    cue = torch.stack([torch.cos(theta), torch.sin(theta), torch.ones_like(theta)], -1)  # (K, B, 3)
    u_cue = (torch.bmm(cue, P["Wc"]) + P["bc"][:, None, :]) * mask_er                 # (K, B, N)
    u_base = P["bc"][:, None, :] * mask_er + P["bv"][:, None, :] * mask_pen          # hold: no velocity
    h = torch.zeros(Kb, B, N, device=dev); out = []
    for t in range(T):
        u = (u_cue + P["bv"][:, None, :] * mask_pen) if t < CUE else u_base
        for _ in range(MICRO):
            h = (1 - ALPHA) * h + ALPHA * (torch.bmm(torch.relu(h), W) + u)
        y = torch.bmm(torch.relu(h) * mask_epg, P["R"]) + P["r"][:, None, :]
        out.append(y)
    return torch.stack(out, 2)


def heading_error(pred, tgt):
    """mean absolute wrapped error (deg) over post-cue steps, per network."""
    ang = torch.atan2(pred[..., 1], pred[..., 0])
    d = torch.remainder(ang - tgt[None] + np.pi, 2 * np.pi) - np.pi
    return torch.rad2deg(d[:, :, CUE:].abs()).mean((1, 2))


@torch.no_grad()
def evaluate(P, data, T, chunk=500):
    th, tgt = data; errs = []
    for i in range(0, len(th), chunk):
        thb = th[i:i + chunk].to(dev)[None].expand(K, -1)
        errs.append(heading_error(run(P, thb, T), tgt[i:i + chunk].to(dev)) * len(thb[0]))
    return (torch.stack(errs).sum(0) / len(th)).cpu().numpy()


# ------------------------------------------------------------------ training
train_sets = [hold_set(N_TRAIN, T_TRAIN, 10_000 + 100 * s + 1000) for _, _, s in conds]
TH = torch.stack([t[0] for t in train_sets]).to(dev); TG = torch.stack([t[1] for t in train_sets]).to(dev)
gens = [torch.Generator(device="cpu").manual_seed(s) for _, _, s in conds]
opt = torch.optim.Adam(P.values(), lr=LR)
best = np.full(K, np.inf); best_state = {k: v.detach().clone() for k, v in P.items()}
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.stack([torch.randint(N_TRAIN, (BATCH,), generator=g) for g in gens]).to(dev)
    th = torch.gather(TH, 1, idx); tg = torch.gather(TG, 1, idx[:, :, None].expand(-1, -1, T_TRAIN))
    pred = run(P, th, T_TRAIN)
    target = torch.stack([torch.cos(tg), torch.sin(tg)], -1)
    loss_k = ((pred - target) ** 2).mean((1, 2, 3))                 # per network
    opt.zero_grad(); loss_k.sum().backward()
    # gradient clipping to norm 1.0 separately for each network
    with torch.no_grad():
        grads = [p for p in P.values() if p.grad is not None]     # velocity weights are unused in hold
        norms = torch.sqrt(sum((p.grad.reshape(K, -1) ** 2).sum(1) for p in grads))
        scale = torch.clamp(1.0 / (norms + 1e-6), max=1.0)
        for p in grads:
            p.grad.mul_(scale.view(-1, *([1] * (p.dim() - 1))))
    opt.step()
    if step % EVAL_EVERY == 0:
        e = evaluate(P, val, T_TRAIN)
        better = e < best
        for k in range(K):
            if better[k]:
                for name in P:
                    best_state[name][k] = P[name][k].detach().clone()
        best = np.minimum(best, e)
        print(f"step {step:5d} | val error median {np.median(e):6.2f} deg | best median {np.median(best):6.2f} | {time.time() - t0:5.0f}s")

with torch.no_grad():
    for name in P:
        P[name].copy_(best_state[name])
rows = []
res = {T: evaluate(P, tests[T], T) for T in tests}
for k, (w, g, s) in enumerate(conds):
    rows.append(dict(wiring=w, gains=g, seed=s, edges_moved=wirings[w][1], val_err=best[k],
                     test40=res[40][k], test80=res[80][k], test200=res[200][k]))
df = pd.DataFrame(rows); df.to_csv("pilot_results.csv", index=False)
print("\n" + df.groupby(["wiring", "gains"])[["test40", "test80", "test200"]].median().round(2).to_string())
np.savez_compressed("pilot_params.npz", **{k: v.detach().cpu().numpy() for k, v in P.items()},
                    W0=W0.cpu().numpy(), conds=np.array([f"{w}|{g}|{s}" for w, g, s in conds]),
                    type_id=type_id, ER=ER, PEN=PEN, EPG=EPG)
print("saved pilot_results.csv and pilot_params.npz")
