"""
stage4_transplant.py -- Stage 4, main experiment: wiring transplant (preregistration
doi:10.5281/zenodo.23382486, hypotheses H1-H3).

Trains 4 flies (mcns, hemibrain, fafb, banc) x 2 gain types (per neuron, per cell type) x 5 seeds
= 40 networks IN PARALLEL on one GPU (hold task, same model and training as the pilot), then
evaluates every trained network on every other fly's wiring WITHOUT retraining. Neuron-specific
parameters (gains, cue/velocity input weights, readout) are carried over through the precomputed
one-to-one maps in networks/transplant_maps.npz (same family + rank of inferred compass angle).

Colab (GPU T4):
    !unzip -oq hd_networks_4flies.zip
    !python stage4_transplant.py --smoke          # 1-2 minute check
    !python stage4_transplant.py                  # full run (~40-50 min at 8000 steps)
Outputs (send back): stage4_results.csv, stage4_params.npz
"""

import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--dir", default="networks")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS, EVAL_EVERY, BATCH, N_TRAIN = (20, 10, 32, 200) if args.smoke else (args.steps, 250, 128, 1000)
N_VAL, N_TEST = (100, 100) if args.smoke else (500, 2000)
SEEDS = [0, 1] if args.smoke else [0, 1, 2, 3, 4]
T_TRAIN, CUE, MICRO, ALPHA, LR = 40, 5, 2, 0.5, 1e-3
FLIES = ["mcns", "hemibrain", "fafb", "banc"]
print(f"device {dev} | smoke={args.smoke} | steps={STEPS}")
if dev.type != "cuda" and not args.smoke:
    raise SystemExit("No GPU: Runtime -> Change runtime type -> T4 GPU, then re-upload the files.")

# ------------------------------------------------------------------ flies
F = {}
for f in FLIES:
    C = sp.load_npz(f"{args.dir}/{f}/subnetwork_adjacency.npz").toarray().astype(np.float64)
    meta = pd.read_csv(f"{args.dir}/{f}/subnetwork_neurons.csv")
    sign = pd.read_csv(f"{args.dir}/{f}/subnetwork_nt.csv").sort_values("index")["sign"].to_numpy().astype(np.float64)
    S = C * sign[:, None]; S = S * (0.9 / np.abs(np.linalg.eigvals(S)).max())
    grp = meta["type"].astype(str).str.extract(r"^(EPG|PEN|PEG|Delta7|ER)")[0].to_numpy()
    F[f] = dict(W=S, n=len(meta), tid=pd.factorize(meta["type"].astype(str))[0],
                ER=np.flatnonzero(grp == "ER"), PEN=np.flatnonzero(grp == "PEN"), EPG=np.flatnonzero(grp == "EPG"))
    print(f"  {f:9s} {F[f]['n']} neurons, {(C > 0).sum()} connections, {F[f]['tid'].max() + 1} cell types")
NMAX = max(F[f]["n"] for f in FLIES); NTMAX = max(F[f]["tid"].max() + 1 for f in FLIES)
MAPS = np.load(f"{args.dir}/transplant_maps.npz", allow_pickle=True)

conds = [(f, g, s) for f in FLIES for g in ("neuron", "celltype") for s in SEEDS]
K = len(conds)


def pad_W(f):
    W = np.zeros((NMAX, NMAX)); n = F[f]["n"]; W[:n, :n] = F[f]["W"]; return W


def masks(f):
    m = np.zeros((3, NMAX))
    for i, key in enumerate(("ER", "PEN", "EPG")):
        m[i, F[f][key]] = 1
    return m


def pad_tid(f):
    t = np.zeros(NMAX, int); t[:F[f]["n"]] = F[f]["tid"]; return t


W_own = torch.tensor(np.stack([pad_W(f) for f, _, _ in conds]), dtype=torch.float32, device=dev)
M_own = torch.tensor(np.stack([masks(f) for f, _, _ in conds]), dtype=torch.float32, device=dev)   # (K, 3, N)
TID = torch.tensor(np.stack([pad_tid(f) for f, _, _ in conds]), device=dev)
is_ct = torch.tensor([g == "celltype" for _, g, _ in conds], device=dev)


# ------------------------------------------------------------------ task
def hold_set(n, T, seed):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, n)
    return torch.tensor(th, dtype=torch.float32), torch.tensor(np.repeat(th[:, None], T, 1), dtype=torch.float32)


val = hold_set(N_VAL, T_TRAIN, 999)
tests = {T: hold_set(N_TEST, T, 1001 + i) for i, T in enumerate((40, 80, 200))}


# ------------------------------------------------------------------ parameters
def init_params():
    P = {k: [] for k in ("g_neuron", "g_type", "Wc", "bc", "wv", "bv", "R", "r")}
    for f, _, s in conds:
        g = torch.Generator().manual_seed(s)
        P["g_neuron"].append(torch.zeros(NMAX)); P["g_type"].append(torch.zeros(NTMAX))
        P["Wc"].append((torch.rand(3, NMAX, generator=g) * 2 - 1) * 0.577)
        P["bc"].append((torch.rand(NMAX, generator=g) * 2 - 1) * 0.577)
        P["wv"].append(torch.rand(NMAX, generator=g) * 2 - 1)
        P["bv"].append(torch.rand(NMAX, generator=g) * 2 - 1)
        P["R"].append((torch.rand(NMAX, 2, generator=g) * 2 - 1) / np.sqrt(len(F[f]["EPG"])))
        P["r"].append(torch.zeros(2))
    return {k: torch.stack(v).to(dev).requires_grad_(True) for k, v in P.items()}


def gain_vector(P, tid=None):
    tid = TID if tid is None else tid
    return torch.where(is_ct[:, None], torch.gather(P["g_type"], 1, tid), P["g_neuron"])


def run(theta, T, W0, M, gain, Wc, bc, bv, R, r):
    Kb, B = theta.shape
    W = W0 * torch.exp(gain)[:, :, None]
    mER, mPEN, mEPG = M[:, 0][:, None, :], M[:, 1][:, None, :], M[:, 2][:, None, :]
    cue = torch.stack([torch.cos(theta), torch.sin(theta), torch.ones_like(theta)], -1)
    u_cue = (torch.bmm(cue, Wc) + bc[:, None, :]) * mER + bv[:, None, :] * mPEN
    u_base = bc[:, None, :] * mER + bv[:, None, :] * mPEN
    h = torch.zeros(Kb, B, W0.shape[-1], device=dev); out = []
    for t in range(T):
        u = u_cue if t < CUE else u_base
        for _ in range(MICRO):
            h = (1 - ALPHA) * h + ALPHA * (torch.bmm(torch.relu(h), W) + u)
        out.append(torch.bmm(torch.relu(h) * mEPG, R) + r[:, None, :])
    return torch.stack(out, 2)


def heading_error(pred, tgt):
    ang = torch.atan2(pred[..., 1], pred[..., 0])
    d = torch.remainder(ang - tgt[None] + np.pi, 2 * np.pi) - np.pi
    return torch.rad2deg(d[:, :, CUE:].abs()).mean((1, 2))


@torch.no_grad()
def evaluate(data, T, W0, M, gain, Wc, bc, bv, R, r, chunk=500):
    th, tgt = data; tot = 0
    for i in range(0, len(th), chunk):
        thb = th[i:i + chunk].to(dev)[None].expand(K, -1)
        tot = tot + heading_error(run(thb, T, W0, M, gain, Wc, bc, bv, R, r), tgt[i:i + chunk].to(dev)) * thb.shape[1]
    return (tot / len(th)).cpu().numpy()


# ------------------------------------------------------------------ training
P = init_params()
train = [hold_set(N_TRAIN, T_TRAIN, 10_000 + 100 * s + 1000) for _, _, s in conds]
TH = torch.stack([t[0] for t in train]).to(dev); TG = torch.stack([t[1] for t in train]).to(dev)
gens = [torch.Generator().manual_seed(1000 * FLIES.index(f) + s) for f, _, s in conds]
opt = torch.optim.Adam(P.values(), lr=LR)
best = np.full(K, np.inf); best_state = {k: v.detach().clone() for k, v in P.items()}
own = lambda: (W_own, M_own, gain_vector(P), P["Wc"], P["bc"], P["bv"], P["R"], P["r"])
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.stack([torch.randint(N_TRAIN, (BATCH,), generator=g) for g in gens]).to(dev)
    th = torch.gather(TH, 1, idx); tg = torch.gather(TG, 1, idx[:, :, None].expand(-1, -1, T_TRAIN))
    pred = run(th, T_TRAIN, *own())
    loss_k = ((pred - torch.stack([torch.cos(tg), torch.sin(tg)], -1)) ** 2).mean((1, 2, 3))
    opt.zero_grad(); loss_k.sum().backward()
    with torch.no_grad():
        grads = [p for p in P.values() if p.grad is not None]
        norms = torch.sqrt(sum((p.grad.reshape(K, -1) ** 2).sum(1) for p in grads))
        scale = torch.clamp(1.0 / (norms + 1e-6), max=1.0)
        for p in grads:
            p.grad.mul_(scale.view(-1, *([1] * (p.dim() - 1))))
    opt.step()
    if step % EVAL_EVERY == 0:
        e = evaluate(val, T_TRAIN, *own())
        for k in np.flatnonzero(e < best):
            for name in P:
                best_state[name][k] = P[name][k].detach().clone()
        best = np.minimum(best, e)
        print(f"step {step:5d} | val median {np.median(e):6.2f} deg | best median {np.median(best):6.2f} | {time.time() - t0:5.0f}s")
with torch.no_grad():
    for name in P:
        P[name].copy_(best_state[name])

# ------------------------------------------------------------------ evaluation: own wiring + transplants
rows = []
for tgt in FLIES:
    with torch.no_grad():
        W_t = torch.tensor(pad_W(tgt), dtype=torch.float32, device=dev)[None].expand(K, -1, -1).contiguous()
        M_t = torch.tensor(masks(tgt), dtype=torch.float32, device=dev)[None].expand(K, -1, -1).contiguous()
        idx = np.zeros((K, NMAX), int); valid = np.zeros((K, NMAX))
        for k, (src, _, _) in enumerate(conds):
            m = MAPS[f"{src}__{tgt}"]; idx[k, :len(m)] = m; valid[k, :len(m)] = 1
        idx_t = torch.tensor(idx, device=dev); valid_t = torch.tensor(valid, dtype=torch.float32, device=dev)
        take = lambda X: torch.gather(X, 1, idx_t) * valid_t                       # (K, N) per-neuron params
        gain = take(gain_vector(P))
        Wc = torch.gather(P["Wc"], 2, idx_t[:, None, :].expand(-1, 3, -1)) * valid_t[:, None, :]
        R = torch.gather(P["R"], 1, idx_t[:, :, None].expand(-1, -1, 2)) * valid_t[:, :, None]
        res = {T: evaluate(tests[T], T, W_t, M_t, gain, Wc, take(P["bc"]), take(P["bv"]), R, P["r"]) for T in tests}
    for k, (src, g, s) in enumerate(conds):
        rows.append(dict(source=src, target=tgt, gains=g, seed=s, own=(src == tgt), val_err=best[k],
                         test40=res[40][k], test80=res[80][k], test200=res[200][k]))
    print(f"evaluated on {tgt}")
df = pd.DataFrame(rows); df.to_csv("stage4_results.csv", index=False)
own_df = df[df.own]
print("\nOwn-wiring hold error (median deg):\n" + own_df.groupby(["source", "gains"])[["test40", "test200"]].median().round(2).to_string())
tr = df[~df.own].groupby(["source", "target", "gains"]).test40.median().unstack("target").round(1)
print("\nTransplant hold error at 40 steps (rows: trained on; columns: tested on):\n" + tr.to_string())
np.savez_compressed("stage4_params.npz", **{k: v.detach().cpu().numpy() for k, v in P.items()},
                    conds=np.array([f"{f}|{g}|{s}" for f, g, s in conds]))
print("saved stage4_results.csv and stage4_params.npz")
