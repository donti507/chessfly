"""
kaggle_phase4.py -- Phase 4: dimensionality test + portal models

PART A (science): participation ratio (PR) = effective number of dimensions of
neural activity across chess positions.
  A1 frozen spiking reservoir (48 inputs): real vs rewired vs degshuffle
  A2 trained flow networks (Phase 3b design): real vs rewired vs degshuffle,
     before and after training, several seeds
  Prediction (ring-attractor compression): PR(real) < PR(controls).

PART B (portal): writes portal_models.json with
  frozen   spiking fly brain + linear move readout
  trained  real fly wiring, policy + value heads (info flows through wiring)
  rewired  same architecture with rewired connections (comparison toggle)
All arrays base64 little-endian (float32 / int32).

Usage:  python kaggle_phase4.py      Env: SEEDS4 (0,1,2) STEPS4 (6000)
"""

import os
import json
import time
import base64

import numpy as np
import pandas as pd
import chess
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

os.environ.setdefault("N_POSITIONS", "100000")
import kaggle_phase2 as p2
import kaggle_phase3b as p3b
from reservoir_core import board_to_features, select_input_neurons, Encoder

SEEDS4 = [int(s) for s in os.environ.get("SEEDS4", "0,1,2").split(",")]
STEPS4 = int(os.environ.get("STEPS4", "6000"))
VALUE_WEIGHT, N_DIM, BATCH, LR = 0.5, 5000, 512, 1e-3
OUT = p2.OUT_DIR
log = p2.log


def participation_ratio(A):
    A = A - A.mean(axis=0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(np.cov(A, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (np.square(ev).sum() + 1e-12))


def b64(a, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a), dtype=dtype).tobytes()).decode()


class PolicyValueBrain(p3b.FlowBrain):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.value = nn.Linear(len(self.outp), 1)

    def weights(self):
        return self.sign * torch.exp(self.log_mag) * self.mask

    def hidden(self, x):
        u = self.inp(x) @ self.P
        h = torch.zeros_like(u)
        W = self.weights()
        for _ in range(p3b.T_STEPS):
            h = (1 - p3b.ALPHA) * h + p3b.ALPHA * (torch.relu(h) @ W + u)
        return torch.relu(h[:, self.outp])

    def forward(self, x):
        z = self.hidden(x)
        return self.readout(z), torch.tanh(self.value(z)).squeeze(1)


def main():
    log("=" * 78)
    log(f"CHESSFLY PHASE 4 -- dimensionality + portal models   seeds={SEEDS4} steps={STEPS4}")
    log("=" * 78)
    sf = p2.ensure_stockfish()
    p2.ensure_network_files()
    fens, moves, cps, gids = p2.build_dataset(sf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adj, meta, signs = p2.load_network()
    n = adj.shape[0]
    C = np.zeros(adj.shape, dtype=np.float32)
    C[adj.row, adj.col] = adj.data
    sensory = np.array(select_input_neurons(adj, meta, p3b.N_SENSORY))
    output = np.setdiff1d(np.arange(n), sensory)

    log("Preparing features, values and legal-move tables ...")
    raw = np.stack([board_to_features(chess.Board(str(f))) for f in fens]).astype(np.float32)
    LF, LT, MASK, TG, valid = p2.legal_tables(fens, moves)
    tr, va, te = p2.split_by_game(gids, valid)
    X = torch.from_numpy(raw).to(device)
    V = torch.from_numpy(np.tanh(np.clip(cps, -2000, 2000) / 400.0).astype(np.float32)).to(device)
    tabs = tuple(torch.from_numpy(a).to(device) for a in (LF, LT, MASK, TG))
    dim_idx = np.random.default_rng(0).permutation(te)[:N_DIM]
    va_sub = np.random.default_rng(1).permutation(va)[:5000]
    rows = []

    # ---------------- A1: frozen spiking reservoir ----------------
    log("A1: frozen spiking reservoir dimensionality ...")
    enc48 = Encoder(48)
    idx48 = np.array(select_input_neurons(adj, meta, 48))
    hid48 = np.ones(n, dtype=bool)
    hid48[idx48] = False
    acts48 = p2.encode_batch(enc48, raw)
    W_real = p2.weight_matrix(adj, signs, 0.22)
    frozen_nets = {"real": W_real, "rewired": p2.rewire(W_real, seed=100),
                   "degshuffle": p3b.degree_preserving_shuffle(C, seed=200) * 0.22 * signs[:, None]}
    for kind, W in frozen_nets.items():
        res = p2.TorchReservoir(W, idx48, 30.0, device)
        rates = np.concatenate([res.run(acts48[dim_idx[s:s + 2048]], seed=7 + s).sum(axis=1)
                                for s in range(0, N_DIM, 2048)])[:, hid48]
        pr = participation_ratio(rates)
        rows.append({"part": "frozen", "condition": kind, "seed": 0, "stage": "frozen", "PR": pr,
                     "mean_rate": float(rates.mean()), "active_pct": 100 * float((rates > 0).mean())})
        log(f"  {kind:<11} PR {pr:6.2f}   mean rate {rates.mean():6.2f} Hz   "
            f"active {100 * (rates > 0).mean():5.1f}%")

    # ---------------- A2: trained flow networks ----------------
    def scores(pol, idx):
        s = pol[:, :64].gather(1, tabs[0][idx]) + pol[:, 64:].gather(1, tabs[1][idx])
        return s.masked_fill(~tabs[2][idx], -1e9)

    @torch.no_grad()
    def evaluate(model, idx_np, k=1):
        model.eval()
        c, preds = 0, []
        for s in range(0, len(idx_np), 4096):
            idx = torch.from_numpy(idx_np[s:s + 4096]).to(device)
            pol, val = model(X[idx])
            c += int((scores(pol, idx).topk(k, dim=1).indices == tabs[3][idx][:, None]).any(1).sum())
            preds.append(val.cpu().numpy())
        model.train()
        return c / len(idx_np), np.concatenate(preds)

    @torch.no_grad()
    def activity_pr(model):
        model.eval()
        z = model.hidden(X[torch.from_numpy(dim_idx).to(device)]).cpu().numpy()
        model.train()
        return participation_ratio(z)

    keep = {}
    log("A2: training flow networks (policy + value) ...")
    for seed in SEEDS4:
        for kind in ["real", "rewired", "degshuffle"]:
            t0 = time.time()
            Ck = C if kind == "real" else (p2.rewire(C, seed=100 + seed) if kind == "rewired"
                                           else p3b.degree_preserving_shuffle(C, seed=200 + seed))
            torch.manual_seed(seed)
            model = PolicyValueBrain(kind, n, sensory, output, C=Ck, signs=signs).to(device)
            pr_init = activity_pr(model)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            tr_t = torch.from_numpy(tr).to(device)
            best, state = -1.0, None
            for step in range(1, STEPS4 + 1):
                b = tr_t[torch.randint(len(tr_t), (BATCH,), device=device)]
                pol, val = model(X[b])
                loss = Fnn.cross_entropy(scores(pol, b), tabs[3][b]) + \
                    VALUE_WEIGHT * Fnn.mse_loss(val, V[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step % 500 == 0:
                    a, _ = evaluate(model, va_sub)
                    if a > best:
                        best, state = a, {k: t.detach().clone() for k, t in model.state_dict().items()}
            model.load_state_dict(state)
            te1, vpred = evaluate(model, te, 1)
            te3, _ = evaluate(model, te, 3)
            vr = float(np.corrcoef(vpred, V[torch.from_numpy(te).to(device)].cpu().numpy())[0, 1])
            pr_tr = activity_pr(model)
            rows.append({"part": "trained", "condition": kind, "seed": seed, "stage": "init", "PR": pr_init})
            rows.append({"part": "trained", "condition": kind, "seed": seed, "stage": "trained",
                         "PR": pr_tr, "test_top1": te1, "test_top3": te3, "value_r": vr})
            log(f"  {kind:<11} seed {seed}  PR init {pr_init:6.2f} -> trained {pr_tr:6.2f}   "
                f"test {te1:.3f} top-3 {te3:.3f} value r {vr:.3f}  [{time.time() - t0:4.0f}s]")
            if seed == SEEDS4[0] and kind in ("real", "rewired"):
                keep[kind] = (model, te1, te3, vr)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase4_dimensionality.csv"), index=False)
    log("\n" + "=" * 78)
    log("DIMENSIONALITY (participation ratio; higher = richer representation)")
    log("=" * 78)
    for part, stage in [("frozen", "frozen"), ("trained", "init"), ("trained", "trained")]:
        g = df[(df.part == part) & (df.stage == stage)]
        line = f"{part + '/' + stage:<18}"
        for kind in ["real", "rewired", "degshuffle"]:
            x = g[g.condition == kind]["PR"]
            line += f"  {kind} {x.mean():6.2f}" + (f"+/-{x.std(ddof=0):.2f}" if len(x) > 1 else "")
        log(line)
        r, c = g[g.condition == "real"]["PR"], g[g.condition != "real"]["PR"]
        log(f"{'':<18}  real lower than every control: {'YES' if r.max() < c.min() else 'no'}")

    # ---------------- B: portal models ----------------
    log("\nB: frozen-fly readout for the portal ...")
    bins = p2.simulate("locked48", W_real, idx48, 30.0, acts48, hid48, device)
    res_frozen = p2.train_eval("portal_frozen_fly48", bins.sum(axis=1).astype(np.float32),
                               tabs, (tr, va, te), device)
    ro = torch.load(os.path.join(OUT, "readout_portal_frozen_fly48.pt"), weights_only=False)
    frozen = {"input_idx": b64(idx48, np.int32), "hidden_idx": b64(np.where(hid48)[0], np.int32),
              "enc_mean": b64(enc48.mean), "enc_proj": b64(enc48.proj), "enc_scale": b64(enc48.scale),
              "i_max": 30.0, "weight_scale": 0.22, "noise_mv": 1.0, "duration_ms": 1000.0,
              "readout_W": b64(ro["state"]["weight"].cpu().numpy()),
              "readout_b": b64(ro["state"]["bias"].cpu().numpy()),
              "mu": b64(ro["mu"]), "sd": b64(ro["sd"]), "test_top1": res_frozen["test_top1"]}

    def export_trained(model, te1, te3, vr):
        m = model.mask.cpu().numpy()
        pre, post = np.nonzero(m)
        W = model.weights().detach().cpu().numpy()
        g = lambda t: t.detach().cpu().numpy()
        return {"sensory": b64(sensory, np.int32), "output": b64(output, np.int32),
                "pre": b64(pre, np.int32), "post": b64(post, np.int32), "w": b64(W[pre, post]),
                "inp_W": b64(g(model.inp.weight)), "inp_b": b64(g(model.inp.bias)),
                "ro_W": b64(g(model.readout.weight)), "ro_b": b64(g(model.readout.bias)),
                "v_W": b64(g(model.value.weight)), "v_b": b64(g(model.value.bias)),
                "T": p3b.T_STEPS, "alpha": p3b.ALPHA,
                "test_top1": te1, "test_top3": te3, "value_r": vr}

    portal = {"format": "chessfly-portal-v1",
              "notes": "base64 little-endian; float32 unless *_idx/pre/post/sensory/output (int32). "
                       "Matrices row-major: enc_proj 48x768, readout_W 128xH, inp_W Sx768, ro_W 128xO. "
                       "Boards are canonical (side to move = White); 768 = 12 planes x 64 squares.",
              "network": {"n": n, "pre": b64(adj.row, np.int32), "post": b64(adj.col, np.int32),
                          "count": b64(adj.data), "sign": b64(signs),
                          "types": meta["type"].astype(str).tolist()},
              "frozen": frozen,
              "trained": {k: export_trained(*keep[k]) for k in ("real", "rewired")}}
    path = os.path.join(OUT, "portal_models.json")
    with open(path, "w") as f:
        json.dump(portal, f)
    log(f"Saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB) and phase4_dimensionality.csv")
    log(f"Portal accuracies: frozen {res_frozen['test_top1']:.3f} | trained real "
        f"{keep['real'][1]:.3f} | trained rewired {keep['rewired'][1]:.3f}")


if __name__ == "__main__":
    main()
