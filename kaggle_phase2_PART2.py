# ================================================================ 2. validation
def validate(adj, meta, signs, device):
    from reservoir_core import FlyReservoir, Encoder, select_input_neurons, random_position

    boards = [chess.Board()]
    b = chess.Board(); b.push_san("e4"); boards.append(b)
    b = chess.Board(); b.push_san("d4"); boards.append(b)
    b = chess.Board()
    for mv in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6"]:
        b.push_san(mv)
    boards.append(b)
    rng = np.random.default_rng(7)
    for plies in [16, 24, 40, 60]:
        boards.append(random_position(rng, plies))
    boards = [canonical(x) for x in boards]

    log("Validation: Brian2 (CPU) vs PyTorch (GPU) on 8 positions ...")
    fly = FlyReservoir()
    brian = np.stack([np.mean([fly.run(x, sim_seed=s)["all_rates"] for s in range(2)], axis=0)
                      for x in boards])

    enc = Encoder(LOCKED_NIN)
    idx = select_input_neurons(adj, meta, LOCKED_NIN)
    acts = np.stack([enc.activation(x) for x in boards])
    tr = TorchReservoir(weight_matrix(adj, signs, LOCKED_SCALE), idx, LOCKED_IMAX, device)
    torch_rates = np.mean([tr.run(acts, seed=s).sum(axis=1) for s in range(8)], axis=0)

    r = float(np.corrcoef(brian.ravel(), torch_rates.ravel())[0, 1])
    mae = float(np.abs(brian - torch_rates).mean())
    log(f"  mean rate  Brian2 {brian.mean():.2f} Hz   PyTorch {torch_rates.mean():.2f} Hz")
    log(f"  per-neuron rate correlation r = {r:.3f}   mean abs diff = {mae:.2f} Hz")
    if r > 0.9:
        log("  PASS: GPU simulator matches Brian2.")
    else:
        log("  WARNING: GPU simulator differs from Brian2 -- treat results with caution "
            "and report this back.")
    return r


# ================================================================ 3. calibration (wide input)
def calibrate_wide(adj, signs, idx, hidden, acts_sample, device):
    log(f"Calibrating wide input ({N_INPUT_WIDE} input neurons) on "
        f"{len(acts_sample)} positions ...")
    rows = []
    for scale in [0.14, 0.22]:
        W = weight_matrix(adj, signs, scale)
        for imax in [10.0, 20.0, 30.0]:
            res = TorchReservoir(W, idx, imax, device)
            r0 = res.run(acts_sample, seed=1).sum(axis=1)[:, hidden]
            r1 = res.run(acts_sample, seed=2).sum(axis=1)[:, hidden]
            active = 100.0 * float((r0 > 0).mean())
            within = float(np.linalg.norm(r0 - r1, axis=1).mean())
            d = cdist(r0, r0)
            between = float(d[~np.eye(len(d), dtype=bool)].mean())
            sep = between / within if within > 1e-9 else float("nan")
            rows.append((scale, imax, active, float(r0.mean()), sep))
            log(f"  scale {scale:.2f}  I_max {imax:4.1f}  active {active:5.1f}%  "
                f"rate {r0.mean():6.2f} Hz  sep {sep:6.2f}")
    useful = [r for r in rows if 15.0 <= r[2] <= 80.0 and not np.isnan(r[4])]
    pick = max(useful or rows, key=lambda r: (r[4] if not np.isnan(r[4]) else -1))
    log(f"  -> using scale {pick[0]:.2f}, I_max {pick[1]:.1f} (sep {pick[4]:.2f})")
    return pick[0], pick[1]


# ================================================================ 4. readout
def legal_tables(fens, moves):
    lf, lt, tg, valid = [], [], [], []
    for fen, mv in zip(fens, moves):
        board = chess.Board(str(fen))
        t = chess.Move.from_uci(str(mv))
        legal = usable_moves(board)
        idx = next((k for k, m in enumerate(legal)
                    if m.from_square == t.from_square and m.to_square == t.to_square), -1)
        lf.append([m.from_square for m in legal])
        lt.append([m.to_square for m in legal])
        tg.append(max(idx, 0))
        valid.append(idx >= 0 and len(legal) > 0)
    L = max(len(x) for x in lf)
    n = len(lf)
    LF = np.zeros((n, L), dtype=np.int64)
    LT = np.zeros((n, L), dtype=np.int64)
    MASK = np.zeros((n, L), dtype=bool)
    for i, (f, t) in enumerate(zip(lf, lt)):
        LF[i, :len(f)] = f
        LT[i, :len(t)] = t
        MASK[i, :len(f)] = True
    return LF, LT, MASK, np.array(tg, dtype=np.int64), np.array(valid)


def split_by_game(gids, valid):
    ug = np.unique(gids)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    n = len(ug)
    g_tr, g_va = ug[: int(0.70 * n)], ug[int(0.70 * n): int(0.85 * n)]
    g_te = ug[int(0.85 * n):]
    pick = lambda g: np.where(np.isin(gids, g) & valid)[0]
    return pick(g_tr), pick(g_va), pick(g_te)


def train_eval(name, X, tabs, splits, device):
    LF, LT, MASK, TG = tabs
    tr, va, te = splits
    X = X.astype(np.float32, copy=False)
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0) + 1e-6
    D = X.shape[1]
    Xg = torch.empty((len(X), D), dtype=torch.float32, device=device)
    for s in range(0, len(X), 8192):
        Xg[s:s + 8192] = torch.from_numpy((X[s:s + 8192] - mu) / sd).to(device)

    def scores(model, idx):
        out = model(Xg[idx])
        s = out[:, :64].gather(1, LF[idx]) + out[:, 64:].gather(1, LT[idx])
        return s.masked_fill(~MASK[idx], -1e9)

    @torch.no_grad()
    def accuracy(model, idx_np, k):
        correct = 0
        for s in range(0, len(idx_np), 4096):
            idx = torch.from_numpy(idx_np[s:s + 4096]).to(device)
            top = scores(model, idx).topk(k, dim=1).indices
            correct += int((top == TG[idx][:, None]).any(dim=1).sum())
        return correct / len(idx_np)

    best = None
    tr_t = torch.from_numpy(tr).to(device)
    for wd in WD_GRID:
        torch.manual_seed(0)
        model = torch.nn.Linear(D, 128).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        best_va, best_state = -1.0, None
        for _ in range(EPOCHS):
            perm = tr_t[torch.randperm(len(tr_t), device=device)]
            for s in range(0, len(perm), 512):
                b = perm[s:s + 512]
                loss = Fnn.cross_entropy(scores(model, b), TG[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            va_acc = accuracy(model, va, 1)
            if va_acc > best_va:
                best_va = va_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"    {name:<28} wd={wd:<5g} best val top-1 = {best_va:.3f}")
        if best is None or best_va > best["val"]:
            best = {"val": best_va, "wd": wd, "state": best_state}

    model = torch.nn.Linear(D, 128).to(device)
    model.load_state_dict(best["state"])
    te1, te3 = accuracy(model, te, 1), accuracy(model, te, 3)
    ci = 1.96 * np.sqrt(te1 * (1 - te1) / len(te))
    torch.save({"state": best["state"], "mu": mu, "sd": sd, "wd": best["wd"]},
               os.path.join(OUT_DIR, f"readout_{name}.pt"))
    del Xg
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"representation": name, "dims": D, "wd": best["wd"], "val_top1": best["val"],
            "test_top1": te1, "test_top1_ci95": ci, "test_top3": te3}


# ================================================================ main
def main():
    log("=" * 78)
    log(f"CHESSFLY PHASE 2   positions={N_POSITIONS}  wide_inputs={N_INPUT_WIDE}  "
        f"bins={N_BINS}  batch={BATCH}")
    log(f"output -> {OUT_DIR}")
    log("=" * 78)

    sf_path = ensure_stockfish()
    log(f"Stockfish: {sf_path}")
    ensure_network_files()

    # 1. dataset (before any CUDA use, so worker processes fork cleanly)
    fens, moves, cps, gids = build_dataset(sf_path)
    n = len(fens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                               "  -- WARNING: no GPU, this will be slow"))

    adj, meta, signs = load_network()
    log(f"Network: {adj.shape[0]} neurons, {adj.nnz} connections, "
        f"{int((signs < 0).sum())} inhibitory")

    # 2. validation
    if VALIDATE:
        validate(adj, meta, signs, device)

    from reservoir_core import Encoder, select_input_neurons, board_to_features

    log("Preparing features and legal-move tables ...")
    raw = np.stack([board_to_features(chess.Board(str(f))) for f in fens]).astype(np.float32)
    enc48, encW = Encoder(LOCKED_NIN), Encoder(N_INPUT_WIDE)
    acts48, actsW = encode_batch(enc48, raw), encode_batch(encW, raw)
    idx48 = select_input_neurons(adj, meta, LOCKED_NIN)
    idxW = select_input_neurons(adj, meta, N_INPUT_WIDE)
    hid48 = np.ones(adj.shape[0], dtype=bool); hid48[idx48] = False
    hidW = np.ones(adj.shape[0], dtype=bool); hidW[idxW] = False

    LF, LT, MASK, TG, valid = legal_tables(fens, moves)
    tr, va, te = split_by_game(gids, valid)
    tabs = tuple(torch.from_numpy(a).to(device) for a in (LF, LT, MASK, TG))
    n_legal = MASK[te].sum(axis=1)
    log(f"Split by game: train {len(tr)} / val {len(va)} / test {len(te)} positions")

    # 3. simulation
    log("Simulating on GPU ...")
    bins48 = simulate("locked48", weight_matrix(adj, signs, LOCKED_SCALE), idx48,
                      LOCKED_IMAX, acts48, hid48, device)
    sample = np.random.default_rng(1).choice(n, size=min(128, n), replace=False)
    scaleW, imaxW = calibrate_wide(adj, signs, idxW, hidW, actsW[sample], device)
    WW = weight_matrix(adj, signs, scaleW)
    tagW = f"wide{N_INPUT_WIDE}_s{scaleW:g}_i{imaxW:g}"
    binsW = simulate(tagW, WW, idxW, imaxW, actsW, hidW, device)
    binsR = simulate("rewired_" + tagW, rewire(WW, seed=0), idxW, imaxW, actsW, hidW, device)

    # 4. readouts
    W_ = N_INPUT_WIDE
    reps = [
        ("raw_board_768", lambda: raw),
        ("encoder_48", lambda: acts48),
        ("fly48_rates", lambda: bins48.sum(axis=1).astype(np.float32)),
        ("fly48_bins+input", lambda: np.hstack([acts48, bins48.reshape(n, -1)])),
        (f"encoder_{W_}", lambda: actsW),
        (f"fly{W_}_bins", lambda: binsW.reshape(n, -1).astype(np.float32)),
        (f"fly{W_}_bins+input", lambda: np.hstack([actsW, binsW.reshape(n, -1)])),
        (f"rewired{W_}_bins+input", lambda: np.hstack([actsW, binsR.reshape(n, -1)])),
    ]
    log("Training readouts ...")
    results = [{"representation": "random legal move", "dims": 0, "wd": float("nan"),
                "val_top1": float("nan"), "test_top1": float(np.mean(1.0 / n_legal)),
                "test_top1_ci95": 0.0,
                "test_top3": float(np.mean(np.minimum(3, n_legal) / n_legal))}]
    for name, make in reps:
        results.append(train_eval(name, make(), tabs, (tr, va, te), device))

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "phase2_results.csv")
    df.to_csv(csv_path, index=False)

    log("\n" + "=" * 78)
    log(f"PHASE 2 RESULTS   ({n} positions, test {len(te)})")
    log("=" * 78)
    log(f"{'representation':<28}{'dims':>6}{'val top-1':>11}{'test top-1':>18}{'test top-3':>12}")
    for r in results:
        va_s = "-" if np.isnan(r["val_top1"]) else f"{r['val_top1']:.3f}"
        te_s = f"{r['test_top1']:.3f} +/- {r['test_top1_ci95']:.3f}"
        log(f"{r['representation']:<28}{r['dims']:>6}{va_s:>11}{te_s:>18}{r['test_top3']:>12.3f}")

    get = {r["representation"]: r for r in results}
    fly, enc, rew = (get[f"fly{W_}_bins+input"], get[f"encoder_{W_}"],
                     get[f"rewired{W_}_bins+input"])
    log("\nKey comparisons (test top-1, val top-1):")
    log(f"  fly+input vs encoder : {fly['test_top1'] - enc['test_top1']:+.3f}, "
        f"{fly['val_top1'] - enc['val_top1']:+.3f}   (does the brain add information?)")
    log(f"  fly+input vs rewired : {fly['test_top1'] - rew['test_top1']:+.3f}, "
        f"{fly['val_top1'] - rew['val_top1']:+.3f}   (does the real wiring matter?)")
    log("  A difference is convincing if it is positive on BOTH val and test and larger "
        "than the test CI.")
    log(f"\nSaved {csv_path} and readout_*.pt")


if __name__ == "__main__":
    main()
