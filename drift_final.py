"""Final E-PG drift analysis, DANDI 000289 contrast experiment (Basnak et al. 2025).
Bump phase = population vector over PB midline ROIs (Basnak angle map). Drift fitted as
MSD(t) = 2 D t + 2 sigma^2 on pooled runs; CIs by bootstrapping flies."""
import numpy as np, pandas as pd
from drift_lib import *
S = pd.read_csv("/mnt/user-data/uploads/dandi_sessions.csv")
T = pd.read_csv("/mnt/user-data/uploads/dandi_trials.csv").merge(S[["path", "subject"]], on="path")
rng = np.random.default_rng(1)

def runs_of(mask, min_len):
    idx = np.flatnonzero(mask)
    if not len(idx): return []
    return [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(r) >= min_len]

def collect(still_deg=10, min_run_s=5, contrast=1, t_window=(0, 1e9), strength_q=0.0):
    per_fly = {}
    for path, g in T.groupby("path"):
        z = np.load("extracted/" + path.replace("/", "__").replace(".nwb", ".npz"), allow_pickle=True)
        F = z["ophys__Fluorescence__RoiResponseSeries__data"]; t = z["ophys__Fluorescence__RoiResponseSeries__timestamps"]
        head = z["behavior__Position__SpatialSeries__data"]
        ph, st = bump_phase(F); dt = float(np.median(np.diff(t)))
        rate = np.gradient(np.unwrap(head), t)
        thr = np.quantile(st, strength_q) if strength_q else -1
        segs = []
        for _, b in g[g.cue_contrast == contrast].iterrows():
            since = t - b.start_time
            m = (t >= b.start_time) & (t < b.stop_time) & (since >= t_window[0]) & (since < t_window[1])
            ok = m & (np.abs(rate) < np.deg2rad(still_deg)) & (st > thr)
            segs += [np.unwrap(ph[r]) for r in runs_of(ok, int(round(min_run_s / dt)))]
        per_fly[g.subject.iloc[0]] = segs
    return per_fly, dt

def estimate(per_fly, dt, fit_s=3.0, max_lag=5.0, n_boot=2000):
    flies = [f for f in per_fly if per_fly[f]]
    allsegs = [s for f in flies for s in per_fly[f]]
    lags, msd, _ = msd_curve(allsegs, dt, max_lag)
    D, sig = fit_diffusion(lags, msd, fit_s)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(flies, len(flies), replace=True)
        bs = [s for f in pick for s in per_fly[f]]
        l, m, _ = msd_curve(bs, dt, max_lag); boots.append(fit_diffusion(l, m, fit_s)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per = {f: fit_diffusion(*msd_curve(per_fly[f], dt, max_lag)[:2], fit_s)[0] for f in flies}
    minutes = sum(len(s) for s in allsegs) * dt / 60
    return dict(D=D, lo=lo, hi=hi, sigma=sig, flies=len(flies), runs=len(allsegs), minutes=minutes, per_fly=per, lags=lags, msd=msd)

def deg(D): return np.degrees(np.sqrt(2 * D))   # spread (deg) after 1 s, Var = 2Dt convention

if __name__ == "__main__":
    out = []
    main_pf, dt = collect()
    M = estimate(main_pf, dt)
    print(f"MAIN darkness, no turning (<10 deg/s), runs>=5 s: D = {M['D']:.4f} rad^2/s  95% CI [{M['lo']:.4f}, {M['hi']:.4f}]"
          f"  = {deg(M['D']):.1f} deg/sqrt(s); noise {M['sigma']:.2f} rad; {M['flies']} flies, {M['runs']} runs, {M['minutes']:.1f} min")
    B = estimate(*collect(contrast=3))
    print(f"CONTROL bright cue, same criteria:         D = {B['D']:.4f} [{B['lo']:.4f}, {B['hi']:.4f}]; {B['flies']} flies, {B['runs']} runs, {B['minutes']:.1f} min")
    Lo = estimate(*collect(contrast=2))
    print(f"Low-contrast cue:                          D = {Lo['D']:.4f} [{Lo['lo']:.4f}, {Lo['hi']:.4f}]; {Lo['runs']} runs")
    print("\nTime since cue offset (darkness):")
    tw = []
    for w in [(0, 50), (50, 100), (100, 150), (150, 200)]:
        E = estimate(*collect(t_window=w), n_boot=1000)
        tw.append((w, E)); print(f"  {w[0]:3d}-{w[1]:3d} s: D = {E['D']:.4f} [{E['lo']:.4f}, {E['hi']:.4f}]  ({E['runs']} runs, {E['minutes']:.1f} min)")
    print("\nRobustness (darkness):")
    for lab, kw, fs in [("still < 5 deg/s", dict(still_deg=5), 3), ("still < 20 deg/s", dict(still_deg=20), 3),
                        ("runs >= 3 s", dict(min_run_s=3), 2), ("runs >= 8 s", dict(min_run_s=8), 3),
                        ("bump strength > median", dict(strength_q=0.5), 3), ("fit window 2 s", {}, 2), ("fit window 4 s", {}, 4)]:
        E = estimate(*collect(**kw), fit_s=fs, n_boot=500)
        print(f"  {lab:24s} D = {E['D']:.4f} [{E['lo']:.4f}, {E['hi']:.4f}]  ({E['runs']} runs)")
    pd.DataFrame([dict(fly=f, D=v) for f, v in M["per_fly"].items()]).to_csv("drift_per_fly.csv", index=False)
    np.savez("drift_results.npz", dark_l=M["lags"], dark_m=M["msd"], bright_l=B["lags"], bright_m=B["msd"],
             low_l=Lo["lags"], low_m=Lo["msd"], D=M["D"], lo=M["lo"], hi=M["hi"], sigma=M["sigma"], Db=B["D"], Dl=Lo["D"],
             tw_mid=[np.mean(w) for w, _ in tw], tw_D=[E["D"] for _, E in tw], tw_lo=[E["lo"] for _, E in tw], tw_hi=[E["hi"] for _, E in tw],
             per_fly=np.array(list(M["per_fly"].values())))
    print(f"\nModel value (Turner-Evans 2017): 0.00182 rad^2/s -> ratio {M['D']/0.00182:.0f}x (if same Var = 2Dt convention)")
