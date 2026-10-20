"""
dandi_extract.py -- Stage 1, step 2: stream the small arrays from every file of DANDI 000289

For each NWB file it reads ONLY:
  session_description, subject id, genotype, imaging rate
  processing/ophys/Fluorescence/RoiResponseSeries   (dF/F along the PB midline, frames x ROIs)
  processing/behavior/...                           (every behaviour series: heading, etc.)
and writes one small .npz per file plus a summary table. Raw movies are never touched.

Run on Google Colab (CPU is fine; about 20-60 min depending on network):
    !pip install -q dandi remfile h5py
    !python dandi_extract.py                       # all 221 files
    !python dandi_extract.py --limit 10            # quick test first
It resumes automatically: files already extracted are skipped, so rerun after a disconnect.

Outputs (send these back):
    dandi_sessions.csv                one row per file: description, frames, duration, ROIs, series found
    extracted/  -> zip it:  !zip -qr extracted.zip extracted   (then download extracted.zip)
"""

import argparse
import csv
import os

import h5py
import numpy as np
import remfile
from dandi.dandiapi import DandiAPIClient

DANDISET, VERSION = "000289", "0.240801.1040"
OUT = "extracted"
MAX_ELEMENTS = 20_000_000      # safety: never read a dataset bigger than this


def text(h, path):
    try:
        v = h[path][()]
        if isinstance(v, np.ndarray):
            v = v.ravel()[0]
        return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:
        return ""


def num(h, path):
    try:
        return float(h[path][()])
    except Exception:
        return float("nan")


def collect_series(h):
    """Every 'data' (+ 'timestamps') under processing/, keyed by its group path."""
    found = {}
    if "processing" not in h:
        return found

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in ("data", "timestamps", "rois"):
            if obj.size <= MAX_ELEMENTS and obj.dtype.kind in "iuf":
                key = name.replace("/", "__")
                found[key] = obj[()]
    h["processing"].visititems(visit)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (for testing)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    assets = list(DandiAPIClient().get_dandiset(DANDISET, VERSION).get_assets())
    assets = sorted([a for a in assets if a.path.endswith(".nwb")], key=lambda a: a.size)
    if args.limit:
        assets = assets[: args.limit]

    rows_path = "dandi_sessions.csv"
    done = set()
    if os.path.exists(rows_path):
        with open(rows_path) as f:
            done = {r["path"] for r in csv.DictReader(f)}
    new_file = not os.path.exists(rows_path)
    fout = open(rows_path, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["path", "subject", "genotype", "description", "imaging_rate", "n_frames",
                    "duration_s", "n_rois", "series", "size_gb", "error"])

    for k, a in enumerate(assets, 1):
        if a.path in done:
            continue
        stem = a.path.replace("/", "__").replace(".nwb", "")
        try:
            url = a.get_content_url(follow_redirects=1, strip_query=True)
            with h5py.File(remfile.File(url), "r") as h:
                desc = text(h, "session_description")
                subj = text(h, "general/subject/subject_id")
                geno = text(h, "general/subject/genotype")
                rate = num(h, "general/optophysiology/imaging_plane/imaging_rate")
                series = collect_series(h)
            np.savez_compressed(os.path.join(OUT, stem + ".npz"), **series,
                                description=desc, subject=subj, genotype=geno, imaging_rate=rate)
            f_key = next((s for s in series if s.endswith("RoiResponseSeries__data")), None)
            t_key = next((s for s in series if s.endswith("RoiResponseSeries__timestamps")), None)
            n_frames = series[f_key].shape[0] if f_key else 0
            n_rois = series[f_key].shape[1] if f_key and series[f_key].ndim == 2 else 0
            dur = float(series[t_key][-1] - series[t_key][0]) if t_key and len(series[t_key]) > 1 else 0.0
            names = ";".join(sorted({s.rsplit("__", 1)[0] for s in series}))
            w.writerow([a.path, subj, geno, desc, rate, n_frames, round(dur, 1), n_rois, names,
                        round(a.size / 1e9, 3), ""])
            print(f"[{k}/{len(assets)}] {a.path}: {n_frames} frames, {dur:.0f}s, {n_rois} ROIs | {desc[:70]}")
        except Exception as e:
            w.writerow([a.path, "", "", "", "", 0, 0, 0, "", round(a.size / 1e9, 3), repr(e)[:200]])
            print(f"[{k}/{len(assets)}] {a.path}: ERROR {e!r}")
        fout.flush()
    fout.close()
    print("done -> dandi_sessions.csv and extracted/")


if __name__ == "__main__":
    main()
