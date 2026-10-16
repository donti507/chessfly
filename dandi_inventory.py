"""
dandi_inventory.py -- Stage 1, step 1: inventory of DANDI 000289 (Basnak et al. 2025)

Lists every file in the dandiset and prints the internal structure (group/dataset names,
shapes, dtypes, attributes) of a few NWB files by STREAMING them. It never loads large
arrays and never downloads the raw imaging movies (TwoPhotonSeries).

Run on Google Colab (CPU is enough):
    !pip install -q dandi remfile h5py pynwb
    !python dandi_inventory.py                 # lists assets + inspects 3 files
    !python dandi_inventory.py --n-inspect 5   # inspect more files

Outputs (send these back):
    dandi_assets.csv          path, size_gb for every file
    dandi_structure.txt       tree of the inspected files
"""

import argparse
import csv

import h5py
import remfile
from dandi.dandiapi import DandiAPIClient

DANDISET, VERSION = "000289", "0.240801.1040"
SKIP_DATA = ("TwoPhotonSeries",)            # never touch raw movies


def describe(name, obj, out, max_attr=6):
    depth = name.count("/")
    pad = "  " * depth
    if isinstance(obj, h5py.Dataset):
        out.append(f"{pad}- {name.split('/')[-1]}  shape={obj.shape} dtype={obj.dtype}")
        # print tiny datasets in full (labels, flags, rates)
        if obj.size and obj.size <= 8 and obj.dtype.kind in "iufSOb":
            try:
                out.append(f"{pad}    value={obj[()]!r}")
            except Exception:
                pass
    else:
        out.append(f"{pad}+ {name.split('/')[-1]}/")
    for k in list(obj.attrs.keys())[:max_attr]:
        try:
            v = obj.attrs[k]
            s = repr(v)
            out.append(f"{pad}    @{k} = {s[:120]}")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inspect", type=int, default=3)
    args = ap.parse_args()

    client = DandiAPIClient()
    ds = client.get_dandiset(DANDISET, VERSION)
    assets = list(ds.get_assets())
    with open("dandi_assets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "size_gb"])
        for a in assets:
            w.writerow([a.path, round(a.size / 1e9, 3)])
    print(f"{len(assets)} assets; total {sum(a.size for a in assets) / 1e9:.1f} GB -> dandi_assets.csv")

    # inspect the smallest NWB files first (cheapest to stream)
    nwb = sorted([a for a in assets if a.path.endswith(".nwb")], key=lambda a: a.size)
    out = []
    for a in nwb[: args.n_inspect]:
        url = a.get_content_url(follow_redirects=1, strip_query=True)
        out.append("=" * 90)
        out.append(f"FILE {a.path}  ({a.size / 1e9:.2f} GB)")
        out.append("=" * 90)
        with h5py.File(remfile.File(url), "r") as h:
            def visit(name, obj):
                if any(s in name for s in SKIP_DATA) and isinstance(obj, h5py.Dataset) and name.endswith("/data"):
                    out.append(f"{'  ' * name.count('/')}- data  shape={obj.shape}  (raw movie, skipped)")
                    return
                describe(name, obj, out)
            h.visititems(visit)
        print(f"inspected {a.path}")
    with open("dandi_structure.txt", "w") as f:
        f.write("\n".join(out))
    print("structure -> dandi_structure.txt")


if __name__ == "__main__":
    main()
