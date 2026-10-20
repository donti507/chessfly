"""
dandi_trials.py -- Stage 1, step 3: read the block table (intervals/trials) of the
contrast experiment in DANDI 000289. cue_contrast: 1 = darkness, 2 = low, 3 = high.

Colab (CPU, about 1 minute; needs dandi_sessions.csv in the same folder):
    !pip install -q dandi remfile h5py pandas
    !python dandi_trials.py
Output: dandi_trials.csv  (path, block, start_time, stop_time, cue_contrast)
"""

import csv

import h5py
import pandas as pd
import remfile
from dandi.dandiapi import DandiAPIClient

DANDISET, VERSION = "000289", "0.240801.1040"

sessions = pd.read_csv("dandi_sessions.csv")
wanted = set(sessions.loc[sessions.description.str.contains("varying contrast", na=False), "path"])
assets = [a for a in DandiAPIClient().get_dandiset(DANDISET, VERSION).get_assets() if a.path in wanted]

with open("dandi_trials.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "block", "start_time", "stop_time", "cue_contrast"])
    for a in assets:
        url = a.get_content_url(follow_redirects=1, strip_query=True)
        with h5py.File(remfile.File(url), "r") as h:
            if "intervals/trials" not in h:
                print(a.path, "-> no trials table")
                continue
            g = h["intervals/trials"]
            cols = {k: g[k][()] for k in ("start_time", "stop_time", "cue_contrast") if k in g}
            n = len(cols["start_time"])
            for i in range(n):
                w.writerow([a.path, i, cols["start_time"][i], cols["stop_time"][i],
                            cols.get("cue_contrast", [None] * n)[i]])
            print(a.path, "->", n, "blocks, contrast order", list(cols.get("cue_contrast", [])))
print("done -> dandi_trials.csv")
