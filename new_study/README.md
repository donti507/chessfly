# new_study: code, results and figures for the paper

Preregistration: [doi:10.5281/zenodo.23382486](https://doi.org/10.5281/zenodo.23382486). Exploratory analyses and deviations are listed in the paper (Methods, Supplement S6).

## Pipeline

| Step | Scripts (`code/`) | Runs on | Output |
| --- | --- | --- | --- |
| 1. Fly imaging data | `dandi_inventory.py`, `dandi_extract.py`, `dandi_trials.py`, `drift_lib.py`, `drift_final.py` | Colab CPU, then local | `results/drift_*.csv`, `results/fly_data/` |
| 2. Connectomes | `build_subnetworks.py` (Codex downloads), `pull_hemibrain.py` (neuPrint), `build_variants.py` (same-fly variants) | local / Colab | `networks/hd_networks_all.zip` |
| 3. Positions, maps, distances | `positions.py`, `transplant_maps.py`, `wiring_maps.py`, `distance_decomposition.py`, `alt_metrics.py` | local CPU | `results/difference_decomposition.csv`, `results/alt_metrics_robustness.csv`, `results/validation_*.csv` |
| 4. Pilot | `pilot_stage3.py` (GPU), `mechanism.py` (CPU) | Colab T4 | `results/pilot_*.csv` |
| 5. Transplants | `stage4_transplant.py`, `stage4b_robust.py`, `stage4c_noise_sweep.py`, `stage4d_final.py` | Colab / Kaggle T4 | `results/stage4*_results.csv`, `results/stage4d_transplants.csv` |
| 6. Analyses | `sim_np.py`, `gain_rescue.py`, `h4_all.py` | local CPU | `results/h4_stage4.csv`, `results/gain_rescue.csv`, `results/stats_table.csv` |
| 7. Figures | `make_final_figures.py` | local CPU | `figures/fig1-6` |

## Networks

`networks/hd_networks_all.zip` holds all nine wiring versions (`networks/<version>/subnetwork_adjacency.npz`, `subnetwork_neurons.csv`, `subnetwork_nt.csv`) plus the cross-fly neuron maps (`transplant_maps.npz`):

| Version | Source |
| --- | --- |
| `mcns` | male CNS v1.0, FlyWire Codex (filtered) |
| `mcns_neuprint` | male CNS v1.0, neuPrint |
| `mcns_v09` | male CNS v0.9, Codex |
| `hemibrain` | hemibrain v1.2.1, neuPrint |
| `fafb` | FlyWire FAFB v783, Princeton synapses, Codex filtered |
| `fafb_unfiltered` | FAFB v783, Codex unfiltered file |
| `fafb_buhmann` | FAFB v783, Buhmann synapses |
| `banc` | BANC v888, Codex |
| `banc_v626` | BANC v626, Codex |

All versions: cell types EPG, PEG, PEN, Delta7, ER1-6; at least 3 synapses per neuron pair after summing over neuropils; signs by cell type.

## Running GPU scripts

```bash
unzip -oq networks/hd_networks_all.zip      # creates networks/<version>/
python code/stage4d_final.py --smoke        # 1-2 minute check (prints "device cuda")
python code/stage4d_final.py                # full run
```

Trained-parameter files (`*.npz`) are not committed; rerun the scripts to regenerate them.
