# Fly Connectome Chess — Technical Documentation

**Status as of Phase 1 (local, CPU):** pipeline complete end-to-end, controls in place, fly reservoir carries chess information but has not yet been shown to add information beyond its own input.
**Next phase:** GPU simulator on Kaggle, larger dataset, redesigned input/readout, research controls.

---

## Table of contents

1. [Goal and framing](#1-goal-and-framing)
2. [Success criteria and current status](#2-success-criteria-and-current-status)
3. [Data source](#3-data-source)
4. [System architecture](#4-system-architecture)
5. [Locked configuration reference](#5-locked-configuration-reference)
6. [Development log and findings](#6-development-log-and-findings)
7. [Results](#7-results)
8. [Interpretation](#8-interpretation)
9. [Related work and novelty](#9-related-work-and-novelty)
10. [Roadmap: Phase 2 (Kaggle GPU)](#10-roadmap-phase-2-kaggle-gpu)
11. [Reproduction](#11-reproduction)
12. [File inventory](#12-file-inventory)
13. [Known gotchas](#13-known-gotchas)

---

## 1. Goal and framing

Use the real synaptic wiring of the *Drosophila* male CNS connectome as a **fixed spiking reservoir**, and train only a thin readout on top of it to play chess.

This is **reservoir computing** (liquid state machine style), not biological chess intelligence. The fly brain does not understand chess. The connectome provides fixed, recurrent, nonlinear dynamics; a trained linear readout maps those dynamics to moves. **Connectome weights are never trained** — that is the premise of the project.

Research question (sharpened during Phase 1):

> Does real connectome wiring transform chess information into a more useful representation, or does it only pass its input through?

---

## 2. Success criteria and current status

| Criterion (from original plan) | Status |
|---|---|
| Only legal moves | ✅ **Met.** Guaranteed by construction; 0 illegal moves in all games |
| Measurably better than random play | ⚠️ **Not yet for the fly bot.** Fly scored 0.55 over 20 games (not significant). Controls score 0.85–0.93 |
| Bonus: beat a greedy material bot | ⬜ Not yet tested (`OPPONENT=greedy` supported) |
| Research: fly reservoir beats its own input (encoder control) | ⚠️ **Not shown.** Ties within noise on Day 4 and Day 6 |

Out of scope for v1: beating Stockfish or humans, full-CNS scale, biological realism beyond LIF neurons.

---

## 3. Data source

- **Dataset:** `male-cns:v1.0` on neuPrint (https://neuprint.janelia.org), queried live via `neuprint-python`.
- **Auth:** personal neuPrint token, read from the `NEUPRINT_TOKEN` environment variable. Never commit tokens.
- **Neurotransmitter labels:** `celltypePredictedNt` (falls back to `predictedNt`). Other available NT-related columns include `consensusNt`, `predictedNtConfidence`, `celltypePredictedNtConfidence`.
- **Data files are not committed**; scripts regenerate them.

### Current subnetwork: fly head-direction circuit

Pulled by `pull_subnetwork_v2.py` with type regex `(EPG.*|PEG.*|PEN.*|Delta7.*|ER[1-6].*)`.

| Property | Value |
|---|---|
| Neurons | 434 |
| Connections (≥ 3 synapses, summed over ROIs) | 33,783 |
| Inhibitory neurons (GABA / glutamate) | 324 (74.7%) |
| Inhibitory share of synaptic weight | 77.7% |
| Mean in-degree | 77.8 |
| Neurons with no inputs | 0 |

Cell types: EPG, PEG, PEN_a (PEN1), PEN_b (PEN2) — cholinergic/excitatory; Δ7 (glutamatergic) and EB ring neurons ER1–ER6 (GABAergic) — inhibitory. Per-neuron types and transmitters are in `subnetwork_nt.csv`.

The earlier v1 subnetwork (68 neurons, 1,629 connections; EPG/PEG/FC/FB only, 100% cholinergic) is backed up as `cx68_subnetwork_*`.

---

## 4. System architecture

```
chess board
   │  canonicalize (mirror if Black to move → always "White to move")
   ▼
768 one-hot features (12 piece planes × 64 squares)
   │  subtract mean board (200 random positions), random Gaussian projection,
   │  per-neuron gain normalization, sigmoid
   ▼
48 activations in [0,1]  ──►  constant current I_ext = activation × 30 mV into 48 input neurons
                                   │
                                   ▼
                  434-neuron LIF network, connectome wiring, 1 s simulation (Brian2)
                                   │
                                   ▼
                  firing rates of 386 hidden (non-input) neurons
                                   │  standardize, linear readout
                                   ▼
             from-square logits (64) + to-square logits (64)
                                   │  score each LEGAL move = from[f] + to[t]
                                   ▼
                     softmax over legal moves → chosen move
```

### 4.1 Neuron model (leaky integrate-and-fire)

```
dv/dt = (v_rest − v + I_ext) / τ + σ·sqrt(2/τ)·ξ(t)     (not integrated while refractory)
spike when v > −50 mV  →  reset v = −65 mV, refractory 3 ms
```

- `v_rest = −65 mV`, `τ = 10 ms`, `σ = 1 mV` membrane noise, Euler integration at Brian2's default `dt` (0.1 ms).
- After synaptic updates each step, `v` is clipped to [−80, 0] mV (via `run_regularly`, keeps strong inhibition bounded and stays vectorized).

### 4.2 Synapses

- Instantaneous voltage jump on each presynaptic spike: `v_post += w`.
- `w = synapse_count × 0.22 mV × sign(pre)`, where `sign = −1` for GABA/glutamate presynaptic neurons, `+1` otherwise.

### 4.3 Input encoding

1. **Canonical board:** if Black is to move, `board.mirror()` is used, so the network always sees the side to move as White. Moves are mirrored back after selection.
2. **Features:** 768-dim one-hot piece-square vector.
3. **Centering:** subtract the mean feature vector of 200 random canonical positions (random play, 0–60 plies, seed 123). This removes the large shared component (most pieces are the same across positions) so between-position differences drive the input.
4. **Projection:** Gaussian random matrix (48 × 768, seed 0).
5. **Gain normalization:** each input's pre-activation is scaled so its std over the reference positions is 1.5; then sigmoid → [0, 1].
6. **Drive:** steady current `I_ext = activation × 30 mV` to 48 input neurons (deterministic — no Poisson spikes).

**Input neuron selection (`diverse_types`):** highest out-degree neuron from each distinct cell type first, then topped up by global out-degree.

### 4.4 Readout

- Representation standardized (train-set mean/std).
- Two linear maps: `F = xW_f + b_f` (64 from-square logits), `T = xW_t + b_t` (64 to-square logits).
- Move score `= F[from] + T[to]`, softmax **over legal moves only** (underpromotions dropped; they share from/to with the queen promotion).
- Loss: cross-entropy vs. Stockfish's move + L2. Optimizer: full-batch Adam, lr 0.01, 400 epochs. L2 chosen from {1e-4, 1e-3, 1e-2, 1e-1} on validation top-1.
- Split **by game** (70/15/15, seed 0) to avoid leakage between near-identical positions.

**Known limitation:** the additive from+to score cannot express piece-and-destination-specific preferences (e.g. "this knight to that square"). This affects all representations equally, so comparisons stay fair, but it caps absolute accuracy.

### 4.5 Controls (same readout, same data)

| Name | Features | Purpose |
|---|---|---|
| `raw_board_768` | 768 one-hot | Upper reference, no fly brain |
| `encoder_input_48` | 48 encoder activations | **Key control**: exactly what enters the fly brain |
| `fly_reservoir` | 386 hidden firing rates | Fly brain only |
| `fly_plus_input` | 48 + 386 concatenated | Hybrid; beats encoder only if the brain adds information |
| `random` | — | Chance baseline |

---

## 5. Locked configuration reference

Defined in `reservoir_core.py` (winner of the Day 3.7 calibration):

| Parameter | Value |
|---|---|
| Network | 434-neuron head-direction circuit, NT-signed |
| Input neurons | 48 (`diverse_types`) |
| Weight scale | 0.22 mV per synapse |
| Max input drive `I_max` | 30 mV |
| Simulation duration | 1000 ms |
| Membrane noise σ | 1 mV |
| Encoder gain | 1.5 |
| Projection seed / reference seed | 0 / 123 |
| Reference positions | 200 |
| Code generation target | Brian2 `numpy` |
| Throughput | ~0.75–2.0 s per position (single CPU core) |

---

## 6. Development log and findings

### Day 1 — Pull subnetwork
`pull_subnetwork.py`: 68 central-complex neurons, 1,629 connections. Only 4 of the 9 requested cell types were returned (exact name matching missed variants like `PEN_a`, `PFNa`).

### Day 2 — LIF simulator
`build_lif_simulator.py`: Brian2 sanity simulation confirming spikes propagate through connectome wiring.

### Day 3 — First chess reservoir
`day3_chess_encoder.py`, `day3_chess_reservoir.py`: 6 input neurons driven by Poisson spikes (5–200 Hz). Only ~6/68 neurons fired; mean pairwise cosine similarity of states ~0.855.

### Day 3.5 — First calibration sweep
`day3_5_calibration.py` (weight scale × input weight × input strategy).
**Finding: a sharp phase transition.** Below scale ≈ 0.08 mV activity stayed near the inputs; at ≥ 0.20 mV the network ignited (64/68 active) and all positions produced identical states (similarity 0.99–1.00). The script's automatic pick (0.20 mV) was wrong.
**Root cause:** every synapse was excitatory → the network could only be silent or runaway.

### Neurotransmitter check
`fetch_nt.py`: all 68 v1 neurons were cholinergic (67 ACh, 1 unclear). Biologically correct for EPG/PEG — the circuit's inhibition comes from Δ7 and ring neurons, which v1 didn't include.

### Subnetwork v2
`pull_subnetwork_v2.py`: regex type matching, added PEN, Δ7, ER1–6 → 434 neurons, 74.7% inhibitory.

### Day 3.6 — Calibration with inhibition
`day3_6_calibration.py`: added proper metrics — separation ratio (between-position / within-position-across-seeds distance), leave-one-seed-out nearest-centroid decoding, hidden neurons only.
**Finding:** separation ratio < 1 and decoding at chance everywhere. Input rates were already ~0.986 similar across positions, and Poisson noise (~18% at 100 Hz / 300 ms) swamped the few-percent differences **before the reservoir saw them**.

### Day 3.7 — Calibration v3 (input redesign)
`day3_7_calibration.py`: deterministic current injection, centered + gain-normalized encoding, 1 mV membrane noise, 8 test positions (chance 12.5%).
- 300 ms runs: best decode 0.71 (sep ratio 5.1).
- After widening the grid, 1000 ms runs, dropping "unsigned" and "inh2x": **decode 1.000, separation ratio 11.6** (scale 0.22, 48 inputs, I_max 30). 19/20 configs in the useful regime.
- Engineering fix: `clip()` inside `on_pre` could not be vectorized (Python-loop fallback); moved to a per-step `run_regularly` on neurons.

### Day 4 — Dataset + readout
`day4_build_dataset.py`: 3,000 positions from 42 Stockfish self-play games (depth 8, 25% random moves), each run through the fly reservoir. `day4_train_readout.py`: readout + controls.

### Day 5 — Games
`day5_play_games.py`: 20 games per bot vs a random-move opponent, 150-ply cap, material adjudication (±3).
**Finding:** fly bot played much worse than its move-prediction accuracy suggested — likely distribution shift (trained on engine-vs-engine positions, played chaotic positions vs random) plus per-move noise.

### Day 6 — Distribution fix + hybrid
`day6_build_vs_random.py`: +2,000 positions from "hero (Stockfish, 25% random) vs random opponent" games (dataset now 5,000 positions / 136 games). `day6_train_readout.py`: adds `fly_plus_input`. `day6_play_games.py`: 40 games/bot, 95% margins, optional run averaging (`FLY_REPEATS`).

---

## 7. Results

### 7.1 Calibration: position decoding (8 positions, chance 12.5%)

| Stage | Decode accuracy | Separation ratio |
|---|---|---|
| Day 3.6 (Poisson input, inhibition) | ~chance (best 33% of 4 positions, chance 25%) | < 1 |
| Day 3.7, 300 ms | 71% | 5.1 |
| **Day 3.7, 1000 ms (locked)** | **100%** | **11.6** |

### 7.2 Day 4: move prediction (3,000 positions; test 524; avg 30.4 legal moves)

| Representation | Test top-1 | Test top-3 |
|---|---|---|
| random legal move | 7.7% | 17.8% |
| encoder_input_48 | 13.0% | 28.4% |
| fly_reservoir | 15.5% | 28.2% |
| raw_board_768 | 16.4% | 33.2% |

Validation top-1 (for comparison): encoder 13.6%, fly 11.4% — the fly/encoder ordering flips between val and test.

### 7.3 Day 5: games vs random opponent (20 games each)

| Bot | W | D | L | Mates W/L | Avg material | Score | Illegal |
|---|---|---|---|---|---|---|---|
| random | 6 | 5 | 9 | 0/2 | −2.5 | 0.42 | 0 |
| raw | 17 | 3 | 0 | 3/0 | +15.6 | 0.93 | 0 |
| encoder | 16 | 2 | 2 | 0/0 | +6.9 | 0.85 | 0 |
| **fly** | 9 | 4 | 7 | 2/0 | +2.6 | **0.55** | 0 |

### 7.4 Day 6: move prediction (5,000 positions; test 607)

| Representation | Val top-1 | Test top-1 | Test top-3 |
|---|---|---|---|
| random legal move | — | 4.7% | 11.5% |
| encoder_input_48 | 16.3% | 12.0% | 26.7% |
| fly_reservoir | 13.5% | 11.9% | 26.0% |
| **fly_plus_input** | 16.1% | 13.7% | 28.2% |
| raw_board_768 | 19.4% | 18.5% | 36.2% |

### 7.5 Day 6: games (40 games each)

_Pending — fill in from `day6_play_games.py`._

---

## 8. Interpretation

1. **The reservoir faithfully carries board information** (100% position decoding, separation 11.6×).
2. **It does not yet add information.** `fly_reservoir` ≈ `encoder_input_48` (11.9% vs 12.0% test), and the hybrid is within noise of the encoder on both splits. Consistent across Day 4 and Day 6 → a real finding, not a fluke.
3. **The encoder is the main bottleneck.** `raw_board_768` beats everything; compressing 768 → 48 loses information the reservoir cannot recover.
4. **Move-prediction accuracy does not translate into play for the fly bot**, unlike the linear controls. Suspected causes: distribution shift and per-move simulation noise (single 1 s run).
5. **Methodological lessons:** an all-excitatory connectome subgraph cannot act as a reservoir; stochastic (Poisson) input can erase small between-input differences before the reservoir sees them; always compare against the reservoir's own input, not just against chance.

---

## 9. Related work and novelty

| Work | Relation |
|---|---|
| Yu et al., *Biological Processing Units* (AGI 2025, LNCS 16058; arXiv 2507.10951) | *Drosophila* **larval** connectome as a fixed recurrent network; MNIST, CIFAR-10 and **ChessBench** chess. Closest prior work |
| Costi et al., *The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction* (Biomimetics 10(5):341, 2025) | Adult fly connectome as an echo state network reservoir |
| Suárez et al., *Connectome-based reservoir computing with the conn2res toolbox* (Nature Communications 15, 2024) | General connectome reservoir framework |

**"A fly brain plays chess" is not novel.** Distinguishing features of this project: adult male CNS dataset, **spiking** LIF dynamics, a specific named circuit (head-direction ring attractor), NT-signed synapses, and an explicit **encoder-input control**. The most defensible contribution is the question in §1, answered with proper controls (see §10.3).

Candidate venues: arXiv + NeurIPS/ICLR NeuroAI workshops, ALIFE (current results); *Biomimetics*, *Frontiers in Computational Neuroscience*, IJCNN (with shuffle controls); *PLoS Computational Biology*, *Neural Computation* (with a strong general finding).

---

## 10. Roadmap: Phase 2 (Kaggle GPU)

### 10.1 Scale: GPU simulator
- Reimplement the locked LIF network in **PyTorch**, simulating a **batch of positions in parallel** on a Kaggle GPU (free tier, ~30 GPU h/week).
- **Acceptance test:** on a fixed set of positions, per-neuron firing rates must closely match Brian2 (report correlation and mean absolute difference) before any results are trusted.
- Target: 50k–100k+ positions.
- Labels: Stockfish on Kaggle CPU, or the public Lichess evaluation database (database.lichess.org) for value targets.

### 10.2 Design improvements (ranked)
1. **Wider input:** 150+ input neurons, or a topographic mapping of board files/ranks onto the ring's columns.
2. **Time-binned readout:** e.g. 10 × 100 ms bins per neuron instead of one total count.
3. **Mushroom body circuit** (Kenyon cells: sparse high-dimensional expansion) as an alternative reservoir.
4. **Value readout:** predict position evaluation; play the legal move whose resulting position scores best.
5. **Better move readout:** low-rank bilinear from×to scoring.
6. **Noise:** average repeated runs per move (`FLY_REPEATS`).

### 10.3 Research controls (needed for publication)
- **Shuffled connectome:** same neurons/weights, randomized wiring.
- **Degree-preserving shuffle:** keeps each neuron's in/out degree.
- **Circuit comparison:** head-direction vs mushroom body vs visual circuit.
- **Spiking vs rate-based** version of the same connectome.
- More games (≥ 100 per bot) with confidence intervals.

### 10.4 Phase 2 targets
| Target | Measure |
|---|---|
| Fly adds information | `fly_plus_input` > `encoder_input_48` on val **and** test, with non-overlapping CIs |
| Fly beats random | score − 95% margin > 0.50 |
| Realistic accuracy | ~20–30% top-1 with frozen reservoir + linear readout |
| Visualization (Day 7) | 3D neurons at real `somaLocation`, spikes lighting up during play; web page and/or video |

---

## 11. Reproduction

```bash
pip install -r requirements.txt
sudo apt install stockfish
export NEUPRINT_TOKEN="<your neuPrint token>"

python3 pull_subnetwork_v2.py        # network + NT signs            (~1 min)
python3 day3_7_calibration.py        # optional calibration sweep    (~10 min)
python3 day4_build_dataset.py        # 3,000 positions               (~40–60 min)
python3 day6_build_vs_random.py      # +2,000 vs-random positions    (~25–70 min)
python3 day6_train_readout.py        # readouts + controls           (few min)
python3 day6_play_games.py           # 40 games/bot                  (~80 min)
```

Useful environment variables: `N_POSITIONS`, `N_TOTAL`, `N_GAMES`, `BOTS`, `OPPONENT` (`random`/`greedy`), `FLY_REPEATS`, `TYPE_REGEX`, `NT_COLUMN`.
Dataset builders are resumable (Ctrl+C saves progress).

---

## 12. File inventory

All scripts live in the repo root because they import each other by module name.

| File | Stage | Purpose |
|---|---|---|
| `pull_subnetwork.py` | Day 1 | v1 subnetwork (68 neurons) — legacy |
| `check_columns.py` | Day 1 | Diagnostic for neuPrint return order — legacy |
| `build_lif_simulator.py` | Day 2 | Brian2 LIF sanity simulation — legacy |
| `day3_chess_encoder.py`, `day3_chess_reservoir.py` | Day 3 | First chess encoding/reservoir — legacy |
| `day3_5_calibration.py` | Day 3.5 | First sweep (found phase transition) |
| `fetch_nt.py` | Day 3.6 | NT lookup for an existing subnetwork |
| `pull_subnetwork_v2.py` | Day 3.6 | **Current network** + NT signs |
| `day3_6_calibration.py` | Day 3.6 | Sweep with separation/decoding metrics |
| `day3_7_calibration.py` | Day 3.7 | Sweep with deterministic input (locked config) |
| `reservoir_core.py` | Day 4+ | **Shared locked reservoir** (encoder, network, simulation, spike recording) |
| `day4_build_dataset.py` | Day 4 | Positions + Stockfish labels + reservoir states |
| `day4_train_readout.py` | Day 4 | Readout + controls |
| `day5_play_games.py` | Day 5 | Game evaluation (random/greedy opponents) |
| `day6_build_vs_random.py` | Day 6 | Distribution-matched positions |
| `day6_train_readout.py` | Day 6 | Retrain + hybrid readout |
| `day6_play_games.py` | Day 6 | 40-game evaluation with CIs |

Generated (not committed): `subnetwork_adjacency.npz`, `subnetwork_neurons.csv`, `subnetwork_bodyids.npy`, `subnetwork_nt.csv`, `cx68_*`, `day4_dataset.npz`, `readout_*.npz`, `day3_*_calibration_results.csv`.

---

## 13. Known gotchas

- `fetch_adjacencies()` returns `(neuron_df, conn_df)` — connections are the **second** value. `fetch_neurons()` returns `(neuron_df, roi_counts_df)`.
- neuPrint type matching is exact unless `regex=True` (subtypes like `PEN_a(PEN1)` are otherwise missed).
- `conn_df` has one row per (pre, post, ROI) — sum weights across ROIs.
- Brian2 `numpy` target cannot vectorize `clip()` inside synaptic `on_pre` → very slow Python loop. Clip on neurons with `run_regularly` instead.
- Pass `namespace={}` to `net.run()` to avoid local variables (e.g. `rates`, `N`) clashing with Brian2 internals.
- Don't paste multi-line Python into bash; write files first. On this Ubuntu, `gedit` isn't installed — use `gnome-text-editor`.
- `sed` substitutions need all three parts: `s/find/replace/`.
- **Never commit the neuPrint token**; use `NEUPRINT_TOKEN`. Rotate it if exposed.
