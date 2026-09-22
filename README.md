# Fly Connectome Chess

A chess player built on the **real wiring of a fruit fly brain**.

The *Drosophila* head-direction circuit (434 neurons, 33,783 connections) from the [male CNS connectome](https://neuprint.janelia.org) (`male-cns:v1.0`) is simulated as a fixed **spiking** neural network and used as a **reservoir computer**. Chess positions go in as input currents; only a small linear readout is trained to choose among legal moves. The connectome weights are never trained.

> The fly brain does not understand chess. This is reservoir computing: the connectome provides fixed nonlinear dynamics, and a thin readout maps them to moves.

📘 **Full technical documentation: [`docs/TECHNICAL.md`](docs/TECHNICAL.md)**

## Status (Phase 1, CPU)

| Goal | Status |
|---|---|
| Only legal moves | ✅ 0 illegal moves (guaranteed by design) |
| Reservoir carries chess information | ✅ 100% decoding of 8 test positions, signal/noise 11.6× |
| Fly bot beats a random player | ⚠️ Not yet significant (score 0.55 over 20 games) |
| Fly brain adds information beyond its input | ⚠️ Not shown — ties the input-only control |

**Next (Phase 2, Kaggle GPU):** batched PyTorch simulator, 50k–100k positions, wider input, time-binned readout, shuffled-connectome controls. See the roadmap in the docs.

## Key results

Move prediction (imitating Stockfish depth 8), 5,000 positions, split by game:

| Representation | Val top-1 | Test top-1 |
|---|---|---|
| random legal move | — | 4.7% |
| encoder input only (48 values, no brain) | 16.3% | 12.0% |
| fly reservoir (386 neurons) | 13.5% | 11.9% |
| fly reservoir + input | 16.1% | 13.7% |
| raw board (768 features, no brain) | 19.4% | 18.5% |

Games vs random opponent (Day 5, 20 games): raw 0.93, encoder 0.85, **fly 0.55**, random 0.42.

## Quick start

```bash
pip install -r requirements.txt
sudo apt install stockfish
export NEUPRINT_TOKEN="<your neuPrint token>"

python3 pull_subnetwork_v2.py
python3 day4_build_dataset.py
python3 day6_build_vs_random.py
python3 day6_train_readout.py
python3 day6_play_games.py
```

## Related work

This project builds on and should be read alongside: Yu et al., *Biological Processing Units* (AGI 2025; larval fly connectome incl. chess), Costi et al., *The Drosophila Connectome as a Computational Reservoir* (Biomimetics 2025), and Suárez et al., *conn2res* (Nature Communications 2024).

## Data & credits

Connectome data: Janelia FlyEM male CNS connectome (`male-cns:v1.0`) via [neuPrint](https://neuprint.janelia.org). Data files are not included; scripts regenerate them. Please cite the dataset according to neuPrint's guidelines.

Built with [Brian2](https://brian2.readthedocs.io), [python-chess](https://python-chess.readthedocs.io), [neuprint-python](https://github.com/connectome-neuprint/neuprint-python), and [Stockfish](https://stockfishchess.org).
