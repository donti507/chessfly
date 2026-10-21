# ChessFly

**Reconstruction differences rival individual differences in the fly compass connectome and break trained circuit models**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Preregistration](https://img.shields.io/badge/Preregistration-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.23382486)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Md Ali Ashraf Yad<sup>1</sup>, Nazifa Tabassum Nabila<sup>2</sup>

<sup>1</sup> Wrocław University of Science and Technology, Poland · <sup>2</sup> Comilla University, Bangladesh

📄 Paper: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) · 📝 Preregistration: [doi:10.5281/zenodo.23382486](https://doi.org/10.5281/zenodo.23382486)

<p align="center"><img src="new_study/figures/fig2_differences_one_scale.png" width="900" alt="Measurement and individual differences on one scale"></p>

## Summary

Neural network models are increasingly built from synapse-resolution connectomes, and the same circuit now exists in reconstructions from several animals and pipelines. We studied the *Drosophila* head-direction (compass) circuit in **nine versions**: four flies (male CNS, hemibrain, FlyWire, BANC) and five variants of the same fly that differ only in synapse filtering, proofreading or synapse detection.

- **Measurement rivals individuality.** Changing the synapse detector within one fly altered the circuit by more than half of the smallest difference between flies, on four distance measures. Proofreading changes were smaller and depended on the measure.
- **Trained models are knife-edge ring attractors.** Networks trained on each wiring to hold a heading formed near-perfect continuous attractors, but a 1% change of all weights broke them.
- **Models do not transfer between connectomes, even of the same fly.** Refitting about thirty cell-type gains repaired only filtering-level differences; the remaining error tracked wiring distance, whatever its source.
- **Robustness has a price.** Training with noise traded precision for robustness and gave only partial transfer.
- **Real fly compasses degrade gradually.** In imaging data (DANDI 000289) the fly compass drifts gradually in darkness, and it works despite wiring differences of the size we measured.

## Figures

| | |
| --- | --- |
| [Fig 1](new_study/figures/fig1_design.png) | Circuit, nine wiring versions, experiments |
| [Fig 2](new_study/figures/fig2_differences_one_scale.png) | All differences on one scale |
| [Fig 3](new_study/figures/fig3_ring_and_fragility.png) | Ring attractors and knife-edge fragility |
| [Fig 4](new_study/figures/fig4_transplant_tradeoff.png) | Transplants and the precision–robustness trade-off |
| [Fig 5](new_study/figures/fig5_transfer_by_source.png) | Transfer by source of difference; cell-type refit |
| [Fig 6](new_study/figures/fig6_fly_drift.png) | Compass drift in real flies |

## Repository layout

| Folder | Contents |
| --- | --- |
| [`new_study/`](new_study/) | Code, results, figures and networks for the paper (start here) |
| [`v1_chessfly/`](v1_chessfly/) | The earlier project: spiking fly reservoir, chess, null-wiring experiments (Phases 1–8), demos |

## Quick start

```bash
git clone https://github.com/donti507/chessfly.git
cd chessfly/new_study
pip install numpy scipy pandas matplotlib torch ripser
unzip -oq networks/hd_networks_all.zip -d .     # nine wiring versions -> networks/<version>/
PYTHONPATH=code python code/distance_decomposition.py   # reproduces the distance table (CPU, minutes)
```

The full pipeline, including the GPU training scripts (free Colab or Kaggle T4, minutes to about an hour each) and the figure script, is described step by step in [`new_study/README.md`](new_study/README.md).

## Data

Janelia FlyEM male CNS and hemibrain connectomes (neuPrint), FlyWire FAFB and BANC (FlyWire Codex), and E-PG imaging data from Basnak et al. 2025 (DANDI 000289). Please cite the original datasets.

## Citation

```bibtex
@article{yad2026reconstruction,
  title   = {Reconstruction differences rival individual differences in the fly compass connectome and break trained circuit models},
  author  = {Yad, Md Ali Ashraf and Nabila, Nazifa Tabassum},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Earlier work

The project began as **ChessFly**: can the fly compass circuit, used as a neural network, learn chess? It learned worse than randomly rewired networks, and that result led to the question studied here. The full earlier project, with interactive demos ([3D mission](https://claude.ai/artifact/5cobwrSTvQWzB88KUtVBZe), [play against the fly](https://claude.ai/artifact/8gRB5DUGNV5WrG3kmYfL8s)), is in [`v1_chessfly/`](v1_chessfly/).

## License

Code is released under the [MIT License](LICENSE). Connectome and imaging data remain under the licenses of their original providers.
