# Aerial Landscape Classification (UNSW COMP9517, 25T1)

Group project for UNSW COMP9517 (Computer Vision), Term 1 2025, revised into a
conference paper in October 2026: 15-way scene classification on the SkyView aerial
landscape dataset, comparing hand-crafted feature pipelines, fine-tuned CNNs and voting
ensembles under balanced, long-tailed and rebalanced training sets.

Author: Nigel Li (UNSW School of Computer Science and Engineering).

**Paper:** [`paper/aerial_scene_classification.pdf`](paper/aerial_scene_classification.pdf)
(IEEEtran conference format, 7 pages; LaTeX source in `paper/`).

## Main results

Macro-F1 (%) on the shared 2,400-image balanced test set (single run, seed 0):

| Model | Balanced | Long-tailed (16:1) | Rebalanced (augmentation) |
|---|---|---|---|
| LBP + kNN | 53.3 | 44.2 | 45.6 |
| Colour histogram + SVM | 69.8 | 56.9 | 62.0 |
| SIFT-BoW + SVM | 62.9 | 53.5 | 54.5 |
| Stacking (SVM, kNN, DT) | 84.4 | 73.1 | 75.4 |
| ResNet-18 | 96.2 | 93.8 (95.0 with weighted CE) | 92.1 |
| EfficientNet-B3 | 96.5 | 92.5 (93.4 with weighted CE) | 91.5 |
| ResNet-18 + EfficientNet-B3, soft vote | **98.0** | **95.3** | **94.1** |

- CNNs lose 2.4 to 4.0 points under the long tail and hand-crafted pipelines lose 9 to 13.
- Offline augmentation of minority classes helps the hand-crafted pipelines but lowers both
  CNNs; inverse-frequency loss weighting raises both CNNs.
- Soft voting of the two CNNs beats the better network in every setting (McNemar
  p < 1e-4); adding hand-crafted members does not help.

## What changed from the April 2025 submission

The original 2025 evaluation had protocol problems that invalidated several reported
numbers:

1. **Voting tie-break bug.** Ensembles used `Counter(votes).most_common(1)`, which breaks
   ties by member order. With two members every disagreement is a tie, so every
   two-member "ensemble" returned its first member's predictions exactly. This produced
   the original conclusion that ensembles "collapse to their weakest member".
2. **Long-tail experiment never ran.** The long-tail evaluation notebook contained the
   same outputs as the balanced one, down to the Optuna timestamps.
3. **Test-set leakage.** Optuna searches and early stopping used the test set.
4. **Mixed splits.** Tables combined numbers from different train/test splits (e.g. the
   colour-histogram SVM was reported at 33.6 % accuracy; its own notebook gives 69.3 %).

`code/experiments/` re-implements the study with a single fixed protocol, and the paper
was rewritten around the new results.

## Layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `sections/`, `refs.bib`), generated `tables/` and `figs/`, compiled PDF |
| `code/experiments/` | Re-run pipeline (see below) |
| `runs/` | Splits, per-model validation/test probabilities, metadata, logs and `results.json` (model weights are not committed) |
| `data/` | Dataset location (not committed, see `data/README.md`) |

## Reproducing

```bash
pip install torch torchvision scikit-learn scikit-image opencv-python-headless optuna matplotlib scipy
zsh code/experiments/run_all.sh
```

`run_all.sh` builds the splits (`common.py`), runs the classical pipelines
(`classical.py`, CPU, about 30 min) and the CNNs (`cnn.py`, about 5 h on an Apple M4
GPU via MPS) in parallel, then writes the paper's tables and figures (`analyze.py`) and
the Grad-CAM figure (`gradcam.py`).

Build the paper with any TeX distribution, e.g. `cd paper && latexmk -pdf main.tex`, or
`tectonic main.tex`.
