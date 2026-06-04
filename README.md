# DPsurv

**Dual-Prototype Evidential Fusion for Uncertainty-Aware and Interpretable Whole Slide Image Survival Prediction**, ICML 2026.
<br><em>Yucheng Xing, Ling Huang†, Jingying Ma, Ruping Hong, Jiangdong Qiu, Pei Liu, Kai He, Huazhu Fu, Mengling Feng</em></br>

[Paper](https://proceedings.mlr.press/v306/) | [Cite](#citation)

**Abstract:** Survival prediction from whole slide images (WSIs) is a fundamental task in computational pathology. Existing approaches either discard morphological structure by flattening patch sets, or aggregate prototype representations without accounting for uncertainty in the predictions. We introduce **DPsurv**, a dual-prototype evidential fusion framework that operates on Gaussian Mixture Model (GMM) representations of WSIs. Each morphological prototype is paired with a dedicated evidence neural network expert that outputs heteroscedastic predictions via Generalised Random Fuzzy Numbers (GRFNs), capturing both aleatoric and epistemic uncertainty. Prototype mixture weights directly participate in a mixture-aware discrete survival loss, making training aware of the underlying GMM structure. On five TCGA cancer-type cohorts, DPsurv consistently improves over deterministic baselines in survival discrimination and calibration, while delivering interpretable, prototype-level survival estimates.

<img src="docs/DPsurv_flowchart.png" width="100%" align="center"/>

## Updates
- **05/2026**: DPsurv codebase is now live.

## Installation

With conda (recommended):

```shell
conda env create -f environment.yml
conda activate dpsurv
```

Or with pip into a Python 3.9 environment:

```shell
# install PyTorch first, matching your CUDA driver (see note below)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **PyTorch / CUDA**: the defaults target CUDA 12.1. Replace `cu121` with `cu118` (or `cpu`) in the index URL / torch package if your driver requires a different version.

### Verify the install

Run the end-to-end smoke test. It generates tiny *synthetic* GMM embeddings and
runs one fold of the full nested-CV pipeline in seconds (no real data needed):

```shell
python tests/smoke_test.py            # add --device cuda to test the GPU path
```

It should print `[smoke_test] PASSED` and a `summary.json` with C-index / IBS / NBLL.

## Data and labels

- **Endpoint**: experiments use **disease-specific survival (DSS)** — columns
  `dss_survival_days` and `dss_censorship` (censorship: `1` = censored, `0` = event).
  The split folders keep the historical name `..._overall_survival_k=<fold>`, but
  training reads the `dss_*` columns.
- **Splits**: 5-fold splits for BLCA, BRCA, KIRC, LUAD, UCEC are provided under
  `data/splits/`. Each fold has `train.csv` and `test.csv`.
- **Embeddings** are *not* committed (they are large). Each fold expects a
  pickled file at `data/splits/<DATASET>_overall_survival_k=<fold>/embeddings/<name>.pkl`
  with the layout:
  ```
  {'train': {'prob': [N,K], 'mean': [N,K,D], 'cov': [N,K,D]},
   'test':  {'prob': [N,K], 'mean': [N,K,D], 'cov': [N,K,D]}}
  ```
  where rows are ordered to match the rows of `train.csv` / `test.csv`.

## Running DPsurv

The full pipeline is: **WSIs → patch features → prototypes → PANTHER GMM
embeddings → DPsurv training**. Steps 0–1 below build on
[PANTHER](https://github.com/mahmoodlab/PANTHER); if you already have the
tokenized embeddings, skip straight to Step 2.

### Step 0a (upstream). Patch features

Tile each WSI and extract patch features with a foundation model (we use
**UNI2**, `vit_large`, 20× magnification, 256 px patches → feature dim **1536**),
saving one `<slide_id>.h5` per slide (dataset key `features`).
See [CLAM](https://github.com/mahmoodlab/CLAM) / [Trident](https://github.com/mahmoodlab/TRIDENT).
This is the only step external to this repo.

### Step 0b. Cluster prototypes (per fold)

Cluster the fold's **training** patch features into **K = 16** prototypes:

```shell
python feature_extraction/cluster_prototypes.py \
    --split_dir data/splits/TCGA_KIRC_overall_survival_k=0 \
    --feat_dir  /path/to/tcga_kirc/feats_h5 \
    --in_dim 1536 --n_proto 16 --mode kmeans --n_proto_patches 100000
```

This writes `data/splits/.../prototypes/prototypes_c16_kmeans_num_1.0e+05.pkl`
(key `prototypes`, shape `[1, K, 1536]`). Use `--mode faiss` for GPU K-means
(needs `faiss-gpu`).

### Step 1. Extract PANTHER GMM embeddings (per fold)

```shell
python feature_extraction/extract_gmm.py \
    --split_dir data/splits/TCGA_KIRC_overall_survival_k=0 \
    --feat_dir  /path/to/tcga_kirc/feats_h5 \
    --proto_path data/splits/TCGA_KIRC_overall_survival_k=0/prototypes/prototypes_c16_kmeans_num_1.0e+05.pkl \
    --in_dim 1536 --n_proto 16 --em_iter 1 --tau 1.0 --ot_eps 1.0 \
    --device cuda
```

This writes the tokenized embedding `.pkl` (in the layout above) into
`data/splits/TCGA_KIRC_overall_survival_k=0/embeddings/`. The default output
name matches the trainer's `--embedding_fname`; repeat for each fold (k=0..4).

### Step 2. Train DPsurv

```shell
bash scripts/run_dpsurv.sh KIRC
```

This runs nested cross-validation over all 5 folds (inner K-selection then
full-train retraining) and writes per-fold metrics and a `summary.json` to
`results/<DATASET>/`. See `python trainer/train_dpsurv.py --help` for all options
(e.g. `--embedding_fname` if you used a custom embedding filename).

## Visualization

DPsurv produces interpretable, prototype-level survival estimates. Prototype assignment maps can be overlaid on the original WSI to identify which morphological patterns drive risk.

<img src="docs/DPsurv_Interpretability.png" width="100%" align="center"/>

The accompanying notebook [`visualization/prototypical_assignment_map_visualization_LUAD.ipynb`](visualization/prototypical_assignment_map_visualization_LUAD.ipynb) reproduces these maps for LUAD slides.

## Citation

If you find this work useful in your research or if you use parts of this code please cite our paper:

```bibtex
@article{xing2025dpsurv,
  title={DPsurv: Dual-Prototype Evidential Fusion for Uncertainty-Aware and Interpretable Whole-Slide Image Survival Prediction},
  author={Xing, Yucheng and Huang, Ling and Ma, Jingying and Hong, Ruping and Qiu, Jiangdong and Liu, Pei and He, Kai and Fu, Huazhu and Feng, Mengling},
  journal={arXiv preprint arXiv:2510.00053},
  year={2025}
}
```

## Acknowledgements

This work builds on [PANTHER](https://github.com/mahmoodlab/PANTHER) (Song et al., CVPR 2024) for prototype representation learning. We thank the TCGA consortium for providing public cancer genomics data.

