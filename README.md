# DPsurv

**Dual-Prototype Evidential Fusion for Uncertainty-Aware and Interpretable Whole Slide Image Survival Prediction**, ICML 2026.
<br><em>Yucheng Xing\*, Ling Huang†, Jingying Ma, Ruping Hong, Jiangdong Qiu, Pei Liu, Kai He, Huazhu Fu, Mengling Feng</em></br>

[Paper](https://proceedings.mlr.press/v306/) | [Cite](#citation)

**Abstract:** Survival prediction from whole slide images (WSIs) is a fundamental task in computational pathology. Existing approaches either discard morphological structure by flattening patch sets, or aggregate prototype representations without accounting for uncertainty in the predictions. We introduce **DPsurv**, a dual-prototype evidential fusion framework that operates on Gaussian Mixture Model (GMM) representations of WSIs. Each morphological prototype is paired with a dedicated evidence neural network expert that outputs heteroscedastic predictions via Generalised Random Fuzzy Numbers (GRFNs), capturing both aleatoric and epistemic uncertainty. Prototype mixture weights directly participate in a mixture-aware discrete survival loss, making training aware of the underlying GMM structure. On five TCGA cancer-type cohorts, DPsurv consistently improves over deterministic baselines in survival discrimination and calibration, while delivering interpretable, prototype-level survival estimates.

<img src="docs/DPsurv_flowchart.png" width="100%" align="center"/>

## Updates
- **05/2026**: DPsurv codebase is now live.

## Installation

```shell
conda env create -f environment.yml
conda activate dpsurv
```

> **PyTorch / CUDA**: the default `environment.yml` targets CUDA 12.1. Replace `cu121` with `cu118` (or `cpu`) in both the index URL and the torch package name if your driver requires a different version.

## Running DPsurv

### Step 1. Extract PANTHER GMM embeddings

```shell
python feature_extraction/extract_gmm.py \
    --feat_dir /path/to/feats_h5 \
    --out_path data/splits/TCGA_KIRC_overall_survival_k=0/embeddings/panther.pkl \
    --proto_path /path/to/kirc_prototypes.pkl \
    --in_dim 1536 --n_proto 16 --device cuda
```

### Step 2. Train DPsurv

```shell
bash scripts/run_dpsurv.sh KIRC
```

## Visualization

DPsurv produces interpretable, prototype-level survival estimates. Prototype assignment maps can be overlaid on the original WSI to identify which morphological patterns drive risk.

<img src="docs/DPsurv_Interpretability.png" width="100%" align="center"/>

The accompanying notebook [`visualization/prototypical_assignment_map_visualization_LUAD.ipynb`](visualization/prototypical_assignment_map_visualization_LUAD.ipynb) reproduces these maps for LUAD slides.

## Citation

If you find this work useful in your research or if you use parts of this code please cite our paper:

```bibtex
@inproceedings{xing2026dpsurv,
  title     = {DPsurv: Dual-Prototype Evidential Fusion for Uncertainty-Aware and Interpretable Whole Slide Image Survival Prediction},
  author    = {Xing, Yucheng and Huang, Ling and Ma, Jingying and Hong, Ruping and Qiu, Jiangdong and Liu, Pei and He, Kai and Fu, Huazhu and Feng, Mengling},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR},
}
```

## Acknowledgements

This work builds on [PANTHER](https://github.com/mahmoodlab/PANTHER) (Song et al., CVPR 2024) for prototype representation learning. We thank the TCGA consortium for providing public cancer genomics data.

