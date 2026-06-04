"""
PANTHER GMM embedding extraction (per split).

Converts per-slide patch feature files (.h5 / .pt) into PANTHER GMM embeddings
(mixture weights pi, component means mu, diagonal covariances Sigma) and saves
them as a single .pkl per fold, in exactly the layout consumed by
``trainer/train_dpsurv.py`` and ``trainer/train_mil.py``:

    {
      'train': {'prob': [N_train, K], 'mean': [N_train, K, D], 'cov': [N_train, K, D]},
      'test':  {'prob': [N_test,  K], 'mean': [N_test,  K, D], 'cov': [N_test,  K, D]},
    }

Rows are ordered to match the rows of the corresponding split CSV (train.csv /
test.csv), so the downstream positional join in ``downstream/dpsurv/data.build_df``
lines up embeddings with survival labels.

This mirrors the original PANTHER embedding step (`training/main_embedding.py`)
used for the paper, with the paper hyper-parameters as defaults
(K=16, em_iter=1, tau=1.0, ot_eps=1.0, out_type='allcat', UNI2 in_dim=1536).

Usage:
    python feature_extraction/extract_gmm.py \
        --split_dir data/splits/TCGA_KIRC_overall_survival_k=0 \
        --feat_dir  /path/to/tcga_kirc/.../feats_h5 \
        --proto_path /path/to/prototypes_c16_..._kmeans_num_1.0e+05.pkl \
        --in_dim 1536 --n_proto 16 --em_iter 1 --tau 1.0 --ot_eps 1.0 \
        --device cuda

The output filename defaults to the same name expected by the trainers
(``--embedding_fname``); override with --out_name if needed.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Make repo-root packages importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction.panther import PANTHERBase
from feature_extraction.tokenizer import PrototypeTokenizer


# Default output filename — must match trainer/train_dpsurv.py:DEFAULT_EMBEDDING_FNAME
DEFAULT_OUT_NAME = (
    "extracted-vit_large_patch16_224.dinov2.uni_mass100k_"
    "PANTHER_embeddings_proto_16_allcat_em_1_eps_1.0_tau_1.0_tokenized.pkl"
)

SPLIT_FILES = {"train": "train.csv", "test": "test.csv"}


# ---------------------------------------------------------------------------
# Feature IO
# ---------------------------------------------------------------------------

def _index_feature_files(feat_dirs: List[Path]) -> Dict[str, Path]:
    """Map slide_id (file stem) -> feature file path across one or more dirs."""
    index: Dict[str, Path] = {}
    for fdir in feat_dirs:
        for pattern in ("*.h5", "*.pt"):
            for fpath in fdir.glob(pattern):
                index.setdefault(fpath.stem, fpath)
    if not index:
        raise FileNotFoundError(f"No .h5/.pt feature files found in: {feat_dirs}")
    return index


def _load_features(path: Path) -> torch.Tensor:
    """Load [M, D] patch features from an .h5 ('features') or .pt file."""
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            feats = f["features"][:]
        feats = np.asarray(feats, dtype=np.float32)
    else:
        feats = torch.load(path, weights_only=False)
        feats = feats.numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats)
        feats = feats.astype(np.float32)
    if feats.ndim == 3:
        assert feats.shape[0] == 1, f"Expected (1, M, D) or (M, D), got {feats.shape}"
        feats = np.squeeze(feats, axis=0)
    return torch.from_numpy(feats)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def _build_encoder(
    in_dim: int,
    n_proto: int,
    em_iter: int,
    tau: float,
    ot_eps: float,
    out_type: str,
    proto_path: str,
    device: torch.device,
) -> PANTHERBase:
    model = PANTHERBase(
        d=in_dim,
        p=n_proto,
        L=em_iter,
        tau=tau,
        out=out_type,
        ot_eps=ot_eps,
        load_proto=True,
        proto_path=proto_path,
        fix_proto=True,
    )
    model.eval()
    model.to(device)
    return model


@torch.inference_mode()
def _embed_slide(model: PANTHERBase, tokenizer: PrototypeTokenizer,
                 feats: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """Run PANTHER EM on one slide and tokenize into (prob, mean, cov)."""
    h = feats.unsqueeze(0).to(device)        # [1, M, D]
    flat_repr, _ = model(h)                  # [1, K + 2*K*D]
    prob, mean, cov = tokenizer(flat_repr)   # [1, K], [1, K, D], [1, K, D]
    return {
        "prob": prob.squeeze(0).cpu().numpy(),
        "mean": mean.squeeze(0).cpu().numpy(),
        "cov":  cov.squeeze(0).cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Per-split extraction
# ---------------------------------------------------------------------------

def _extract_split(
    csv_path: Path,
    file_index: Dict[str, Path],
    model: PANTHERBase,
    tokenizer: PrototypeTokenizer,
    slide_col: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    if slide_col not in df.columns:
        raise KeyError(f"Column '{slide_col}' not in {csv_path} (columns: {list(df.columns)[:6]}...)")

    probs, means, covs = [], [], []
    missing = []
    # Iterate in CSV row order so the saved arrays align row-for-row with the CSV.
    for slide_id in tqdm(df[slide_col].astype(str).tolist(), desc=f"  {csv_path.name}", leave=False):
        fpath = file_index.get(slide_id)
        if fpath is None:
            missing.append(slide_id)
            continue
        emb = _embed_slide(model, tokenizer, _load_features(fpath), device)
        probs.append(emb["prob"])
        means.append(emb["mean"])
        covs.append(emb["cov"])

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} slide(s) in {csv_path.name} have no feature file, "
            f"e.g. {missing[:5]}. Embedding rows would no longer align with the CSV."
        )

    return {
        "prob": np.stack(probs, axis=0),   # [N, K]
        "mean": np.stack(means, axis=0),   # [N, K, D]
        "cov":  np.stack(covs,  axis=0),   # [N, K, D]
    }


def extract_fold(
    split_dir: Path,
    feat_dirs: List[Path],
    proto_path: str,
    out_name: str,
    in_dim: int,
    n_proto: int,
    em_iter: int,
    tau: float,
    ot_eps: float,
    out_type: str,
    slide_col: str,
    device_str: str,
) -> Path:
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[extract_gmm] device={device}  K={n_proto}  em_iter={em_iter}  tau={tau}  ot_eps={ot_eps}")

    model = _build_encoder(in_dim, n_proto, em_iter, tau, ot_eps, out_type, proto_path, device)
    tokenizer = PrototypeTokenizer(proto_model_type="PANTHER", out_type=out_type, p=n_proto)
    file_index = _index_feature_files(feat_dirs)
    print(f"[extract_gmm] indexed {len(file_index)} feature files")

    embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for split, fname in SPLIT_FILES.items():
        csv_path = split_dir / fname
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        print(f"[extract_gmm] extracting '{split}' from {csv_path.name}")
        embeddings[split] = _extract_split(csv_path, file_index, model, tokenizer, slide_col, device)

    out_path = split_dir / "embeddings" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f, protocol=4)

    for split, d in embeddings.items():
        print(f"[extract_gmm]   {split}: prob{d['prob'].shape} mean{d['mean'].shape} cov{d['cov'].shape}")
    print(f"[extract_gmm] saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract PANTHER GMM embeddings (per fold) for DPsurv.")
    p.add_argument("--split_dir", type=Path, required=True,
                   help="Fold directory containing train.csv and test.csv "
                        "(e.g. data/splits/TCGA_KIRC_overall_survival_k=0).")
    p.add_argument("--feat_dir", type=Path, nargs="+", required=True,
                   help="One or more directories of per-slide .h5/.pt patch features "
                        "(filenames are <slide_id>.h5).")
    p.add_argument("--proto_path", type=str, required=True,
                   help="Path to PANTHER prototype .pkl (key 'prototypes') or .npy.")
    p.add_argument("--out_name", type=str, default=DEFAULT_OUT_NAME,
                   help="Output filename inside <split_dir>/embeddings/ "
                        "(default matches the trainers' --embedding_fname).")
    p.add_argument("--in_dim", type=int, default=1536, help="Patch feature dim (UNI2: 1536).")
    p.add_argument("--n_proto", type=int, default=16, help="Number of GMM prototypes K.")
    p.add_argument("--em_iter", type=int, default=1, help="PANTHER EM iterations (paper: 1).")
    p.add_argument("--tau", type=float, default=1.0, help="PANTHER tau (paper: 1.0).")
    p.add_argument("--ot_eps", type=float, default=1.0, help="PANTHER initial covariance eps (paper: 1.0).")
    p.add_argument("--out_type", type=str, default="allcat", help="PANTHER output mode.")
    p.add_argument("--slide_col", type=str, default="slide_id", help="Slide-id column in the split CSVs.")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_fold(
        split_dir=args.split_dir,
        feat_dirs=[Path(d) for d in args.feat_dir],
        proto_path=args.proto_path,
        out_name=args.out_name,
        in_dim=args.in_dim,
        n_proto=args.n_proto,
        em_iter=args.em_iter,
        tau=args.tau,
        ot_eps=args.ot_eps,
        out_type=args.out_type,
        slide_col=args.slide_col,
        device_str=args.device,
    )
