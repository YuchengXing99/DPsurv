"""
GMM Feature Extraction via PANTHER.

Converts per-slide patch feature files (.h5) into PANTHER GMM embeddings
(mixture weights π, component means μ, diagonal covariances Σ) and saves
them as a single .pkl file compatible with GMMEmbeddingDataset.

Usage:
    python feature_extraction/extract_gmm.py \
        --feat_dir /path/to/h5_features \
        --out_path  data/splits/TCGA_KIRC_.../embeddings/panther_embeddings.pkl \
        --proto_path data/prototypes/kirc_prototypes.pkl \
        --in_dim 1536 --n_proto 16 --n_iters 3 --device cuda
"""

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from feature_extraction.panther import PANTHERBase
from feature_extraction.tokenizer import PrototypeTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_h5_features(path: Path) -> torch.Tensor:
    with h5py.File(path, 'r') as f:
        feats = f['features'][:]
    return torch.from_numpy(feats.astype(np.float32))


def _build_encoder(
    in_dim: int,
    n_proto: int,
    n_iters: int,
    proto_path: str,
    device: torch.device,
) -> PANTHERBase:
    model = PANTHERBase(
        d=in_dim,
        p=n_proto,
        L=n_iters,
        out='allcat',
        load_proto=True,
        proto_path=proto_path,
        fix_proto=True,
    )
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Per-slide extraction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_slide(
    model: PANTHERBase,
    tokenizer: PrototypeTokenizer,
    feats: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Run PANTHER EM on one slide's patch features.

    PANTHERBase.forward returns (flat_repr, qqs) where flat_repr is the
    allcat concatenation [π, μ, Σ].  The tokenizer splits it back into
    the three named components.

    Args:
        feats: [M, D] float tensor of patch features
    Returns:
        dict with keys 'prob' [C], 'mean' [C, D], 'cov' [C, D]
    """
    h = feats.unsqueeze(0).to(device)       # [1, M, D]
    flat_repr, _ = model(h)                 # [1, K + 2*K*D]

    prob, mean, cov = tokenizer(flat_repr)  # [1,K], [1,K,D], [1,K,D]

    return {
        'prob': prob.squeeze(0).cpu().numpy(),   # [C]
        'mean': mean.squeeze(0).cpu().numpy(),   # [C, D]
        'cov':  cov.squeeze(0).cpu().numpy(),    # [C, D]
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def extract_dataset(
    feat_dir: Path,
    out_path: Path,
    proto_path: str,
    in_dim: int = 1536,
    n_proto: int = 16,
    n_iters: int = 3,
    device_str: str = 'cuda',
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"[extract_gmm] Using device: {device}")

    model = _build_encoder(in_dim, n_proto, n_iters, proto_path, device)
    tokenizer = PrototypeTokenizer(proto_model_type='PANTHER', out_type='allcat', p=n_proto)

    h5_files = sorted(feat_dir.glob('*.h5'))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {feat_dir}")
    print(f"[extract_gmm] Found {len(h5_files)} slides.")

    probs, means, covs, slide_ids = [], [], [], []
    failed = []

    for h5_path in tqdm(h5_files, desc='Extracting GMM features'):
        slide_id = h5_path.stem
        try:
            feats = _load_h5_features(h5_path)
            emb   = extract_slide(model, tokenizer, feats, device)
            probs.append(emb['prob'])
            means.append(emb['mean'])
            covs.append(emb['cov'])
            slide_ids.append(slide_id)
        except Exception as exc:
            print(f"  [WARN] {slide_id}: {exc}")
            failed.append(slide_id)

    result = {
        'prob':     np.stack(probs,  axis=0),   # [N, C]
        'mean':     np.stack(means,  axis=0),   # [N, C, D]
        'cov':      np.stack(covs,   axis=0),   # [N, C, D]
        'slide_ids': slide_ids,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(result, f, protocol=4)

    print(f"[extract_gmm] Saved {len(slide_ids)} slides → {out_path}")
    if failed:
        print(f"[extract_gmm] Failed slides ({len(failed)}): {failed}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract PANTHER GMM embeddings from patch .h5 features.")
    p.add_argument('--feat_dir',    type=Path, required=True,  help='Directory of .h5 patch feature files')
    p.add_argument('--out_path',    type=Path, required=True,  help='Output .pkl file path')
    p.add_argument('--proto_path',  type=str,  required=True,  help='Path to prototype .pkl or .npy file')
    p.add_argument('--in_dim',      type=int,  default=1536,   help='Patch feature dimension (default: 1536 for UNI2)')
    p.add_argument('--n_proto',     type=int,  default=16,     help='Number of GMM prototypes K (default: 16)')
    p.add_argument('--n_iters',     type=int,  default=3,      help='EM iterations (default: 3)')
    p.add_argument('--device',      type=str,  default='cuda', help='Device: cuda or cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    extract_dataset(
        feat_dir=args.feat_dir,
        out_path=args.out_path,
        proto_path=args.proto_path,
        in_dim=args.in_dim,
        n_proto=args.n_proto,
        n_iters=args.n_iters,
        device_str=args.device,
    )
