"""
Prototype construction via K-means clustering of patch features.

Step 0 of the DPsurv pipeline: cluster the *training* patch features of a fold
into K prototypes. The resulting `.pkl` (key 'prototypes', shape [1, K, D]) is
then passed to `feature_extraction/extract_gmm.py` as --proto_path.

This mirrors the original PANTHER prototype step (`training/main_prototype.py`
+ `utils/proto_utils.cluster`): it samples up to `n_proto * n_proto_patches`
patch features (evenly across the fold's training slides) and runs K-means.

Usage:
    python feature_extraction/cluster_prototypes.py \
        --split_dir data/splits/TCGA_KIRC_overall_survival_k=0 \
        --feat_dir  /path/to/tcga_kirc/feats_h5 \
        --in_dim 1536 --n_proto 16 --mode kmeans --n_proto_patches 100000

By default the prototypes are written to
`<split_dir>/prototypes/prototypes_c<K>_<mode>_num_<n_proto_patches>.pkl`.

`--mode faiss` uses GPU FAISS K-means (requires the `faiss-gpu` package);
`--mode kmeans` (default) uses scikit-learn and needs no extra dependency.
"""

import argparse
import pickle
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

# Make repo-root packages importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction.extract_gmm import _index_feature_files, _load_features


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sample_patches(
    slide_ids: List[str],
    file_index: dict,
    feature_dim: int,
    n_total: int,
) -> torch.Tensor:
    """Sample an even number of patches from each training slide (up to n_total)."""
    n_slides = max(1, len(slide_ids))
    per_slide = (n_total + n_slides - 1) // n_slides
    print(f"Sampling up to {n_total} patches: ~{per_slide} from each of {n_slides} slides")

    patches = torch.empty(n_total, feature_dim, dtype=torch.float32)
    n = 0
    for slide_id in tqdm(slide_ids, desc="Sampling patches"):
        if n >= n_total:
            break
        fpath = file_index.get(slide_id)
        if fpath is None:
            raise FileNotFoundError(f"No feature file for slide '{slide_id}'")
        feats = _load_features(fpath).reshape(-1, feature_dim).numpy()
        np.random.shuffle(feats)
        take = min(per_slide, feats.shape[0], n_total - n)
        patches[n:n + take] = torch.from_numpy(feats[:take])
        n += take
    print(f"Aggregated {n} patches total")
    return patches[:n]


def cluster_prototypes(
    split_dir: Path,
    feat_dirs: List[Path],
    in_dim: int,
    n_proto: int,
    mode: str,
    n_proto_patches: int,
    n_iter: int,
    n_init: int,
    seed: int,
    out_name: str,
) -> Path:
    _seed_everything(seed)

    train_csv = split_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}")
    import pandas as pd
    slide_ids = pd.read_csv(train_csv)["slide_id"].astype(str).tolist()

    file_index = _index_feature_files(feat_dirs)
    patches = _sample_patches(slide_ids, file_index, in_dim, n_proto * n_proto_patches)

    t0 = time.time()
    if mode == "kmeans":
        print(f"Running scikit-learn KMeans (K={n_proto}, max_iter={n_iter})...")
        km = KMeans(n_clusters=n_proto, max_iter=n_iter, n_init=n_init, random_state=seed)
        km.fit(patches.numpy())
        weight = km.cluster_centers_[np.newaxis, ...]            # [1, K, D]
    elif mode == "faiss":
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("--mode faiss requires the faiss-gpu package.") from exc
        n_gpus = torch.cuda.device_count()
        print(f"Running FAISS KMeans on {n_gpus} GPU(s) (K={n_proto}, niter={n_iter})...")
        km = faiss.Kmeans(in_dim, n_proto, niter=n_iter, nredo=n_init, verbose=True,
                          max_points_per_centroid=n_proto_patches, gpu=n_gpus)
        km.train(patches.numpy())
        weight = km.centroids[np.newaxis, ...]
    else:
        raise NotImplementedError(f"Unknown clustering mode: {mode}")
    print(f"Clustering took {time.time() - t0:.1f}s")

    out_path = split_dir / "prototypes" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"prototypes": weight.astype(np.float32)}, f, protocol=4)
    print(f"Saved prototypes {weight.shape} -> {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster training patch features into DPsurv prototypes.")
    p.add_argument("--split_dir", type=Path, required=True,
                   help="Fold directory containing train.csv.")
    p.add_argument("--feat_dir", type=Path, nargs="+", required=True,
                   help="One or more directories of per-slide .h5/.pt patch features.")
    p.add_argument("--in_dim", type=int, default=1536, help="Patch feature dim (UNI2: 1536).")
    p.add_argument("--n_proto", type=int, default=16, help="Number of prototypes K.")
    p.add_argument("--mode", type=str, default="kmeans", choices=["kmeans", "faiss"])
    p.add_argument("--n_proto_patches", type=int, default=100000,
                   help="Patches sampled per prototype (total = n_proto * this).")
    p.add_argument("--n_iter", type=int, default=50, help="K-means iterations.")
    p.add_argument("--n_init", type=int, default=5, help="K-means re-initialisations.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_name", type=str, default=None,
                   help="Output filename inside <split_dir>/prototypes/ "
                        "(default: prototypes_c<K>_<mode>_num_<n_proto_patches>.pkl).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_name = args.out_name or (
        f"prototypes_c{args.n_proto}_{args.mode}_num_{args.n_proto_patches:.1e}.pkl"
    )
    cluster_prototypes(
        split_dir=args.split_dir,
        feat_dirs=[Path(d) for d in args.feat_dir],
        in_dim=args.in_dim,
        n_proto=args.n_proto,
        mode=args.mode,
        n_proto_patches=args.n_proto_patches,
        n_iter=args.n_iter,
        n_init=args.n_init,
        seed=args.seed,
        out_name=out_name,
    )
