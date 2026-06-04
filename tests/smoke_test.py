"""
End-to-end smoke test for DPsurv.

Generates tiny *synthetic* PANTHER GMM embeddings (in the exact on-disk layout
produced by feature_extraction/extract_gmm.py and consumed by the trainers),
then runs one fold of the full nested-CV training + evaluation pipeline.

It uses small dimensions so it finishes in seconds on CPU and needs no real
WSI data — its purpose is to verify that the environment and the training/
evaluation code path work after installation, NOT to reproduce paper numbers.

Usage:
    python tests/smoke_test.py            # CPU
    python tests/smoke_test.py --device cuda
"""

import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

N_PROTO = 6      # GMM components (one ENNreg expert per component)
DIM     = 32     # GMM mean/cov dimension (tiny; real UNI2 is 1536)
DATASET = "SYN"  # resolve_dataset_name leaves unknown names unchanged


def _make_split(n: int, rng: np.random.Generator) -> dict:
    prob = rng.random((n, N_PROTO)).astype(np.float32)
    prob /= prob.sum(axis=1, keepdims=True)
    return {
        "prob": prob,
        "mean": rng.standard_normal((n, N_PROTO, DIM)).astype(np.float32),
        "cov":  (rng.random((n, N_PROTO, DIM)).astype(np.float32) + 0.1),
    }


def _make_csv(n: int, rng: np.random.Generator, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({
        "case_id":           [f"{prefix}-{i:04d}" for i in range(n)],
        "slide_id":          [f"{prefix}-{i:04d}-DX1" for i in range(n)],
        "dss_survival_days": rng.integers(30, 3000, size=n).astype(float),
        # ~60% events (censorship 0 = event, 1 = censored)
        "dss_censorship":    (rng.random(n) > 0.6).astype(int),
    })


def build_synthetic_fold(splits_root: Path, embedding_fname: str,
                         n_train: int = 60, n_test: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fold_dir = splits_root / f"{DATASET}_overall_survival_k=0"
    (fold_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    _make_csv(n_train, rng, "TR").to_csv(fold_dir / "train.csv", index=False)
    _make_csv(n_test,  rng, "TE").to_csv(fold_dir / "test.csv",  index=False)

    embeddings = {"train": _make_split(n_train, rng), "test": _make_split(n_test, rng)}
    with (fold_dir / "embeddings" / embedding_fname).open("wb") as f:
        pickle.dump(embeddings, f, protocol=4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    embedding_fname = "synthetic_tokenized.pkl"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        splits_root = tmp / "splits"
        build_synthetic_fold(splits_root, embedding_fname)

        cmd = [
            sys.executable, str(REPO_ROOT / "trainer" / "train_dpsurv.py"),
            "--datasets", DATASET,
            "--folds", "0",
            "--k_values", "1", "2",
            "--splits_root", str(splits_root),
            "--results_dir", str(tmp / "results"),
            "--embedding_fname", embedding_fname,
            "--max_epochs", "2", "--min_epochs", "1", "--patience", "1",
            "--batch_size", "8", "--eval_batch_size", "32",
            "--n_label_bins", "4", "--kmeans_nstart", "2",
            "--device", args.device,
        ]
        print("Running:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))

        summary = tmp / "results" / DATASET / "summary.json"
        if proc.returncode == 0 and summary.exists():
            print("\n[smoke_test] PASSED — full nested-CV pipeline ran end to end.")
            print(summary.read_text())
            return 0
        print("\n[smoke_test] FAILED — see traceback above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
