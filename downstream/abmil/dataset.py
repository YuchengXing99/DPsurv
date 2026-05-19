"""
PatchBagDataset: loads pre-extracted patch feature bags for supervised MIL survival.

Supported feature file formats:
    .h5  — HDF5 with dataset key "features" shaped [M, D]
    .pt  — PyTorch tensor file, shape [M, D]

Typical use with ABMIL:
    ds = PatchBagDataset(label_df, feat_dir="/path/to/features", n_bins=4)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_bags, shuffle=True)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


def _compute_qbins(df: pd.DataFrame, n_bins: int) -> np.ndarray:
    uncensored = df[df["dss_censorship"].astype(float) == 0.0]
    src = uncensored if len(uncensored) >= 2 else df
    for q in range(n_bins, 0, -1):
        try:
            _, bins = pd.qcut(
                src["dss_survival_days"].astype(float),
                q=q, retbins=True, labels=False, duplicates="drop",
            )
            bins = np.unique(bins.astype(np.float32))
            if len(bins) >= 2:
                bins[0]  = min(bins[0],  1e-6)
                bins[-1] = max(bins[-1], 1e6)
                return bins
        except ValueError:
            continue
    v = df["dss_survival_days"].astype(float)
    return np.array([min(float(v.min()), 1e-6), max(float(v.max()), 1e6)], dtype=np.float32)


class PatchBagDataset(Dataset):
    """
    Dataset pairing a bag of patch features with a survival label.

    Args:
        label_df:     DataFrame with columns [case_id, dss_survival_days, dss_censorship]
        feat_dir:     directory containing <case_id>.h5 or <case_id>.pt files
        n_bins:       number of discrete survival time bins
        qbins:        pre-computed bin edges (np.ndarray, length n_bins+1);
                      computed from uncensored training data if None
        max_patches:  if > 0, randomly sub-sample bags to this size at __getitem__
                      (training augmentation — set 0 for evaluation)
    """

    def __init__(
        self,
        label_df: pd.DataFrame,
        feat_dir: Path,
        n_bins: int = 4,
        qbins: np.ndarray | None = None,
        max_patches: int = 0,
    ):
        self.feat_dir    = Path(feat_dir)
        self.max_patches = max_patches
        self.df          = label_df.reset_index(drop=True)

        if qbins is None:
            qbins = _compute_qbins(self.df, n_bins)
        self.qbins = np.asarray(qbins, dtype=np.float32)

        raw_labels = pd.cut(
            self.df["dss_survival_days"].astype(float),
            bins=self.qbins, labels=False, include_lowest=True,
        ).fillna(len(self.qbins) - 2).astype(np.int64)

        self.labels = torch.tensor(raw_labels.to_numpy(), dtype=torch.long)
        self.times  = torch.tensor(self.df["dss_survival_days"].to_numpy(), dtype=torch.float32)
        self.cens   = torch.tensor(self.df["dss_censorship"].to_numpy(),    dtype=torch.float32)

    def _load_features(self, case_id: str) -> torch.Tensor:
        for ext in (".h5", ".pt"):
            p = self.feat_dir / f"{case_id}{ext}"
            if p.exists():
                if ext == ".h5":
                    import h5py
                    with h5py.File(p, "r") as f:
                        return torch.from_numpy(f["features"][:]).float()
                else:
                    obj = torch.load(p, map_location="cpu")
                    if isinstance(obj, torch.Tensor):
                        return obj.float()
                    return torch.from_numpy(np.array(obj)).float()
        raise FileNotFoundError(
            f"No feature file found for case '{case_id}' in {self.feat_dir}.\n"
            f"Expected {self.feat_dir}/{case_id}.h5 or .pt"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        case_id = self.df["case_id"].iloc[idx]
        feats   = self._load_features(case_id)

        if self.max_patches > 0 and feats.shape[0] > self.max_patches:
            idx_sel = torch.randperm(feats.shape[0])[:self.max_patches]
            feats   = feats[idx_sel]

        return {
            "features":      feats,
            "survival_time": self.times[idx],
            "censorship":    self.cens[idx],
            "label":         self.labels[idx],
            "case_id":       case_id,
        }


def collate_bags(batch: list) -> dict:
    """
    DataLoader collate for variable-length feature bags.

    Features are returned as a list (bags have different numbers of patches).
    All scalar fields are stacked into tensors.
    """
    return {
        "features":      [b["features"]      for b in batch],
        "survival_time": torch.stack([b["survival_time"] for b in batch]),
        "censorship":    torch.stack([b["censorship"]    for b in batch]),
        "label":         torch.stack([b["label"]         for b in batch]),
        "case_id":       [b["case_id"] for b in batch],
    }
