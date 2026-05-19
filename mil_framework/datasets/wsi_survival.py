"""
WSISurvivalDataset — loads raw patch feature bags for the ABMIL (non-GMM) pathway.

Supports .h5 and .pt feature files. Discretizes survival times into bins for NLL loss.
Compatible with downstream/abmil/model.py (PatchBagDataset is an alternative
that wraps per-slide bags; this version handles multi-slide cases and bin computation).

Adapted from src/wsi_datasets/wsi_survival.py.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Survival-time discretization
# ---------------------------------------------------------------------------

def compute_discretization(
    df: pd.DataFrame,
    survival_time_col: str = 'dss_survival_days',
    censorship_col: str = 'dss_censorship',
    n_bins: int = 4,
    bins: np.ndarray = None,
) -> tuple:
    """
    Discretize continuous survival times into n_bins quantile bins.

    Bins are computed from uncensored patients only (standard practice).
    If `bins` is provided, it is used directly (for test/val splits).

    Returns:
        disc_labels: pd.Series of bin indices
        bins: np.ndarray of bin edges (length n_bins+1)
    """
    if bins is not None:
        disc_labels, bins = pd.cut(
            df[survival_time_col], bins=bins, retbins=True,
            labels=False, include_lowest=True,
        )
        disc_labels.name = 'disc_label'
        return disc_labels, bins

    uncensored = df[df[censorship_col].astype(float) == 0.0]
    src = uncensored if len(uncensored) >= 2 else df

    for q in range(n_bins, 0, -1):
        try:
            _, edges = pd.qcut(
                src[survival_time_col].astype(float),
                q=q, retbins=True, labels=False, duplicates='drop',
            )
            edges = np.unique(edges.astype(np.float32))
            if len(edges) >= 2:
                break
        except Exception:
            continue

    edges[0]  = -1e-6
    edges[-1] = 1e6

    disc_labels, edges = pd.cut(
        df[survival_time_col].astype(float), bins=edges,
        retbins=True, labels=False, include_lowest=True,
    )
    disc_labels = disc_labels.fillna(0).astype(int)
    disc_labels.name = 'disc_label'
    return disc_labels, edges


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WSISurvivalDataset(Dataset):
    """
    Dataset for raw-patch-feature-based survival models (ABMIL pathway).

    Each sample loads the full patch bag for one patient slide from an .h5 or .pt file.

    Args:
        df:               DataFrame with columns [case_id, slide_id, survival_time_col, censorship_col]
        feat_dir:         Directory containing feature files (*.h5 or *.pt)
        survival_time_col: Column name for survival time
        censorship_col:   Column name for censorship flag (0=event, 1=censored)
        n_bins:           Number of discrete survival bins
        bins:             Pre-computed bin edges (use for val/test with train bins)
        use_h5:           True for .h5 files, False for .pt files
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_dir: str,
        survival_time_col: str = 'dss_survival_days',
        censorship_col: str = 'dss_censorship',
        n_bins: int = 4,
        bins: np.ndarray = None,
        use_h5: bool = True,
    ):
        self.feat_dir = Path(feat_dir)
        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.use_h5 = use_h5

        df = df.dropna(subset=[survival_time_col, censorship_col]).copy()
        df = df[df[censorship_col].isin([0, 1])]
        df = df[df[survival_time_col] >= 0]

        disc_labels, self.bins = compute_discretization(
            df, survival_time_col, censorship_col, n_bins, bins
        )
        df = df.copy()
        df['disc_label'] = disc_labels.values

        self.df = df.reset_index(drop=True)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        slide_id = row.get('slide_id', row.get('case_id'))

        if self.use_h5:
            feat_path = self.feat_dir / f"{slide_id}.h5"
            with h5py.File(feat_path, 'r') as f:
                features = torch.from_numpy(f['features'][:].astype(np.float32))
        else:
            feat_path = self.feat_dir / f"{slide_id}.pt"
            features = torch.load(feat_path, weights_only=False)
            if features.ndim == 3:
                features = features.squeeze(0)

        return {
            'features':      features,                                    # [M, D]
            'label':         torch.tensor(int(row['disc_label'])),        # scalar
            'event_time':    torch.tensor(float(row[self.survival_time_col])),
            'censorship':    torch.tensor(int(row[self.censorship_col])),
            'case_id':       str(row.get('case_id', slide_id)),
        }
