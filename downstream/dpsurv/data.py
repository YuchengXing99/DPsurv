"""
DPsurv dataset utilities for GMM-based WSI embeddings.

GMMEmbeddingDataset — PyTorch Dataset loading PANTHER GMM embeddings + survival labels.
build_df            — merge embedding dict with label CSV into a single DataFrame.
collate_flat        — DataLoader collate function that stacks per-sample dicts.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_df(embed_data: dict, label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge PANTHER embedding dict with a survival label DataFrame.

    embed_data keys: 'prob'  [N, C]     mixture weights
                     'mean'  [N, C, D]  component means
                     'cov'   [N, C, D]  component (diagonal) covariances
    label_df columns: case_id, dss_survival_days, dss_censorship
    """
    return pd.DataFrame({
        'prob':               list(embed_data['prob']),
        'mean':               list(embed_data['mean']),
        'cov':                list(embed_data['cov']),
        'case_id':            label_df['case_id'].values,
        'dss_survival_days':  label_df['dss_survival_days'].values,
        'dss_censorship':     label_df['dss_censorship'].values,
    })


def collate_flat(batch: list) -> dict:
    """Stack a list of sample dicts into batched tensors."""
    prob  = torch.stack([b['prob'] for b in batch], dim=0)
    mean  = torch.stack([b['mean'] for b in batch], dim=0)
    cov   = torch.stack([b['cov']  for b in batch], dim=0)
    time  = torch.stack([b['survival_time'] for b in batch], dim=0).squeeze()
    cens  = torch.stack([b['censorship']    for b in batch], dim=0).squeeze()

    out = {'prob': prob, 'mean': mean, 'cov': cov, 'survival_time': time, 'censorship': cens}

    if 'labels' in batch[0]:
        out['labels'] = torch.stack([b['labels'] for b in batch], dim=0)

    return out


class GMMEmbeddingDataset(Dataset):
    """
    Dataset wrapping PANTHER GMM embeddings and survival labels.

    Each sample exposes:
        prob  [C]     mixture weights
        mean  [C, D]  component means
        cov   [C, D]  component (diagonal) covariances
        survival_time scalar
        censorship    scalar  (1 = censored, 0 = event)
    """

    def __init__(self, df: pd.DataFrame):
        self.df   = df.reset_index(drop=True)
        prob = np.stack(df['prob'].to_numpy())   # [N, C]
        mean = np.stack(df['mean'].to_numpy())   # [N, C, D]
        cov  = np.stack(df['cov'].to_numpy())    # [N, C, D]

        self.prob   = torch.tensor(prob, dtype=torch.float32)
        self.mean   = torch.tensor(mean, dtype=torch.float32)
        self.cov    = torch.tensor(cov,  dtype=torch.float32)
        self.y_time = torch.tensor(df['dss_survival_days'].to_numpy(), dtype=torch.float32)
        self.y_cens = torch.tensor(df['dss_censorship'].to_numpy(),    dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        out = {
            'prob':          self.prob[idx],
            'mean':          self.mean[idx],
            'cov':           self.cov[idx],
            'survival_time': self.y_time[idx],
            'censorship':    self.y_cens[idx],
        }
        if hasattr(self, 'labels'):
            out['labels'] = self.labels[idx]
        return out
