"""
GMMSurvivalDataset — thin wrapper over downstream.dpsurv.data.GMMEmbeddingDataset.

Use this for the GMM-embedding pathway (DPsurv / LinearEmb / IndivMLPEmb).
"""

from downstream.dpsurv.data import GMMEmbeddingDataset, build_df, collate_flat

__all__ = ["GMMSurvivalDataset", "build_df", "collate_flat"]

GMMSurvivalDataset = GMMEmbeddingDataset
