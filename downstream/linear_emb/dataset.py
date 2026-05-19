"""Re-export GMMEmbeddingDataset for use with LinearEmb / IndivMLPEmb."""
from downstream.dpsurv.data import GMMEmbeddingDataset, build_df, collate_flat

__all__ = ["GMMEmbeddingDataset", "build_df", "collate_flat"]
