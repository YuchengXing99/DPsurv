"""
Attention-Based Multiple Instance Learning (ABMIL) for survival prediction.

Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.
Survival head: discrete NLL following HIPT / SurvPath (Chen et al., CVPR 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al. Eq. 4)."""

    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.U    = nn.Linear(dim, hidden_dim)
        self.V    = nn.Linear(dim, hidden_dim)
        self.w    = nn.Linear(hidden_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple:
        """
        Args:
            h: [M, dim]  patch features
        Returns:
            z: [dim]     slide-level embedding (weighted sum)
            a: [M, 1]    normalised attention weights
        """
        a = torch.tanh(self.U(h)) * torch.sigmoid(self.V(h))
        a = self.drop(self.w(a))           # [M, 1]
        a = F.softmax(a, dim=0)
        z = (a * h).sum(dim=0)             # [dim]
        return z, a


class ABMIL(nn.Module):
    """
    ABMIL survival model operating on raw patch feature bags.

    Input:   bag of patch features  [M, in_dim]
    Output:  discrete hazard logits [n_bins]

    Pipeline:
        1. MLP encoder:        in_dim → feat_dim
        2. Gated attention pooling → slide embedding [feat_dim]
        3. Hazard head:        feat_dim → n_bins (raw logits)

    The n_bins discrete hazards model survival as:
        S(t_k) = prod_{j<=k} (1 - sigmoid(logit_j))
    """

    def __init__(
        self,
        in_dim: int   = 1024,
        feat_dim: int = 512,
        n_bins: int   = 4,
        dropout: float = 0.25,
        attn_hidden: int = 256,
    ):
        super().__init__()
        self.n_bins = n_bins

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        )
        self.attn_pool = GatedAttentionPool(feat_dim, hidden_dim=attn_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_bins),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [M, in_dim]  patch feature bag (single slide)
        Returns:
            logits:  [n_bins]  raw pre-sigmoid scores
            hazards: [n_bins]  predicted hazard probabilities in (0,1)
            S:       [n_bins]  survival S(t_k) = cumprod_{j<=k}(1 - h_j)
            attn:    [M, 1]    attention weights
        """
        h = self.encoder(x)                         # [M, feat_dim]
        z, attn = self.attn_pool(h)                 # [feat_dim], [M, 1]
        logits  = self.head(z)                      # [n_bins]
        hazards = torch.sigmoid(logits)
        S       = torch.cumprod(1 - hazards, dim=0)
        return {'logits': logits, 'hazards': hazards, 'S': S, 'attn': attn}
