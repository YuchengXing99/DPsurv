"""
Linear downstream models operating on pre-computed GMM embeddings.

LinearEmb   — simple linear head on the full flattened GMM embedding (π, μ, Σ).
IndivMLPEmb — per-prototype MLP with optional shared and post-concat layers.

Both expect the flattened 'allcat' PANTHER output as input:
    [π_1..π_K, μ_1,1..μ_K,D, Σ_1,1..Σ_K,D]  →  K + 2*K*D  dims
"""

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dims: list, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class LinearEmb(nn.Module):
    """
    Direct linear projection from flattened GMM embedding to survival bins.

    Input:  [B, flat_gmm_dim]   — flattened π + μ + Σ concatenation
    Output: [B, n_bins]         — raw logit scores
    """

    def __init__(self, in_dim: int, n_bins: int = 4):
        super().__init__()
        self.classifier = nn.Linear(in_dim, n_bins, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        """x: [B, in_dim] → {'logits': [B, n_bins]}"""
        return {'logits': self.classifier(x)}


class IndivMLPEmb(nn.Module):
    """
    Per-prototype MLP on tokenized GMM embeddings, then concat and classify.

    Input:  [B, n_proto, proto_dim]  — tokenized (one row per GMM component)
    Output: [B, n_bins]              — raw logit scores

    Pipeline:
        shared_mlp (optional):  proto_dim → shared_dim   (shared across prototypes)
        indiv_mlps:             shared_dim → indiv_dim    (one MLP per prototype)
        concat:                 n_proto × indiv_dim       (flattened)
        postcat_mlp (optional): n_proto*indiv_dim → post_dim
        classifier:             post_dim → n_bins
    """

    def __init__(
        self,
        n_proto: int,
        proto_dim: int,
        n_bins: int = 4,
        shared_dim: int = 0,
        indiv_dim: int = 256,
        post_dim: int = 0,
        dropout: float = 0.25,
        n_fc_layers: int = 2,
    ):
        super().__init__()
        self.n_proto = n_proto

        hidden = [indiv_dim] * (n_fc_layers - 1)

        if shared_dim > 0:
            self.shared_mlp = _mlp(proto_dim, hidden, shared_dim, dropout)
            next_dim = shared_dim
        else:
            self.shared_mlp = nn.Identity()
            next_dim = proto_dim

        self.indiv_mlps = nn.ModuleList([
            _mlp(next_dim, hidden, indiv_dim, dropout) for _ in range(n_proto)
        ])
        concat_dim = n_proto * indiv_dim

        if post_dim > 0:
            self.postcat_mlp = _mlp(concat_dim, [post_dim], post_dim, dropout)
            clf_in = post_dim
        else:
            self.postcat_mlp = nn.Identity()
            clf_in = concat_dim

        self.classifier = nn.Linear(clf_in, n_bins, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: [B, n_proto, proto_dim]
        Returns: {'logits': [B, n_bins]}
        """
        x = self.shared_mlp(x)                                          # [B, n_proto, dim]
        x = torch.stack(
            [self.indiv_mlps[k](x[:, k, :]) for k in range(self.n_proto)],
            dim=1,
        )                                                                # [B, n_proto, indiv_dim]
        x = x.reshape(x.shape[0], -1)                                   # [B, n_proto*indiv_dim]
        x = self.postcat_mlp(x)
        return {'logits': self.classifier(x)}
