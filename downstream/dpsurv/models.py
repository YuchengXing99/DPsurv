"""
DPsurv model components.

ENNreg_new   — single evidence neural network regressor for one GMM component.
mixture_ENNreg_new — ensemble of K per-prototype ENNreg_new experts (the DPsurv model).
ENNreg_init_cosine — cosine-distance KMeans initialisation for prototype parameters.
ENNreg_init        — Euclidean KMeans initialisation (alternative).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


def mixture_input_dim_from_mean_dim(mean_dim: int) -> int:
    """Compute per-expert input dim from GMM mean dim: 1 (prob) + mean_dim + cov_dim."""
    return 1 + 2 * int(mean_dim)


def ENNreg_init_cosine(prob, X, y, K, nstart=100, c=1.0, eps=1e-8, prob_thresh=0.1):
    """
    Initialise prototype parameters using cosine-distance weighted KMeans.

    Only samples with prob > prob_thresh are used for clustering.
    Falls back to using all samples directly when the filtered count < K.

    Args:
        prob:        [N]     per-sample mixture weight (GMM pi for component i)
        X:           [N, p]  per-sample features (GMM mean for component i)
        y:           [N]     log-survival times
        K:           number of prototypes (inner KMeans clusters)
        nstart:      KMeans re-starts
        c:           gamma scale factor
        prob_thresh: minimum prob to include a sample in clustering

    Returns:
        dict with keys: alpha, Beta, sig, eta, gam, W  (all torch tensors)
    """
    device = X.device
    dtype = torch.float64

    mask = prob > prob_thresh
    X_sel = X[mask]
    y_sel = y[mask]
    prob_sel = prob[mask]

    N_sel = X_sel.shape[0]
    if N_sel < K:
        print(f"[Warning] Effective samples {N_sel} < K={K}; using all filtered samples as prototypes.")
        x_norm = torch.norm(X_sel, p=2, dim=1, keepdim=True).clamp_min(eps)
        X_unit = (X_sel / x_norm).to(dtype=dtype, device=device)
        W = X_unit
        K = N_sel

        input_dim = mixture_input_dim_from_mean_dim(X_sel.shape[1])
        Beta  = torch.zeros(K, input_dim, dtype=dtype, device=device)
        alpha = y_sel.to(dtype=dtype, device=device).clone()
        sig   = torch.ones(K, dtype=dtype, device=device)
        gam   = torch.ones(K, dtype=dtype, device=device)
        eta   = 2 * torch.sqrt(torch.clamp(prob_sel.to(dtype=dtype, device=device), min=1e-6))
        return {'alpha': alpha, 'Beta': Beta, 'sig': sig, 'eta': eta, 'gam': gam, 'W': W}

    X_sel = X_sel.to(dtype=dtype)
    prob_sel = prob_sel.to(dtype=dtype)
    y_sel = y_sel.to(dtype=dtype)

    x_norm = torch.norm(X_sel, p=2, dim=1, keepdim=True).clamp_min(eps)
    X_unit = (X_sel / x_norm).cpu().numpy().astype(np.float64)
    prob_np = prob_sel.cpu().numpy()

    clus = KMeans(n_clusters=K, max_iter=5000, n_init=nstart, random_state=0).fit(
        X_unit, sample_weight=prob_np
    )

    W_np = clus.cluster_centers_
    W_np = W_np / (np.linalg.norm(W_np, axis=1, keepdims=True) + 1e-12)

    input_dim = mixture_input_dim_from_mean_dim(X_sel.shape[1])
    Beta  = torch.zeros(K, input_dim, dtype=dtype)
    alpha = torch.zeros(K, dtype=dtype)
    sig   = torch.ones(K, dtype=dtype)
    W     = torch.from_numpy(W_np).to(dtype=dtype, device=device)
    gam   = torch.ones(K, dtype=dtype, device=device)

    labels   = clus.labels_
    labels_t = torch.from_numpy(labels).to(device)
    X_unit_t = torch.from_numpy(X_unit).to(device=device, dtype=dtype)

    for k in range(K):
        ii = torch.nonzero(labels_t == k, as_tuple=True)[0]
        nk = ii.numel()
        if nk == 0:
            gam[k] = 1.0
            continue

        w_k = prob_sel[ii]
        alpha[k] = torch.sum(w_k * y_sel[ii]) / (torch.sum(w_k) + 1e-8)

        if nk > 1:
            x_i = X_unit_t[ii]
            cos_sim = torch.sum(x_i * W[k].unsqueeze(0), dim=1).clamp(-1.0, 1.0)
            d_cos = (1.0 - cos_sim) / 2.0
            mean_dcos = torch.sum(w_k * d_cos) / (torch.sum(w_k) + 1e-8)
            gam[k] = 1.0 / torch.sqrt(2.0 * torch.clamp(mean_dcos, min=1e-8))

            var_y = torch.sum(w_k * (y_sel[ii] - alpha[k]) ** 2) / (torch.sum(w_k) + 1e-8)
            sig[k] = torch.sqrt(torch.clamp(var_y, min=1e-12))
        else:
            gam[k] = torch.tensor(1.0, dtype=dtype, device=device)

    gam = gam * c

    cnt      = np.bincount(labels, minlength=K)
    sum_prob = np.bincount(labels, weights=prob_np, minlength=K)
    mean_prob = sum_prob / np.maximum(cnt, 1)
    eta = 2 * torch.sqrt(
        torch.clamp(torch.from_numpy(mean_prob).to(device=device, dtype=dtype), min=1e-6)
    )

    return {
        'alpha': alpha.to(device),
        'Beta':  Beta.to(device),
        'sig':   sig.to(device),
        'eta':   eta.to(device),
        'gam':   gam.to(device),
        'W':     W.to(device),
    }


def ENNreg_init(prob, X, y, K, nstart=100, c=1):
    """
    Initialise prototype parameters using Euclidean weighted KMeans.

    Alternative to ENNreg_init_cosine; kept for reference.
    """
    X_np   = X.cpu().numpy()
    prob_np = prob.cpu().numpy()

    clus = KMeans(n_clusters=K, max_iter=5000, n_init=nstart, random_state=0).fit(
        X_np, sample_weight=prob_np
    )

    input_dim = mixture_input_dim_from_mean_dim(X.shape[1])
    Beta  = torch.zeros(K, input_dim, dtype=torch.float64)
    alpha = torch.zeros(K, dtype=torch.float64)
    sig   = torch.ones(K, dtype=torch.float64)
    W     = torch.tensor(clus.cluster_centers_, dtype=torch.float64)
    gam   = torch.ones(K, dtype=torch.float64)

    X_t    = X.to(dtype=torch.float64)
    prob_t = prob.to(dtype=torch.float64)
    y_t    = y.to(dtype=torch.float64)

    for k in range(K):
        mask = torch.eq(torch.tensor(clus.labels_), k)
        ii = torch.nonzero(mask, as_tuple=True)[0]
        if ii.numel() > 0:
            w_k = prob_t[ii]
            alpha[k] = torch.sum(w_k * y_t[ii]) / (torch.sum(w_k) + 1e-8)
            if ii.numel() > 1:
                dist2 = torch.sum((X_t[ii] - W[k]) ** 2, dim=1)
                inertia_k = torch.sum(w_k * dist2)
                gam[k] = 1.0 / torch.sqrt(1e-3 + (inertia_k / (torch.sum(w_k) + 1e-8)) / 2)
                sig[k] = torch.sqrt(
                    torch.sum(w_k * (y_t[ii] - alpha[k]) ** 2) / (torch.sum(w_k) + 1e-8)
                )

    gam *= c

    cnt      = np.bincount(clus.labels_, minlength=K)
    sum_prob = np.bincount(clus.labels_, weights=prob_np, minlength=K)
    mean_prob = sum_prob / np.maximum(cnt, 1)
    eta = 2 * torch.sqrt(
        torch.clamp(torch.from_numpy(mean_prob), min=1e-6)
    ).to(dtype=torch.float64)

    return {'alpha': alpha, 'Beta': Beta, 'sig': sig, 'eta': eta, 'gam': gam, 'W': W}


class ENNreg_new(nn.Module):
    """
    Single-component evidence neural network regressor.

    Outputs a Generalised Random Fuzzy Number (GRFN) prediction:
      mux   — predictive mean
      sig2x — predictive variance
      hx    — aggregated confidence (prototype-weighted)
    """

    def __init__(self, input_dim: int, prototype_dim: int):
        super().__init__()
        self.input_dim    = input_dim
        self.prototype_dim = prototype_dim
        self.mean_dim = (input_dim - 1) // 2

        self.alpha = nn.Parameter(torch.randn(1, prototype_dim, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.randn(prototype_dim, input_dim, dtype=torch.float32))
        self.sig   = nn.Parameter(torch.randn(1, prototype_dim, dtype=torch.float32))
        self.eta   = nn.Parameter(torch.randn(1, prototype_dim, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.randn(prototype_dim, dtype=torch.float32))
        self.w     = nn.Parameter(torch.randn(prototype_dim, self.mean_dim, dtype=torch.float32))

    def reset_parameters(self, prototype: dict, device: torch.device) -> None:
        self.alpha.data.copy_(prototype['alpha'].to(device, dtype=torch.float32))
        self.beta.data.copy_(prototype['Beta'].to(device, dtype=torch.float32))
        self.sig.data.copy_(prototype['sig'].to(device, dtype=torch.float32))
        self.eta.data.copy_(prototype['eta'].to(device, dtype=torch.float32))
        self.gamma.data.copy_(prototype['gam'].to(device, dtype=torch.float32))
        self.w.data.copy_(prototype['W'].to(device, dtype=torch.float32))

    def forward(self, input: torch.Tensor, prob: torch.Tensor) -> dict:
        nt = input.size(0)
        h = self.eta ** 2

        mean_slice = input[:, 1:1 + self.mean_dim]
        a = torch.stack([
            torch.exp(
                -self.gamma[k] ** 2 * (
                    1 - F.cosine_similarity(mean_slice, self.w[k].unsqueeze(0), dim=1, eps=1e-8)
                )
            )
            for k in range(self.prototype_dim)
        ], dim=1)  # [nt, prototype_dim]

        H  = h.expand(nt, -1)
        hx = torch.clamp(torch.sum(a * H, dim=1), min=1e-8)
        mu = torch.mm(input, self.beta.T) + self.alpha.expand(nt, -1)

        mux   = torch.sum(mu * a * H, dim=1) / hx
        sig2x = torch.clamp(
            torch.sum((self.sig ** 2).expand(nt, -1) * (a ** 2) * (H ** 2), dim=1) / (hx ** 2),
            min=1e-8,
        )

        return {
            'mux':      mux,
            'sig2x':    sig2x,
            'hx':       hx,
            'penalty1': torch.mean(h),
            'penalty2': torch.mean(self.gamma ** 2),
        }


class mixture_ENNreg_new(nn.Module):
    """
    DPsurv: ensemble of one ENNreg_new expert per GMM component.

    Each expert processes the features of its assigned prototype (pi, mean, cov).
    The mixture weights (pi) are used inside the loss to aggregate expert outputs.
    """

    def __init__(self, input_dim: int = 3073, prototype_list=None, num_models: int = 16):
        super().__init__()
        self.models = nn.ModuleList([
            ENNreg_new(input_dim=input_dim, prototype_dim=prototype_list[i]['W'].shape[0])
            for i in range(num_models)
        ])

    def forward(self, x: torch.Tensor, prob: torch.Tensor) -> dict:
        outputs = [model(x[:, i, :], prob[:, i]) for i, model in enumerate(self.models)]
        return {key: torch.stack([o[key] for o in outputs], dim=0) for key in outputs[0]}

    def reset_parameters(self, prototypes_list: list, device: torch.device) -> None:
        for i, model in enumerate(self.models):
            model.reset_parameters(prototypes_list[i], device)
