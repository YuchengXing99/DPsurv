"""
Discrete NLL survival loss and evaluation metrics for ABMIL.

Loss formulation follows HIPT / SurvPath (Chen et al., CVPR 2023).
Evaluation uses pycox EvalSurv with bin-midpoint time axis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sksurv.metrics import concordance_index_censored
from pycox.evaluation import EvalSurv


class SurvNLLLoss(nn.Module):
    """
    Discrete negative log-likelihood survival loss.

    For each patient with event time bin t and censorship c:
        uncensored (c=0): loss = -log h_t - sum_{k<t} log(1 - h_k)
        censored   (c=1): loss = -sum_{k<=t} log(1 - h_k)

    Combined: loss = censored_nll + (1-alpha) * uncensored_nll + alpha * uncensored_nll
                   = censored_nll + uncensored_nll   when alpha=0 (default)

    alpha:  weight for the extra uncensored NLL term (0 = standard, 0.5 = HIPT setting)
    """

    def __init__(self, alpha: float = 0.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.eps   = eps

    def forward(
        self,
        hazards:    torch.Tensor,   # [B, n_bins]  sigmoid probabilities
        labels:     torch.Tensor,   # [B]           int bin index of event/censor time
        censorship: torch.Tensor,   # [B]           1 = censored, 0 = event
    ) -> torch.Tensor:
        B = hazards.shape[0]
        S        = torch.cumprod(1 - hazards, dim=1)        # [B, n_bins]
        S_padded = torch.cat([torch.ones(B, 1, device=hazards.device), S], dim=1)

        lab = labels.unsqueeze(1)
        s_prev = S_padded.gather(1, lab    ).clamp(min=self.eps).squeeze(1)
        h_t    = hazards.gather(1,  lab    ).clamp(min=self.eps).squeeze(1)
        s_t    = S.gather(1,        lab    ).clamp(min=self.eps).squeeze(1)

        c = censorship
        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_t))
        censored_loss   = -c       *  torch.log(s_t)

        loss = censored_loss + uncensored_loss
        return ((1 - self.alpha) * loss + self.alpha * uncensored_loss).mean()


def evaluate_abmil(
    model:      nn.Module,
    dataloader,
    device:     torch.device,
    qbins:      np.ndarray,
) -> dict:
    """
    Compute C-index, C-index_td, IBS, and NBLL for a trained ABMIL model.

    qbins: [n_bins+1] bin edges in days (from PatchBagDataset.qbins).
           Used to map discrete S(bin) values to actual survival times.
    """
    model.eval()
    all_S, all_times, all_cens = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            for feats in batch["features"]:
                out = model(feats.to(device))
                all_S.append(out["S"].cpu().float().numpy())
            all_times.extend(batch["survival_time"].numpy())
            all_cens.extend(batch["censorship"].numpy())

    S_all    = np.stack(all_S)              # [N, n_bins]
    times_np = np.array(all_times, dtype=np.float64)
    cens_np  = np.array(all_cens,  dtype=np.float64)
    events   = (1 - cens_np).astype(bool)

    # Use bin midpoints as the time axis for the survival function
    bin_times = 0.5 * (qbins[:-1] + qbins[1:]).astype(np.float64)  # [n_bins]

    risk    = -S_all.sum(axis=1)
    c_index = concordance_index_censored(events, times_np, risk, tied_tol=1e-8)[0]

    # EvalSurv: rows = time points (bin midpoints), cols = patients
    surv_df  = pd.DataFrame(S_all.T, index=bin_times)
    ev       = EvalSurv(surv_df, times_np, events.astype(float), censor_surv="km")

    t_min = max(bin_times[0], times_np.min())
    t_max = min(bin_times[-1], times_np.max())
    if t_min >= t_max:
        t_min, t_max = bin_times[0], bin_times[-1]
    time_grid = np.linspace(t_min, t_max, 100)

    return {
        "c_index":    c_index,
        "c_index_td": ev.concordance_td("adj_antolini"),
        "ibs":        ev.integrated_brier_score(time_grid),
        "nbll":       ev.integrated_nbll(time_grid),
    }
