"""
DPsurv loss functions and survival evaluation utilities.

Mixture_EvidentialSurvLoss   — continuous evidential survival loss (not used in final paper).
Mixture_Evidential_nll_Loss  — discrete NLL survival loss with evidential uncertainty (paper).
evaluate_nll_batch_survival  — evaluation metrics for the NLL-trained model.
evaluate_batch_survival      — evaluation metrics for the continuous loss (reference).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored
from pycox.evaluation import EvalSurv


class Mixture_EvidentialSurvLoss(nn.Module):
    """
    Continuous evidential survival loss using belief/plausibility functions.

    lambd: weight between BEL and PL survival formulations.
    xi, rho: regularisation weights for eta and gamma penalty terms.
    """

    def __init__(self, lambd: float = 0.5, xi: float = 0.1, rho: float = 0.1):
        super().__init__()
        self.lambd = lambd
        self.xi    = xi
        self.rho   = rho

    def forward(self, GRFN_dict: dict, y: torch.Tensor, events: torch.Tensor, prob: torch.Tensor):
        lambd = self.lambd
        nu    = 1e-16

        mux_all      = GRFN_dict['mux'].squeeze()      # [num_components, B]
        sig2x_all    = GRFN_dict['sig2x'].squeeze()
        hx_all       = GRFN_dict['hx'].squeeze()
        penalty1_all = GRFN_dict['penalty1']
        penalty2_all = GRFN_dict['penalty2']

        final_Fy1 = final_Fy2 = final_fy1 = final_fy2 = 0

        for i in range(len(mux_all)):
            mux   = mux_all[i]
            sig2x = sig2x_all[i]
            sigx  = torch.sqrt(sig2x)
            hx    = hx_all[i]

            Z2   = hx * sig2x + 1
            Z    = torch.sqrt(Z2)
            sig1 = sigx * Z
            pl   = 1 / Z * torch.exp(-0.5 * hx * (y - mux) ** 2 / Z2)

            eps  = 1e-2 * torch.std(y)
            nd   = Normal(mux, sigx)
            nd1  = Normal(mux, sig1)

            Fy1 = nd.cdf(y) - pl * nd1.cdf(y)
            Fy2 = Fy1 + pl

            pl1 = 1 / Z * torch.exp(-0.5 * hx * (y - eps - mux) ** 2 / Z2)
            pl2 = 1 / Z * torch.exp(-0.5 * hx * (y + eps - mux) ** 2 / Z2)

            Fy2_1 = nd.cdf(y + eps) + pl1 * nd1.cdf(y - eps)
            Fy2_2 = nd.cdf(y - eps) - pl2 * (1 - nd1.cdf(y + eps))
            fy2   = Fy2_1 - Fy2_2
            fy1   = (fy2
                     - pl1 * nd1.cdf(y + 2 * eps * hx * sig2x / 2)
                     - pl2 * (1 - nd1.cdf(y - 2 * eps * hx * sig2x / 2)))

            final_Fy1 += prob[:, i] * Fy1
            final_Fy2 += prob[:, i] * Fy2
            final_fy1 += prob[:, i] * fy1
            final_fy2 += prob[:, i] * fy2

        Sy1 = torch.clamp(1 - final_Fy1, min=0.0)
        Sy2 = torch.clamp(1 - final_Fy2, min=0.0)
        fy1 = torch.clamp(final_fy1, min=0.0)
        fy2 = torch.clamp(final_fy2, min=0.0)

        loss = (
            -lambd       * torch.mean(torch.log(fy1 + nu) * events + torch.log(Sy1 + nu) * (1 - events))
            -(1 - lambd) * torch.mean(torch.log(fy2 + nu) * events + torch.log(Sy2 + nu) * (1 - events))
            + self.xi  * torch.sum(penalty1_all)
            + self.rho * torch.sum(penalty2_all)
        )
        return loss


class Mixture_Evidential_nll_Loss(nn.Module):
    """
    Discrete NLL survival loss with evidential uncertainty quantification.

    Survival probabilities are evaluated at quantile bin boundaries derived from
    the training data.  The loss interpolates between censored and uncensored
    NLL contributions via the alpha parameter.

    qbins:  [n_bins+1]  quantile bin edges (float32 tensor, in original time units)
    alpha:  weight for the uncensored NLL term
    lambd:  weight between BEL and PL survival formulations
    xi, rho: regularisation weights
    """

    def __init__(
        self,
        qbins: torch.Tensor,
        alpha: float = 0.0,
        eps: float = 1e-7,
        reduction: str = 'mean',
        lambd: float = 0.5,
        xi: float = 0.1,
        rho: float = 0.1,
    ):
        super().__init__()
        self.qbins     = qbins
        self.alpha     = alpha
        self.eps       = eps
        self.reduction = reduction
        self.lambd     = lambd
        self.xi        = xi
        self.rho       = rho

    def forward(self, GRFN_dict: dict, y_label: torch.Tensor, c: torch.Tensor, prob: torch.Tensor):
        mux_all      = GRFN_dict['mux'].squeeze()      # [num_components, B]
        sig2x_all    = GRFN_dict['sig2x'].squeeze()
        hx_all       = GRFN_dict['hx'].squeeze()
        penalty1_all = GRFN_dict['penalty1']
        penalty2_all = GRFN_dict['penalty2']

        log_bins = torch.log(self.qbins).view(1, -1).to(mux_all.device)

        Final_S = 0
        for i in range(len(mux_all)):
            mux   = mux_all[i].view(-1, 1)
            sig2x = sig2x_all[i].view(-1, 1)
            sigx  = torch.sqrt(sig2x)
            hx    = hx_all[i].view(-1, 1)

            Z2   = hx * sig2x + 1
            Z    = torch.sqrt(Z2)
            sig1 = sigx * Z
            pl   = 1 / Z * torch.exp(-0.5 * hx * (log_bins - mux) ** 2 / Z2)

            Fy1 = Normal(mux, sigx).cdf(log_bins) - pl * Normal(mux, sig1).cdf(log_bins)
            Fy2 = Fy1 + pl
            S   = self.lambd * (1 - Fy1) + self.lambd * (1 - Fy2)

            Final_S += prob[:, i].view(-1, 1) * S

        S_prev = Final_S[:, :-1].clamp(min=self.eps)
        S_next = Final_S[:, 1:].clamp(min=self.eps)
        hazards = (S_prev - S_next) / S_prev

        y_label = y_label.long().unsqueeze(1)
        c = c.long()

        s_prev = torch.gather(Final_S, 1, y_label).clamp(min=self.eps)
        h_this = torch.gather(hazards, 1, y_label).clamp(min=self.eps)
        s_this = torch.gather(Final_S, 1, y_label + 1).clamp(min=self.eps)

        uncensored_loss = -(1 - c) * (torch.log(s_prev.squeeze()) + torch.log(h_this.squeeze()))
        censored_loss   = -c * torch.log(s_this.squeeze())

        neg_l = censored_loss + uncensored_loss
        loss  = ((1 - self.alpha) * neg_l
                 + self.alpha * uncensored_loss
                 + self.xi  * torch.mean(penalty1_all)
                 + self.rho * torch.mean(penalty2_all))

        if self.reduction == 'mean':
            return {'loss': loss.mean(), 'uncensored_loss': uncensored_loss.mean(), 'censored_loss': censored_loss.mean()}
        elif self.reduction == 'sum':
            return {'loss': loss.sum(), 'uncensored_loss': uncensored_loss.sum(), 'censored_loss': censored_loss.sum()}
        return {'loss': loss, 'uncensored_loss': uncensored_loss, 'censored_loss': censored_loss}


def evaluate_nll_batch_survival(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    qbins: torch.Tensor,
    weight: float = 0.5,
) -> dict:
    """
    Compute survival metrics for the NLL-trained DPsurv model.

    Returns dict with keys: c_index, c_index_td, ibs, nbll.
    """
    model.eval()
    mux_all = []; sig2x_all = []; hx_all = []
    time_all = []; censorship_all = []; prob_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            prob = batch['prob'].to(device)
            mean = batch['mean'].to(device)
            cov  = batch['cov'].to(device)
            time = batch['survival_time'].to(device)
            cens = batch['censorship'].to(device)

            input_feat = torch.cat([prob.unsqueeze(2), mean, cov], dim=2)
            out = model(input_feat, prob)

            mux   = out['mux'].squeeze()
            sig2x = out['sig2x'].squeeze()
            hx    = out['hx'].squeeze()
            time  = time.to(dtype=mux.dtype).view(-1)

            # handle edge case of batch size 1
            for t in [mux, sig2x, hx]:
                if t.dim() == 1:
                    t = t.unsqueeze(1)
            time       = time.unsqueeze(0) if time.dim() == 0 else time
            cens       = cens.unsqueeze(0) if cens.dim() == 0 else cens
            prob       = prob.unsqueeze(0) if prob.dim() == 0 else prob

            mux_all.append(mux.detach().cpu())
            sig2x_all.append(sig2x.detach().cpu())
            hx_all.append(hx.detach().cpu())
            time_all.append(time.detach().cpu())
            censorship_all.append(cens.detach().cpu())
            prob_all.append(prob.detach().cpu())

    mux_all        = torch.cat(mux_all,        dim=1)
    sig2x_all      = torch.cat(sig2x_all,      dim=1)
    hx_all         = torch.cat(hx_all,         dim=1)
    time_all       = torch.cat(time_all,        dim=0)
    censorship_all = torch.cat(censorship_all,  dim=0)
    prob_all       = torch.cat(prob_all,        dim=0)

    # C-index via risk score from qbin survival function
    log_bins = torch.log(qbins).view(1, -1).to(mux_all.device)
    Final_S = 0
    for i in range(len(mux_all)):
        mux   = mux_all[i].view(-1, 1)
        sig2x = sig2x_all[i].view(-1, 1)
        sigx  = torch.sqrt(sig2x)
        hx    = hx_all[i].view(-1, 1)
        Z2   = hx * sig2x + 1
        Z    = torch.sqrt(Z2)
        sig1 = sigx * Z
        pl   = 1 / Z * torch.exp(-0.5 * hx * (log_bins - mux) ** 2 / Z2)
        Fy1  = Normal(mux, sigx).cdf(log_bins) - pl * Normal(mux, sig1).cdf(log_bins)
        Fy2  = Fy1 + pl
        S    = weight * (1 - Fy1) + weight * (1 - Fy2)
        Final_S += prob_all[:, i].view(-1, 1) * S

    risk = -torch.sum(Final_S, dim=1)
    c_index = concordance_index_censored(
        (1 - censorship_all).cpu().numpy().astype(bool),
        time_all.cpu().numpy(),
        risk.cpu().numpy(),
        tied_tol=1e-08,
    )[0]

    # IBS / NBLL / c_index_td via full survival curve on observed times
    final_Fy1 = final_Fy2 = 0
    for i in range(len(mux_all)):
        mux   = mux_all[i]
        sig2x = sig2x_all[i]
        sigx  = torch.sqrt(sig2x)
        hx    = hx_all[i]
        Z2   = hx * sig2x + 1
        Z    = torch.sqrt(Z2)
        sig1 = sigx * Z
        D, M  = torch.meshgrid(torch.log(time_all), mux, indexing='ij')
        diff  = D - M
        pl    = 1 / Z * torch.exp(-0.5 * hx * diff ** 2 / Z2)
        Fy1   = Normal(mux, sigx).cdf(D) - pl * Normal(mux, sig1).cdf(D)
        Fy2   = Fy1 + pl
        final_Fy1 += prob_all[:, i] * Fy1
        final_Fy2 += prob_all[:, i] * Fy2

    surv = 1 - (weight * final_Fy1 + (1 - weight) * final_Fy2)
    surv_df = pd.DataFrame(surv.detach().cpu().numpy(), index=time_all.detach().cpu().numpy())
    surv_df = surv_df.sort_index()

    ev = EvalSurv(surv_df, time_all.cpu().numpy(), 1 - censorship_all.cpu().numpy().squeeze(), censor_surv='km')
    time_grid  = np.linspace(time_all.cpu().numpy().min(), time_all.cpu().numpy().max(), 100)
    c_index_td = ev.concordance_td('adj_antolini')
    ibs        = ev.integrated_brier_score(time_grid)
    nbll       = ev.integrated_nbll(time_grid)

    return {'c_index': c_index, 'c_index_td': c_index_td, 'ibs': ibs, 'nbll': nbll}


def evaluate_batch_survival(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    weight: float = 0.5,
) -> dict:
    """
    Compute survival metrics for the continuous evidential loss variant.

    Returns dict with keys: c_index, c_index_td, ibs, nbll.
    """
    model.eval()
    mux_all = []; sig2x_all = []; hx_all = []
    time_all = []; censorship_all = []; prob_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            prob = batch['prob'].to(device)
            mean = batch['mean'].to(device)
            cov  = batch['cov'].to(device)
            time = batch['survival_time'].to(device)
            cens = batch['censorship'].to(device)

            input_feat = torch.cat([prob.unsqueeze(2), mean, cov], dim=2)
            out = model(input_feat, prob)

            mux   = out['mux'].squeeze()
            sig2x = out['sig2x'].squeeze()
            hx    = out['hx'].squeeze()
            time  = time.to(dtype=mux.dtype).view(-1)

            mux_all.append(mux.detach().cpu())
            sig2x_all.append(sig2x.detach().cpu())
            hx_all.append(hx.detach().cpu())
            time_all.append(time.detach().cpu())
            censorship_all.append(cens.detach().cpu())
            prob_all.append(prob.detach().cpu())

    mux_all        = torch.cat(mux_all,        dim=1)
    sig2x_all      = torch.cat(sig2x_all,      dim=1)
    hx_all         = torch.cat(hx_all,         dim=1)
    time_all       = torch.cat(time_all,        dim=0)
    censorship_all = torch.cat(censorship_all,  dim=0)
    prob_all       = torch.cat(prob_all,        dim=0)

    final_Fy1 = final_Fy2 = 0
    for i in range(len(mux_all)):
        mux   = mux_all[i]
        sig2x = sig2x_all[i]
        sigx  = torch.sqrt(sig2x)
        hx    = hx_all[i]
        Z2   = hx * sig2x + 1
        Z    = torch.sqrt(Z2)
        sig1 = sigx * Z
        D, M  = torch.meshgrid(torch.log(time_all), mux, indexing='ij')
        diff  = D - M
        pl    = 1 / Z * torch.exp(-0.5 * hx * diff ** 2 / Z2)
        Fy1   = Normal(mux, sigx).cdf(D) - pl * Normal(mux, sig1).cdf(D)
        Fy2   = Fy1 + pl
        final_Fy1 += prob_all[:, i] * Fy1
        final_Fy2 += prob_all[:, i] * Fy2

    surv = 1 - (weight * final_Fy1 + (1 - weight) * final_Fy2)
    surv_df = pd.DataFrame(surv.detach().cpu().numpy(), index=time_all.detach().cpu().numpy())
    surv_df = surv_df.sort_index()

    ev = EvalSurv(surv_df, time_all.cpu().numpy(), 1 - censorship_all.cpu().numpy().squeeze(), censor_surv='km')
    risk       = -np.trapz(y=surv_df.values, x=surv_df.index.values, axis=0)
    c_index    = concordance_index_censored(
        (1 - censorship_all).cpu().numpy().astype(bool),
        time_all.cpu().numpy(),
        risk,
        tied_tol=1e-08,
    )[0]
    time_grid  = np.linspace(time_all.cpu().numpy().min(), time_all.cpu().numpy().max(), 100)
    c_index_td = ev.concordance_td('adj_antolini')
    ibs        = ev.integrated_brier_score(time_grid)
    nbll       = ev.integrated_nbll(time_grid)

    return {'c_index': c_index, 'c_index_td': c_index_td, 'ibs': ibs, 'nbll': nbll}
