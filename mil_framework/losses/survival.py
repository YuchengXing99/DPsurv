"""
Survival loss functions for the raw-patch MIL pathway (ABMIL).

NLLSurvLoss — discrete NLL with censorship weighting (Zadeh & Schmid, 2020).
CoxLoss     — partial log-likelihood (Cox proportional hazards).

Adapted from src/utils/losses.py.
"""

import torch
import torch.nn as nn


class NLLSurvLoss(nn.Module):
    """
    Negative log-likelihood loss for discrete-time survival models.

    hazards = sigmoid(logits)
    S(t_k) = prod_{j<=k} (1 - hazards_j)

    Args:
        alpha:     weight on the uncensored term (default 0.0 = standard NLL)
        eps:       numerical floor for log
        reduction: 'mean' or 'sum'
    """

    def __init__(self, alpha: float = 0.0, eps: float = 1e-7, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, times: torch.Tensor, censorships: torch.Tensor) -> dict:
        """
        Args:
            logits:      [B, n_bins]  raw model outputs
            times:       [B]          discrete bin index (0-based)
            censorships: [B]          1=censored, 0=event
        Returns:
            dict with keys 'loss', 'censored_loss', 'uncensored_loss'
        """
        y = times.long().unsqueeze(1)      # [B, 1]
        c = censorships.long().unsqueeze(1)

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        S_padded = torch.cat([torch.ones(S.shape[0], 1, device=S.device), S], dim=1)

        s_prev = torch.gather(S_padded, 1, y).clamp(min=self.eps)
        h_this = torch.gather(hazards, 1, y).clamp(min=self.eps)
        s_this = torch.gather(S_padded, 1, y + 1).clamp(min=self.eps)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss   = -c * torch.log(s_this)
        neg_l = censored_loss + uncensored_loss
        loss  = (1 - self.alpha) * neg_l + self.alpha * uncensored_loss

        if self.reduction == 'mean':
            return {
                'loss': loss.mean(),
                'censored_loss': censored_loss.mean(),
                'uncensored_loss': uncensored_loss.mean(),
            }
        return {
            'loss': loss.sum(),
            'censored_loss': censored_loss.sum(),
            'uncensored_loss': uncensored_loss.sum(),
        }


class CoxLoss(nn.Module):
    """Cox partial log-likelihood loss."""

    def forward(self, logits: torch.Tensor, times: torch.Tensor, censorships: torch.Tensor) -> dict:
        """
        Args:
            logits:      [B, 1]  log-risk scores
            times:       [B]     survival times (continuous or bin index)
            censorships: [B]     1=censored, 0=event
        """
        lrisks = logits.squeeze(1)
        events = (1 - censorships).float()
        n_events = events.sum()
        if n_events == 0:
            return {'loss': lrisks.sum() * 0}

        order = torch.argsort(-times)
        lrisks = lrisks[order]
        events = events[order]

        log_risk = torch.logcumsumexp(lrisks, dim=0)
        loss = -((lrisks - log_risk) * events).sum() / n_events
        return {'loss': loss}
