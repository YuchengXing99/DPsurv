"""
Model factory for the MIL framework.

create_downstream_model(name, cfg) instantiates any supported downstream model.

Input pathway:
    'abmil'      — raw patch features  [M, D]      → downstream/abmil/
    'linear_emb' — flat GMM embedding  [B, K+2KD]  → downstream/linear_emb/
    'indiv_mlp'  — tokenized GMM       [B, K, D]   → downstream/linear_emb/
    'dpsurv'     — GMM (prob,mean,cov) per-expert  → downstream/dpsurv/
"""

from downstream.abmil.model import ABMIL
from downstream.linear_emb.model import LinearEmb, IndivMLPEmb
from downstream.dpsurv.models import mixture_ENNreg_new


def create_downstream_model(model_name: str, cfg: dict):
    """
    Instantiate a downstream survival model.

    Args:
        model_name: one of 'abmil', 'linear_emb', 'indiv_mlp', 'dpsurv'
        cfg:        keyword arguments forwarded to the model constructor

    Returns:
        nn.Module
    """
    name = model_name.lower()
    if name == 'abmil':
        return ABMIL(**cfg)
    if name == 'linear_emb':
        return LinearEmb(**cfg)
    if name == 'indiv_mlp':
        return IndivMLPEmb(**cfg)
    if name == 'dpsurv':
        return mixture_ENNreg_new(**cfg)
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: abmil, linear_emb, indiv_mlp, dpsurv."
    )
