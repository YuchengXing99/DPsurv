"""
Unified MIL Training Script — two input pathways.

  Raw-patch pathway (--input_type raw):
      .h5/.pt patch features → ABMIL → NLL/Cox survival loss
      Dataset: mil_framework/datasets/wsi_survival.py  (WSISurvivalDataset)
      Model:   downstream/abmil/model.py                (ABMIL)

  GMM-embedding pathway (--input_type gmm):
      pre-computed PANTHER .pkl → LinearEmb / IndivMLPEmb / DPsurv
      Dataset: downstream/dpsurv/data.py                (GMMEmbeddingDataset)
      Model:   downstream/linear_emb/ or downstream/dpsurv/

Usage examples:
    # ABMIL on raw patch features (KIRC, fold 0)
    python trainer/train_mil.py --dataset KIRC --model abmil \\
        --input_type raw --feat_dir /path/to/feats_h5 --fold 0

    # LinearEmb on pre-computed GMM embeddings
    python trainer/train_mil.py --dataset KIRC --model linear_emb \\
        --input_type gmm --embedding_fname panther.pkl --fold 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split

# Add repo root to path so downstream/mil_framework imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mil_framework.factory import create_downstream_model
from mil_framework.losses.survival import NLLSurvLoss, CoxLoss
from mil_framework.utils import seed_torch, EarlyStopping
from mil_framework.datasets.wsi_survival import WSISurvivalDataset, compute_discretization
from downstream.abmil.losses import SurvNLLLoss as ABMILSurvNLLLoss, evaluate_abmil
from downstream.abmil.dataset import PatchBagDataset, collate_bags
from downstream.dpsurv.data import GMMEmbeddingDataset, build_df, collate_flat
from downstream.linear_emb.model import LinearEmb, IndivMLPEmb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42

DATASET_ALIASES = {
    "BLCA": "TCGA_BLCA",
    "BRCA": "TCGA_BRCA",
    "KIRC": "TCGA_KIRC",
    "LUAD": "TCGA_LUAD",
    "UCEC": "TCGA_UCEC",
}

DEFAULT_EMBEDDING_FNAME = (
    "extracted-vit_large_patch16_224.dinov2.uni_mass100k_"
    "PANTHER_embeddings_proto_16_allcat_em_1_eps_1.0_tau_1.0_tokenized.pkl"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split_csv(split_dir: Path) -> tuple:
    train_df = pd.read_csv(split_dir / "train.csv")
    test_df  = pd.read_csv(split_dir / "test.csv")
    return train_df, test_df


def _split_train_val(df, val_fraction=0.15, seed=SEED):
    """Split a training frame into (inner-train, validation), keeping events in both.

    Selection is by event-presence only (a validity requirement), never by metric,
    so model selection / early stopping can use the validation set instead of test.
    """
    events = (1.0 - df['dss_censorship'].astype(float)).astype(int)
    strat = events if (events.nunique() > 1 and events.value_counts().min() >= 2) else None
    tr, va = None, None
    for offset in range(32):
        tr, va = train_test_split(df, test_size=val_fraction, random_state=seed + offset,
                                  shuffle=True, stratify=strat)
        has = lambda d: ((1.0 - d['dss_censorship'].astype(float)) > 0.5).any()
        if has(tr) and has(va):
            break
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def _build_abmil_loaders(train_df, test_df, feat_dir, n_bins, batch_size, num_workers, val_fraction):
    tr_df, val_df = _split_train_val(train_df, val_fraction)
    train_ds = PatchBagDataset(tr_df,   feat_dir=feat_dir, n_bins=n_bins)
    val_ds   = PatchBagDataset(val_df,  feat_dir=feat_dir, n_bins=n_bins, bins=train_ds.qbins)
    test_ds  = PatchBagDataset(test_df, feat_dir=feat_dir, n_bins=n_bins, bins=train_ds.qbins)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_bags, num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_bags, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_bags, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds.qbins


def _assign_discrete_labels(ds, df, qbins):
    """Discretize survival days into bin indices and attach them to the dataset."""
    labels = pd.cut(df['dss_survival_days'].astype(float), bins=qbins,
                    labels=False, include_lowest=True)
    labels = labels.fillna(len(qbins) - 2).astype(np.int64)
    ds.labels = torch.tensor(labels.to_numpy(), dtype=torch.float32)


def _build_gmm_loaders(train_df, test_df, split_dir, embedding_fname, n_bins, batch_size, num_workers, val_fraction):
    import pickle

    emb_path = split_dir / "embeddings" / embedding_fname
    with open(emb_path, 'rb') as f:
        embed_data = pickle.load(f)

    # Embeddings are stored per split: {'train': {prob,mean,cov}, 'test': {...}}
    # build_df pairs each split's embedding rows with that split's CSV rows (same order).
    train_gmm = build_df(embed_data['train'], train_df).reset_index(drop=True)
    test_gmm  = build_df(embed_data['test'],  test_df).reset_index(drop=True)
    # Hold out a validation set from train (embeddings + labels stay row-aligned).
    tr_gmm, val_gmm = _split_train_val(train_gmm, val_fraction)

    # Compute bins from (uncensored) inner-training data only
    uncensored = tr_gmm[tr_gmm['dss_censorship'].astype(float) == 0.0]
    src = uncensored if len(uncensored) >= 2 else tr_gmm
    _, qbins = pd.qcut(src['dss_survival_days'].astype(float), q=n_bins,
                       retbins=True, labels=False, duplicates='drop')
    qbins = np.unique(qbins.astype(np.float32))
    qbins[0]  = min(qbins[0],  1e-6)
    qbins[-1] = max(qbins[-1], 1e6)

    train_ds, val_ds, test_ds = (GMMEmbeddingDataset(tr_gmm),
                                 GMMEmbeddingDataset(val_gmm),
                                 GMMEmbeddingDataset(test_gmm))
    _assign_discrete_labels(train_ds, tr_gmm,  qbins)
    _assign_discrete_labels(val_ds,   val_gmm, qbins)
    _assign_discrete_labels(test_ds,  test_gmm, qbins)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_flat, num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_flat, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_flat, num_workers=num_workers)
    return train_loader, val_loader, test_loader, qbins


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _train_epoch_abmil(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        feats_list = batch['features']
        labels     = batch['label'].to(device)
        cens       = batch['censorship'].to(device)

        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        for i, feats in enumerate(feats_list):
            out    = model(feats.to(device))
            hazards = out['hazards'].unsqueeze(0)
            lbl    = labels[i].unsqueeze(0)
            c      = cens[i].unsqueeze(0)
            batch_loss = batch_loss + loss_fn(hazards, lbl, c)
        batch_loss = batch_loss / len(feats_list)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    return total_loss / max(len(loader), 1)


def _flatten_gmm_batch(batch):
    """Concatenate (prob, mean, cov) into a flat [B, K + 2*K*D] tensor (for LinearEmb)."""
    return torch.cat([
        batch['prob'],
        batch['mean'].reshape(batch['mean'].shape[0], -1),
        batch['cov'].reshape(batch['cov'].shape[0], -1),
    ], dim=1)


def _tokenize_gmm_batch(batch):
    """Stack (prob, mean, cov) into a tokenized [B, K, 1 + 2*D] tensor (for IndivMLPEmb)."""
    return torch.cat([batch['prob'].unsqueeze(2), batch['mean'], batch['cov']], dim=2)


def _build_gmm_input(batch, tokenize):
    return _tokenize_gmm_batch(batch) if tokenize else _flatten_gmm_batch(batch)


def _train_epoch_gmm(model, loader, loss_fn, optimizer, device, tokenize=False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x      = _build_gmm_input(batch, tokenize).to(device)
        labels = batch['labels'].long().to(device)
        cens   = batch['censorship'].to(device)

        optimizer.zero_grad()
        out  = model(x)
        loss_dict = loss_fn(out['logits'], labels, cens)
        loss_dict['loss'].backward()
        optimizer.step()
        total_loss += loss_dict['loss'].item()
    return total_loss / max(len(loader), 1)


def _evaluate_emb(model, loader, device, qbins, tokenize=False):
    """Survival metrics for embedding-based models (LinearEmb / IndivMLPEmb).

    Mirrors evaluate_abmil but consumes GMM batches and {'logits'} model output.
    """
    model.eval()
    all_S, all_times, all_cens = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = _build_gmm_input(batch, tokenize).to(device)
            logits = model(x)['logits']
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            all_S.append(S.cpu().float().numpy())
            all_times.extend(batch['survival_time'].reshape(-1).numpy())
            all_cens.extend(batch['censorship'].reshape(-1).numpy())

    S_all    = np.concatenate(all_S, axis=0)                 # [N, n_bins]
    times_np = np.array(all_times, dtype=np.float64)
    cens_np  = np.array(all_cens,  dtype=np.float64)
    events   = (1 - cens_np).astype(bool)

    bin_times = 0.5 * (qbins[:-1] + qbins[1:]).astype(np.float64)
    risk    = -S_all.sum(axis=1)
    c_index = concordance_index_censored(events, times_np, risk, tied_tol=1e-8)[0]

    surv_df = pd.DataFrame(S_all.T, index=bin_times)
    ev      = EvalSurv(surv_df, times_np, events.astype(float), censor_surv='km')
    t_min, t_max = max(bin_times[0], times_np.min()), min(bin_times[-1], times_np.max())
    if t_min >= t_max:
        t_min, t_max = bin_times[0], bin_times[-1]
    time_grid = np.linspace(t_min, t_max, 100)
    return {
        'c_index':    c_index,
        'c_index_td': ev.concordance_td('adj_antolini'),
        'ibs':        ev.integrated_brier_score(time_grid),
        'nbll':       ev.integrated_nbll(time_grid),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified MIL survival training.")
    p.add_argument('--dataset',        type=str,   default='KIRC',
                   help='Short dataset name: LUAD, KIRC, BRCA, BLCA, UCEC')
    p.add_argument('--model',          type=str,   default='abmil',
                   choices=['abmil', 'linear_emb', 'indiv_mlp'],
                   help='Downstream model')
    p.add_argument('--input_type',     type=str,   default='raw',
                   choices=['raw', 'gmm'],
                   help='raw: patch features (for abmil); gmm: pre-computed embeddings')
    p.add_argument('--fold',           type=int,   default=0,  help='Outer fold index')
    p.add_argument('--splits_root',    type=Path,  default=None)
    p.add_argument('--embedding_fname',type=str,   default=DEFAULT_EMBEDDING_FNAME)
    p.add_argument('--feat_dir',       type=str,   default=None,
                   help='Directory of .h5/.pt patch features (raw pathway only)')
    p.add_argument('--in_dim',         type=int,   default=1024, help='Patch feature dim (raw pathway)')
    p.add_argument('--feat_dim',       type=int,   default=512)
    p.add_argument('--n_bins',         type=int,   default=4)
    p.add_argument('--val_fraction',   type=float, default=0.15,
                   help='Fraction of train held out for early stopping (test is never used for selection)')
    p.add_argument('--loss',           type=str,   default='nll', choices=['nll', 'cox'])
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--epochs',         type=int,   default=20)
    p.add_argument('--patience',       type=int,   default=10)
    p.add_argument('--batch_size',     type=int,   default=16)
    p.add_argument('--num_workers',    type=int,   default=0)
    p.add_argument('--device',         type=str,   default='cuda')
    p.add_argument('--results_dir',    type=Path,  default=None)
    p.add_argument('--base_dir',       type=Path,
                   default=Path(__file__).resolve().parent.parent)
    return p.parse_args()


def main():
    args = parse_args()
    seed_torch(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset_full = DATASET_ALIASES.get(args.dataset, args.dataset)
    if args.splits_root is None:
        args.splits_root = args.base_dir / 'data' / 'splits'
    split_dir = args.splits_root / f"{dataset_full}_overall_survival_k={args.fold}"

    train_df, test_df = _load_split_csv(split_dir)

    # ---- Build data loaders ------------------------------------------------
    if args.input_type == 'raw':
        if args.feat_dir is None:
            raise ValueError("--feat_dir is required for --input_type raw")
        train_loader, val_loader, test_loader, qbins = _build_abmil_loaders(
            train_df, test_df, args.feat_dir, args.n_bins, args.batch_size, args.num_workers,
            args.val_fraction
        )
        model_cfg = dict(in_dim=args.in_dim, feat_dim=args.feat_dim, n_bins=args.n_bins)
        loss_fn   = ABMILSurvNLLLoss(alpha=0.0)

        def train_epoch(m, ldr, lfn, opt, dev):
            return _train_epoch_abmil(m, ldr, lfn, opt, dev)

        def evaluate(m, loader):
            return evaluate_abmil(m, loader, device, qbins)

    else:  # gmm
        train_loader, val_loader, test_loader, qbins = _build_gmm_loaders(
            train_df, test_df, split_dir, args.embedding_fname,
            args.n_bins, args.batch_size, args.num_workers, args.val_fraction
        )
        # infer GMM shapes from first batch: prob [B, K], mean/cov [B, K, D]
        sample = next(iter(train_loader))
        n_proto   = sample['prob'].shape[1]
        mean_dim  = sample['mean'].shape[2]
        flat_dim  = n_proto + 2 * n_proto * mean_dim   # K + 2*K*D  (LinearEmb input)
        proto_dim = 1 + 2 * mean_dim                   # per-token dim (IndivMLPEmb input)

        # IndivMLPEmb consumes the tokenized [B, K, 1+2D] tensor; LinearEmb the flat one.
        tokenize = (args.model == 'indiv_mlp')
        if args.model == 'indiv_mlp':
            model_cfg = dict(n_proto=n_proto, proto_dim=proto_dim, n_bins=args.n_bins)
        else:
            model_cfg = dict(in_dim=flat_dim, n_bins=args.n_bins)

        if args.loss == 'nll':
            loss_fn = NLLSurvLoss(alpha=0.0)
        else:
            loss_fn = CoxLoss()

        def train_epoch(m, ldr, lfn, opt, dev):
            return _train_epoch_gmm(m, ldr, lfn, opt, dev, tokenize=tokenize)

        def evaluate(m, loader):
            return _evaluate_emb(m, loader, device, qbins, tokenize=tokenize)

    # ---- Build model --------------------------------------------------------
    # model_cfg is fully prepared above per (input_type, model).
    model = create_downstream_model(args.model, model_cfg)
    model.to(device)

    optimizer    = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stop   = EarlyStopping(patience=args.patience, mode='max')

    # ---- Training loop ------------------------------------------------------
    print(f"\n[train_mil] Dataset={args.dataset}, model={args.model}, "
          f"input={args.input_type}, fold={args.fold}, device={device}")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader)   # model selection on validation only
        print(f"  Epoch {epoch:3d} | loss={tr_loss:.4f} | "
              f"val C-index={val_metrics['c_index']:.4f} | val IBS={val_metrics['ibs']:.4f}")
        if early_stop(val_metrics['c_index'], epoch):
            print(f"  Early stopping at epoch {epoch}.")
            break

    metrics = evaluate(model, test_loader)          # test touched once, for reporting only
    print(f"\n[Results] C-index={metrics['c_index']:.4f} | "
          f"C-index_td={metrics['c_index_td']:.4f} | "
          f"IBS={metrics['ibs']:.4f} | NBLL={metrics['nbll']:.4f}")

    # ---- Save results -------------------------------------------------------
    if args.results_dir is None:
        args.results_dir = args.base_dir / 'results' / 'mil'
    out_dir = args.results_dir / args.dataset / args.model / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"[Saved] {out_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
