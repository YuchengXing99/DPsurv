"""
DPsurv — Nested cross-validation training and evaluation.

Runs per-fold nested model selection (inner-loop K search) followed by
full outer-train retraining, and reports C-index / NBLL / IBS / C-index_td
for each dataset and fold.

Usage example:
    python trainer/train_dpsurv.py --datasets BRCA --device cuda
    bash scripts/run_dpsurv.sh KIRC

See python trainer/train_dpsurv.py --help for all options.
"""

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from downstream.dpsurv.models import ENNreg_init_cosine, mixture_ENNreg_new, mixture_input_dim_from_mean_dim
from downstream.dpsurv.losses import Mixture_Evidential_nll_Loss, evaluate_nll_batch_survival
from downstream.dpsurv.data import GMMEmbeddingDataset, build_df, collate_flat


DEFAULT_DATASETS = ["BLCA", "BRCA", "KIRC", "LUAD", "UCEC"]
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

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DPsurv: nested cross-validation training. "
            "Performs inner K selection then full outer-train retraining."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: directory containing this script).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset short names. Defaults to BLCA BRCA KIRC LUAD UCEC.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Outer folds to evaluate.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Directory to save outputs (default: results/ next to this script).",
    )
    parser.add_argument(
        "--splits_root",
        type=Path,
        default=None,
        help="Directory containing <DATASET>_overall_survival_k=<fold> folders.",
    )
    parser.add_argument(
        "--embedding_fname",
        type=str,
        default=DEFAULT_EMBEDDING_FNAME,
        help="Embedding filename inside each fold's embeddings/ directory.",
    )
    parser.add_argument("--k_values",           nargs="+", type=int,  default=[1, 2, 3, 4])
    parser.add_argument("--inner_val_fraction", type=float, default=0.15)
    parser.add_argument("--n_label_bins",       type=int,   default=8)
    parser.add_argument("--max_epochs",         type=int,   default=50)
    parser.add_argument("--min_epochs",         type=int,   default=5)
    parser.add_argument("--patience",           type=int,   default=5)
    parser.add_argument("--batch_size",         type=int,   default=32)
    parser.add_argument("--eval_batch_size",    type=int,   default=1000)
    parser.add_argument("--num_workers",        type=int,   default=0)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--weight_decay",       type=float, default=2e-4)
    parser.add_argument("--weight",             type=float, default=0.5,
                        help="Lambda: weight between BEL and PL survival formulations.")
    parser.add_argument("--alpha",              type=float, default=0.5,
                        help="Weight for the uncensored NLL term.")
    parser.add_argument("--xi",                 type=float, default=0.0,
                        help="Regularisation weight for eta penalty.")
    parser.add_argument("--rho",                type=float, default=0.0,
                        help="Regularisation weight for gamma penalty.")
    parser.add_argument("--warmup_ratio",       type=float, default=0.1)
    parser.add_argument("--eta_min",            type=float, default=1e-6)
    parser.add_argument("--gamma_scale",        type=float, default=0.5)
    parser.add_argument("--prob_thresh",        type=float, default=0.01)
    parser.add_argument("--kmeans_nstart",      type=int,   default=100)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset.strip().upper(), dataset.strip())


def fold_dir(splits_root: Path, dataset_name: str, fold: int) -> Path:
    return splits_root / f"{dataset_name}_overall_survival_k={fold}"


def metrics_key(metrics: Dict, k_value: int) -> tuple:
    return (-float(metrics["c_index"]), float(metrics["nbll"]), float(metrics["ibs"]), float(k_value))


def has_any_event(df: pd.DataFrame) -> bool:
    return bool(((1.0 - df["dss_censorship"].astype(float)) > 0.5).any())


def deduplicate_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["dss_survival_days", "dss_censorship"])
    df = df.drop_duplicates(subset=["case_id"])
    return df.reset_index(drop=True)


def robust_quantile_bins(survival_days: pd.Series, n_bins: int) -> np.ndarray:
    values = survival_days.dropna().astype(float)
    if values.empty:
        return np.array([1e-6, 1e6], dtype=np.float32)
    max_bins = max(1, min(int(n_bins), int(values.nunique())))
    for candidate_bins in range(max_bins, 0, -1):
        try:
            _, bins = pd.qcut(values, q=candidate_bins, retbins=True, labels=False, duplicates="drop")
            bins = np.unique(np.asarray(bins, dtype=np.float64))
            if bins.size >= 2:
                bins[0]  = min(bins[0],  1e-6)
                bins[-1] = max(bins[-1], 1e6)
                return bins.astype(np.float32)
        except ValueError:
            continue
    min_v = float(max(values.min(), 1e-6))
    max_v = float(max(values.max(), min_v + 1.0))
    return np.array([min(min_v, 1e-6), max(max_v, 1e6)], dtype=np.float32)


def assign_discrete_labels(df: pd.DataFrame, qbins: np.ndarray) -> torch.Tensor:
    labels = pd.cut(df["dss_survival_days"].astype(float), bins=qbins, labels=False, include_lowest=True)
    labels = labels.fillna(len(qbins) - 2).astype(np.int64)
    return torch.tensor(labels.to_numpy(), dtype=torch.float32)


def build_dataset_with_bins(
    df: pd.DataFrame,
    n_label_bins: int,
    qbins: Optional[np.ndarray] = None,
) -> Tuple[GMMEmbeddingDataset, torch.Tensor]:
    dataset = GMMEmbeddingDataset(df)
    if qbins is None:
        uncensored = df[df["dss_censorship"].astype(float) == 0.0]
        source = uncensored if len(uncensored) >= 2 else df
        qbins = robust_quantile_bins(source["dss_survival_days"], n_label_bins)
    qbins_t = torch.tensor(qbins, dtype=torch.float32)
    dataset.labels = assign_discrete_labels(df, qbins)
    return dataset, qbins_t


def stratify_labels(df: pd.DataFrame, n_time_bins: int = 4) -> List[Optional[pd.Series]]:
    event_labels = (1.0 - df["dss_censorship"].astype(float)).astype(int).astype(str)
    candidates: List[Optional[pd.Series]] = []
    max_bins = min(n_time_bins, int(df["dss_survival_days"].nunique()))
    if max_bins >= 2:
        try:
            time_bins = pd.qcut(df["dss_survival_days"].astype(float), q=max_bins, labels=False, duplicates="drop")
            combo = event_labels + "_" + time_bins.astype(int).astype(str)
            if combo.value_counts().min() >= 2:
                candidates.append(combo)
        except ValueError:
            pass
    if event_labels.value_counts().min() >= 2:
        candidates.append(event_labels)
    candidates.append(None)
    return candidates


def split_inner_train_val(
    train_df: pd.DataFrame,
    val_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    indices = np.arange(len(train_df))
    best_split: Optional[Tuple] = None
    for stratify in stratify_labels(train_df):
        n_val = max(1, int(round(len(train_df) * val_fraction)))
        if stratify is not None and pd.Series(stratify).nunique() > n_val:
            continue
        for offset in range(32):
            try:
                tr_idx, va_idx = train_test_split(
                    indices,
                    test_size=val_fraction,
                    random_state=SEED + offset,
                    shuffle=True,
                    stratify=stratify if stratify is None else stratify.iloc[indices],
                )
            except ValueError:
                continue
            inner_tr = train_df.iloc[tr_idx].reset_index(drop=True)
            inner_va = train_df.iloc[va_idx].reset_index(drop=True)
            best_split = (inner_tr, inner_va)
            if has_any_event(inner_tr) and has_any_event(inner_va):
                return inner_tr, inner_va
    if best_split is None:
        raise RuntimeError("Failed to create an inner train/validation split.")
    return best_split


def make_loader(dataset: GMMEmbeddingDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_flat, num_workers=num_workers, pin_memory=False)


def load_fold_frames(
    splits_root: Path,
    dataset_name: str,
    fold: int,
    embedding_fname: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fdir = fold_dir(splits_root, dataset_name, fold)
    embed_path = fdir / "embeddings" / embedding_fname
    train_csv  = fdir / "train.csv"
    test_csv   = fdir / "test.csv"

    if not embed_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {embed_path}")
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing train/test CSVs under: {fdir}")

    with embed_path.open("rb") as f:
        emb = pickle.load(f)

    train_df = deduplicate_and_filter(build_df(emb["train"], pd.read_csv(train_csv)))
    test_df  = deduplicate_and_filter(build_df(emb["test"],  pd.read_csv(test_csv)))
    return train_df, test_df


def infer_input_dim(train_df: pd.DataFrame) -> int:
    mean = np.asarray(train_df["mean"].iloc[0])
    cov  = np.asarray(train_df["cov"].iloc[0])
    if mean.shape != cov.shape:
        raise ValueError(f"Mean/Cov shape mismatch: {mean.shape} vs {cov.shape}")
    return 1 + 2 * int(mean.shape[1])


def init_prototypes(
    train_df: pd.DataFrame,
    k_value: int,
    nstart: int,
    gamma_scale: float,
    prob_thresh: float,
) -> List[Dict]:
    prob_np = np.stack(train_df["prob"].to_numpy())
    mean_np = np.stack(train_df["mean"].to_numpy())
    log_surv = torch.log(torch.tensor(train_df["dss_survival_days"].values, dtype=torch.float32))
    n_components = prob_np.shape[1]
    return [
        ENNreg_init_cosine(
            torch.tensor(prob_np[:, i], dtype=torch.float32),
            torch.tensor(mean_np[:, i, :], dtype=torch.float32),
            log_surv,
            k_value,
            nstart=nstart,
            c=gamma_scale,
            prob_thresh=prob_thresh,
        )
        for i in range(n_components)
    ]


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    beta_params, w_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "beta" in name:
            beta_params.append(param)
        elif "w" in name:
            w_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.AdamW(
        [{"params": beta_params,  "weight_decay": weight_decay},
         {"params": w_params,     "weight_decay": weight_decay},
         {"params": other_params, "weight_decay": weight_decay}],
        lr=lr,
    )


def build_scheduler(
    optimizer, epochs: int, steps_per_epoch: int, warmup_ratio: float, eta_min: float
) -> Optional[object]:
    total_steps  = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = min(max(1, int(round(total_steps * warmup_ratio))), total_steps)
    if total_steps <= 1:
        return None
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def build_model_bundle(train_df, k_value, qbins, epochs, args, device, steps_per_epoch):
    protos   = init_prototypes(train_df, k_value, args.kmeans_nstart, args.gamma_scale, args.prob_thresh)
    input_dim = infer_input_dim(train_df)
    model = mixture_ENNreg_new(input_dim=input_dim, prototype_list=protos, num_models=len(protos)).to(device)
    model.reset_parameters(protos, device)
    loss_fn   = Mixture_Evidential_nll_Loss(qbins=qbins, alpha=args.alpha, eps=1e-7, reduction="mean",
                                            lambd=args.weight, xi=args.xi, rho=args.rho)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, epochs, steps_per_epoch, args.warmup_ratio, args.eta_min)
    return model, loss_fn, optimizer, scheduler


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, desc) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        input_feat = torch.cat([batch["prob"].unsqueeze(2), batch["mean"], batch["cov"]], dim=2)
        loss = loss_fn(model(input_feat, batch["prob"]), batch["labels"], batch["censorship"], batch["prob"])["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach().cpu())
        n += 1
    return total / max(1, n)


def eval_metrics(model, loader, qbins, weight, device) -> Dict:
    raw = evaluate_nll_batch_survival(model, loader, device=device, qbins=qbins, weight=weight)
    return {k: float(v) for k, v in raw.items()}


def should_update_best(cur: Dict, best: Optional[Dict], tol: float = 1e-8) -> bool:
    if best is None:
        return True
    if cur["c_index"] > best["c_index"] + tol:
        return True
    if abs(cur["c_index"] - best["c_index"]) <= tol:
        if cur["nbll"] < best["nbll"] - tol:
            return True
        if abs(cur["nbll"] - best["nbll"]) <= tol and cur["ibs"] < best["ibs"] - tol:
            return True
    return False


def clear_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# Inner selection
# ---------------------------------------------------------------------------

def run_inner_selection(outer_train_df, dataset_name, fold, args, device):
    inner_tr, inner_va = split_inner_train_val(outer_train_df, args.inner_val_fraction)

    tr_ds, qbins = build_dataset_with_bins(inner_tr, args.n_label_bins)
    va_ds, _     = build_dataset_with_bins(inner_va, args.n_label_bins, qbins.cpu().numpy())

    tr_loader = make_loader(tr_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    va_loader = make_loader(va_ds, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    history_rows, selection_rows = [], []
    best_overall, best_overall_key = None, None

    for k in args.k_values:
        set_seed(SEED + fold * 997 + k * 53)
        model, loss_fn, optimizer, scheduler = build_model_bundle(
            inner_tr, k, qbins, args.max_epochs, args, device, len(tr_loader)
        )
        best_k, best_epoch_k, patience_ctr = None, None, 0

        for epoch in range(1, args.max_epochs + 1):
            tr_loss  = train_one_epoch(model, tr_loader, loss_fn, optimizer, scheduler, device,
                                       f"{dataset_name} fold={fold} K={k} epoch={epoch}")
            va_met   = eval_metrics(model, va_loader, qbins, args.weight, device)
            history_rows.append({"dataset": dataset_name, "fold": fold,
                                  "k": k, "epoch": epoch, "train_loss": tr_loss,
                                  **{f"val_{m}": v for m, v in va_met.items()}})

            if should_update_best(va_met, best_k):
                best_k, best_epoch_k, patience_ctr = va_met, epoch, 0
            elif epoch >= args.min_epochs:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    break

        row = {"dataset": dataset_name, "fold": fold, "k": k,
               "best_epoch": best_epoch_k, **{f"val_{m}": v for m, v in best_k.items()}}
        selection_rows.append(row)

        key = (-float(best_k["c_index"]), float(best_k["nbll"]), float(best_k["ibs"]), float(k))
        if best_overall is None or key < best_overall_key:
            best_overall, best_overall_key = {**row, "inner_train_size": len(inner_tr), "inner_val_size": len(inner_va)}, key

        del model, loss_fn, optimizer, scheduler
        gc.collect(); clear_cache()

    split_summary = pd.DataFrame([{
        "dataset": dataset_name, "fold": fold,
        "inner_train_size": len(inner_tr), "inner_val_size": len(inner_va),
        "inner_train_events": int((1.0 - inner_tr["dss_censorship"].astype(float)).sum()),
        "inner_val_events":   int((1.0 - inner_va["dss_censorship"].astype(float)).sum()),
    }])
    return best_overall, pd.DataFrame(selection_rows), pd.DataFrame(history_rows), split_summary


# ---------------------------------------------------------------------------
# Final training
# ---------------------------------------------------------------------------

def run_final_training(outer_train_df, test_df, selected_k, selected_epoch, dataset_name, fold, args, device):
    set_seed(SEED + fold * 1231 + selected_k * 61)
    tr_ds, qbins = build_dataset_with_bins(outer_train_df, args.n_label_bins)
    te_ds        = GMMEmbeddingDataset(test_df)

    tr_loader = make_loader(tr_ds, args.batch_size,      shuffle=True,  num_workers=args.num_workers)
    te_loader = make_loader(te_ds, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model, loss_fn, optimizer, scheduler = build_model_bundle(
        outer_train_df, selected_k, qbins, selected_epoch, args, device, len(tr_loader)
    )
    epoch_rows = []
    for epoch in range(1, selected_epoch + 1):
        tr_loss = train_one_epoch(model, tr_loader, loss_fn, optimizer, scheduler, device,
                                  f"{dataset_name} fold={fold} final epoch={epoch}")
        epoch_rows.append({"dataset": dataset_name, "fold": fold,
                            "k": selected_k, "epoch": epoch, "train_loss": tr_loss})

    test_met = eval_metrics(model, te_loader, qbins, args.weight, device)
    del model, loss_fn, optimizer, scheduler
    gc.collect(); clear_cache()

    return {
        "dataset": dataset_name, "fold": fold,
        "selected_k": selected_k, "selected_epoch": selected_epoch,
        "train_size":   len(outer_train_df), "test_size": len(test_df),
        "train_events": int((1.0 - outer_train_df["dss_censorship"].astype(float)).sum()),
        "test_events":  int((1.0 - test_df["dss_censorship"].astype(float)).sum()),
        **test_met,
        "final_train_curve": epoch_rows,
    }


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset_name: str, args: argparse.Namespace, device: torch.device) -> None:
    out_dir = args.results_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, all_sel, all_hist, all_splits, curves = [], [], [], [], []

    for fold in args.folds:
        print(f"\n{'='*60}")
        print(f"  Dataset={dataset_name}  Fold={fold}")
        print(f"{'='*60}")

        outer_tr, test_df = load_fold_frames(args.splits_root, dataset_name, fold, args.embedding_fname)
        best_cfg, sel_df, hist_df, split_df = run_inner_selection(outer_tr, dataset_name, fold, args, device)
        result = run_final_training(outer_tr, test_df, int(best_cfg["k"]), int(best_cfg["best_epoch"]),
                                    dataset_name, fold, args, device)

        print(f"  Selected K={result['selected_k']} epoch={result['selected_epoch']}"
              f" | C-index={result['c_index']:.4f} C-index_td={result['c_index_td']:.4f}"
              f" IBS={result['ibs']:.4f} NBLL={result['nbll']:.4f}")

        fold_rows.append({
            "dataset": dataset_name, "fold": fold,
            "selected_k": result["selected_k"], "selected_epoch": result["selected_epoch"],
            "inner_val_c_index":    float(best_cfg["val_c_index"]),
            "inner_val_c_index_td": float(best_cfg["val_c_index_td"]),
            "inner_val_ibs":        float(best_cfg["val_ibs"]),
            "inner_val_nbll":       float(best_cfg["val_nbll"]),
            "train_size":   result["train_size"],   "test_size":   result["test_size"],
            "train_events": result["train_events"], "test_events": result["test_events"],
            "c_index":      result["c_index"],      "c_index_td":  result["c_index_td"],
            "ibs":          result["ibs"],          "nbll":        result["nbll"],
        })
        all_sel.append(sel_df); all_hist.append(hist_df); all_splits.append(split_df)
        curves.extend(result["final_train_curve"])

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.concat(all_sel,    ignore_index=True).to_csv(out_dir / "inner_selection_summary.csv", index=False)
    pd.concat(all_hist,   ignore_index=True).to_csv(out_dir / "inner_training_history.csv",  index=False)
    pd.concat(all_splits, ignore_index=True).to_csv(out_dir / "inner_split_summary.csv",     index=False)
    pd.DataFrame(curves).to_csv(out_dir / "final_train_curve.csv", index=False)

    metric_cols = ["c_index", "c_index_td", "ibs", "nbll"]
    summary = {
        "dataset": dataset_name,
        "folds": list(args.folds), "k_values": list(args.k_values),
        **{f"{m}_mean": float(fold_df[m].mean()) for m in metric_cols},
        **{f"{m}_std":  float(fold_df[m].std(ddof=1)) if len(fold_df) > 1 else 0.0 for m in metric_cols},
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nPer-fold results:")
    print(fold_df[["fold", "selected_k", "c_index", "c_index_td", "ibs", "nbll"]].to_string(index=False))
    print("\nAggregate:")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("folds", "k_values")}, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.base_dir = args.base_dir.resolve()

    if args.splits_root is None:
        args.splits_root = args.base_dir / "data" / "splits"
    args.splits_root = args.splits_root.resolve()

    if args.results_dir is None:
        args.results_dir = args.base_dir / "results"
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device} | Datasets: {args.datasets}")

    for ds in args.datasets:
        run_dataset(resolve_dataset_name(ds), args, device)


if __name__ == "__main__":
    main()
