"""
SSPM on all 9 modern backbones (3 models x 3 feature sets) for all 4 datasets.

Hyperparameters strictly match run_feature_comparison.py so the SSPM-vs-baseline
comparison is "same model, same features, only training strategy differs".

For each (dataset, horizon, backbone):
  1. Masked pretraining (30 epochs)
  2. Alpha search: 6 dataset-specific alphas, select best by validation loss
  3. Evaluate best model on test set

Outputs:
  predictions/{ds}/H{h}_SSPM_{backbone}.npz  (q10/q50/q90/true)
  checkpoints/{ds}/H{h}_SSPM_{backbone}.pt   (model weights)
  sspm_all_backbones_results.json            (metrics + selected alpha)

Usage:
  python run_sspm_all_backbones.py                  # all datasets
  python run_sspm_all_backbones.py --dataset MIT
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json
from common.config import PROJECT_ROOT

from train_msra import set_seed
from run_baselines import (DLinear, PatchTST, iTransformer,
                           train_one_model, evaluate_on_test)
from msra.physics_constraints import pinball_loss
from run_feature_comparison import (DATASET_CONFIGS, HPARAMS, LOOKBACK,
                                    HORIZONS, SEED, QUANTILE_LEVELS,
                                    N_QUANTILES, build_feature_sets,
                                    load_and_engineer,
                                    build_dataset_for_feature_set,
                                    make_model)

logger = get_logger("sspm_all")

# Dataset-specific alpha search grid (user-specified)
ALPHA_GRID = {
    "MIT":       [0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
    "Helios":    [0.05, 1.0, 2.0, 3.0, 5.0, 7.0],
    "Alibaba":   [0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
    "Inference": [0.005, 0.01, 0.02, 0.05, 0.07, 0.1],
}

PRETRAIN_EPOCHS = 30
MASK_RATIO = 0.3


def pretrain_masked(model, train_X, hparams, device, epochs=PRETRAIN_EPOCHS,
                    mask_ratio=MASK_RATIO):
    """Self-supervised masked reconstruction pretraining.

    Mask mask_ratio of input features, predict masked values from the rest.
    """
    X_t = torch.from_numpy(train_X).float()
    B_total, L, C = X_t.shape

    # Flex reconstruction head
    dummy = model(X_t[:2].to(device))
    out_dim = dummy.reshape(2, -1).shape[1]
    recon = nn.Linear(out_dim, L * C).to(device)

    params = list(model.parameters()) + list(recon.parameters())
    opt = torch.optim.AdamW(params, lr=hparams["lr"] * 2, weight_decay=1e-5)
    dl = torch.utils.data.DataLoader(X_t, batch_size=hparams["batch_size"],
                                      shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb in dl:
            xb = xb.to(device)
            B = xb.shape[0]
            mask = torch.rand(B, L, C, device=device) < mask_ratio
            xb_masked = xb.clone()
            xb_masked[mask] = 0.0
            out = model(xb_masked).reshape(B, -1)
            rec = recon(out).view(B, L, C)
            loss = nn.functional.mse_loss(rec[mask], xb[mask])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    del recon
    return model


def train_mixup(model, data, hparams, alpha, device):
    """Finetune with temporal mixup. Returns (model, best_val_loss)."""
    ql = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32).to(device)
    tr_X = torch.from_numpy(data["train_X"]).float()
    tr_y = torch.from_numpy(data["train_y"]).float()
    va_X = torch.from_numpy(data["val_X"]).float()
    va_y = torch.from_numpy(data["val_y"]).float()

    tr_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_X, tr_y),
        batch_size=hparams["batch_size"], shuffle=True, drop_last=True)
    va_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(va_X, va_y),
        batch_size=hparams["batch_size"])

    opt = torch.optim.AdamW(model.parameters(), lr=hparams["lr"],
                             weight_decay=hparams["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=hparams["lr"],
        total_steps=len(tr_dl) * hparams["epochs"], pct_start=0.2)

    best_val = float("inf")
    best_st = None
    no_imp = 0

    for _ in range(hparams["epochs"]):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            B = xb.shape[0]
            lam = np.random.beta(alpha, alpha)
            idx = torch.randperm(B, device=device)
            xm = lam * xb + (1 - lam) * xb[idx]
            ym = lam * yb + (1 - lam) * yb[idx]
            loss = pinball_loss(model(xm), ym, ql)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip"])
            opt.step()
            sched.step()

        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl += pinball_loss(model(xb), yb, ql).item() * len(xb)
                nv += len(xb)
        vl /= max(nv, 1)

        if vl < best_val:
            best_val = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= hparams["patience"]:
                break

    if best_st:
        model.load_state_dict(best_st)
    return model, best_val


def run_one(ds_name, horizon, model_name, fs_name, df, feature_sets,
            target, device):
    """Run SSPM on a single (dataset, horizon, backbone) combination."""
    fs_cols = feature_sets[fs_name]
    data = build_dataset_for_feature_set(df, fs_cols, horizon, target)
    nf = data["train_X"].shape[2]

    # Phase 1: Pretrain (done once, reused across alpha search)
    set_seed(SEED)
    model_pre = make_model(model_name, nf, horizon).to(device)
    t0 = time.time()
    model_pre = pretrain_masked(model_pre, data["train_X"], HPARAMS, device)
    pretrain_sd = {k: v.cpu().clone() for k, v in model_pre.state_dict().items()}
    del model_pre
    torch.cuda.empty_cache()
    t_pre = time.time() - t0

    # Phase 2: Alpha search with mixup
    alphas = ALPHA_GRID[ds_name]
    best_val = float("inf")
    best_alpha = None
    best_sd = None

    for alpha in alphas:
        set_seed(SEED)
        m = make_model(model_name, nf, horizon).to(device)
        m.load_state_dict(pretrain_sd)
        m, val_loss = train_mixup(m, data, HPARAMS, alpha, device)
        if val_loss < best_val:
            best_val = val_loss
            best_alpha = alpha
            best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        del m
        torch.cuda.empty_cache()

    # Final evaluation on test set
    m = make_model(model_name, nf, horizon).to(device)
    m.load_state_dict(best_sd)
    metrics, pred, true = evaluate_on_test(
        m, data["test_X"], data["test_y"], data["norm_params"],
        device, return_predictions=True)
    metrics["alpha"] = best_alpha
    metrics["val_loss"] = round(best_val, 6)
    metrics["pretrain_time_s"] = round(t_pre, 1)
    return metrics, pred, true, best_sd


def main(ds_name="all"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    out_dir = os.path.join(results_path("case10c_cluster_forecast"), "data")

    model_names = ["DLinear", "PatchTST", "iTransformer"]
    feat_set_names = ["Calendar", "All", "Workload"]

    ds_list = [ds_name] if ds_name != "all" else list(DATASET_CONFIGS.keys())

    results = {}
    t_start = time.time()

    for ds in ds_list:
        cfg = DATASET_CONFIGS[ds]
        logger.info(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")
        df, feature_sets, target = load_and_engineer(cfg)

        pred_dir = os.path.join(out_dir, "predictions", ds)
        ckpt_dir = os.path.join(out_dir, "checkpoints", ds)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        results[ds] = {}

        for horizon in HORIZONS:
            logger.info(f"\n--- {ds} H={horizon} ---")
            results[ds][str(horizon)] = {}

            for model_name in model_names:
                for fs_name in feat_set_names:
                    backbone = f"{model_name}-{fs_name}"
                    key = f"SSPM_{backbone}"
                    t0 = time.time()

                    metrics, pred, true, best_sd = run_one(
                        ds, horizon, model_name, fs_name,
                        df, feature_sets, target, device)
                    elapsed = time.time() - t0

                    results[ds][str(horizon)][backbone] = metrics

                    # Save predictions
                    np.savez_compressed(
                        os.path.join(pred_dir, f"H{horizon}_{key}.npz"),
                        pred_q10=pred[:, :, 0],
                        pred_q50=pred[:, :, 1],
                        pred_q90=pred[:, :, 2],
                        true=true)
                    # Save checkpoint
                    torch.save(best_sd,
                               os.path.join(ckpt_dir, f"H{horizon}_{key}.pt"))

                    logger.info(f"  {backbone}: R²={metrics['r2']:.4f}  "
                                f"RMSE={metrics['rmse']}  α={metrics['alpha']}  "
                                f"({elapsed:.1f}s)")

    # Save aggregated results
    results["total_time_s"] = round(time.time() - t_start, 1)
    out_path = os.path.join(out_dir, "sspm_all_backbones_results.json")
    save_json(results, out_path)
    logger.info(f"\nSaved to {out_path}")
    logger.info(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASET_CONFIGS.keys()) + ["all"])
    args = parser.parse_args()
    main(args.dataset)
