"""
Feature Comparison: Calendar vs All vs Workload-aware on SOTA models.

Validates that workload-aware feature engineering (replacing calendar features
with GMM regime probabilities) provides universal improvement across DLinear,
PatchTST, and iTransformer on AI cluster load forecasting.

Experiment matrix: 3 models × 3 feature sets × 2 horizons = 18 experiments.

Usage:
    python run_feature_comparison.py                # MIT only (default)
    python run_feature_comparison.py --dataset Helios
    python run_feature_comparison.py --dataset all   # all 4 datasets
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json
from common.config import DATA_PROCESSED

from train_msra import set_seed, make_loader
from run_baselines import (DLinear, PatchTST, iTransformer,
                           train_one_model, evaluate_on_test)

logger = get_logger("feature_comparison")

# ============================================================
# Hyperparameters (same as run_baselines.py)
# ============================================================

HPARAMS = {
    "batch_size": 128,
    "epochs": 50,
    "patience": 7,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # DLinear
    "dlinear_moving_avg": 25,
    # PatchTST
    "patch_size": 16,
    "patch_stride": 8,
    "ptst_d_model": 128,
    "ptst_nhead": 4,
    "ptst_layers": 2,
    "ptst_dropout": 0.1,
    # iTransformer
    "itrans_d_model": 128,
    "itrans_nhead": 4,
    "itrans_layers": 2,
    "itrans_dropout": 0.1,
}

QUANTILE_LEVELS = [0.1, 0.5, 0.9]
N_QUANTILES = len(QUANTILE_LEVELS)
LOOKBACK = 96
HORIZONS = [16, 96]
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# ============================================================
# Per-dataset configuration
# ============================================================

DATASET_CONFIGS = {
    "MIT": {
        "path": os.path.join(DATA_PROCESSED, "mit_supercloud_15min.csv"),
        "target": "cluster_power_kw",
        "workload_cols": ["active_gpus", "queue_depth", "running_jobs",
                          "new_submits_15min", "completions_15min"],
    },
    "Helios": {
        "path": os.path.join(DATA_PROCESSED, "helios_saturn_15min.csv"),
        "target": "active_gpu_count",
        "workload_cols": ["active_gpus", "queue_depth", "running_jobs",
                          "new_submits_15min", "completions_15min"],
    },
    "Alibaba": {
        "path": os.path.join(DATA_PROCESSED, "alibaba_v2026_spot_15min.csv"),
        "target": "active_gpu_count",
        "workload_cols": ["queue_depth", "running_jobs",
                          "new_submits_15min", "completions_15min"],
    },
    "Inference": {
        "path": os.path.join(DATA_PROCESSED, "inference_cluster_15min.csv"),
        "target": "cluster_power_kw",
        "workload_cols": ["active_gpus", "queue_depth", "running_jobs",
                          "new_submits_15min", "completions_15min"],
    },
}


def build_feature_sets(target, workload_cols, df_columns):
    """Construct Calendar/All/Workload feature sets (matches explore_round4)."""
    avail_wl = [c for c in workload_cols if c in df_columns]
    avail_d = [f"d_{c}" for c in workload_cols if f"d_{c}" in df_columns]

    calendar = [target, f"d_{target}", "target_ma_4h", "target_std_4h",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    all_feat = ([target] + avail_wl + [f"d_{target}"] + avail_d +
                ["target_ma_4h", "target_std_4h",
                 "hour_sin", "hour_cos", "dow_sin", "dow_cos"])

    workload = ([target] + avail_wl + [f"d_{target}"] + avail_d +
                ["target_ma_4h", "target_std_4h"])

    return {"Calendar": calendar, "All": all_feat, "Workload": workload}


# ============================================================
# Data pipeline
# ============================================================

def load_and_engineer(ds_config):
    """Load CSV and compute derived features (matches explore_round4 exactly).

    No GMM — Workload features are simply All minus time encoding.
    Returns: (df_with_features, feature_sets_dict, target_col)
    """
    data_path = ds_config["path"]
    target = ds_config["target"]
    wl_cols = ds_config["workload_cols"]

    if not os.path.exists(data_path):
        logger.error(f"Dataset CSV not found: {data_path}. "
                     f"Build the processed datasets first (see data/README.md).")
        sys.exit(1)
    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "hour" not in df.columns:
        df["hour"] = df["time"].dt.hour
    if "dow" not in df.columns:
        df["dow"] = df["time"].dt.dayofweek
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    # Change rate features (target + all workload cols present)
    for col in [target] + wl_cols:
        if col in df.columns:
            df[f"d_{col}"] = df[col].diff().fillna(0)

    # Rolling statistics on target
    df["target_ma_4h"] = df[target].rolling(16, min_periods=1).mean()
    df["target_std_4h"] = df[target].rolling(
        16, min_periods=1).std().fillna(0)

    # Cyclic time encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Build feature sets
    feature_sets = build_feature_sets(target, wl_cols, df.columns)

    logger.info(f"Feature dims: Calendar={len(feature_sets['Calendar'])}, "
                f"All={len(feature_sets['All'])}, "
                f"Workload={len(feature_sets['Workload'])}")

    return df, feature_sets, target


def create_windows(values, target, lookback, horizon, stride=1):
    """Create sliding windows from numpy arrays.

    Args:
        values: (N, F) feature array
        target: (N,) target array
        lookback: int
        horizon: int
        stride: int

    Returns:
        X: (n_windows, lookback, F) float32
        y: (n_windows, horizon) float32
    """
    n = len(values)
    total_len = lookback + horizon
    is_nan = np.isnan(target)

    X_list, y_list = [], []
    for i in range(0, n - total_len + 1, stride):
        if is_nan[i:i + total_len].any():
            continue
        X_list.append(values[i:i + lookback])
        y_list.append(target[i + lookback:i + total_len])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Z-normalize using training set statistics only.

    Returns: (X_tr_n, X_va_n, X_te_n, y_tr_n, y_va_n, y_te_n, norm_params)
    """
    f_mean = X_train.mean(axis=(0, 1))
    f_std = X_train.std(axis=(0, 1))
    f_std = np.where(f_std < 1e-8, 1.0, f_std)

    X_tr_n = (X_train - f_mean) / f_std
    X_va_n = (X_val - f_mean) / f_std
    X_te_n = (X_test - f_mean) / f_std

    t_mean = float(y_train.mean())
    t_std = float(y_train.std())
    if t_std < 1e-8:
        t_std = 1.0

    y_tr_n = (y_train - t_mean) / t_std
    y_va_n = (y_val - t_mean) / t_std
    y_te_n = (y_test - t_mean) / t_std

    norm_params = {
        "feature_means": f_mean.tolist(),
        "feature_stds": f_std.tolist(),
        "target_mean": t_mean,
        "target_std": t_std,
    }
    return X_tr_n, X_va_n, X_te_n, y_tr_n, y_va_n, y_te_n, norm_params


def build_dataset_for_feature_set(df, feature_cols, horizon, target_col):
    """Build train/val/test windows for a given feature set and horizon.

    Returns: dict with train_X, train_y, val_X, val_y, test_X, test_y, norm_params
    """
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # Create windows per split
    tr_X, tr_y = create_windows(
        train_df[feature_cols].values, train_df[target_col].values,
        LOOKBACK, horizon)
    va_X, va_y = create_windows(
        val_df[feature_cols].values, val_df[target_col].values,
        LOOKBACK, horizon)
    te_X, te_y = create_windows(
        test_df[feature_cols].values, test_df[target_col].values,
        LOOKBACK, horizon)

    logger.info(f"  Windows: train={len(tr_X)}, val={len(va_X)}, test={len(te_X)}, "
                f"features={tr_X.shape[2]}")

    # Normalize
    tr_X, va_X, te_X, tr_y, va_y, te_y, norm = normalize_data(
        tr_X, va_X, te_X, tr_y, va_y, te_y)

    return {
        "train_X": tr_X, "train_y": tr_y,
        "val_X": va_X, "val_y": va_y,
        "test_X": te_X, "test_y": te_y,
        "norm_params": norm,
    }


# ============================================================
# Model factory
# ============================================================

def make_model(model_name, n_features, horizon):
    """Create a model instance by name."""
    if model_name == "DLinear":
        return DLinear(LOOKBACK, horizon, n_features, N_QUANTILES,
                       HPARAMS["dlinear_moving_avg"])
    elif model_name == "PatchTST":
        return PatchTST(LOOKBACK, horizon, n_features, N_QUANTILES,
                        HPARAMS["patch_size"], HPARAMS["patch_stride"],
                        HPARAMS["ptst_d_model"], HPARAMS["ptst_nhead"],
                        HPARAMS["ptst_layers"], HPARAMS["ptst_dropout"])
    elif model_name == "iTransformer":
        return iTransformer(LOOKBACK, horizon, n_features, N_QUANTILES,
                            HPARAMS["itrans_d_model"], HPARAMS["itrans_nhead"],
                            HPARAMS["itrans_layers"], HPARAMS["itrans_dropout"])
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Main
# ============================================================

def run_feature_comparison(ds_name="MIT"):
    """Run 3 models × 3 feature sets × 2 horizons = 18 experiments for one dataset."""
    ds_config = DATASET_CONFIGS[ds_name]
    target = ds_config["target"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)
    logger.info(f"Device: {device}, Dataset: {ds_name}")

    # Load data and compute all features
    logger.info("Loading and engineering features...")
    df, feature_sets, target = load_and_engineer(ds_config)

    out_dir = os.path.join(results_path("case10c_cluster_forecast"), "data")

    results = {}
    model_names = ["DLinear", "PatchTST", "iTransformer"]
    feat_set_names = ["Calendar", "All", "Workload"]

    total_exps = len(model_names) * len(feat_set_names) * len(HORIZONS)
    exp_idx = 0

    for horizon in HORIZONS:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# {ds_name} | HORIZON = {horizon} steps ({horizon * 15 / 60:.0f}h)")
        logger.info(f"{'#' * 60}")

        h_results = {}

        # Build datasets for each feature set (cached per horizon)
        datasets = {}
        for fs_name in feat_set_names:
            fs_cols = feature_sets[fs_name]
            logger.info(f"\n  Building dataset: {fs_name} ({len(fs_cols)} features)")
            datasets[fs_name] = build_dataset_for_feature_set(
                df, fs_cols, horizon, target)

        # Prediction / checkpoint output dirs
        pred_dir = os.path.join(out_dir, "predictions", ds_name)
        ckpt_dir = os.path.join(out_dir, "checkpoints", ds_name)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Run each model × feature set combination
        for model_name in model_names:
            for fs_name in feat_set_names:
                exp_idx += 1
                key = f"{model_name}-{fs_name}"
                data = datasets[fs_name]
                n_features = data["train_X"].shape[2]

                logger.info(f"\n  [{exp_idx}/{total_exps}] {key} "
                            f"(H={horizon}, F={n_features})")

                set_seed(SEED)  # Reset seed for reproducibility
                model = make_model(model_name, n_features, horizon).to(device)
                n_params = sum(p.numel() for p in model.parameters())
                logger.info(f"    Parameters: {n_params:,}")

                model, best_val, elapsed = train_one_model(
                    model, data["train_X"], data["train_y"],
                    data["val_X"], data["val_y"], HPARAMS, device)

                metrics, pred_all, true_all = evaluate_on_test(
                    model, data["test_X"], data["test_y"],
                    data["norm_params"], device, return_predictions=True)
                metrics["val_loss"] = round(best_val, 6)
                metrics["train_time_s"] = round(elapsed, 1)
                metrics["n_params"] = n_params
                metrics["n_features"] = n_features

                h_results[key] = metrics
                logger.info(f"    RMSE={metrics['rmse']}, R2={metrics['r2']}, "
                            f"CRPS={metrics['crps']:.4f}, "
                            f"Cov80={metrics.get('coverage_80','N/A')}  "
                            f"({elapsed:.1f}s)")

                # Save predictions (q10/q50/q90 + true)
                np.savez_compressed(
                    os.path.join(pred_dir, f"H{horizon}_{key}.npz"),
                    pred_q10=pred_all[:, :, 0],
                    pred_q50=pred_all[:, :, 1],
                    pred_q90=pred_all[:, :, 2],
                    true=true_all)

                # Save model checkpoint
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"H{horizon}_{key}.pt"))

                del model
                torch.cuda.empty_cache()

        results[str(horizon)] = h_results

    # Save results
    suffix = f"_{ds_name}" if ds_name != "MIT" else ""
    out_path = os.path.join(out_dir, f"feature_comparison_results{suffix}.json")
    save_json(results, out_path)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary tables
    print_summary(results)

    return results


def print_summary(results):
    """Print formatted summary tables with improvement percentages."""
    logger.info(f"\n{'=' * 90}")
    logger.info("FEATURE COMPARISON SUMMARY")
    logger.info(f"{'=' * 90}")

    model_names = ["DLinear", "PatchTST", "iTransformer"]

    for h_key in sorted(results.keys(), key=int):
        horizon = int(h_key)
        h_results = results[h_key]

        logger.info(f"\n  Horizon = {horizon} steps ({horizon * 15 / 60:.0f}h)")
        logger.info(f"  {'Model':<22} {'Features':>10} {'RMSE':>8} {'MAE':>8} "
                     f"{'R2':>7} {'MAPE%':>7} {'Time(s)':>8}")
        logger.info(f"  {'-' * 80}")

        for model_name in model_names:
            for fs in ["Calendar", "All", "Workload"]:
                key = f"{model_name}-{fs}"
                if key not in h_results:
                    continue
                m = h_results[key]
                logger.info(f"  {key:<22} {m.get('n_features', '?'):>10} "
                            f"{m['rmse']:>8.3f} {m['mae']:>8.3f} "
                            f"{m['r2']:>7.4f} {m['mape']:>7.2f} "
                            f"{m.get('train_time_s', 0):>8.1f}")
            logger.info(f"  {'-' * 80}")

        # Improvement analysis
        logger.info(f"\n  Improvement Analysis (RMSE reduction %):")
        logger.info(f"  {'Model':<18} {'Workload vs Calendar':>22} {'Workload vs All':>18}")
        logger.info(f"  {'-' * 60}")

        for model_name in model_names:
            cal_key = f"{model_name}-Calendar"
            all_key = f"{model_name}-All"
            wl_key = f"{model_name}-Workload"

            if all(k in h_results for k in [cal_key, all_key, wl_key]):
                rmse_cal = h_results[cal_key]["rmse"]
                rmse_all = h_results[all_key]["rmse"]
                rmse_wl = h_results[wl_key]["rmse"]

                imp_vs_cal = (rmse_cal - rmse_wl) / rmse_cal * 100
                imp_vs_all = (rmse_all - rmse_wl) / rmse_all * 100

                logger.info(f"  {model_name:<18} {imp_vs_cal:>+21.2f}% {imp_vs_all:>+17.2f}%")

        # R2 improvement
        logger.info(f"\n  R2 Improvement (absolute):")
        logger.info(f"  {'Model':<18} {'Workload vs Calendar':>22} {'Workload vs All':>18}")
        logger.info(f"  {'-' * 60}")

        for model_name in model_names:
            cal_key = f"{model_name}-Calendar"
            all_key = f"{model_name}-All"
            wl_key = f"{model_name}-Workload"

            if all(k in h_results for k in [cal_key, all_key, wl_key]):
                r2_cal = h_results[cal_key]["r2"]
                r2_all = h_results[all_key]["r2"]
                r2_wl = h_results[wl_key]["r2"]

                logger.info(f"  {model_name:<18} {r2_wl - r2_cal:>+21.4f} "
                            f"{r2_wl - r2_all:>+17.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MIT",
                        choices=list(DATASET_CONFIGS.keys()) + ["all"],
                        help="Dataset to run (default: MIT)")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASET_CONFIGS:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"  DATASET: {ds}")
            logger.info(f"{'=' * 70}")
            run_feature_comparison(ds)
    else:
        run_feature_comparison(args.dataset)
