"""
SSPM ablation aligned with feature_comparison HPARAMS.

For each (dataset, horizon), runs the 4 ablation variants on the best
SSPM backbone (matching Table IV's SSPM row):
  1. Baseline       — reuse feature_comparison_results
  2. H_only         — pretrain → standard finetune (NEW)
  3. I_only         — no pretrain → mixup with alpha search (NEW)
  4. HI_combined    — reuse sspm_all_backbones_results

All runs use the SAME hyperparameters as feature_comparison (d_model=128,
2 layers, 50 epochs, lr=5e-4) so Baseline matches Table IV exactly.

Outputs:
  predictions/{ds}/H{h}_SSPM_H_only_aligned.npz
  predictions/{ds}/H{h}_SSPM_I_only_aligned.npz
  checkpoints/{ds}/H{h}_SSPM_*_aligned.pt
  sspm_ablation_aligned_results.json (summary including baseline + HI from existing)
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json, load_json
from common.config import PROJECT_ROOT

from train_msra import set_seed
from run_baselines import (DLinear, PatchTST, iTransformer,
                           train_one_model, evaluate_on_test)
from msra.physics_constraints import pinball_loss
from run_feature_comparison import (DATASET_CONFIGS, HPARAMS, LOOKBACK,
                                    HORIZONS, SEED, QUANTILE_LEVELS,
                                    N_QUANTILES, load_and_engineer,
                                    build_dataset_for_feature_set,
                                    make_model)
from run_sspm_all_backbones import (ALPHA_GRID, PRETRAIN_EPOCHS, MASK_RATIO,
                                     pretrain_masked, train_mixup)

logger = get_logger("sspm_ablation")

# Best SSPM backbone per (dataset, horizon) — matches Table IV's SSPM row
BEST_BACKBONE = {
    ("MIT",       16): ("PatchTST",     "Workload"),
    ("MIT",       96): ("DLinear",      "Calendar"),
    ("Helios",    16): ("iTransformer", "Calendar"),
    ("Helios",    96): ("iTransformer", "Workload"),
    ("Alibaba",   16): ("PatchTST",     "Calendar"),
    ("Alibaba",   96): ("DLinear",      "All"),
    ("Inference", 16): ("iTransformer", "All"),
    ("Inference", 96): ("iTransformer", "Calendar"),
}


def train_h_only(model, data, hparams, device):
    """H_only: pretrain (30 ep) → standard supervised finetune (no mixup).

    Uses train_one_model from run_baselines.py for the supervised stage.
    """
    # Phase 1: masked pretraining
    model = pretrain_masked(model, data["train_X"], hparams, device)
    # Phase 2: standard supervised
    model, val_loss, elapsed = train_one_model(
        model, data["train_X"], data["train_y"],
        data["val_X"], data["val_y"], hparams, device)
    return model, val_loss, elapsed


def train_i_only(model_factory, data, hparams, alphas, device):
    """I_only: no pretrain, just mixup with alpha search by validation loss."""
    best_val = float("inf")
    best_alpha = None
    best_sd = None
    for alpha in alphas:
        set_seed(SEED)
        m = model_factory().to(device)
        m, val_loss = train_mixup(m, data, hparams, alpha, device)
        if val_loss < best_val:
            best_val = val_loss
            best_alpha = alpha
            best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        del m
        torch.cuda.empty_cache()
    return best_sd, best_val, best_alpha


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)
    logger.info(f"Device: {device}")

    out_dir = os.path.join(results_path("case10c_cluster_forecast"), "data")
    fc_existing = {}
    for ds in DATASET_CONFIGS:
        sf = f'_{ds}' if ds != 'MIT' else ''
        fc_path = os.path.join(out_dir, f'feature_comparison_results{sf}.json')
        if not os.path.exists(fc_path):
            logger.error(f"Missing {fc_path} — run "
                         f"experiments/run_feature_comparison.py --dataset {ds} first.")
            sys.exit(1)
        fc_existing[ds] = load_json(fc_path)
    sspm_path = os.path.join(out_dir, 'sspm_all_backbones_results.json')
    if not os.path.exists(sspm_path):
        logger.error(f"Missing {sspm_path} — run "
                     f"experiments/run_sspm_all_backbones.py first.")
        sys.exit(1)
    sspm_existing = load_json(sspm_path)

    results = {}
    for ds, cfg in DATASET_CONFIGS.items():
        results[ds] = {}
        df, feature_sets, target = load_and_engineer(cfg)

        pred_dir = os.path.join(out_dir, "predictions", ds)
        ckpt_dir = os.path.join(out_dir, "checkpoints", ds)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        for horizon in HORIZONS:
            backbone_model, backbone_fs = BEST_BACKBONE[(ds, horizon)]
            backbone_key = f"{backbone_model}-{backbone_fs}"
            logger.info(f"\n=== {ds} H={horizon} | backbone: {backbone_key} ===")

            fs_cols = feature_sets[backbone_fs]
            data = build_dataset_for_feature_set(df, fs_cols, horizon, target)
            nf = data["train_X"].shape[2]

            def model_factory():
                return make_model(backbone_model, nf, horizon)

            h_results = {}

            # 1) Baseline — reuse feature_comparison
            bl = fc_existing[ds][str(horizon)][backbone_key]
            h_results["baseline"] = {k: v for k, v in bl.items()
                                      if k in ("rmse", "mae", "r2", "mape", "smape")}
            h_results["baseline"]["backbone"] = backbone_key
            logger.info(f"  Baseline (reused): R²={bl['r2']:.4f}")

            # 2) H_only — NEW
            t0 = time.time()
            set_seed(SEED)
            m = model_factory().to(device)
            m, _, _ = train_h_only(m, data, HPARAMS, device)
            metrics, pred, true = evaluate_on_test(
                m, data["test_X"], data["test_y"], data["norm_params"],
                device, return_predictions=True)
            metrics["backbone"] = backbone_key
            h_results["H_only"] = metrics
            np.savez_compressed(
                os.path.join(pred_dir, f"H{horizon}_SSPM_H_only_aligned.npz"),
                pred_q10=pred[:,:,0], pred_q50=pred[:,:,1], pred_q90=pred[:,:,2], true=true)
            torch.save(m.state_dict(),
                       os.path.join(ckpt_dir, f"H{horizon}_SSPM_H_only_aligned.pt"))
            del m; torch.cuda.empty_cache()
            logger.info(f"  H_only: R²={metrics['r2']:.4f} ({time.time()-t0:.1f}s)")

            # 3) I_only — NEW (alpha search)
            t0 = time.time()
            alphas = ALPHA_GRID[ds]
            best_sd, best_val, best_a = train_i_only(
                model_factory, data, HPARAMS, alphas, device)
            m = model_factory().to(device)
            m.load_state_dict(best_sd)
            metrics, pred, true = evaluate_on_test(
                m, data["test_X"], data["test_y"], data["norm_params"],
                device, return_predictions=True)
            metrics["backbone"] = backbone_key
            metrics["alpha"] = best_a
            h_results["I_only"] = metrics
            np.savez_compressed(
                os.path.join(pred_dir, f"H{horizon}_SSPM_I_only_aligned.npz"),
                pred_q10=pred[:,:,0], pred_q50=pred[:,:,1], pred_q90=pred[:,:,2], true=true)
            torch.save(best_sd,
                       os.path.join(ckpt_dir, f"H{horizon}_SSPM_I_only_aligned.pt"))
            del m, best_sd; torch.cuda.empty_cache()
            logger.info(f"  I_only: R²={metrics['r2']:.4f} α={best_a} ({time.time()-t0:.1f}s)")

            # 4) HI_combined — reuse sspm_all_backbones
            hi = sspm_existing[ds][str(horizon)][backbone_key]
            h_results["HI_combined"] = {k: v for k, v in hi.items()
                                         if k in ("rmse", "mae", "r2", "mape", "smape", "alpha")}
            h_results["HI_combined"]["backbone"] = backbone_key
            logger.info(f"  HI_combined (reused): R²={hi['r2']:.4f}")

            # Summary
            bl_r2 = h_results["baseline"]["r2"]
            for v in ["H_only", "I_only", "HI_combined"]:
                d = h_results[v]["r2"] - bl_r2
                flag = "✓" if d > 0.005 else ("~" if d > -0.005 else "✗")
                logger.info(f"    {v}: Δ={d:+.4f} {flag}")

            results[ds][str(horizon)] = h_results

    # Save aggregated results
    out_path = os.path.join(out_dir, "sspm_ablation_aligned_results.json")
    save_json(results, out_path)
    logger.info(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
