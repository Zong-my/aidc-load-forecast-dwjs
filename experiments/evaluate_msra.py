"""
MSRA Evaluation: metrics, ablation, comparison, and analysis.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger

logger = get_logger("eval_msra")


def compute_point_metrics(pred, true):
    """Compute standard point forecast metrics.

    Args:
        pred: (N, H) predictions in original scale
        true: (N, H) ground truth in original scale

    Returns:
        dict with rmse, mae, r2, mape
    """
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))

    mask = np.abs(true) > 0.5  # avoid division by near-zero
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100)
    else:
        mape = float("nan")

    return {"rmse": round(rmse, 3), "mae": round(mae, 3),
            "r2": round(r2, 4), "mape": round(mape, 2)}


def compute_step_metrics(pred, true):
    """Per-step RMSE across the forecast horizon.

    Args:
        pred: (N, H) or (N, H, Q) — if 3D, uses median quantile
        true: (N, H)

    Returns:
        list of H RMSE values
    """
    if pred.ndim == 3:
        pred = pred[:, :, pred.shape[2] // 2]  # median quantile
    H = pred.shape[1]
    return [round(float(np.sqrt(np.mean((pred[:, h] - true[:, h]) ** 2))), 3)
            for h in range(H)]


def compute_crps(pred_quantiles, true, quantile_levels):
    """Approximate CRPS from quantile predictions.

    Args:
        pred_quantiles: (N, H, Q) predicted quantiles
        true: (N, H) ground truth
        quantile_levels: list of Q quantile levels (e.g., [0.1, 0.5, 0.9])

    Returns:
        float: mean CRPS
    """
    N, H, Q = pred_quantiles.shape
    true_expanded = true[:, :, None]  # (N, H, 1)
    tau = np.array(quantile_levels).reshape(1, 1, -1)

    errors = true_expanded - pred_quantiles
    pinball = np.maximum(tau * errors, (tau - 1) * errors)
    return float(2 * pinball.mean())


def compute_coverage(pred_quantiles, true, quantile_levels):
    """Compute prediction interval coverage.

    Args:
        pred_quantiles: (N, H, Q) with at least q_low and q_high
        true: (N, H)
        quantile_levels: e.g., [0.1, 0.5, 0.9]

    Returns:
        dict with coverage_80 (fraction in [q10, q90])
    """
    q_low = pred_quantiles[:, :, 0]   # q10
    q_high = pred_quantiles[:, :, -1]  # q90
    inside = (true >= q_low) & (true <= q_high)
    nominal = quantile_levels[-1] - quantile_levels[0]
    return {
        f"coverage_{int(nominal*100)}": round(float(inside.mean()), 4),
        "interval_width": round(float((q_high - q_low).mean()), 3),
    }


def evaluate_model(model, test_X, test_y, norm_params, device="cuda",
                   quantile_levels=None, is_baseline_fn=False):
    """Evaluate a model on test data.

    Args:
        model: nn.Module or None (for persistence baselines)
        test_X: (N, L, F) normalized
        test_y: (N, H) normalized
        norm_params: dict with target_mean, target_std
        device: cuda/cpu
        quantile_levels: list of quantile levels
        is_baseline_fn: if True, model is a callable predict_fn(X, H) -> (N,H,Q)

    Returns:
        dict with point_metrics, step_rmse, probabilistic_metrics
    """
    t_mean = norm_params["target_mean"]
    t_std = norm_params["target_std"]
    H = test_y.shape[1]

    if is_baseline_fn:
        # Persistence-type baseline: callable(test_X, horizon) -> (N, H, Q)
        pred_norm = model(test_X, H)
        pred_kw = pred_norm * t_std + t_mean
    else:
        model.eval()
        X_t = torch.from_numpy(test_X).float().to(device)
        with torch.no_grad():
            pred_norm = model(X_t).cpu().numpy()  # (N, H, Q)
        pred_kw = pred_norm * t_std + t_mean

    true_kw = test_y * t_std + t_mean

    # Point metrics using median quantile
    if pred_kw.ndim == 3:
        median_idx = pred_kw.shape[2] // 2
        pred_point = pred_kw[:, :, median_idx]
    else:
        pred_point = pred_kw

    result = {
        "point": compute_point_metrics(pred_point, true_kw),
        "step_rmse": compute_step_metrics(pred_point, true_kw),
    }

    # Probabilistic metrics
    if pred_kw.ndim == 3 and quantile_levels is not None:
        result["crps"] = compute_crps(pred_kw, true_kw, quantile_levels)
        result["coverage"] = compute_coverage(pred_kw, true_kw, quantile_levels)

    return result


def evaluate_by_regime(model, test_X, test_y, test_regimes, norm_params,
                       n_regimes, device="cuda"):
    """Evaluate model performance broken down by regime.

    Returns:
        dict: {regime_k: point_metrics}
    """
    t_mean = norm_params["target_mean"]
    t_std = norm_params["target_std"]

    model.eval()
    X_t = torch.from_numpy(test_X).float().to(device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()
    if pred_norm.ndim == 3:
        pred_norm = pred_norm[:, :, pred_norm.shape[2] // 2]

    pred_kw = pred_norm * t_std + t_mean
    true_kw = test_y * t_std + t_mean

    regime_metrics = {}
    for k in range(n_regimes):
        mask = test_regimes == k
        if mask.sum() < 10:
            continue
        regime_metrics[f"regime_{k}"] = compute_point_metrics(
            pred_kw[mask], true_kw[mask])
        regime_metrics[f"regime_{k}"]["n_samples"] = int(mask.sum())

    return regime_metrics


def evaluate_transition_impact(model, test_X, test_y, test_regimes,
                                norm_params, horizon, device="cuda"):
    """Compare error for windows with vs without regime transitions.

    A "transition window" is one where the regime at prediction moment
    differs from the majority regime in the preceding lookback.
    """
    t_mean = norm_params["target_mean"]
    t_std = norm_params["target_std"]

    model.eval()
    X_t = torch.from_numpy(test_X).float().to(device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()
    if pred_norm.ndim == 3:
        pred_norm = pred_norm[:, :, pred_norm.shape[2] // 2]

    pred_kw = pred_norm * t_std + t_mean
    true_kw = test_y * t_std + t_mean

    errors = np.sqrt(np.mean((pred_kw - true_kw) ** 2, axis=1))  # per-window RMSE

    # Simple heuristic: transition = regime differs from 8 steps ago
    # (We only have regime at prediction moment, not full sequence)
    # Use variance of power in first half of lookback as proxy
    power_first_half = test_X[:, :test_X.shape[1] // 2, 0]
    power_second_half = test_X[:, test_X.shape[1] // 2:, 0]
    std_change = np.abs(power_second_half.std(axis=1) - power_first_half.std(axis=1))
    threshold = np.percentile(std_change, 75)

    transition_mask = std_change > threshold
    stable_mask = ~transition_mask

    return {
        "transition": {
            "n": int(transition_mask.sum()),
            "rmse": round(float(errors[transition_mask].mean()), 3),
        },
        "stable": {
            "n": int(stable_mask.sum()),
            "rmse": round(float(errors[stable_mask].mean()), 3),
        },
        "ratio": round(float(errors[transition_mask].mean() /
                              max(errors[stable_mask].mean(), 1e-8)), 3),
    }


def run_ablation(data, cfg, trained_results, device="cuda"):
    """Run ablation study: disable each module and measure impact.

    Returns dict of {variant_name: {horizon: metrics}}
    """
    logger.info("\n=== Ablation Study ===")
    ablation_results = {}

    for horizon in cfg["horizons"]:
        hdata = data["horizons"][horizon]
        norm = hdata["norm_params"]
        ql = cfg["quantile_levels"]

        msra = trained_results[horizon]["msra"]
        msra.eval()

        # Full MSRA
        full_metrics = evaluate_model(
            msra, hdata["test_X"], hdata["test_y"], norm, device, ql)

        ablation_results.setdefault("MSRA-full", {})[horizon] = full_metrics

        # Ablation: equal weights (replace transition predictor output)
        class EqualWeightWrapper:
            def __init__(self, base_model, K):
                self.base = base_model
                self.K = K

            def eval(self): self.base.eval()

            def __call__(self, x):
                self.base.eval()
                with torch.no_grad():
                    expert_preds = self.base.expert_ensemble(x)
                    batch, K, H, Q = expert_preds.shape
                    uniform_w = torch.ones(batch, H, K, device=x.device) / K
                    pred = self.base.mixture_output(expert_preds, uniform_w)
                return pred

        eq_wrapper = EqualWeightWrapper(msra, cfg["n_regimes"])
        eq_metrics = evaluate_model(
            eq_wrapper, hdata["test_X"], hdata["test_y"], norm, device, ql)
        ablation_results.setdefault("MSRA-equal-wt", {})[horizon] = eq_metrics

        logger.info(f"  H={horizon}: full RMSE={full_metrics['point']['rmse']}, "
                    f"equal-wt RMSE={eq_metrics['point']['rmse']}")

    return ablation_results


def evaluate_all(data, cfg, trained_results, device="cuda"):
    """Run complete evaluation suite.

    Returns nested dict with all metrics.
    """
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    all_metrics = {}

    for horizon in cfg["horizons"]:
        h_label = f"{horizon*15//60}h"
        logger.info(f"\n--- Horizon = {horizon} ({h_label}) ---")

        hdata = data["horizons"][horizon]
        norm = hdata["norm_params"]
        ql = cfg["quantile_levels"]
        K = cfg["n_regimes"]

        msra = trained_results[horizon]["msra"]
        baselines = trained_results[horizon]["baselines"]

        h_results = {}

        # MSRA
        msra_m = evaluate_model(msra, hdata["test_X"], hdata["test_y"],
                                norm, device, ql)
        h_results["MSRA"] = msra_m
        logger.info(f"  MSRA: RMSE={msra_m['point']['rmse']}, "
                    f"R²={msra_m['point']['r2']}")

        # Regime-conditioned evaluation
        regime_m = evaluate_by_regime(
            msra, hdata["test_X"], hdata["test_y"],
            hdata["test_regimes"], norm, K, device)
        h_results["MSRA_by_regime"] = regime_m

        # Transition impact
        trans_m = evaluate_transition_impact(
            msra, hdata["test_X"], hdata["test_y"],
            hdata["test_regimes"], norm, horizon, device)
        h_results["MSRA_transition"] = trans_m
        logger.info(f"  Transition RMSE ratio: {trans_m['ratio']}")

        # Baselines
        for name, bl in baselines.items():
            if "predict_fn" in bl:
                bl_m = evaluate_model(
                    bl["predict_fn"], hdata["test_X"], hdata["test_y"],
                    norm, device, ql, is_baseline_fn=True)
            else:
                test_X = hdata["test_X"]
                if bl.get("input") == "uv":
                    test_X = test_X[:, :, :1]
                bl_m = evaluate_model(
                    bl["model"], test_X, hdata["test_y"],
                    norm, device, ql)
            h_results[name] = bl_m
            logger.info(f"  {name}: RMSE={bl_m['point']['rmse']}, "
                        f"R²={bl_m['point']['r2']}")

        all_metrics[horizon] = h_results

    # Ablation
    all_metrics["ablation"] = run_ablation(data, cfg, trained_results, device)

    return all_metrics
