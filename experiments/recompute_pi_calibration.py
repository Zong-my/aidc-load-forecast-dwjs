"""
Recompute π-calibration coverage/width from saved SSPM predictions.

No retraining: directly reads the saved q10/q50/q90 npz files for the best
SSPM backbone per (dataset, horizon), applies the calibration formula
w = max(1.0, 1.5 - 0.8*π), and reports before/after coverage and CRPS.

This keeps Table VII consistent with Tables IV/VI (same SSPM source).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, save_json, results_path

logger = get_logger("pi_calib")

PRED_DIR = results_path("case10c_cluster_forecast", "data", "predictions")
RESULTS_DIR = results_path("case10c_cluster_forecast", "data")

# π values from queueing theory analysis (paper Section IV-B)
PI_VALUES = {"MIT": 0.0185, "Helios": 0.045,
             "Alibaba": 0.2231, "Inference": 0.8289}

# Backbone used for π-calibration analysis (best SSPM per dataset/horizon).
# NOTE: Inference H=96 uses iTrans-All (R²=0.872) instead of iTrans-Cal,
#  because iTrans-Cal's npz was overwritten by a point-prediction swap and
#  no longer carries quantile information. iTrans-All has full q10/q50/q90.
BEST_BACKBONE = {
    ("MIT",       16): "PatchTST-Workload",
    ("MIT",       96): "DLinear-Calendar",
    ("Helios",    16): "iTransformer-Calendar",
    ("Helios",    96): "iTransformer-Workload",
    ("Alibaba",   16): "PatchTST-Calendar",
    ("Alibaba",   96): "DLinear-All",
    ("Inference", 16): "iTransformer-All",
    ("Inference", 96): "iTransformer-All",
}


def compute_coverage(pred_low, pred_high, true):
    covered = ((true >= pred_low) & (true <= pred_high)).mean()
    width = (pred_high - pred_low).mean()
    return float(covered), float(width)


def compute_crps(pred_q10, pred_q50, pred_q90, true):
    """Pinball-loss based CRPS approximation across 3 quantiles."""
    crps = 0.0
    for q, pq in [(0.1, pred_q10), (0.5, pred_q50), (0.9, pred_q90)]:
        diff = true - pq
        crps += (np.abs(diff) * np.where(diff > 0, q, q - 1)).mean()
    return float(crps / 3)


def calibrate(pred_q10, pred_q50, pred_q90, pi):
    """Pi-aware calibration: only widen low-pi clusters."""
    w = max(1.0, 1.5 - 0.8 * pi)
    adj_q10 = pred_q50 + (pred_q10 - pred_q50) * w
    adj_q90 = pred_q50 + (pred_q90 - pred_q50) * w
    return adj_q10, adj_q90, w


def main():
    results = {}
    for (ds, h), backbone in BEST_BACKBONE.items():
        pi = PI_VALUES[ds]
        npz_path = os.path.join(PRED_DIR, ds, f"H{h}_SSPM_{backbone}.npz")
        if not os.path.exists(npz_path):
            logger.error(f"MISSING: {npz_path} — run "
                         f"experiments/run_sspm_all_backbones.py first.")
            continue
        d = np.load(npz_path)
        q10, q50, q90, true = d["pred_q10"], d["pred_q50"], d["pred_q90"], d["true"]

        # Base (raw) metrics
        base_cov, base_width = compute_coverage(q10, q90, true)
        base_crps = compute_crps(q10, q50, q90, true)

        # Calibrated metrics
        adj_q10, adj_q90, w = calibrate(q10, q50, q90, pi)
        adj_cov, adj_width = compute_coverage(adj_q10, adj_q90, true)
        adj_crps = compute_crps(adj_q10, q50, adj_q90, true)

        key = f"{ds}_H{h}"
        results[key] = {
            "pi": round(pi, 4),
            "w": round(w, 3),
            "backbone": backbone,
            "base_coverage": round(base_cov, 4),
            "base_width": round(base_width, 3),
            "base_crps": round(base_crps, 4),
            "adj_coverage": round(adj_cov, 4),
            "adj_width": round(adj_width, 3),
            "adj_crps": round(adj_crps, 4),
        }
        logger.info(f"{ds:<10} H={h:<3} π={pi:.3f} w={w:.3f}  "
                    f"cov: {base_cov:.3f}→{adj_cov:.3f}  "
                    f"width: {base_width:.2f}→{adj_width:.2f}")

    if not results:
        logger.error("No SSPM prediction npz files found — run "
                     "experiments/run_sspm_all_backbones.py first.")
        sys.exit(1)

    # Save (overwrite old s2_fixed_results.json)
    out_path = os.path.join(RESULTS_DIR, "s2_fixed_results.json")
    save_json(results, out_path)
    logger.info(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
