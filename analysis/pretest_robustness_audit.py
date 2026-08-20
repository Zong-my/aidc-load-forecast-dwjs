#!/usr/bin/env python3
"""Audit periodicity claims using pre-test history only.

This script is intentionally independent of the forecasting test set.  It
recomputes the six operational periodicity indices, STL cross terms, threshold
classification margins, and four-week rolling-window stability on the first
85% of each time series.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reanalysis_suite import (SCENARIOS, active_scenarios,  # noqa: E402
                              load_series, stl_decompose)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "analysis")
OUT = os.path.join(OUT_DIR, "pretest_robustness_audit_results.json")


THETA_LOW = 15.0
THETA_HIGH = 50.0
LOW_GRID = [8.0, 10.0, 12.0, 15.0, 18.0]
HIGH_GRID = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
ROLL_WINDOW = 4 * 7 * 96
ROLL_STEP = 7 * 96


def classify(pi_pct, low=THETA_LOW, high=THETA_HIGH):
    if pi_pct < low:
        return "train_dominant"
    if pi_pct > high:
        return "inference_dominant"
    return "mixed"


def covariance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def stl_summary(series, robust):
    trend, seasonal, resid = stl_decompose(series, robust)
    y = np.asarray(series, dtype=float)
    var_y = float(np.var(y))
    shares = {
        "trend_pct": float(np.var(trend) / var_y * 100.0),
        "pi_pct": float(np.var(seasonal) / var_y * 100.0),
        "residual_pct": float(np.var(resid) / var_y * 100.0),
    }
    cross = 2.0 * (
        covariance(trend, seasonal)
        + covariance(trend, resid)
        + covariance(seasonal, resid)
    ) / var_y * 100.0
    shares["cross_signed_pct"] = float(cross)
    shares["cross_abs_pct"] = float(abs(cross))
    return shares


def main():
    scenarios = {}
    baseline_classes = {}
    rolling_stability = {}

    # Field scenarios are skipped automatically when their (proprietary)
    # CSVs are absent — see field/README.md.
    names = active_scenarios()
    for name in names:
        cfg = SCENARIOS[name]
        full = load_series(name)
        cut = int(len(full) * 0.85)
        pretest = full.iloc[:cut]
        summary = stl_summary(pretest, cfg["robust"])
        pi_pct = summary["pi_pct"]
        cls = classify(pi_pct)
        baseline_classes[name] = cls

        if cls == "train_dominant":
            margin = THETA_LOW - pi_pct
        elif cls == "inference_dominant":
            margin = pi_pct - THETA_HIGH
        else:
            margin = min(pi_pct - THETA_LOW, THETA_HIGH - pi_pct)

        rolling = []
        for start in range(0, len(pretest) - ROLL_WINDOW + 1, ROLL_STEP):
            window = pretest.iloc[start:start + ROLL_WINDOW]
            win_pi = stl_summary(window, cfg["robust"])["pi_pct"]
            rolling.append(win_pi)
        stable = [classify(value) == cls for value in rolling]
        stability_pct = 100.0 * sum(stable) / len(stable) if stable else float("nan")
        rolling_stability[name] = stability_pct

        scenarios[name] = {
            "n_full": int(len(full)),
            "n_pretest": int(len(pretest)),
            "stl_robust": bool(cfg["robust"]),
            **{key: round(value, 6) for key, value in summary.items()},
            "class": cls,
            "threshold_margin_pp": round(float(margin), 6),
            "rolling_window_count": len(rolling),
            "rolling_pi_pct_min": round(float(min(rolling)), 6),
            "rolling_pi_pct_median": round(float(np.median(rolling)), 6),
            "rolling_pi_pct_max": round(float(max(rolling)), 6),
            "rolling_class_stability_pct": round(float(stability_pct), 6),
        }

    grid_rows = []
    for low in LOW_GRID:
        for high in HIGH_GRID:
            classes = {
                name: classify(row["pi_pct"], low, high)
                for name, row in scenarios.items()
            }
            grid_rows.append({
                "theta_low_pct": low,
                "theta_high_pct": high,
                "unchanged": classes == baseline_classes,
                "classes": classes,
            })

    grid_margins = []
    for row in scenarios.values():
        value = row["pi_pct"]
        if row["class"] == "train_dominant":
            grid_margins.append(min(LOW_GRID) - value)
        elif row["class"] == "inference_dominant":
            grid_margins.append(value - max(HIGH_GRID))
        else:
            grid_margins.append(min(value - max(LOW_GRID), min(HIGH_GRID) - value))

    pi_values = [row["pi_pct"] for row in scenarios.values()]
    result = {
        "definition": "first 85% of each series; rolling windows remain inside that history",
        "thresholds_pct": [THETA_LOW, THETA_HIGH],
        "scenarios": scenarios,
        "cross_abs_range_pct": [
            round(min(row["cross_abs_pct"] for row in scenarios.values()), 6),
            round(max(row["cross_abs_pct"] for row in scenarios.values()), 6),
        ],
        "threshold_grid": {
            "n_combinations": len(grid_rows),
            "n_unchanged": sum(row["unchanged"] for row in grid_rows),
            "minimum_default_threshold_margin_pp": round(
                min(row["threshold_margin_pp"] for row in scenarios.values()), 6
            ),
            "minimum_full_grid_margin_pp": round(min(grid_margins), 6),
        },
        "rolling_stability_pct_range": [
            round(min(rolling_stability.values()), 6),
            round(max(rolling_stability.values()), 6),
        ],
        "extreme_pi_ratio": round(max(pi_values) / min(pi_values), 6),
    }

    assert result["threshold_grid"]["n_unchanged"] == len(grid_rows)
    assert result["threshold_grid"]["minimum_full_grid_margin_pp"] >= 2.0
    assert np.isclose(
        scenarios["MIT"]["pi_pct"], 1.6187, atol=5e-4
    )
    assert np.isclose(
        scenarios["Inference"]["pi_pct"], 82.2543, atol=5e-4
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
