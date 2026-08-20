#!/usr/bin/env python3
"""Recompute STL variance shares on pre-test history for the six paper scenarios.

The proprietary field CSVs are not redistributed; missing field scenarios are
skipped with a warning (see field/README.md).
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(REPO_ROOT, "results", "analysis")
OUT = os.path.join(OUT_DIR, "pretest_stl_audit_results.json")
sys.path.insert(0, _HERE)

from reanalysis_suite import (SCENARIOS, active_scenarios,  # noqa: E402
                              load_series, stl_decompose)


def covariance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def main():
    rows = {}
    for name in active_scenarios():
        cfg = SCENARIOS[name]
        full = load_series(name)
        cut = int(len(full) * 0.85)
        series = full.iloc[:cut]
        trend, seasonal, resid = stl_decompose(series, cfg["robust"])
        var_y = float(np.var(series.to_numpy()))
        shares = {
            "trend": float(np.var(trend) / var_y),
            "seasonal": float(np.var(seasonal) / var_y),
            "residual": float(np.var(resid) / var_y),
        }
        cross_signed = 2.0 * (
            covariance(trend, seasonal)
            + covariance(trend, resid)
            + covariance(seasonal, resid)
        ) / var_y
        rows[name] = {
            "n_full": len(full),
            "n_pretest": len(series),
            "stl_robust": cfg["robust"],
            "trend_share_pct": round(shares["trend"] * 100, 4),
            "pi_pretest_pct": round(shares["seasonal"] * 100, 4),
            "residual_share_pct": round(shares["residual"] * 100, 4),
            "component_sum_pct": round(sum(shares.values()) * 100, 4),
            "cross_signed_pct": round(cross_signed * 100, 4),
            "cross_abs_pct": round(abs(cross_signed) * 100, 4),
        }
    result = {
        "definition": "first 85% of each series (train+validation, pre-test history)",
        "scenarios": rows,
        "cross_abs_range_pct": [
            min(v["cross_abs_pct"] for v in rows.values()),
            max(v["cross_abs_pct"] for v in rows.values()),
        ],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
