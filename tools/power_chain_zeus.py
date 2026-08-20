#!/usr/bin/env python3
"""
Zeus GPU Power-Law Model Fitting: P = a * PL^b

Fits power-law models (P = a * PL^b) for 4 GPU types (V100, A40, RTX6000, P100)
using Zeus NSDI'23 profiling traces, and compares with Case 9 Blackwell DVFS
measurements (P = a * f^b).

Zeus data uses power_limit (PL) as the control variable instead of frequency.
Since PL constrains GPU frequency via DVFS, PL serves as a monotonic proxy for f.
For the Blackwell RTX PRO 6000, actual clock frequency is available from Case 9.

Output:
  - JSON: zeus_power_params.json (fitted parameters per GPU)
  - Figure: zeus_gpu_power_model.svg (4+1 subplot comparison)
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json, load_json
from common.config import DATA_RAW
from common.plotting import save_figure

import matplotlib.pyplot as plt

logger = get_logger("power_chain_zeus")

# ── Paths ──────────────────────────────────────────────────────

# Zeus NSDI'23 traces: clone https://github.com/ml-energy/zeus into data/zeus
# (only examples/research_reproducibility/zeus_nsdi23/trace is needed).
ZEUS_TRACE_DIR = os.path.join(DATA_RAW, "zeus", "examples",
                              "research_reproducibility", "zeus_nsdi23", "trace")
# Optional local DVFS sweep (Blackwell) — not distributable; section skipped
# when this file is absent.
CASE9_JSON = os.path.join(
    os.path.dirname(__file__), "..", "results", "case9_dvfs_flexibility",
    "data", "dvfs_results.json"
)
OUTPUT_DIR = results_path("case10c_cluster_forecast", "data")
FIG_DIR = results_path("case10c_cluster_forecast", "figures")

# GPU metadata: (filename_suffix, display_name, TDP_W, architecture)
GPU_META = {
    "v100":    ("V100",            300, "Volta"),
    "a40":     ("A40",             300, "Ampere"),
    "rtx6000": ("RTX 6000 (Ada)",  260, "Ada Lovelace"),
    "p100":    ("P100",            250, "Pascal"),
}

# Heavy-compute workloads for power-law fitting (exclude trivially light ones)
HEAVY_WORKLOADS = {"resnet50", "bert_base_uncased", "deepspeech2"}


def power_law(x, a, b):
    """Power-law model: P = a * x^b."""
    return a * np.power(x, b)


def load_zeus_data(gpu_key: str) -> pd.DataFrame:
    """Load and return Zeus profiling CSV for a given GPU."""
    path = os.path.join(ZEUS_TRACE_DIR, f"summary_power_{gpu_key}.csv")
    df = pd.read_csv(path)
    logger.info(f"Loaded {gpu_key}: {len(df)} rows")
    return df


def aggregate_zeus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Zeus data: for each power_limit, compute mean power across
    heavy-compute workloads (all batch sizes and optimizers).

    Returns DataFrame with columns [power_limit, avg_power, std_power, n_points].
    """
    # Filter to heavy-compute workloads only
    heavy = df[df["network"].isin(HEAVY_WORKLOADS)].copy()
    logger.info(f"  Heavy-compute subset: {len(heavy)} rows "
                f"(workloads: {sorted(heavy['network'].unique())})")

    agg = heavy.groupby("power_limit")["average_power"].agg(
        avg_power="mean", std_power="std", n_points="count"
    ).reset_index()
    return agg


def fit_power_law(x, y, x_label="PL"):
    """
    Fit P = a * x^b via log-linear regression for initial guess, then curve_fit.

    Returns dict with {a, b, r2, rmse}.
    """
    # Initial guess from log-linear regression
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    a0, b0 = np.exp(intercept), slope

    # Nonlinear least-squares refinement
    popt, pcov = curve_fit(power_law, x, y, p0=[a0, b0], maxfev=10000)
    a, b = popt

    # Goodness of fit
    y_pred = power_law(x, a, b)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    logger.info(f"  Fit: P = {a:.4f} * {x_label}^{b:.4f}  "
                f"(R² = {r2:.6f}, RMSE = {rmse:.2f} W)")
    return {"a": float(a), "b": float(b), "r2": float(r2), "rmse": float(rmse)}


def load_case9_blackwell():
    """
    Load Case 9 DVFS results and compute per-frequency mean power.

    Returns dict with freq_mhz[], power_mean[], and fitted params.
    """
    data = load_json(CASE9_JSON)
    results = data["results"]

    # Group by frequency, average across runs (skip run_idx=0 warmup artifact)
    from collections import defaultdict
    freq_power = defaultdict(list)
    for r in results:
        if r["run_idx"] > 0:  # skip first run (warmup transient)
            freq_power[r["freq_mhz"]].append(r["gpu_power_mean_w"])

    freqs = sorted(freq_power.keys())
    powers = [np.mean(freq_power[f]) for f in freqs]

    freqs = np.array(freqs, dtype=float)
    powers = np.array(powers, dtype=float)

    params = fit_power_law(freqs, powers, x_label="f")
    return {
        "freq_mhz": freqs.tolist(),
        "power_mean_w": powers.tolist(),
        "params": params,
        "n_points": len(freqs),
        "x_label": "Frequency (MHz)",
        "gpu_name": "RTX PRO 6000 (Blackwell)",
    }


def plot_all(zeus_results: dict, case9: dict):
    """
    Create a 5-panel figure: 4 Zeus GPUs + 1 Blackwell (Case 9).
    Each panel shows measured points + fitted curve.
    """
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=False)

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Zeus GPUs
    gpu_order = ["v100", "a40", "rtx6000", "p100"]
    for i, gpu_key in enumerate(gpu_order):
        ax = axes[i]
        info = zeus_results[gpu_key]
        x = np.array(info["power_limits"])
        y = np.array(info["avg_powers"])
        yerr = np.array(info["std_powers"])
        params = info["params"]

        # Scatter with error bars
        ax.errorbar(x, y, yerr=yerr, fmt="o", color=colors[i],
                     markersize=4, capsize=2, capthick=0.6, linewidth=0.6,
                     label="Measured", zorder=3)

        # Fitted curve
        x_fit = np.linspace(x.min() * 0.95, x.max() * 1.05, 200)
        y_fit = power_law(x_fit, params["a"], params["b"])
        ax.plot(x_fit, y_fit, "--", color=colors[i], linewidth=1.0,
                label=f"$P = {params['a']:.3f} \\cdot PL^{{{params['b']:.3f}}}$",
                zorder=2)

        name = GPU_META[gpu_key][0]
        ax.set_title(f"{name}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Power Limit (W)", fontsize=8)
        if i == 0:
            ax.set_ylabel("Average Power (W)", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3, linewidth=0.4)

        # R² annotation
        ax.text(0.97, 0.05, f"$R^2 = {params['r2']:.4f}$",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="0.8", alpha=0.9))

    # Case 9: Blackwell
    ax = axes[4]
    x = np.array(case9["freq_mhz"])
    y = np.array(case9["power_mean_w"])
    params = case9["params"]

    ax.plot(x, y, "o", color=colors[4], markersize=4, label="Measured", zorder=3)

    x_fit = np.linspace(x.min() * 0.95, x.max() * 1.05, 200)
    y_fit = power_law(x_fit, params["a"], params["b"])
    ax.plot(x_fit, y_fit, "--", color=colors[4], linewidth=1.0,
            label=f"$P = {params['a']:.3f} \\cdot f^{{{params['b']:.3f}}}$",
            zorder=2)

    ax.set_title("RTX PRO 6000\n(Blackwell)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Frequency (MHz)", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.text(0.97, 0.05, f"$R^2 = {params['r2']:.4f}$",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="0.8", alpha=0.9))

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "zeus_gpu_power_model")
    save_figure(fig, out_path)
    logger.info(f"Figure saved: {out_path}.svg / .png")


def main():
    logger.info("=" * 60)
    logger.info("Zeus GPU Power-Law Model Fitting")
    logger.info("=" * 60)

    # ── 1. Fit Zeus GPUs ──
    zeus_results = {}
    for gpu_key, (display_name, tdp, arch) in GPU_META.items():
        logger.info(f"\n--- {display_name} ({arch}, TDP={tdp}W) ---")
        df = load_zeus_data(gpu_key)
        agg = aggregate_zeus(df)

        x = agg["power_limit"].values.astype(float)
        y = agg["avg_power"].values.astype(float)

        params = fit_power_law(x, y, x_label="PL")

        zeus_results[gpu_key] = {
            "gpu_name": display_name,
            "architecture": arch,
            "tdp_w": tdp,
            "power_limits": x.tolist(),
            "avg_powers": y.tolist(),
            "std_powers": agg["std_power"].values.tolist(),
            "n_datapoints": int(agg["n_points"].sum()),
            "n_pl_levels": len(x),
            "params": params,
        }

    # ── 2. Load Case 9 Blackwell (optional local DVFS sweep) ──
    case9 = None
    if os.path.exists(CASE9_JSON):
        logger.info(f"\n--- RTX PRO 6000 (Blackwell, Case 9 DVFS) ---")
        case9 = load_case9_blackwell()
    else:
        logger.info("\nCase-9 DVFS sweep JSON not found — skipping the "
                    "Blackwell comparison section (Zeus fits are unaffected).")

    # ── 3. Save JSON ──
    output = {
        "description": (
            "Power-law model P = a * x^b fitted per GPU. "
            "Zeus GPUs use power_limit (PL, W) as x-variable; "
            "Blackwell uses clock frequency (f, MHz) from Case 9 DVFS sweep."
        ),
        "zeus_gpus": {},
        "summary_table": [],
    }
    if case9 is not None:
        output["blackwell"] = {
            "gpu_name": case9["gpu_name"],
            "architecture": "Blackwell",
            "x_variable": "frequency_mhz",
            "freq_range": [min(case9["freq_mhz"]), max(case9["freq_mhz"])],
            "n_freq_levels": case9["n_points"],
            "params": case9["params"],
        }

    for gpu_key, info in zeus_results.items():
        output["zeus_gpus"][gpu_key] = {
            "gpu_name": info["gpu_name"],
            "architecture": info["architecture"],
            "tdp_w": info["tdp_w"],
            "x_variable": "power_limit_w",
            "pl_range": [min(info["power_limits"]), max(info["power_limits"])],
            "n_pl_levels": info["n_pl_levels"],
            "n_datapoints": info["n_datapoints"],
            "params": info["params"],
        }

        output["summary_table"].append({
            "gpu": info["gpu_name"],
            "arch": info["architecture"],
            "x_var": "PL (W)",
            "a": info["params"]["a"],
            "b": info["params"]["b"],
            "R2": info["params"]["r2"],
            "RMSE_W": info["params"]["rmse"],
        })

    if case9 is not None:
        output["summary_table"].append({
            "gpu": "RTX PRO 6000",
            "arch": "Blackwell",
            "x_var": "f (MHz)",
            "a": case9["params"]["a"],
            "b": case9["params"]["b"],
            "R2": case9["params"]["r2"],
            "RMSE_W": case9["params"]["rmse"],
        })

    json_path = os.path.join(OUTPUT_DIR, "zeus_power_params.json")
    save_json(output, json_path)
    logger.info(f"\nResults saved: {json_path}")

    # ── 4. Print summary table ──
    logger.info("\n" + "=" * 72)
    logger.info("Summary: Power-Law Model P = a * x^b")
    logger.info("-" * 72)
    logger.info(f"{'GPU':<22} {'Arch':<14} {'x-var':<10} {'a':>8} {'b':>7} {'R²':>8} {'RMSE(W)':>8}")
    logger.info("-" * 72)
    for row in output["summary_table"]:
        logger.info(
            f"{row['gpu']:<22} {row['arch']:<14} {row['x_var']:<10} "
            f"{row['a']:>8.4f} {row['b']:>7.4f} {row['R2']:>8.5f} {row['RMSE_W']:>8.2f}"
        )
    logger.info("=" * 72)

    # ── 5. Plot (needs the optional Case-9 panel) ──
    if case9 is not None:
        plot_all(zeus_results, case9)
    else:
        logger.info("Plot skipped (5th panel needs the Case-9 DVFS sweep).")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
