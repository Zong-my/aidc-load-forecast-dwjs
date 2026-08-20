"""
Queueing-theory validation: compare theoretical daily/weekly seasonality
attenuation (Section II, Theorem 1) against measured seasonality from
four heterogeneous cluster traces.

Theorem 1 (spectral peak attenuation):

    S_P(omega_d) / S_P(0)  ~  alpha_d^2 / [1 + (omega_d * E[S])^2 * h(rho)]

where  omega_d = 2*pi / T_d,   h(rho) = rho^2 / (1 - rho)^2

Four scenarios
--------------
1. MIT Supercloud  (academic training, 448xV100)
2. Helios Saturn   (industrial training, ~2072 GPU)
3. Alibaba PAI     (mixed ML platform)
4. Azure LMM       (commercial inference service)
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json, load_json
from common.plotting import save_figure, setup_journal_style
from common.config import COLORS, DATA_PROCESSED

log = get_logger("queueing_theory_validation")

# ── Constants ────────────────────────────────────────────────────────

T_D = 24.0        # daily period in hours
T_W = 168.0       # weekly period in hours
OMEGA_D = 2 * np.pi / T_D   # rad/h
OMEGA_W = 2 * np.pi / T_W   # rad/h

DATA_DIR = DATA_PROCESSED
RESULT_DIR = "case10c_cluster_forecast"


# ── Queueing theory functions ────────────────────────────────────────

def h_rho(rho):
    """Queue congestion factor h(rho) = rho^2 / (1 - rho)^2."""
    return rho**2 / (1 - rho)**2


def spectral_attenuation(alpha, omega, E_S, rho):
    """
    Theoretical spectral attenuation ratio at frequency omega.

    Returns S_P(omega) / S_P(0) propto alpha^2 / [1 + (omega * E_S)^2 * h(rho)]
    The numerator alpha^2 is the seasonality strength of the arrival process.
    """
    return alpha**2 / (1 + (omega * E_S)**2 * h_rho(rho))


def spectral_curve(E_S_range, alpha, omega, rho):
    """Compute attenuation over a range of E[S] values."""
    return np.array([spectral_attenuation(alpha, omega, es, rho) for es in E_S_range])


# ── Measured seasonality extraction ──────────────────────────────────

def compute_measured_seasonality(ts, values, col_name, period_h=24.0, freq_min=15):
    """
    Extract measured seasonality metrics from a 15-min time series.

    Returns dict with:
      - daily_var_frac:  var(seasonal) / var(total)
      - peak_trough_ratio
      - acf_24h
      - psd_peak_normalized (PSD at omega_d normalized by PSD(0))
    """
    result = {}
    samples_per_period = int(period_h * 60 / freq_min)

    # Drop NaN
    valid = values.dropna()
    if len(valid) < 2 * samples_per_period:
        log.warning(f"  {col_name}: too few valid points ({len(valid)})")
        return None

    # STL decomposition
    try:
        stl = STL(valid, period=samples_per_period, robust=True)
        res = stl.fit()
        seasonal = res.seasonal
        resid = res.resid
        trend = res.trend

        total_var = np.var(valid)
        seasonal_var = np.var(seasonal)
        result["daily_var_frac"] = float(seasonal_var / total_var) if total_var > 0 else 0.0
        result["seasonal_std"] = float(np.std(seasonal))
        result["total_std"] = float(np.std(valid))
    except Exception as e:
        log.warning(f"  STL failed for {col_name}: {e}")
        result["daily_var_frac"] = np.nan

    # Peak-to-trough ratio from hourly profile
    hourly = valid.groupby(valid.index.hour).mean()
    peak = hourly.max()
    trough = hourly.min()
    result["peak_trough_ratio"] = float(peak / trough) if trough > 0 else np.inf
    result["peak_hour"] = int(hourly.idxmax())
    result["trough_hour"] = int(hourly.idxmin())

    # ACF at 24h lag (= samples_per_period)
    try:
        max_lag = min(samples_per_period + 5, len(valid) // 2)
        acf_vals = acf(valid.values, nlags=max_lag, fft=True)
        if samples_per_period < len(acf_vals):
            result["acf_24h"] = float(acf_vals[samples_per_period])
        else:
            result["acf_24h"] = np.nan
    except Exception:
        result["acf_24h"] = np.nan

    # PSD via Welch, normalized
    try:
        fs = 1.0 / (freq_min * 60)  # sampling freq in Hz
        nperseg = min(2 * samples_per_period * 4, len(valid))
        freqs, psd = signal.welch(valid.values, fs=fs, nperseg=nperseg)
        # Target frequency for daily cycle
        f_daily = 1.0 / (period_h * 3600)  # Hz
        idx_daily = np.argmin(np.abs(freqs - f_daily))
        # Normalize by DC component (or max)
        psd_dc = psd[0] if psd[0] > 0 else psd.max()
        result["psd_peak_norm"] = float(psd[idx_daily] / psd_dc) if psd_dc > 0 else 0.0
        result["psd_daily_raw"] = float(psd[idx_daily])
    except Exception:
        result["psd_peak_norm"] = np.nan

    return result


# ── Load and process each dataset ────────────────────────────────────

def _require_csv(path):
    """Guard: give a clear message when a built 15-min CSV is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Built 15-min dataset not found: {path}. "
            f"Run the dataset-building steps first (see data/README.md)."
        )


def load_mit_supercloud():
    """Load MIT Supercloud 15min data and return utilization series."""
    path = os.path.join(DATA_DIR, "mit_supercloud_15min.csv")
    _require_csv(path)
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    # Use cluster power as the main signal
    col = "cluster_power_kw"
    series = df[col].dropna()
    # Handle tz-aware timestamps: convert to UTC then strip tz
    if hasattr(series.index, 'tz') and series.index.tz is not None:
        series.index = series.index.tz_convert("UTC").tz_localize(None)
    else:
        try:
            series.index = pd.DatetimeIndex(series.index, utc=True).tz_localize(None)
        except Exception:
            series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)
    return series, col


def load_helios_saturn():
    """Load Helios Saturn 15min data (active_gpus as proxy)."""
    path = os.path.join(DATA_DIR, "helios_saturn_15min.csv")
    _require_csv(path)
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    col = "active_gpus"
    series = df[col].dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series, col


def load_alibaba():
    """Load Alibaba v2026 spot GPU 15min data (matches paper CSV).

    Paper uses the newer v2026 spot-GPU trace; column active_gpu_count
    (number of concurrently running jobs) serves as the utilisation proxy.
    """
    path = os.path.join(DATA_DIR, "alibaba_v2026_spot_15min.csv")
    _require_csv(path)
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    col = "active_gpu_count"
    series = df[col].dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series, col


def load_inference():
    """Load the synthesised Inference cluster 15min data (matches paper CSV).

    Paper uses inference_cluster_15min.csv: cluster power synthesised from
    Azure LLM request traces + Zeus power model + GreenSKU chain. Column
    cluster_power_kw is the target signal for periodicity extraction.
    """
    path = os.path.join(DATA_DIR, "inference_cluster_15min.csv")
    _require_csv(path)
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    col = "cluster_power_kw"
    series = df[col].dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series, col


# ── Main computation ─────────────────────────────────────────────────

def main():
    log.info("=== Queueing Theory Validation: Theory vs Measured ===")

    # Load queueing parameters
    param_dir = results_path(RESULT_DIR, "data", "dummy.txt")
    param_dir = os.path.dirname(param_dir)

    mit_json = os.path.join(param_dir, "queueing_params_mit.json")
    helios_json = os.path.join(param_dir, "queueing_params_helios.json")
    for p, producer in [(mit_json, "experiments/queueing_theory_analysis.py"),
                        (helios_json, "experiments/queueing_theory_helios.py")]:
        if not os.path.exists(p):
            log.error(f"Queueing parameters not found: {p}. "
                      f"Run {producer} first to generate it "
                      f"(input data: see data/README.md).")
            sys.exit(1)

    mit_params = load_json(mit_json)
    helios_params = load_json(helios_json)

    # ── Define scenarios ─────────────────────────────────────────
    # Each scenario: name, alpha_d, alpha_w, E_S (hours), rho, color, marker
    scenarios = {
        "MIT Supercloud": {
            "alpha_d": mit_params["arrival_rate"]["alpha_d"],
            "alpha_w": mit_params["arrival_rate"]["alpha_w"],
            "E_S_h": mit_params["service_time"]["gpu_only"]["mean_h"],
            "rho": mit_params["utilization"]["mean"],
            "color": COLORS["primary"],
            "marker": "o",
            "workload": "Academic training",
        },
        "Helios Saturn": {
            "alpha_d": helios_params["arrival_rate"]["seasonal_model"]["alpha_d"],
            "alpha_w": helios_params["arrival_rate"]["seasonal_model"]["alpha_w"],
            "E_S_h": helios_params["service_time"]["gpu_jobs"]["E_S_hours"],
            "rho": helios_params["utilization"]["rho_mean"],
            "color": COLORS["tertiary"],
            "marker": "s",
            "workload": "Industrial training",
        },
        "Alibaba PAI": {
            # No direct job logs; reverse-engineer from utilization patterns.
            # Alibaba is a mixed platform (training + inference + data processing).
            # From measured data: peak/trough ~ 1.19x, moderate daily pattern.
            # Assume moderate alpha_d (weaker than Azure user-driven, but present)
            # E[S] intermediate: mix of short inference + medium training jobs
            "alpha_d": 0.25,  # moderate daily modulation (mixed workload)
            "alpha_w": 0.10,  # weak weekly (production cluster runs 24/7)
            "E_S_h": 0.5,    # mixed: many short + some long jobs
            "rho": 0.70,     # typical high-util production cluster
            "color": COLORS["secondary"],
            "marker": "D",
            "workload": "Mixed ML platform",
            "assumed": True,
        },
        "Inference": {
            # Synthesised inference cluster (Azure-trace + Zeus + GreenSKU)
            "alpha_d": 0.60,  # strong daily user-demand pattern
            "alpha_w": 0.55,  # moderate weekly (weekday/weekend)
            "E_S_h": 0.003,  # aggregate estimate (matches Table II of the paper)
            "rho": 0.45,     # moderate utilization (elastic scaling)
            "color": COLORS["quaternary"],
            "marker": "^",
            "workload": "Inference serving",
            "assumed": True,
        },
    }

    # ── Compute theoretical attenuation for each scenario ────────
    log.info("\n--- Theoretical Spectral Attenuation ---")
    for name, s in scenarios.items():
        omega_d_ES = OMEGA_D * s["E_S_h"]
        omega_w_ES = OMEGA_W * s["E_S_h"]
        h_val = h_rho(s["rho"])
        att_daily = spectral_attenuation(s["alpha_d"], OMEGA_D, s["E_S_h"], s["rho"])
        att_weekly = spectral_attenuation(s["alpha_w"], OMEGA_W, s["E_S_h"], s["rho"])

        s["omega_d_ES"] = omega_d_ES
        s["omega_w_ES"] = omega_w_ES
        s["h_rho"] = h_val
        s["theory_daily"] = att_daily
        s["theory_weekly"] = att_weekly

        log.info(f"  {name:20s}: E[S]={s['E_S_h']:.3f}h, "
                 f"omega_d*E[S]={omega_d_ES:.4f}, h(rho)={h_val:.3f}, "
                 f"daily_att={att_daily:.6f}, weekly_att={att_weekly:.6f}")

    # ── Compute measured seasonality ─────────────────────────────
    log.info("\n--- Measured Seasonality from 15-min Data ---")
    loaders = {
        "MIT Supercloud": load_mit_supercloud,
        "Helios Saturn": load_helios_saturn,
        "Alibaba PAI": load_alibaba,
        "Inference": load_inference,
    }

    for name, loader in loaders.items():
        log.info(f"  Processing {name}...")
        try:
            series, col = loader()
            log.info(f"    Loaded {len(series)} points, column={col}")

            # Daily seasonality
            daily = compute_measured_seasonality(series, series, f"{name}_daily",
                                                 period_h=24.0, freq_min=15)
            if daily:
                scenarios[name]["measured_daily"] = daily
                log.info(f"    Daily var frac={daily['daily_var_frac']:.4f}, "
                         f"peak/trough={daily['peak_trough_ratio']:.3f}, "
                         f"ACF(24h)={daily['acf_24h']:.4f}")

            # Weekly seasonality (period=168h = 672 samples)
            weekly = compute_measured_seasonality(series, series, f"{name}_weekly",
                                                  period_h=168.0, freq_min=15)
            if weekly:
                scenarios[name]["measured_weekly"] = weekly

        except FileNotFoundError as e:
            log.warning(f"    Skipped: {e}")
        except Exception as e:
            log.warning(f"    Failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Build comparison table ───────────────────────────────────
    log.info("\n--- Comparison Table ---")
    header = (f"{'Scenario':20s} | {'E[S](h)':>8s} | {'omega_d*E[S]':>12s} | "
              f"{'h(rho)':>8s} | {'Theory_D':>10s} | {'Meas_D_var%':>11s} | "
              f"{'Meas_P/T':>8s} | {'ACF(24h)':>8s} | {'PSD_norm':>8s}")
    log.info(header)
    log.info("-" * len(header))

    table_rows = []
    for name, s in scenarios.items():
        md = s.get("measured_daily", {})
        row = {
            "scenario": name,
            "workload_type": s["workload"],
            "E_S_h": s["E_S_h"],
            "rho": s["rho"],
            "alpha_d": s["alpha_d"],
            "alpha_w": s.get("alpha_w", np.nan),
            "omega_d_ES": s["omega_d_ES"],
            "omega_w_ES": s.get("omega_w_ES", np.nan),
            "h_rho": s["h_rho"],
            "theory_daily_attenuation": s["theory_daily"],
            "theory_weekly_attenuation": s.get("theory_weekly", np.nan),
            "measured_daily_var_frac": md.get("daily_var_frac", np.nan),
            "measured_peak_trough_ratio": md.get("peak_trough_ratio", np.nan),
            "measured_acf_24h": md.get("acf_24h", np.nan),
            "measured_psd_norm": md.get("psd_peak_norm", np.nan),
            "assumed_params": s.get("assumed", False),
        }
        mw = s.get("measured_weekly", {})
        row["measured_weekly_var_frac"] = mw.get("daily_var_frac", np.nan)
        table_rows.append(row)

        log.info(f"  {name:20s} | {s['E_S_h']:8.3f} | {s['omega_d_ES']:12.4f} | "
                 f"{s['h_rho']:8.3f} | {s['theory_daily']:10.6f} | "
                 f"{md.get('daily_var_frac', np.nan):11.4f} | "
                 f"{md.get('peak_trough_ratio', np.nan):8.3f} | "
                 f"{md.get('acf_24h', np.nan):8.4f} | "
                 f"{md.get('psd_peak_norm', np.nan):8.4f}")

    # ── Ranking validation ───────────────────────────────────────
    log.info("\n--- Ranking Validation ---")
    log.info("Expected: Azure > Alibaba > Helios > MIT (daily seasonality)")

    theory_rank = sorted(scenarios.items(), key=lambda x: x[1]["theory_daily"], reverse=True)
    log.info(f"  Theory ranking: {[r[0] for r in theory_rank]}")

    measured_rank = sorted(
        [(n, s) for n, s in scenarios.items() if "measured_daily" in s],
        key=lambda x: x[1]["measured_daily"].get("daily_var_frac", 0),
        reverse=True
    )
    log.info(f"  Measured ranking (var frac): {[r[0] for r in measured_rank]}")

    acf_rank = sorted(
        [(n, s) for n, s in scenarios.items() if "measured_daily" in s],
        key=lambda x: x[1]["measured_daily"].get("acf_24h", 0),
        reverse=True
    )
    log.info(f"  Measured ranking (ACF 24h): {[r[0] for r in acf_rank]}")

    # ── Save results JSON ────────────────────────────────────────
    output = {
        "description": "Queueing theory validation: Theorem 1 spectral attenuation vs measured",
        "formula": "S_P(omega_d)/S_P(0) ~ alpha_d^2 / [1 + (omega_d*E[S])^2 * h(rho)]",
        "h_rho_formula": "h(rho) = rho^2 / (1-rho)^2",
        "omega_d_rad_per_h": OMEGA_D,
        "omega_w_rad_per_h": OMEGA_W,
        "scenarios": table_rows,
        "theory_ranking_daily": [r[0] for r in theory_rank],
        "measured_ranking_daily_var": [r[0] for r in measured_rank],
        "measured_ranking_acf24h": [r[0] for r in acf_rank],
    }

    json_path = results_path(RESULT_DIR, "data", "queueing_theory_validation.json")
    save_json(output, json_path)
    log.info(f"\nSaved results to {json_path}")

    # ── Generate figure ──────────────────────────────────────────
    plot_theory_vs_measured(scenarios, table_rows)

    log.info("\n=== Done ===")


# ── Plotting ─────────────────────────────────────────────────────────

def plot_theory_vs_measured(scenarios, table_rows):
    """
    Figure: Theory vs measured daily seasonality.

    Panel (a): Theoretical attenuation curves + measured points
               x-axis = omega_d * E[S],  y-axis = seasonality strength
    Panel (b): Bar chart comparing theory and measured for each scenario
    """
    setup_journal_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # --- Panel (a): Attenuation curves in omega*E[S] space ---

    # Draw theoretical curves for representative alpha_d and rho values
    ES_range = np.logspace(-3, 1.5, 500)  # E[S] from 0.001 to ~30 hours
    omega_ES_range = OMEGA_D * ES_range

    # Reference curves for different rho values
    rho_ref = [0.5, 0.75, 0.9]
    alpha_ref = 0.5  # reference alpha_d
    for rho_val in rho_ref:
        att_curve = alpha_ref**2 / (1 + (omega_ES_range)**2 * h_rho(rho_val))
        # Normalize to max=1
        att_curve_norm = att_curve / (alpha_ref**2)
        ax1.plot(omega_ES_range, att_curve_norm, '--', linewidth=0.6,
                 color='gray', alpha=0.5)
        # Label at right end
        ax1.text(omega_ES_range[-1] * 1.05, att_curve_norm[-1],
                 f"$\\rho$={rho_val:.2f}", fontsize=6, color='gray', va='center')

    # Plot each scenario as a point
    for name, s in scenarios.items():
        x_val = s["omega_d_ES"]
        y_theory = s["theory_daily"]
        # Normalize: divide by alpha_d^2 to get the 1/[1+(omega*ES)^2*h(rho)] part
        y_att_only = 1.0 / (1 + x_val**2 * s["h_rho"])

        ax1.scatter(x_val, y_att_only, color=s["color"], marker=s["marker"],
                    s=80, zorder=5, edgecolors='black', linewidths=0.5,
                    label=f"{name}")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\omega_d \cdot E[S]$")
    ax1.set_ylabel(r"$\frac{1}{1 + (\omega_d E[S])^2 \cdot h(\rho)}$", fontsize=9)
    ax1.set_title("(a) Spectral Attenuation Factor", fontsize=10)
    ax1.legend(fontsize=6.5, loc="lower left")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlim(1e-3, 50)
    ax1.set_ylim(1e-4, 1.5)

    # Add annotation: high omega*E[S] = strong smoothing = weak seasonality
    ax1.annotate("Strong\nsmoothing",
                 xy=(10, 5e-4), fontsize=6, ha="center", color="gray")
    ax1.annotate("Weak\nsmoothing",
                 xy=(0.005, 0.5), fontsize=6, ha="center", color="gray")

    # --- Panel (b): Grouped bar chart: theory_daily vs measured metrics ---

    names = list(scenarios.keys())
    short_names = ["MIT", "Helios", "Alibaba", "Azure"]
    x_pos = np.arange(len(names))
    bar_w = 0.25

    # Theory: normalized attenuation (alpha_d^2 * attenuation_factor)
    theory_vals = [scenarios[n]["theory_daily"] for n in names]
    # Normalize all to max for visual comparison
    max_theory = max(theory_vals) if max(theory_vals) > 0 else 1
    theory_norm = [v / max_theory for v in theory_vals]

    # Measured: daily variance fraction
    meas_var = []
    for n in names:
        md = scenarios[n].get("measured_daily", {})
        meas_var.append(md.get("daily_var_frac", 0))
    max_meas = max(meas_var) if max(meas_var) > 0 else 1
    meas_norm = [v / max_meas for v in meas_var]

    # Measured: ACF(24h)
    meas_acf = []
    for n in names:
        md = scenarios[n].get("measured_daily", {})
        meas_acf.append(max(0, md.get("acf_24h", 0)))
    max_acf = max(meas_acf) if max(meas_acf) > 0 else 1
    acf_norm = [v / max_acf for v in meas_acf]

    bars1 = ax2.bar(x_pos - bar_w, theory_norm, bar_w, label="Theory (norm.)",
                    color=[scenarios[n]["color"] for n in names], alpha=0.9,
                    edgecolor="black", linewidth=0.4)
    bars2 = ax2.bar(x_pos, meas_norm, bar_w, label="STL Var. Frac. (norm.)",
                    color=[scenarios[n]["color"] for n in names], alpha=0.5,
                    edgecolor="black", linewidth=0.4, hatch="//")
    bars3 = ax2.bar(x_pos + bar_w, acf_norm, bar_w, label="ACF(24h) (norm.)",
                    color=[scenarios[n]["color"] for n in names], alpha=0.5,
                    edgecolor="black", linewidth=0.4, hatch="\\\\")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, fontsize=8)
    ax2.set_ylabel("Normalized Seasonality Strength")
    ax2.set_title("(b) Theory vs Measured Daily Seasonality", fontsize=10)
    ax2.legend(fontsize=6, loc="upper left")
    ax2.set_ylim(0, 1.35)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = results_path(RESULT_DIR, "figures", "theory_vs_measured")
    save_figure(fig, fig_path)
    log.info(f"Saved figure to {fig_path}.svg")


if __name__ == "__main__":
    main()
