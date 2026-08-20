"""
Queueing-theoretic parameter extraction from MIT Supercloud slurm-log.

Validates the M/G/k model in Section II of the paper:
  - Arrival rate lambda(t) with daily/weekly sinusoidal modulation
  - Service time distribution E[S]
  - System utilization rho(t) = occupied_GPUs / total_GPUs
  - Weekly arrival pattern by day-of-week

Reference cluster: TX-Gaia, 448 x V100 GPUs, ~234 days (2021 Jan-Oct).
"""

import os
import sys
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import lognorm

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json
from common.plotting import save_figure, setup_journal_style
from common.config import SUPERCLOUD_DIR, COLORS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

LOG = get_logger("queueing_theory")

# ── Constants ──────────────────────────────────────────────────────────────
SLURM_LOG = os.path.join(SUPERCLOUD_DIR, "slurm-log.csv")
TOTAL_GPUS = 448  # TX-Gaia cluster
CASE_NAME = "case10c_cluster_forecast"

# SLURM state codes
STATE_COMPLETED = 3
STATE_CANCELLED = 4
STATE_FAILED = 5
STATE_TIMEOUT = 6

# SLURM TRES keys (Trackable Resources)
TRES_KEY_GPU = "1002"  # gres/gpu


# ── Helpers ────────────────────────────────────────────────────────────────

def parse_tres_gpu(tres_str):
    """Extract GPU count from TRES allocation string like '1=20,2=170000,1002=2'."""
    if pd.isna(tres_str):
        return 0
    for kv in str(tres_str).split(","):
        parts = kv.split("=")
        if len(parts) == 2 and parts[0].strip() == TRES_KEY_GPU:
            return int(parts[1])
    return 0


def sinusoidal_daily_weekly(t_hours, lam_bar, alpha_d, phi_d, alpha_w, phi_w):
    """
    lambda(t) = lam_bar * [1 + alpha_d * sin(2*pi*t/24 + phi_d)
                              + alpha_w * sin(2*pi*t/168 + phi_w)]
    """
    return lam_bar * (
        1
        + alpha_d * np.sin(2 * np.pi * t_hours / 24.0 + phi_d)
        + alpha_w * np.sin(2 * np.pi * t_hours / 168.0 + phi_w)
    )


# ── Data Loading ───────────────────────────────────────────────────────────

def load_slurm_log():
    """Load and preprocess slurm-log.csv."""
    if not os.path.exists(SLURM_LOG):
        LOG.error(
            "MIT SuperCloud slurm-log not found: %s\n"
            "Download the MIT SuperCloud Datacenter Challenge dataset "
            "(see data/README.md) and place slurm-log.csv there.",
            SLURM_LOG,
        )
        sys.exit(1)
    LOG.info("Loading slurm-log from %s", SLURM_LOG)
    df = pd.read_csv(SLURM_LOG)
    LOG.info("Raw rows: %d", len(df))

    # Parse timestamps to datetime (UTC)
    for col in ["time_submit", "time_eligible", "time_start", "time_end"]:
        df[col + "_dt"] = pd.to_datetime(df[col], unit="s", utc=True)

    # Parse GPU count from tres_alloc
    df["gpu_count"] = df["tres_alloc"].apply(parse_tres_gpu)

    # Service time (seconds) for completed jobs
    df["service_time_s"] = df["time_end"] - df["time_start"]

    # Wait time (seconds)
    df["wait_time_s"] = df["time_start"] - df["time_submit"]

    # Filter: only jobs that actually ran (state=COMPLETED, FAILED, TIMEOUT, CANCELLED with runtime)
    ran_mask = df["state"].isin([STATE_COMPLETED, STATE_FAILED, STATE_TIMEOUT, STATE_CANCELLED])
    df_ran = df[ran_mask & (df["service_time_s"] > 0)].copy()
    LOG.info("Jobs that ran (service_time > 0): %d / %d", len(df_ran), len(df))

    return df, df_ran


# ── 1. Arrival Rate Analysis ──────────────────────────────────────────────

def compute_arrival_rate(df):
    """
    Compute hourly arrival rate lambda(t) and fit sinusoidal model.
    Returns dict with lambda_bar, alpha_d, alpha_w, phi_d, phi_w, hourly_rate.
    """
    LOG.info("Computing arrival rate lambda(t)...")

    submit_dt = df["time_submit_dt"]
    t_min, t_max = submit_dt.min(), submit_dt.max()
    total_days = (t_max - t_min).total_seconds() / 86400
    LOG.info("Time span: %.1f days (%s to %s)", total_days, t_min, t_max)

    # Hourly bins
    hourly = df.set_index("time_submit_dt").resample("1h").size()
    hourly = hourly.reindex(
        pd.date_range(t_min.floor("h"), t_max.ceil("h"), freq="1h", tz="UTC"),
        fill_value=0,
    )

    # Average by hour-of-day (24h profile)
    hourly_df = hourly.reset_index()
    hourly_df.columns = ["timestamp", "count"]
    hourly_df["hour"] = hourly_df["timestamp"].dt.hour
    profile_24h = hourly_df.groupby("hour")["count"].mean()

    # Average by day-of-week
    hourly_df["dow"] = hourly_df["timestamp"].dt.dayofweek  # Mon=0, Sun=6
    profile_dow = hourly_df.groupby("dow")["count"].mean()

    # Global mean arrival rate (jobs/hour)
    lambda_bar = hourly_df["count"].mean()
    LOG.info("lambda_bar = %.2f jobs/hour", lambda_bar)

    # Fit sinusoidal model to hourly time series
    t_hours = (hourly_df["timestamp"] - hourly_df["timestamp"].iloc[0]).dt.total_seconds() / 3600.0
    y = hourly_df["count"].values.astype(float)

    try:
        popt, pcov = curve_fit(
            sinusoidal_daily_weekly,
            t_hours.values,
            y,
            p0=[lambda_bar, 0.2, 0.0, 0.1, 0.0],
            bounds=(
                [0, 0, -np.pi, 0, -np.pi],
                [np.inf, 1.0, np.pi, 1.0, np.pi],
            ),
            maxfev=20000,
        )
        lam_fit, alpha_d, phi_d, alpha_w, phi_w = popt
        perr = np.sqrt(np.diag(pcov))
        LOG.info("Fit: lambda_bar=%.2f, alpha_d=%.4f, alpha_w=%.4f", lam_fit, alpha_d, alpha_w)
        LOG.info("Fit stderr: %.4f, %.4f, %.4f, %.4f, %.4f", *perr)

        # R-squared
        y_pred = sinusoidal_daily_weekly(t_hours.values, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        LOG.info("R^2 = %.4f", r_squared)
    except Exception as e:
        LOG.warning("Sinusoidal fit failed: %s", e)
        lam_fit, alpha_d, phi_d, alpha_w, phi_w = lambda_bar, 0, 0, 0, 0
        r_squared = 0
        perr = [0] * 5

    return {
        "lambda_bar": float(lam_fit),
        "alpha_d": float(alpha_d),
        "phi_d": float(phi_d),
        "alpha_w": float(alpha_w),
        "phi_w": float(phi_w),
        "r_squared": float(r_squared),
        "fit_stderr": [float(x) for x in perr],
        "profile_24h": {int(k): float(v) for k, v in profile_24h.items()},
        "profile_dow": {int(k): float(v) for k, v in profile_dow.items()},
        "total_days": float(total_days),
        "total_jobs": int(len(df)),
        # For plotting
        "_hourly_df": hourly_df,
        "_t_hours": t_hours,
        "_popt": [lam_fit, alpha_d, phi_d, alpha_w, phi_w],
    }


# ── 2. Service Time Distribution ──────────────────────────────────────────

def compute_service_time(df_ran):
    """Compute service time statistics for all jobs and GPU-only jobs."""
    LOG.info("Computing service time distribution...")

    results = {}

    for label, mask_fn in [
        ("all", lambda d: d),
        ("gpu_only", lambda d: d[d["gpu_count"] > 0]),
    ]:
        subset = mask_fn(df_ran)
        s = subset["service_time_s"]
        LOG.info("  %s: %d jobs", label, len(subset))

        stats = {
            "count": int(len(subset)),
            "mean_s": float(s.mean()),
            "median_s": float(s.median()),
            "std_s": float(s.std()),
            "p25_s": float(s.quantile(0.25)),
            "p75_s": float(s.quantile(0.75)),
            "p90_s": float(s.quantile(0.90)),
            "p99_s": float(s.quantile(0.99)),
            "min_s": float(s.min()),
            "max_s": float(s.max()),
            "mean_h": float(s.mean() / 3600),
            "median_h": float(s.median() / 3600),
        }
        results[label] = stats
        LOG.info("    E[S] = %.1f s (%.2f h), median = %.1f s", stats["mean_s"], stats["mean_h"], stats["median_s"])

    # By GPU count
    gpu_groups = df_ran[df_ran["gpu_count"] > 0].groupby("gpu_count")["service_time_s"]
    by_gpu = {}
    for ngpu, group in gpu_groups:
        by_gpu[int(ngpu)] = {
            "count": int(len(group)),
            "mean_s": float(group.mean()),
            "median_s": float(group.median()),
            "mean_h": float(group.mean() / 3600),
        }
    results["by_gpu_count"] = by_gpu

    # Store raw arrays for plotting
    results["_all_service_s"] = df_ran["service_time_s"].values
    results["_gpu_service_s"] = df_ran[df_ran["gpu_count"] > 0]["service_time_s"].values

    return results


# ── 3. Utilization Time Series ─────────────────────────────────────────────

def compute_utilization(df_ran, resolution_min=15):
    """
    Compute GPU utilization rho(t) at given resolution.
    For each time bin, count how many GPUs are occupied.
    """
    LOG.info("Computing utilization rho(t) at %d-min resolution...", resolution_min)

    # Only GPU jobs
    gpu_jobs = df_ran[df_ran["gpu_count"] > 0].copy()
    LOG.info("  GPU jobs for utilization: %d", len(gpu_jobs))

    t_min = df_ran["time_start"].min()
    t_max = df_ran["time_end"].max()

    # Time bins (unix timestamps)
    bin_edges = np.arange(t_min, t_max + resolution_min * 60, resolution_min * 60)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # For each bin, sum GPUs of jobs that overlap [bin_start, bin_end]
    gpu_starts = gpu_jobs["time_start"].values
    gpu_ends = gpu_jobs["time_end"].values
    gpu_counts = gpu_jobs["gpu_count"].values

    n_bins = len(bin_centers)
    utilization = np.zeros(n_bins)

    # Vectorized: for each bin check overlap
    # This is O(n_bins * n_jobs) which is too slow. Use event-based approach.
    LOG.info("  Building event-based utilization (efficient)...")

    events = []
    for i in range(len(gpu_starts)):
        events.append((gpu_starts[i], gpu_counts[i]))   # job start: +gpus
        events.append((gpu_ends[i], -gpu_counts[i]))     # job end: -gpus
    events.sort(key=lambda x: x[0])

    # Walk through events to build piecewise-constant GPU count
    event_times = np.array([e[0] for e in events])
    event_deltas = np.array([e[1] for e in events])

    # Cumulative GPU count at each event
    cum_gpus = np.cumsum(event_deltas)

    # Sample at bin centers using searchsorted
    idx = np.searchsorted(event_times, bin_centers, side="right") - 1
    idx = np.clip(idx, 0, len(cum_gpus) - 1)
    utilization = cum_gpus[idx] / TOTAL_GPUS

    # Convert to datetime
    bin_datetimes = pd.to_datetime(bin_centers, unit="s", utc=True)

    rho_mean = float(np.mean(utilization))
    rho_median = float(np.median(utilization))
    rho_max = float(np.max(utilization))
    rho_p95 = float(np.percentile(utilization, 95))

    LOG.info("  rho: mean=%.3f, median=%.3f, max=%.3f, p95=%.3f", rho_mean, rho_median, rho_max, rho_p95)

    return {
        "mean": rho_mean,
        "median": rho_median,
        "max": rho_max,
        "p95": rho_p95,
        "std": float(np.std(utilization)),
        "resolution_min": resolution_min,
        "n_bins": int(n_bins),
        "_bin_datetimes": bin_datetimes,
        "_utilization": utilization,
    }


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_arrival_rate_24h(arrival, fig_dir):
    """Plot 24h arrival rate profile with sinusoidal fit."""
    setup_journal_style()

    fig, ax = plt.subplots(figsize=(5, 3.2))

    hours = np.arange(24)
    rates = [arrival["profile_24h"][h] for h in hours]

    # Bar chart of empirical rates
    ax.bar(hours, rates, width=0.7, color=COLORS["primary"], alpha=0.6, label="Empirical", zorder=2)

    # Sinusoidal fit overlay
    t_fine = np.linspace(0, 24, 200)
    popt = arrival["_popt"]
    # For 24h profile, evaluate fit at each hour of day averaged over all days
    # Simpler: just plot the daily component
    lam_bar, alpha_d, phi_d, alpha_w, phi_w = popt
    y_fit = lam_bar * (1 + alpha_d * np.sin(2 * np.pi * t_fine / 24.0 + phi_d))
    ax.plot(t_fine, y_fit, color=COLORS["quaternary"], linewidth=1.5, label="Sinusoidal fit", zorder=3)

    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Average Arrival Rate (jobs/h)")
    ax.set_title("Job Arrival Rate: 24-Hour Profile")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    # Annotation box
    text = (
        f"$\\bar{{\\lambda}}$ = {lam_bar:.1f} jobs/h\n"
        f"$\\alpha_d$ = {alpha_d:.4f}\n"
        f"$\\alpha_w$ = {alpha_w:.4f}\n"
        f"$R^2$ = {arrival['r_squared']:.4f}"
    )
    ax.text(
        0.98, 0.97, text,
        transform=ax.transAxes, fontsize=7,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "queueing_arrival_rate_24h"))
    LOG.info("Saved arrival rate 24h plot")


def plot_service_time_dist(service, fig_dir):
    """Plot service time distribution histogram (log scale)."""
    setup_journal_style()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    for ax, key, title, color in [
        (axes[0], "_all_service_s", "All Jobs", COLORS["primary"]),
        (axes[1], "_gpu_service_s", "GPU Jobs Only", COLORS["tertiary"]),
    ]:
        data = service[key]
        data_h = data / 3600.0  # Convert to hours

        # Log-spaced bins
        bins = np.logspace(np.log10(max(data_h.min(), 1/60)), np.log10(data_h.max()), 80)
        ax.hist(data_h, bins=bins, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Service Time (hours)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Stats
        stats_key = "all" if "all" in key else "gpu_only"
        s = service[stats_key]
        text = (
            f"E[S] = {s['mean_h']:.2f} h\n"
            f"Median = {s['median_h']:.2f} h\n"
            f"N = {s['count']:,}"
        )
        ax.text(
            0.97, 0.97, text,
            transform=ax.transAxes, fontsize=7,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
        )

    fig.suptitle("Service Time Distribution (MIT Supercloud)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "queueing_service_time_dist"))
    LOG.info("Saved service time distribution plot")


def plot_utilization_ts(util, fig_dir):
    """Plot GPU utilization rho(t) time series."""
    setup_journal_style()

    fig, ax = plt.subplots(figsize=(8, 3.0))

    dt = util["_bin_datetimes"]
    rho = util["_utilization"]

    # Downsample for plotting if too many points
    if len(dt) > 50000:
        step = len(dt) // 50000
        dt_plot = dt[::step]
        rho_plot = rho[::step]
    else:
        dt_plot = dt
        rho_plot = rho

    ax.plot(dt_plot, rho_plot, color=COLORS["primary"], linewidth=0.4, rasterized=True)

    # Rolling average (daily)
    window = int(24 * 60 / util["resolution_min"])  # 1 day
    if len(rho) > window:
        rho_smooth = pd.Series(rho).rolling(window, center=True, min_periods=1).mean().values
        if len(dt) > 50000:
            ax.plot(dt_plot, rho_smooth[::step], color=COLORS["quaternary"], linewidth=1.0,
                    label="24h moving avg", zorder=3)
        else:
            ax.plot(dt, rho_smooth, color=COLORS["quaternary"], linewidth=1.0,
                    label="24h moving avg", zorder=3)

    ax.axhline(util["mean"], color=COLORS["gray"], linestyle="--", linewidth=0.8,
               label=f"Mean = {util['mean']:.3f}")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("GPU Utilization $\\rho(t)$")
    ax.set_title("GPU Cluster Utilization (TX-Gaia, 448 V100s)")
    ax.set_ylim(0, min(rho.max() * 1.1, 1.5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True)

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "queueing_utilization_ts"))
    LOG.info("Saved utilization time series plot")


def plot_arrival_weekly(arrival, fig_dir):
    """Plot weekly arrival pattern by day-of-week."""
    setup_journal_style()

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dows = list(range(7))
    rates = [arrival["profile_dow"][d] for d in dows]

    bars = ax.bar(dows, rates, width=0.6, color=COLORS["primary"], alpha=0.7, zorder=2)

    # Highlight weekend
    for i in [5, 6]:
        bars[i].set_color(COLORS["quaternary"])
        bars[i].set_alpha(0.7)

    ax.set_xticks(dows)
    ax.set_xticklabels(dow_labels)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Arrival Rate (jobs/h)")
    ax.set_title("Weekly Arrival Pattern")
    ax.grid(True, axis="y")

    # Weekday vs weekend ratio
    wd_rate = np.mean([rates[i] for i in range(5)])
    we_rate = np.mean([rates[i] for i in [5, 6]])
    ratio = wd_rate / we_rate if we_rate > 0 else float("inf")
    text = f"Weekday avg: {wd_rate:.1f}\nWeekend avg: {we_rate:.1f}\nRatio: {ratio:.2f}x"
    ax.text(
        0.97, 0.97, text,
        transform=ax.transAxes, fontsize=7,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "queueing_arrival_weekly"))
    LOG.info("Saved weekly arrival pattern plot")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    LOG.info("=" * 60)
    LOG.info("Queueing Theory Parameter Extraction — MIT Supercloud")
    LOG.info("=" * 60)

    # Output directories
    data_dir = results_path(CASE_NAME, "data", "placeholder")[:-len("/placeholder")]
    fig_dir = results_path(CASE_NAME, "figures", "placeholder")[:-len("/placeholder")]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    df, df_ran = load_slurm_log()

    # 1. Arrival rate
    arrival = compute_arrival_rate(df)

    # 2. Service time
    service = compute_service_time(df_ran)

    # 3. Utilization
    util = compute_utilization(df_ran, resolution_min=15)

    # ── Save JSON (exclude internal arrays) ──
    json_out = {
        "dataset": "MIT Supercloud TX-Gaia",
        "total_gpus": TOTAL_GPUS,
        "arrival_rate": {
            "lambda_bar_jobs_per_hour": arrival["lambda_bar"],
            "alpha_d": arrival["alpha_d"],
            "phi_d_rad": arrival["phi_d"],
            "alpha_w": arrival["alpha_w"],
            "phi_w_rad": arrival["phi_w"],
            "r_squared": arrival["r_squared"],
            "fit_stderr": arrival["fit_stderr"],
            "total_days": arrival["total_days"],
            "total_jobs": arrival["total_jobs"],
            "profile_24h_jobs_per_hour": arrival["profile_24h"],
            "profile_dow_jobs_per_hour": arrival["profile_dow"],
        },
        "service_time": {
            k: v for k, v in service.items() if not k.startswith("_")
        },
        "utilization": {
            k: v for k, v in util.items() if not k.startswith("_")
        },
        "queueing_model": {
            "model": "M/G/k",
            "k_servers": TOTAL_GPUS,
            "spectral_attenuation_note": (
                "Daily spectral peak ~ alpha_d^2 / [1 + (omega_d * E[S])^2 * h(rho)]. "
                "When E[S] is large (training jobs), the daily peak is attenuated."
            ),
            "omega_d_rad_per_s": 2 * np.pi / (24 * 3600),
            "omega_d_E_S_product_all": (
                2 * np.pi / (24 * 3600) * service["all"]["mean_s"]
            ),
            "omega_d_E_S_product_gpu": (
                2 * np.pi / (24 * 3600) * service["gpu_only"]["mean_s"]
            ),
        },
    }
    json_path = os.path.join(data_dir, "queueing_params_mit.json")
    save_json(json_out, json_path)
    LOG.info("Saved queueing parameters to %s", json_path)

    # ── Generate plots ──
    plot_arrival_rate_24h(arrival, fig_dir)
    plot_service_time_dist(service, fig_dir)
    plot_utilization_ts(util, fig_dir)
    plot_arrival_weekly(arrival, fig_dir)

    # ── Summary ──
    LOG.info("=" * 60)
    LOG.info("SUMMARY")
    LOG.info("=" * 60)
    LOG.info("lambda_bar = %.2f jobs/h", arrival["lambda_bar"])
    LOG.info("alpha_d (daily modulation) = %.4f", arrival["alpha_d"])
    LOG.info("alpha_w (weekly modulation) = %.4f", arrival["alpha_w"])
    LOG.info("E[S] all = %.1f s (%.2f h)", service["all"]["mean_s"], service["all"]["mean_h"])
    LOG.info("E[S] GPU = %.1f s (%.2f h)", service["gpu_only"]["mean_s"], service["gpu_only"]["mean_h"])
    LOG.info("rho_mean = %.3f, rho_max = %.3f", util["mean"], util["max"])
    LOG.info(
        "omega_d * E[S] (GPU) = %.4f  -->  spectral attenuation factor",
        json_out["queueing_model"]["omega_d_E_S_product_gpu"],
    )
    LOG.info("Results in: %s", data_dir)
    LOG.info("Figures in: %s", fig_dir)


if __name__ == "__main__":
    main()
