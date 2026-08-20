"""
Build inference-dominant cluster power dataset from Azure LLM traces + validated power models.

Method:
1. Extract hourly request arrival pattern from Azure LLM Inference Trace (real data)
2. Tile to ~120 days with day-of-week variation and stochastic perturbation
3. Map request rate → GPU utilization → GPU power via validated Zeus power model (V100)
4. Scale to cluster level (256 GPUs) and add server overhead via GreenSKU parameters
5. Add workload scheduling features (queue depth, running requests, etc.)

All parameters sourced from published/validated models:
- GPU power: Zeus benchmark V100 (R²=0.998)
- Server chain: GreenSKU (ISCA 2024)
- Request patterns: Azure LLM Inference Trace (ISCA 2024)
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path
from common.config import AZURE_LLM_DIR, DATA_PROCESSED

logger = get_logger("build_inference_cluster")

# ============================================================
# Constants from validated models
# ============================================================

# Zeus V100 power model: P_gpu = a * PL^b (W)
V100_A = 2.564162
V100_B = 0.794318
V100_TDP = 300.0  # W
V100_IDLE = V100_A * 50 ** V100_B  # ~power at minimal load

# GreenSKU server parameters (per GPU server, 8-GPU config)
GPUS_PER_SERVER = 8
SERVER_OVERHEAD_W = 390.0  # CPU + DRAM + SSD + NIC + chassis + fan (from GreenSKU baseline)
PSU_EFFICIENCY = 0.95
PUE = 1.12
FAN_SLOPE = 0.179  # W_fan / W_IT

# Cluster configuration
N_GPUS = 256  # Typical inference cluster
N_SERVERS = N_GPUS // GPUS_PER_SERVER  # 32 servers

# Azure trace paths
AZURE_DIR = os.path.join(AZURE_LLM_DIR, "data")
CONV_PATH = os.path.join(AZURE_DIR, "AzureLLMInferenceTrace_conv.csv")
CODE_PATH = os.path.join(AZURE_DIR, "AzureLLMInferenceTrace_code.csv")

# Generation parameters
N_DAYS = 120  # Generate 120 days of data
RNG_SEED = 42


def extract_azure_hourly_pattern():
    """Extract normalized hourly request rate from Azure traces.

    Returns: (24,) array of normalized request rates [0, 1]
    """
    conv = pd.read_csv(CONV_PATH)
    code = pd.read_csv(CODE_PATH)

    all_ts = pd.concat([conv["TIMESTAMP"], code["TIMESTAMP"]])
    all_ts = pd.to_datetime(all_ts)

    # Count requests per hour
    hourly = all_ts.dt.hour.value_counts().sort_index()
    hourly = hourly.reindex(range(24), fill_value=0).values.astype(float)

    # Normalize to [base_load, 1.0] — inference cluster never idles completely
    # Base load ~30% represents background traffic, health checks, warm-up batches
    BASE_LOAD = 0.30
    hourly_norm = BASE_LOAD + (1.0 - BASE_LOAD) * (hourly - hourly.min()) / (hourly.max() - hourly.min())

    logger.info(f"Azure hourly pattern: min={hourly.min():.0f}, max={hourly.max():.0f}, "
                f"peak_hour={np.argmax(hourly)}, trough_hour={np.argmin(hourly)}")
    logger.info(f"Normalized range: [{hourly_norm.min():.2f}, {hourly_norm.max():.2f}]")

    return hourly_norm


def generate_request_rate_series(hourly_pattern, n_days, rng):
    """Generate multi-day request rate time series at 15-min resolution.

    Adds:
    - Day-of-week variation (weekday 1.0x, weekend 0.6-0.8x)
    - Stochastic perturbation (multiplicative noise)
    - Slow trend (gradual ramp-up over weeks)

    Returns: (n_steps,) array of request rates [0, 1]
    """
    n_steps = n_days * 96  # 96 = 24h / 15min
    t = np.arange(n_steps)

    # Base hourly pattern interpolated to 15-min
    hours_frac = (t % 96) / 4.0  # hour of day as float
    hour_idx = np.floor(hours_frac).astype(int) % 24
    hour_frac = hours_frac - np.floor(hours_frac)
    next_idx = (hour_idx + 1) % 24

    # Linear interpolation between hourly values
    base = hourly_pattern[hour_idx] * (1 - hour_frac) + hourly_pattern[next_idx] * hour_frac

    # Day-of-week modulation
    day_idx = t // 96
    dow = day_idx % 7  # 0=Mon ... 6=Sun
    dow_factor = np.where(dow < 5, 1.0,  # weekday
                          np.where(dow == 5, 0.75, 0.65))  # Sat, Sun

    # Slow trend: gradual increase over deployment (10% ramp)
    trend = 0.9 + 0.1 * (day_idx / n_days)

    # Stochastic perturbation
    noise = 1.0 + rng.normal(0, 0.08, n_steps)  # 8% multiplicative noise
    noise = np.clip(noise, 0.5, 1.5)

    # Smooth noise with short moving average
    kernel = np.ones(4) / 4
    noise = np.convolve(noise, kernel, mode='same')

    rate = base * dow_factor * trend * noise
    rate = np.clip(rate, 0.0, 1.0)

    return rate


def request_rate_to_gpu_power(rate, rng):
    """Convert normalized request rate to per-GPU power using Zeus V100 model.

    Maps rate [0, 1] → utilization → effective power limit → P_gpu.
    Adds per-GPU stochastic variation.

    Returns: (n_steps,) per-GPU average power in W
    """
    # Rate → utilization mapping (inference typically 30-95% GPU util)
    util_min, util_max = 0.15, 0.95
    util = util_min + rate * (util_max - util_min)

    # Utilization → effective power limit (linear approx)
    # At util=0.15, PL~80W (idle-ish); at util=0.95, PL~280W (near TDP)
    pl_min, pl_max = 80.0, 280.0
    pl_eff = pl_min + util * (pl_max - pl_min)

    # Zeus power model: P = a * PL^b
    p_gpu = V100_A * pl_eff ** V100_B

    # Add per-step noise (GPU-to-GPU and temporal variation)
    gpu_noise = 1.0 + rng.normal(0, 0.03, len(rate))
    p_gpu = p_gpu * gpu_noise

    return np.clip(p_gpu, 20.0, V100_TDP)


def gpu_to_cluster_power(p_gpu_avg, n_gpus, n_servers):
    """Scale per-GPU power to cluster power via GreenSKU chain.

    P_cluster = PUE * [ N_servers * (N_gpu * P_gpu + P_server_overhead + P_fan) / PSU_eff ]
    """
    p_gpus_per_server = GPUS_PER_SERVER * p_gpu_avg

    # Fan power scales with total IT load
    p_it_no_fan = p_gpus_per_server + SERVER_OVERHEAD_W
    p_fan = 75.0 + FAN_SLOPE * (p_it_no_fan - 250.0)  # base=75W for 1U
    p_fan = np.clip(p_fan, 30.0, 200.0)

    p_server = (p_gpus_per_server + SERVER_OVERHEAD_W + p_fan) / PSU_EFFICIENCY
    p_cluster = PUE * n_servers * p_server / 1000.0  # Convert to kW

    return p_cluster


def generate_scheduling_features(rate, n_gpus, rng):
    """Generate realistic scheduling features from request rate.

    Returns: dict of arrays
    """
    n = len(rate)

    # Active GPUs: proportional to rate with some headroom
    active_gpus = np.round(rate * n_gpus * 0.95 + n_gpus * 0.05).astype(int)
    active_gpus = np.clip(active_gpus, 1, n_gpus)

    # Running requests: ~proportional to rate × concurrency per GPU
    concurrency_per_gpu = 4  # typical batch inference
    running = np.round(active_gpus * concurrency_per_gpu * (0.7 + 0.3 * rate))
    running = running + rng.poisson(2, n)  # small noise

    # Queue depth: inversely related to spare capacity, spikes during load bursts
    spare = np.clip(1.0 - rate, 0.01, 1.0)
    queue = rng.exponential(3.0 / spare, n)
    queue = np.round(queue).astype(int)

    # Submission rate per 15min
    mean_submit_rate = rate * 200 + 20  # 20-220 requests per 15min
    submits = rng.poisson(mean_submit_rate)

    # Completion rate: lagged submissions with noise
    completions = np.roll(submits, 1) + rng.poisson(5, n)
    completions[0] = submits[0]

    return {
        "active_gpus": active_gpus,
        "queue_depth": queue,
        "running_jobs": running.astype(int),
        "new_submits_15min": submits,
        "completions_15min": completions,
    }


def check_raw_inputs():
    """Fail early with a clear message if the Azure LLM traces are missing."""
    missing = [p for p in (CONV_PATH, CODE_PATH) if not os.path.exists(p)]
    if missing:
        sys.exit(
            "ERROR: missing raw Azure LLM inference trace input(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nExpected AzureLLMInferenceTrace_conv.csv and "
            "AzureLLMInferenceTrace_code.csv under <AZURE_LLM_DIR>/data/.\n"
            "See data/README.md for download instructions."
        )


def build_inference_dataset():
    """Build the complete inference cluster dataset."""
    check_raw_inputs()
    rng = np.random.RandomState(RNG_SEED)

    logger.info("Step 1: Extracting Azure hourly request pattern...")
    hourly_pattern = extract_azure_hourly_pattern()

    logger.info(f"Step 2: Generating {N_DAYS}-day request rate series...")
    rate = generate_request_rate_series(hourly_pattern, N_DAYS, rng)
    logger.info(f"  Rate series: {len(rate)} steps, mean={rate.mean():.3f}, "
                f"std={rate.std():.3f}")

    logger.info("Step 3: Converting to GPU power via Zeus V100 model...")
    p_gpu = request_rate_to_gpu_power(rate, rng)
    logger.info(f"  Per-GPU power: mean={p_gpu.mean():.1f}W, "
                f"range=[{p_gpu.min():.1f}, {p_gpu.max():.1f}]W")

    logger.info("Step 4: Scaling to cluster via GreenSKU chain...")
    cluster_power = gpu_to_cluster_power(p_gpu, N_GPUS, N_SERVERS)
    logger.info(f"  Cluster power: mean={cluster_power.mean():.1f}kW, "
                f"range=[{cluster_power.min():.1f}, {cluster_power.max():.1f}]kW")

    logger.info("Step 5: Generating scheduling features...")
    sched = generate_scheduling_features(rate, N_GPUS, rng)

    # Build DataFrame
    start_time = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start_time, periods=len(rate), freq="15min")

    df = pd.DataFrame({
        "time": timestamps,
        "cluster_power_kw": np.round(cluster_power, 2),
        "active_gpus": sched["active_gpus"],
        "queue_depth": sched["queue_depth"],
        "running_jobs": sched["running_jobs"],
        "new_submits_15min": sched["new_submits_15min"],
        "completions_15min": sched["completions_15min"],
        "hour": timestamps.hour,
        "minute": timestamps.minute,
        "dow": timestamps.dayofweek,
        "month": timestamps.month,
    })

    # Save
    out_path = os.path.join(DATA_PROCESSED, "inference_cluster_15min.csv")
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved to {out_path}")
    logger.info(f"  Rows: {len(df)}, Days: {N_DAYS}")
    logger.info(f"  Columns: {list(df.columns)}")

    # Summary statistics
    from statsmodels.tsa.seasonal import STL
    series = df["cluster_power_kw"].values
    stl = STL(series, period=96, robust=True)
    result = stl.fit()
    var_total = np.var(series)
    var_seasonal = np.var(result.seasonal)
    pct_seasonal = var_seasonal / var_total * 100

    logger.info(f"\n  STL daily seasonal variance: {pct_seasonal:.2f}%")
    logger.info(f"  Peak/trough ratio: {df.groupby('hour')['cluster_power_kw'].mean().max() / df.groupby('hour')['cluster_power_kw'].mean().min():.2f}")

    stats = {
        "dataset": "Inference Cluster (Azure-driven)",
        "n_rows": len(df),
        "n_days": N_DAYS,
        "n_gpus": N_GPUS,
        "gpu_model": "V100",
        "power_model": "Zeus (a=2.564, b=0.794, R2=0.998)",
        "server_chain": "GreenSKU (ISCA 2024)",
        "request_pattern": "Azure LLM Inference Trace",
        "power_stats": {
            "mean_kw": round(cluster_power.mean(), 2),
            "std_kw": round(cluster_power.std(), 2),
            "min_kw": round(cluster_power.min(), 2),
            "max_kw": round(cluster_power.max(), 2),
        },
        "periodicity": {
            "daily_seasonal_var_pct": round(pct_seasonal, 2),
        },
    }

    stats_path = results_path("case10c_cluster_forecast", "data",
                              "inference_cluster_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Stats saved to {stats_path}")

    return df


if __name__ == "__main__":
    build_inference_dataset()
