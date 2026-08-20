"""
Build 15-minute resolution time series for Helios Saturn and Alibaba PAI clusters.

Produces datasets matching the MIT Supercloud format (the original build_dataset.py (now in case10c)):
  time, cluster_load, active_gpus, queue_depth, running_jobs,
  new_submits_15min, completions_15min, hour, minute, dow, month

Data sources:
  1. Helios Saturn: job-level traces reconstructed into 15-min GPU occupancy
     - cluster_load = active_gpu_count (GPUs occupied by running jobs)
     - Total cluster capacity: 2072 GPUs
  2. Alibaba PAI (v2020): machine-level GPU metrics + job scheduling
     - cluster_load = cluster_gpu_util (mean GPU utilization across machines)
     - 1720 GPU machines in cluster

Output:
  <DATA_PROCESSED>/helios_saturn_15min.csv
  <DATA_PROCESSED>/alibaba_pai_15min.csv   (only with --with-pai)
"""

import argparse
import os
import sys
import time as time_module

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger
from common.config import DATA_PROCESSED, HELIOS_TRACE_DIR, ALIBABA_TRACE_DIR

logger = get_logger("build_multi_datasets")

OUTPUT_DIR = DATA_PROCESSED
BIN_SEC = 900  # 15 minutes

# ============================================================
# Helios Saturn
# ============================================================

HELIOS_CSV = os.path.join(HELIOS_TRACE_DIR, "data", "Saturn", "cluster_log.csv")
HELIOS_CAPACITY = 2072  # total GPUs in Saturn cluster


def build_helios_saturn():
    """
    Reconstruct 15-min active GPU count from Helios Saturn job traces.

    For each 15-min bin, count:
      - active_gpu_count: sum of gpu_num for jobs where start_time <= bin < end_time
      - queue_depth: jobs where submit_time <= bin < start_time (waiting)
      - running_jobs: count of running jobs
      - new_submits_15min: jobs submitted in [bin-15min, bin)
      - completions_15min: jobs ended in [bin-15min, bin)

    Uses vectorized bin assignment for efficiency: for each job, determine
    which bins it overlaps, then use np.add.at for accumulation.
    """
    logger.info("=== Building Helios Saturn dataset ===")
    t0 = time_module.time()

    # Load job traces
    logger.info(f"Loading {HELIOS_CSV}...")
    df = pd.read_csv(HELIOS_CSV)
    logger.info(f"  Raw rows: {len(df):,}")

    # Filter out CANCELLED jobs (they never ran)
    n_before = len(df)
    df = df[df["state"] != "CANCELLED"].copy()
    logger.info(f"  After removing CANCELLED: {len(df):,} ({n_before - len(df):,} removed)")

    # Parse timestamps
    df["submit_time"] = pd.to_datetime(df["submit_time"])
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Drop rows with missing critical times
    df = df.dropna(subset=["start_time", "end_time"]).copy()
    logger.info(f"  After dropping NaT: {len(df):,}")

    # Convert to Unix seconds for fast bin computation
    submit_s = (df["submit_time"].astype(np.int64) // 10**9).values
    start_s = (df["start_time"].astype(np.int64) // 10**9).values
    end_s = (df["end_time"].astype(np.int64) // 10**9).values
    gpu_num = df["gpu_num"].values.astype(np.int64)

    # Determine time range (use start/end of actual activity)
    t_min = int(np.min(start_s))
    t_max = int(np.max(end_s))
    # Align to 15-min boundaries
    bin_min = (t_min // BIN_SEC) * BIN_SEC
    bin_max = ((t_max // BIN_SEC) + 1) * BIN_SEC
    n_bins = (bin_max - bin_min) // BIN_SEC
    logger.info(f"  Time range: {pd.Timestamp(bin_min, unit='s')} to "
                f"{pd.Timestamp(bin_max, unit='s')}")
    logger.info(f"  Number of 15-min bins: {n_bins:,}")

    # Initialize arrays
    active_gpus = np.zeros(n_bins, dtype=np.float64)
    running_jobs = np.zeros(n_bins, dtype=np.float64)
    queue_depth = np.zeros(n_bins, dtype=np.float64)
    new_submits = np.zeros(n_bins, dtype=np.float64)
    completions = np.zeros(n_bins, dtype=np.float64)

    # --- Active GPUs and Running Jobs ---
    # For each job, find the bins it overlaps [start_bin, end_bin)
    # and add gpu_num to those bins
    logger.info("  Computing active GPUs and running jobs...")
    start_bins = np.clip((start_s - bin_min) // BIN_SEC, 0, n_bins - 1).astype(np.int64)
    end_bins = np.clip((end_s - bin_min) // BIN_SEC, 0, n_bins - 1).astype(np.int64)

    # Use difference array technique for interval updates: O(N_jobs) instead of O(N_jobs * N_bins)
    # For active_gpus: add gpu_num at start_bin, subtract at end_bin+1
    active_diff = np.zeros(n_bins + 1, dtype=np.float64)
    running_diff = np.zeros(n_bins + 1, dtype=np.float64)

    np.add.at(active_diff, start_bins, gpu_num)
    np.add.at(active_diff, end_bins + 1, -gpu_num)
    active_gpus = np.cumsum(active_diff[:n_bins])

    np.add.at(running_diff, start_bins, 1)
    np.add.at(running_diff, end_bins + 1, -1)
    running_jobs = np.cumsum(running_diff[:n_bins])

    # --- Queue Depth ---
    # Jobs waiting: submit_time <= bin_time < start_time
    # Use difference array: +1 at submit_bin, -1 at start_bin
    logger.info("  Computing queue depth...")
    submit_bins = np.clip((submit_s - bin_min) // BIN_SEC, 0, n_bins - 1).astype(np.int64)
    queue_diff = np.zeros(n_bins + 1, dtype=np.float64)
    np.add.at(queue_diff, submit_bins, 1)
    np.add.at(queue_diff, start_bins, -1)
    queue_depth = np.cumsum(queue_diff[:n_bins])
    # Clip negative values (can happen at boundaries)
    queue_depth = np.maximum(queue_depth, 0)

    # --- New Submits per 15-min bin ---
    logger.info("  Computing submission and completion rates...")
    np.add.at(new_submits, submit_bins, 1)

    # --- Completions per 15-min bin ---
    completion_bins = np.clip((end_s - bin_min) // BIN_SEC, 0, n_bins - 1).astype(np.int64)
    np.add.at(completions, completion_bins, 1)

    # Cap active_gpus at cluster capacity
    # Some TIMEOUT jobs have imprecise end times causing phantom overlap
    active_gpus = np.minimum(active_gpus, HELIOS_CAPACITY)

    # Build DataFrame
    bin_times = pd.date_range(
        start=pd.Timestamp(bin_min, unit="s"),
        periods=n_bins,
        freq="15min",
    )

    result = pd.DataFrame({
        "time": bin_times,
        "active_gpu_count": active_gpus.astype(np.int64),
        "active_gpus": active_gpus.astype(np.int64),  # alias for compatibility
        "queue_depth": queue_depth.astype(np.int64),
        "running_jobs": running_jobs.astype(np.int64),
        "new_submits_15min": new_submits.astype(np.int64),
        "completions_15min": completions.astype(np.int64),
        "hour": bin_times.hour,
        "minute": bin_times.minute,
        "dow": bin_times.dayofweek,
        "month": bin_times.month,
    })

    # Trim leading/trailing zeros (before first job and after last job)
    mask = result["active_gpu_count"] > 0
    if mask.any():
        first_active = mask.idxmax()
        last_active = mask[::-1].idxmax()
        result = result.loc[first_active:last_active].reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, "helios_saturn_15min.csv")
    result.to_csv(out_path, index=False)
    elapsed = time_module.time() - t0

    logger.info(f"\n  Helios Saturn dataset saved to {out_path}")
    logger.info(f"  Rows: {len(result):,}")
    logger.info(f"  Time range: {result['time'].iloc[0]} to {result['time'].iloc[-1]}")
    logger.info(f"  Duration: {(result['time'].iloc[-1] - result['time'].iloc[0]).days} days")
    logger.info(f"  Active GPUs: mean={result['active_gpu_count'].mean():.0f}, "
                f"max={result['active_gpu_count'].max()}, "
                f"capacity={HELIOS_CAPACITY}")
    logger.info(f"  Queue depth: mean={result['queue_depth'].mean():.0f}, "
                f"max={result['queue_depth'].max()}")
    logger.info(f"  Running jobs: mean={result['running_jobs'].mean():.0f}, "
                f"max={result['running_jobs'].max()}")
    logger.info(f"  New submits/15min: mean={result['new_submits_15min'].mean():.1f}")
    logger.info(f"  Completions/15min: mean={result['completions_15min'].mean():.1f}")
    logger.info(f"  Build time: {elapsed:.1f}s")

    return result


# ============================================================
# Alibaba PAI (v2020)
# ============================================================

ALIBABA_DIR = os.path.join(ALIBABA_TRACE_DIR, "cluster-trace-gpu-v2020", "data")
ALIBABA_METRIC = os.path.join(ALIBABA_DIR, "pai_machine_metric.csv")
ALIBABA_JOB = os.path.join(ALIBABA_DIR, "pai_job_table.csv")
ALIBABA_TASK = os.path.join(ALIBABA_DIR, "pai_task_table.csv")
ALIBABA_SPEC = os.path.join(ALIBABA_DIR, "pai_machine_spec.csv")


def build_alibaba_pai():
    """
    Build 15-min aggregated GPU utilization from Alibaba PAI v2020 traces.

    Machine metrics (pai_machine_metric.csv):
      - machine_gpu is total GPU utilization across all GPUs on a machine
        (e.g., 8-GPU machine can have values up to ~800)
      - Timestamps are relative seconds from an epoch (not Unix timestamps)

    Job table (pai_job_table.csv):
      - job_name, inst_id, user, status, start_time, end_time
      - start_time/end_time are same relative seconds

    We normalize machine_gpu by each machine's GPU count (from pai_machine_spec.csv)
    to get per-GPU utilization (0-100 scale), then average across all active machines.
    """
    logger.info("\n=== Building Alibaba PAI dataset ===")
    t0 = time_module.time()

    # Load machine specs to get GPU count per machine
    # Note: Alibaba CSVs have no header row; column names come from .header files
    logger.info("Loading machine specs...")
    spec_cols = ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]
    spec = pd.read_csv(ALIBABA_SPEC, header=None, names=spec_cols)
    gpu_machines = spec[spec["cap_gpu"] > 0][["machine", "cap_gpu"]].copy()
    gpu_count_map = dict(zip(gpu_machines["machine"], gpu_machines["cap_gpu"]))
    logger.info(f"  GPU machines: {len(gpu_count_map)} "
                f"(GPU counts: {sorted(set(gpu_count_map.values()))})")

    # Load machine metrics
    logger.info(f"Loading {ALIBABA_METRIC}...")
    metric_cols = ["worker_name", "machine", "start_time", "end_time",
                   "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr",
                   "machine_gpu", "machine_load_1", "machine_net_receive",
                   "machine_num_worker", "machine_cpu"]
    metric = pd.read_csv(ALIBABA_METRIC, header=None, names=metric_cols)
    logger.info(f"  Raw rows: {len(metric):,}")

    # Filter to GPU machines only (machines with cap_gpu > 0)
    metric = metric[metric["machine"].isin(gpu_count_map)].copy()
    logger.info(f"  After filtering to GPU machines: {len(metric):,}")

    # Drop rows with missing GPU utilization
    metric = metric.dropna(subset=["machine_gpu"]).copy()
    metric = metric[metric["machine_gpu"] >= 0].copy()
    logger.info(f"  After dropping NaN/negative GPU: {len(metric):,}")

    # Normalize GPU utilization by machine GPU count
    # machine_gpu is sum across GPUs, so divide by cap_gpu to get per-GPU average
    metric["cap_gpu"] = metric["machine"].map(gpu_count_map).fillna(1)
    metric["gpu_per_card"] = metric["machine_gpu"] / metric["cap_gpu"]

    # Use the midpoint of [start_time, end_time) as the measurement time
    metric["mid_time"] = (metric["start_time"] + metric["end_time"]) / 2.0

    # Bin into 15-min intervals
    metric["bin"] = (metric["mid_time"] // BIN_SEC).astype(np.int64) * BIN_SEC

    # Aggregate per 15-min bin: mean GPU util, count of records and machines
    logger.info("  Aggregating to 15-min bins...")
    agg = metric.groupby("bin").agg(
        cluster_gpu_util=("gpu_per_card", "mean"),
        gpu_util_max=("gpu_per_card", "max"),
        cpu_util_mean=("machine_cpu", "mean"),
        n_records=("machine_gpu", "count"),
        n_machines=("machine", "nunique"),
    ).reset_index()

    # Convert bin (relative seconds) to datetime
    # The Alibaba trace uses relative seconds; we create synthetic timestamps
    # starting from an arbitrary epoch. The key patterns are diurnal/weekly cycles.
    # Use epoch 0 = 2020-01-01 00:00:00 as reference
    epoch_offset = pd.Timestamp("2020-01-01").timestamp()
    agg["time"] = pd.to_datetime(agg["bin"] + epoch_offset, unit="s")

    agg["hour"] = agg["time"].dt.hour
    agg["minute"] = agg["time"].dt.minute
    agg["dow"] = agg["time"].dt.dayofweek
    agg["month"] = agg["time"].dt.month

    # --- Job-level features from pai_job_table ---
    logger.info(f"Loading {ALIBABA_JOB}...")
    job_cols = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
    jobs = pd.read_csv(ALIBABA_JOB, header=None, names=job_cols)
    logger.info(f"  Raw jobs: {len(jobs):,}")

    # Parse start/end times (relative seconds, float with possible NaN)
    jobs["start_time"] = pd.to_numeric(jobs["start_time"], errors="coerce")
    jobs["end_time"] = pd.to_numeric(jobs["end_time"], errors="coerce")

    # Jobs with valid start_time
    jobs_valid = jobs.dropna(subset=["start_time"]).copy()
    # For running jobs (no end_time), use the max observed time as proxy
    max_time = metric["end_time"].max()
    jobs_valid["end_time_filled"] = jobs_valid["end_time"].fillna(max_time)

    # Bin boundaries
    bin_min = int(agg["bin"].min())
    bin_max = int(agg["bin"].max())
    n_bins = (bin_max - bin_min) // BIN_SEC + 1
    bin_edges = np.arange(bin_min, bin_max + BIN_SEC, BIN_SEC)

    # Running jobs per bin (difference array technique)
    logger.info("  Computing running jobs and queue depth...")
    start_s = jobs_valid["start_time"].values
    end_s = jobs_valid["end_time_filled"].values

    start_bin_idx = np.clip(((start_s - bin_min) / BIN_SEC).astype(np.int64), 0, n_bins - 1)
    end_bin_idx = np.clip(((end_s - bin_min) / BIN_SEC).astype(np.int64), 0, n_bins - 1)

    running_diff = np.zeros(n_bins + 1, dtype=np.float64)
    np.add.at(running_diff, start_bin_idx, 1)
    np.add.at(running_diff, end_bin_idx + 1, -1)
    running_jobs = np.cumsum(running_diff[:n_bins])

    # New submits per bin
    new_submits = np.zeros(n_bins, dtype=np.float64)
    # Use start_time as submit proxy (job_table doesn't have a separate submit_time)
    np.add.at(new_submits, start_bin_idx, 1)

    # Completions per bin
    completions = np.zeros(n_bins, dtype=np.float64)
    # Only count terminated/failed jobs (not still running)
    terminated = jobs_valid[jobs_valid["status"].isin(["Terminated", "Failed"])].copy()
    if len(terminated) > 0:
        term_end_s = terminated["end_time"].dropna().values
        term_end_bin = np.clip(((term_end_s - bin_min) / BIN_SEC).astype(np.int64), 0, n_bins - 1)
        np.add.at(completions, term_end_bin, 1)

    # Queue depth: for Alibaba, we estimate from "Waiting" status jobs
    # Since job_table doesn't have explicit submit_time separate from start_time,
    # queue_depth is approximated as: jobs in "Waiting" or "Running" with no actual start
    # A simpler proxy: total submitted - total started - total completed at each bin
    # But the data doesn't cleanly support this. Use running_jobs as the primary signal.
    # Queue depth = 0 placeholder (not reliably reconstructable from this dataset)
    queue_depth_arr = np.zeros(n_bins, dtype=np.float64)

    # Build the bin-indexed job features
    job_features = pd.DataFrame({
        "bin": bin_edges[:n_bins],
        "running_jobs": running_jobs.astype(np.int64),
        "queue_depth": queue_depth_arr.astype(np.int64),
        "new_submits_15min": new_submits.astype(np.int64),
        "completions_15min": completions.astype(np.int64),
    })

    # Merge with metric aggregation
    result = agg.merge(job_features, on="bin", how="left")
    result = result.fillna(0)

    # Add active_gpus (n_machines is a proxy; each machine has multiple GPUs)
    # Compute total active GPUs from n_machines and their GPU counts
    # Since we can't easily map per-bin machine identity to GPU count,
    # use n_machines * median_gpu_count as approximation
    median_gpus_per_machine = np.median(list(gpu_count_map.values()))
    result["active_gpus"] = (result["n_machines"] * median_gpus_per_machine).astype(np.int64)

    # Select and order columns to match MIT format
    result = result[[
        "time", "cluster_gpu_util", "active_gpus", "queue_depth",
        "running_jobs", "new_submits_15min", "completions_15min",
        "hour", "minute", "dow", "month",
        # Extra Alibaba-specific columns
        "gpu_util_max", "cpu_util_mean", "n_records", "n_machines",
    ]].copy()

    # Sort by time and ensure contiguous
    result = result.sort_values("time").reset_index(drop=True)

    # Trim rows with no data
    mask = result["n_records"] > 0
    if mask.any():
        first_valid = mask.idxmax()
        last_valid = mask[::-1].idxmax()
        result = result.loc[first_valid:last_valid].reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, "alibaba_pai_15min.csv")
    result.to_csv(out_path, index=False)
    elapsed = time_module.time() - t0

    logger.info(f"\n  Alibaba PAI dataset saved to {out_path}")
    logger.info(f"  Rows: {len(result):,}")
    logger.info(f"  Time range: {result['time'].iloc[0]} to {result['time'].iloc[-1]}")
    valid = result[result["n_records"] > 0]
    logger.info(f"  Valid rows (n_records > 0): {len(valid):,} / {len(result):,}")
    logger.info(f"  Cluster GPU util: mean={valid['cluster_gpu_util'].mean():.1f}%, "
                f"max={valid['cluster_gpu_util'].max():.1f}%")
    logger.info(f"  Active GPUs: mean={valid['active_gpus'].mean():.0f}, "
                f"max={valid['active_gpus'].max()}")
    logger.info(f"  Running jobs: mean={valid['running_jobs'].mean():.0f}, "
                f"max={valid['running_jobs'].max()}")
    logger.info(f"  New submits/15min: mean={valid['new_submits_15min'].mean():.1f}")
    logger.info(f"  Completions/15min: mean={valid['completions_15min'].mean():.1f}")
    logger.info(f"  Build time: {elapsed:.1f}s")

    return result


# ============================================================
# Main
# ============================================================

def check_helios_inputs():
    """Fail early with a clear message if the Helios raw trace is missing."""
    if not os.path.exists(HELIOS_CSV):
        sys.exit(
            "ERROR: missing raw Helios input:\n"
            f"  - {HELIOS_CSV}\n"
            "Expected the HeliosData repository checkout under <HELIOS_TRACE_DIR> "
            "(data/Saturn/cluster_log.csv).\n"
            "See data/README.md for download instructions."
        )


def check_alibaba_pai_inputs():
    """Fail early with a clear message if the Alibaba PAI v2020 raw traces are missing."""
    expected = [ALIBABA_SPEC, ALIBABA_METRIC, ALIBABA_JOB]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        sys.exit(
            "ERROR: missing raw Alibaba PAI v2020 input(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nExpected the clusterdata cluster-trace-gpu-v2020 files under "
            "<ALIBABA_TRACE_DIR>/cluster-trace-gpu-v2020/data/ "
            "(a separate large download; only needed for --with-pai).\n"
            "See data/README.md for download instructions."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build 15-min cluster time series. By default only the "
                    "Helios Saturn series (used in the paper) is built.")
    parser.add_argument(
        "--with-pai", action="store_true",
        help="Additionally build the Alibaba-PAI v2020 series (exploration-only; "
             "its raw data is a separate large download).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Building multi-cluster datasets at 15-min resolution\n")

    check_helios_inputs()
    helios = build_helios_saturn()

    alibaba = None
    if args.with_pai:
        check_alibaba_pai_inputs()
        alibaba = build_alibaba_pai()

    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info(f"  Helios Saturn: {len(helios):,} rows, "
                f"{helios['time'].iloc[0]} to {helios['time'].iloc[-1]}")
    if alibaba is not None:
        logger.info(f"  Alibaba PAI:   {len(alibaba):,} rows, "
                    f"{alibaba['time'].iloc[0]} to {alibaba['time'].iloc[-1]}")
    else:
        logger.info("  Alibaba PAI:   skipped (use --with-pai to build)")
    logger.info("=" * 60)
    logger.info("Done!")
