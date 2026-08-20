"""
Build 15-min resolution time series from Alibaba v2026-spot-gpu cluster trace.

Uses difference-array technique for O(N_jobs + N_bins) complexity:
  - For each job, increment at start bin and decrement at end bin
  - Cumulative sum yields per-bin active counts

Source: cluster-trace-v2026-spot-gpu/job_info_df.csv (466K jobs, 184 days)
Anchor: 2024-06-01 00:00:00 UTC (arbitrary)

Output columns (matching MIT Supercloud format):
  time, active_gpu_count, queue_depth, running_jobs,
  new_submits_15min, completions_15min, hour, minute, dow, month
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger
from common.config import ALIBABA_TRACE_DIR, DATA_PROCESSED

logger = get_logger("build_alibaba_v2026")

# Paths
JOB_CSV = os.path.join(ALIBABA_TRACE_DIR, "cluster-trace-v2026-spot-gpu", "job_info_df.csv")
NODE_CSV = os.path.join(ALIBABA_TRACE_DIR, "cluster-trace-v2026-spot-gpu", "node_info_df.csv")
OUTPUT_PATH = os.path.join(DATA_PROCESSED, "alibaba_v2026_spot_15min.csv")

BIN_SIZE_S = 900  # 15 minutes
ANCHOR = pd.Timestamp("2024-06-01 00:00:00", tz="UTC")


def load_data():
    """Load job and node data."""
    logger.info("Loading job data from %s", JOB_CSV)
    jobs = pd.read_csv(JOB_CSV)
    logger.info("  %d jobs loaded", len(jobs))

    logger.info("Loading node data from %s", NODE_CSV)
    nodes = pd.read_csv(NODE_CSV)
    total_gpus = nodes["gpu_capacity_num"].sum()
    logger.info("  %d nodes, %d total GPU capacity", len(nodes), total_gpus)

    return jobs, total_gpus


def build_timeseries(jobs, total_gpu_capacity):
    """Build 15-min resolution time series using difference arrays."""

    # Compute start/end in seconds
    start_s = jobs["submit_time"].values
    duration_s = jobs["duration"].values
    end_s = start_s + duration_s

    # GPU count per job
    gpu_per_job = (jobs["gpu_request"] * jobs["worker_num"]).values

    # Bin range
    max_time_s = end_s.max()
    n_bins = int(np.ceil(max_time_s / BIN_SIZE_S)) + 1
    logger.info("Time span: %.1f days, %d bins (15-min each)", max_time_s / 86400, n_bins)

    # --- 1. active_gpu_count via difference array ---
    logger.info("Building active_gpu_count via difference array...")
    diff_gpu = np.zeros(n_bins + 1, dtype=np.float64)
    start_bins = (start_s / BIN_SIZE_S).astype(np.int64)
    end_bins = (end_s / BIN_SIZE_S).astype(np.int64)

    # Vectorized: use np.add.at for scatter-add
    np.add.at(diff_gpu, start_bins, gpu_per_job)
    np.add.at(diff_gpu, end_bins, -gpu_per_job)
    active_gpu = np.cumsum(diff_gpu[:n_bins])
    active_gpu = np.clip(active_gpu, 0, total_gpu_capacity)

    # --- 2. running_jobs via difference array ---
    logger.info("Building running_jobs via difference array...")
    diff_jobs = np.zeros(n_bins + 1, dtype=np.float64)
    np.add.at(diff_jobs, start_bins, 1)
    np.add.at(diff_jobs, end_bins, -1)
    running_jobs = np.cumsum(diff_jobs[:n_bins])
    running_jobs = np.clip(running_jobs, 0, None)

    # --- 3. new_submits_15min: count of jobs submitted in each bin ---
    logger.info("Building new_submits_15min...")
    new_submits = np.zeros(n_bins, dtype=np.int64)
    np.add.at(new_submits, start_bins, 1)

    # --- 4. completions_15min: count of jobs ending in each bin ---
    logger.info("Building completions_15min...")
    completions = np.zeros(n_bins, dtype=np.int64)
    end_bins_clipped = np.clip(end_bins, 0, n_bins - 1)
    np.add.at(completions, end_bins_clipped, 1)

    # --- 5. queue_depth = 0 (no explicit start_time; assume immediate start) ---
    queue_depth = np.zeros(n_bins, dtype=np.int64)

    # --- Build DataFrame ---
    logger.info("Assembling DataFrame...")
    bin_ids = np.arange(n_bins)
    timestamps = ANCHOR + pd.to_timedelta(bin_ids * BIN_SIZE_S, unit="s")

    df = pd.DataFrame({
        "time": timestamps,
        "active_gpu_count": active_gpu.astype(np.int64),
        "queue_depth": queue_depth,
        "running_jobs": running_jobs.astype(np.int64),
        "new_submits_15min": new_submits,
        "completions_15min": completions,
    })

    # Time features
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["dow"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    return df


def print_summary(df, total_gpu_capacity):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("Alibaba v2026-spot-gpu 15-min Time Series Summary")
    print("=" * 70)
    print(f"  Time range:  {df['time'].iloc[0]}  to  {df['time'].iloc[-1]}")
    print(f"  Duration:    {(df['time'].iloc[-1] - df['time'].iloc[0]).days} days")
    print(f"  Rows:        {len(df):,}")
    print(f"  GPU capacity:{total_gpu_capacity:,}")
    print()
    print("Active GPU count:")
    print(f"  mean = {df['active_gpu_count'].mean():.1f},"
          f"  std = {df['active_gpu_count'].std():.1f},"
          f"  min = {df['active_gpu_count'].min()},"
          f"  max = {df['active_gpu_count'].max()}")
    print(f"  avg utilization = {df['active_gpu_count'].mean() / total_gpu_capacity * 100:.1f}%")
    print()
    print("Running jobs:")
    print(f"  mean = {df['running_jobs'].mean():.1f},"
          f"  std = {df['running_jobs'].std():.1f},"
          f"  min = {df['running_jobs'].min()},"
          f"  max = {df['running_jobs'].max()}")
    print()
    print("New submits per 15min:")
    print(f"  mean = {df['new_submits_15min'].mean():.2f},"
          f"  std = {df['new_submits_15min'].std():.2f},"
          f"  max = {df['new_submits_15min'].max()}")
    print()
    print("Completions per 15min:")
    print(f"  mean = {df['completions_15min'].mean():.2f},"
          f"  std = {df['completions_15min'].std():.2f},"
          f"  max = {df['completions_15min'].max()}")
    print()
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 70)


def check_raw_inputs():
    """Fail early with a clear message if the Alibaba v2026 raw traces are missing."""
    missing = [p for p in (JOB_CSV, NODE_CSV) if not os.path.exists(p)]
    if missing:
        sys.exit(
            "ERROR: missing raw Alibaba v2026-spot-gpu input(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nExpected job_info_df.csv and node_info_df.csv under "
            "<ALIBABA_TRACE_DIR>/cluster-trace-v2026-spot-gpu/.\n"
            "See data/README.md for download instructions."
        )


def main():
    check_raw_inputs()
    jobs, total_gpu_capacity = load_data()
    df = build_timeseries(jobs, total_gpu_capacity)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(df), OUTPUT_PATH)

    print_summary(df, total_gpu_capacity)

    # Show first/last rows
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
