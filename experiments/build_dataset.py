"""
Build unified cluster-level dataset at 15-min resolution (96 points/day).

Processes ALL 120K GPU CSV files (1.5TB+) and slurm-log.csv to produce
a single merged dataset with:

  1. Real cluster GPU power (kW) — summed across all active GPUs per 15-min bin
  2. Active GPU count — number of GPUs reporting data per bin
  3. Queue depth — jobs submitted but not yet started
  4. Submission rate — new submissions in the 15-min window
  5. Completion rate — jobs completed in the 15-min window
  6. Running job count — jobs currently executing
  7. Time features — hour, day-of-week, month (raw integers for downstream encoding)

All timestamps aligned to 15-min bins in US/Eastern timezone.

Output:
  <DATA_PROCESSED>/mit_supercloud_15min.csv
"""

import os
import sys
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger
from common.config import SUPERCLOUD_DIR, DATA_PROCESSED

logger = get_logger("build_dataset")

GPU_DIR = os.path.join(SUPERCLOUD_DIR, "gpu")
SLURM_LOG = os.path.join(SUPERCLOUD_DIR, "slurm-log.csv")
OUTPUT_PATH = os.path.join(DATA_PROCESSED, "mit_supercloud_15min.csv")


def check_raw_inputs():
    """Fail early with a clear message if the raw MIT SuperCloud data is missing."""
    missing = [p for p in (GPU_DIR, SLURM_LOG) if not os.path.exists(p)]
    if missing:
        sys.exit(
            "ERROR: missing raw MIT SuperCloud input(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nExpected files: <SUPERCLOUD_DIR>/gpu/**.csv and <SUPERCLOUD_DIR>/slurm-log.csv."
            "\nSee data/README.md for download instructions."
        )

# 15-min bin = 900 seconds
BIN_SIZE_S = 900
TZ = "US/Eastern"


# ============================================================
# Part 1: GPU Power Aggregation (from 120K CSV files)
# ============================================================

def process_gpu_file(filepath):
    """Process one GPU CSV: return {bin_id: (mean_power_W, sample_count)}."""
    try:
        df = pd.read_csv(filepath, usecols=["timestamp", "power_draw_W"])
        if len(df) == 0:
            return {}

        df["bin_id"] = (df["timestamp"] // BIN_SIZE_S).astype(np.int64)
        agg = df.groupby("bin_id")["power_draw_W"].agg(["mean", "count"])

        return {int(b): (float(r["mean"]), int(r["count"]))
                for b, r in agg.iterrows()}
    except Exception:
        return {}


def collect_gpu_files():
    """List all GPU CSV file paths."""
    files = []
    for sub in sorted(os.listdir(GPU_DIR)):
        subpath = os.path.join(GPU_DIR, sub)
        if os.path.isdir(subpath):
            for f in sorted(os.listdir(subpath)):
                if f.endswith(".csv"):
                    files.append(os.path.join(subpath, f))
    return files


def aggregate_gpu_power():
    """Process all GPU files and aggregate to 15-min bins."""
    all_files = collect_gpu_files()
    n_files = len(all_files)
    logger.info(f"GPU files to process: {n_files}")

    n_workers = min(cpu_count(), 16)
    logger.info(f"Using {n_workers} workers, bin size = {BIN_SIZE_S}s (15 min)")

    # Accumulate: bin_id -> [total_power_W, gpu_count]
    cluster = defaultdict(lambda: [0.0, 0])

    batch_size = 1000
    processed = 0
    t0 = time.time()

    with Pool(n_workers) as pool:
        for i in range(0, n_files, batch_size):
            batch = all_files[i:i + batch_size]
            results = pool.map(process_gpu_file, batch)

            for file_result in results:
                for bin_id, (mean_pw, count) in file_result.items():
                    cluster[bin_id][0] += mean_pw   # sum of per-GPU mean power
                    cluster[bin_id][1] += 1          # number of GPU-files active

            processed += len(batch)
            elapsed = time.time() - t0
            rate = processed / elapsed
            remaining = (n_files - processed) / max(rate, 1)
            logger.info(f"  [{processed}/{n_files}] "
                        f"{rate:.0f} files/s, ~{remaining/60:.0f} min left")

    logger.info(f"GPU aggregation done in {(time.time()-t0)/60:.1f} min, "
                f"{len(cluster)} time bins")

    # Convert to DataFrame
    rows = []
    for bin_id in sorted(cluster.keys()):
        total_w, n_gpus = cluster[bin_id]
        rows.append({
            "bin_id": bin_id,
            "cluster_power_kw": total_w / 1000,
            "active_gpus": n_gpus,
        })

    return pd.DataFrame(rows)


# ============================================================
# Part 2: Queue & Scheduling Features (from slurm-log.csv)
# ============================================================

def build_queue_features(time_bins_unix):
    """Build queue depth, submission rate, completion rate, running jobs
    for each 15-min bin."""
    logger.info(f"Loading slurm-log...")
    df = pd.read_csv(SLURM_LOG)

    # Keep only jobs with valid duration
    df = df[(df["time_end"] - df["time_start"]) > 0].copy()
    logger.info(f"  Valid jobs: {len(df)}")

    submit = df["time_submit"].values
    start = df["time_start"].values
    end = df["time_end"].values

    n_bins = len(time_bins_unix)
    queue_depth = np.zeros(n_bins, dtype=np.int32)
    running_jobs = np.zeros(n_bins, dtype=np.int32)
    new_submits = np.zeros(n_bins, dtype=np.int32)
    completions = np.zeros(n_bins, dtype=np.int32)

    logger.info(f"Computing scheduling features for {n_bins} bins...")

    for i, t in enumerate(time_bins_unix):
        t_prev = t - BIN_SIZE_S

        # Queue depth: submitted before t, not started by t
        queue_depth[i] = int(((submit <= t) & (start > t)).sum())

        # Running: started before t, not ended by t
        running_jobs[i] = int(((start <= t) & (end > t)).sum())

        # New submissions in this bin
        new_submits[i] = int(((submit > t_prev) & (submit <= t)).sum())

        # Completions in this bin
        completions[i] = int(((end > t_prev) & (end <= t)).sum())

        if (i + 1) % 2000 == 0:
            logger.info(f"  [{i+1}/{n_bins}]")

    return queue_depth, running_jobs, new_submits, completions


# ============================================================
# Part 3: Merge and Save
# ============================================================

def main():
    check_raw_inputs()

    # Step 1: GPU power aggregation
    gpu_df = aggregate_gpu_power()

    # Convert bin_id to timestamp
    gpu_df["time_utc"] = pd.to_datetime(
        gpu_df["bin_id"] * BIN_SIZE_S, unit="s", utc=True)
    gpu_df["time"] = gpu_df["time_utc"].dt.tz_convert(TZ)

    # Step 2: Queue features — aligned to same bins
    time_bins_unix = (gpu_df["bin_id"].values * BIN_SIZE_S).astype(np.float64)
    queue_depth, running_jobs, new_submits, completions = build_queue_features(
        time_bins_unix)

    gpu_df["queue_depth"] = queue_depth
    gpu_df["running_jobs"] = running_jobs
    gpu_df["new_submits_15min"] = new_submits
    gpu_df["completions_15min"] = completions

    # Step 2.5: Reindex onto the complete 15-min grid. Telemetry gaps become
    # all-NaN rows; downstream windowing treats them as window breakers, so
    # they must be present (the paper's dataset includes them).
    full_bins = np.arange(int(gpu_df["bin_id"].min()),
                          int(gpu_df["bin_id"].max()) + 1)
    gpu_df = (gpu_df.set_index("bin_id").reindex(full_bins)
                    .rename_axis("bin_id").reset_index())
    gpu_df["time_utc"] = pd.to_datetime(
        gpu_df["bin_id"] * BIN_SIZE_S, unit="s", utc=True)
    gpu_df["time"] = gpu_df["time_utc"].dt.tz_convert(TZ)

    # Step 3: Time features
    gpu_df["hour"] = gpu_df["time"].dt.hour
    gpu_df["minute"] = gpu_df["time"].dt.minute
    gpu_df["dow"] = gpu_df["time"].dt.dayofweek       # 0=Mon, 6=Sun
    gpu_df["month"] = gpu_df["time"].dt.month
    gpu_df["date"] = gpu_df["time"].dt.date

    # Filter out leading/trailing empty periods
    first_valid = (gpu_df["cluster_power_kw"] > 0.5).idxmax()
    last_valid = (gpu_df["cluster_power_kw"] > 0.5)[::-1].idxmax()
    gpu_df = gpu_df.loc[first_valid:last_valid].reset_index(drop=True)

    # Select and order columns
    output_cols = [
        "time", "cluster_power_kw", "active_gpus",
        "queue_depth", "running_jobs", "new_submits_15min", "completions_15min",
        "hour", "minute", "dow", "month",
    ]
    out = gpu_df[output_cols].copy()

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"Period: {out['time'].iloc[0]} to {out['time'].iloc[-1]}")
    logger.info(f"Rows: {len(out)} ({len(out)/96:.0f} days × 96 points/day)")
    logger.info(f"")
    logger.info(f"--- Cluster Power ---")
    logger.info(f"  Mean: {out['cluster_power_kw'].mean():.1f} kW")
    logger.info(f"  Std:  {out['cluster_power_kw'].std():.1f} kW")
    logger.info(f"  Max:  {out['cluster_power_kw'].max():.1f} kW")
    logger.info(f"  Min:  {out['cluster_power_kw'].min():.1f} kW")
    logger.info(f"")
    logger.info(f"--- Active GPUs ---")
    logger.info(f"  Mean: {out['active_gpus'].mean():.0f}")
    logger.info(f"  Max:  {out['active_gpus'].max()}")
    logger.info(f"")
    logger.info(f"--- Queue ---")
    logger.info(f"  Queue depth mean: {out['queue_depth'].mean():.0f}")
    logger.info(f"  Queue depth max:  {out['queue_depth'].max()}")
    logger.info(f"  Running jobs mean: {out['running_jobs'].mean():.0f}")
    logger.info(f"  Submits/15min mean: {out['new_submits_15min'].mean():.1f}")
    logger.info(f"  Completions/15min mean: {out['completions_15min'].mean():.1f}")
    logger.info(f"")
    logger.info(f"Columns: {list(out.columns)}")


if __name__ == "__main__":
    main()
