#!/usr/bin/env python3
"""Rebuild Azure LMM 15-minute multimodal inference-demand dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.config import AZURE_LLM_DIR, DATA_PROCESSED

RAW_PATH = Path(AZURE_LLM_DIR) / "data" / "AzureLMMInferenceTrace_multimodal.csv.gz"
OUT_PATH = Path(DATA_PROCESSED) / "azure_lmm_15min.csv"


def check_raw_inputs() -> None:
    """Fail early with a clear message if the raw Azure LMM trace is missing."""
    if not RAW_PATH.exists():
        sys.exit(
            "ERROR: missing raw Azure LMM input:\n"
            f"  - {RAW_PATH}\n"
            "Expected AzureLMMInferenceTrace_multimodal.csv.gz under <AZURE_LLM_DIR>/data/.\n"
            "See data/README.md for download instructions."
        )


def build() -> pd.DataFrame:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    raw["timestamp_utc"] = pd.to_datetime(raw["TIMESTAMP"], utc=True, errors="coerce")
    raw["ContextTokens"] = pd.to_numeric(raw["ContextTokens"], errors="coerce")
    raw["GeneratedTokens"] = pd.to_numeric(raw["GeneratedTokens"], errors="coerce")
    raw["NumImages"] = pd.to_numeric(raw["NumImages"], errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc", "ContextTokens", "GeneratedTokens", "NumImages"]).copy()
    raw["bin"] = raw["timestamp_utc"].dt.floor("15min")
    raw["total_tokens_request"] = raw["ContextTokens"] + raw["GeneratedTokens"]
    grouped = raw.groupby("bin", sort=True)
    out = grouped.agg(
        request_count=("TIMESTAMP", "count"),
        context_tokens_sum=("ContextTokens", "sum"),
        generated_tokens_sum=("GeneratedTokens", "sum"),
        total_tokens=("total_tokens_request", "sum"),
        image_count_sum=("NumImages", "sum"),
        avg_context=("ContextTokens", "mean"),
        avg_generated=("GeneratedTokens", "mean"),
        avg_images=("NumImages", "mean"),
    ).reset_index()
    out = out.rename(columns={"bin": "time"})
    out["hour"] = out["time"].dt.hour
    out["dow"] = out["time"].dt.dayofweek
    # Column set / order matches the dataset used for the paper exactly.
    out = out[["time", "request_count", "total_tokens",
               "avg_context", "avg_generated", "avg_images", "hour", "dow"]]
    out.to_csv(OUT_PATH, index=False)
    return out


def main() -> None:
    check_raw_inputs()
    out = build()
    print(f"Saved {len(out)} rows to {OUT_PATH}")
    print(f"  Time span: {out['time'].min()} to {out['time'].max()}")
    print(f"  Requests:  {int(out['request_count'].sum()):,}")


if __name__ == "__main__":
    main()
