"""
MSRA (Multi-Scale Regime-Aware) Forecasting Configuration.

All hyperparameters centralized here. Algorithm is designed to be
general-purpose — regime count K, feature set, and physical bounds
are configurable, not hardcoded to any specific dataset.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.config import DATA_PROCESSED

MSRA_CONFIG = {
    # ── Data ──
    "data_path": os.path.join(DATA_PROCESSED, "mit_supercloud_15min.csv"),
    "target_col": "cluster_power_kw",

    # Features used by expert predictors (input to Module C)
    "feature_cols": [
        "cluster_power_kw", "active_gpus",
        "queue_depth", "running_jobs",
        "new_submits_15min", "completions_15min",
    ],

    # Features used by regime identifier (input to Module A)
    "regime_cols": [
        "cluster_power_kw", "active_gpus",
        "running_jobs", "queue_depth",
    ],

    # Derived features (computed in data pipeline)
    "add_change_rates": True,       # d_power, d_queue, d_running, d_submits
    "add_rolling_stats": True,      # power_ma_4h, power_std_4h
    "add_cyclic_time": True,        # hour_sin/cos, dow_sin/cos

    # ── Windows ──
    "lookback": 96,                 # 96 × 15min = 24h
    "horizons": [16, 96],           # 4h, 24h
    "stride": 1,

    # ── Split (chronological, no shuffle) ──
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    # test_ratio = 0.15 (implicit)

    # ── Regime (Module A) ──
    "n_regimes": 4,                 # auto-selectable via BIC if set to "auto"
    "gmm_covariance_type": "full",
    "gmm_n_init": 5,

    # ── Expert Predictors (Module C) ──
    "expert_hidden_dim": 128,
    "expert_gru_layers": 2,
    "expert_conv_kernel": 3,
    "expert_dropout": 0.1,
    "n_quantiles": 3,
    "quantile_levels": [0.1, 0.5, 0.9],

    # ── Transition Predictor (Module B) ──
    "trans_gru_hidden": 32,
    "trans_gru_layers": 1,

    # ── Physical Constraints (Module D) ──
    # Set to None to auto-detect from training data
    "p_min": None,
    "p_max": None,
    "max_slew_rate": None,          # kW per 15-min step
    "lambda_slew": 0.05,
    "lambda_bound": 0.1,
    "lambda_trans": 0.1,

    # ── Training ──
    "batch_size": 128,
    "epochs_phase1": 30,
    "epochs_phase2": 20,
    "epochs_phase3": 60,
    "lr_expert": 1e-3,
    "lr_transition": 5e-4,
    "lr_finetune": 2e-4,
    "lr_base_T": 5e-5,              # slow lr for Markov matrix
    "weight_decay": 1e-4,
    "patience_phase1": 7,
    "patience_phase2": 5,
    "patience_phase3": 10,
    "grad_clip": 1.0,
    "seed": 42,
    "device": "cuda",

    # ── Baselines ──
    "baseline_epochs": 50,
    "baseline_patience": 7,
    "baseline_hidden": 128,
    "baseline_layers": 2,
    "baseline_lr": 1e-3,

    # ── Output ──
    "case_name": "case10c_cluster_forecast",
}
