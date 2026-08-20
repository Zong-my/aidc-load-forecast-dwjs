"""
MSRA Data Pipeline.

Load 15-min dataset → derive features → chronological split →
fit regime identifier → label regimes → create sliding windows
(skipping NaN) → normalize → package into ready-to-train dict.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json

# RegimeIdentifier belongs to the MSRA exploration framework, which is not
# shipped; only build_msra_dataset's legacy path needs it.
try:
    from msra.regime_identifier import RegimeIdentifier
except ImportError:
    RegimeIdentifier = None

logger = get_logger("prepare_data")


def load_and_derive(cfg):
    """Load CSV and compute derived features.

    Returns DataFrame with all raw + derived columns.
    """
    if not os.path.exists(cfg["data_path"]):
        logger.error(f"Dataset CSV not found: {cfg['data_path']}. "
                     f"Build the processed datasets first (see data/README.md).")
        sys.exit(1)
    df = pd.read_csv(cfg["data_path"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    logger.info(f"Loaded {len(df)} rows, NaN={df['cluster_power_kw'].isna().sum()}")

    # Change rate features
    if cfg.get("add_change_rates", True):
        for col in ["cluster_power_kw", "queue_depth", "running_jobs",
                     "new_submits_15min"]:
            df[f"d_{col}"] = df[col].diff()
        # First row diff is NaN, fill with 0
        df = df.fillna({f"d_{col}": 0 for col in
                        ["cluster_power_kw", "queue_depth", "running_jobs",
                         "new_submits_15min"]})

    # Rolling statistics
    if cfg.get("add_rolling_stats", True):
        df["power_ma_4h"] = df["cluster_power_kw"].rolling(16, min_periods=1).mean()
        df["power_std_4h"] = df["cluster_power_kw"].rolling(16, min_periods=1).std().fillna(0)

    # Cyclic time encoding
    if cfg.get("add_cyclic_time", True):
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    return df


def get_feature_cols(cfg):
    """Build full feature column list based on config."""
    cols = list(cfg["feature_cols"])

    if cfg.get("add_change_rates", True):
        cols += ["d_cluster_power_kw", "d_queue_depth",
                 "d_running_jobs", "d_new_submits_15min"]

    if cfg.get("add_rolling_stats", True):
        cols += ["power_ma_4h", "power_std_4h"]

    if cfg.get("add_cyclic_time", True):
        cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    return cols


def chronological_split(df, train_ratio=0.7, val_ratio=0.15):
    """Split by time order (no shuffle)."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def create_windows(df, feature_cols, target_col, lookback, horizon, stride=1):
    """Create sliding windows, skipping windows containing NaN.

    Returns:
        X: (N, lookback, n_features) float32
        y: (N, horizon) float32 — target column
        regime_at_pred: (N,) int — regime label at prediction moment (last lookback step)
    """
    features = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    is_nan = df[target_col].isna().values

    n = len(df)
    total_len = lookback + horizon
    n_features = len(feature_cols)

    X_list, y_list = [], []
    indices = []  # track which rows were used (for regime labeling)

    for i in range(0, n - total_len + 1, stride):
        window = slice(i, i + total_len)
        # Skip if any NaN in the window's target column
        if is_nan[window].any():
            continue

        X_list.append(features[i:i + lookback])
        y_list.append(target[i + lookback:i + total_len])
        indices.append(i + lookback - 1)  # last lookback step = prediction moment

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"Windows: {len(X)} (from {n} rows, lookback={lookback}, "
                f"horizon={horizon}, stride={stride})")
    return X, y, np.array(indices)


def normalize(X_train, X_val, X_test, y_train, y_val, y_test):
    """Z-normalize using train statistics only.

    Returns normalized arrays + norm_params dict.
    """
    # Feature normalization
    f_mean = X_train.mean(axis=(0, 1))  # (n_features,)
    f_std = X_train.std(axis=(0, 1))
    f_std = np.where(f_std < 1e-8, 1.0, f_std)

    X_train_n = (X_train - f_mean) / f_std
    X_val_n = (X_val - f_mean) / f_std
    X_test_n = (X_test - f_mean) / f_std

    # Target normalization
    t_mean = float(y_train.mean())
    t_std = float(y_train.std())
    if t_std < 1e-8:
        t_std = 1.0

    y_train_n = (y_train - t_mean) / t_std
    y_val_n = (y_val - t_mean) / t_std
    y_test_n = (y_test - t_mean) / t_std

    norm_params = {
        "feature_means": f_mean.tolist(),
        "feature_stds": f_std.tolist(),
        "target_mean": t_mean,
        "target_std": t_std,
    }

    return X_train_n, X_val_n, X_test_n, y_train_n, y_val_n, y_test_n, norm_params


def build_msra_dataset(cfg):
    """Full pipeline: load → derive → split → fit GMM → windows → normalize.

    Returns dict ready for training.
    """
    # Load and derive features
    df = load_and_derive(cfg)
    feature_cols = get_feature_cols(cfg)
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Find regime column indices in feature_cols
    regime_col_indices = [feature_cols.index(c) for c in cfg["regime_cols"]
                          if c in feature_cols]
    logger.info(f"Regime column indices: {regime_col_indices}")

    # Chronological split
    train_df, val_df, test_df = chronological_split(
        df, cfg["train_ratio"], cfg["val_ratio"])

    # Fit regime identifier on training data
    if RegimeIdentifier is None:
        raise ImportError(
            "build_msra_dataset needs the MSRA exploration framework, which "
            "is not part of this repository (the paper's pipeline does not "
            "use it).")
    regime_id = RegimeIdentifier(
        n_regimes=cfg["n_regimes"],
        covariance_type=cfg.get("gmm_covariance_type", "full"),
        n_init=cfg.get("gmm_n_init", 5),
        random_state=cfg.get("seed", 42),
    )
    train_regime_feats = train_df[cfg["regime_cols"]].dropna().values
    regime_id.fit(train_regime_feats)
    logger.info(f"Regime stats: {regime_id.get_regime_stats()}")
    logger.info(f"Transition matrix:\n{regime_id.get_transition_matrix()}")

    # Label regimes for all splits
    for split_df in [train_df, val_df, test_df]:
        feats = split_df[cfg["regime_cols"]].values
        # Handle NaN: assign -1 for NaN rows
        nan_mask = np.isnan(feats).any(axis=1)
        labels = np.full(len(split_df), -1, dtype=np.int32)
        if (~nan_mask).sum() > 0:
            labels[~nan_mask] = regime_id.predict_hard(feats[~nan_mask])
        split_df["regime_label"] = labels

    # Compute physical constraint parameters from training data
    train_valid = train_df.dropna(subset=[cfg["target_col"]])
    p_vals = train_valid[cfg["target_col"]].values
    p_min = cfg.get("p_min") or float(np.percentile(p_vals, 0.5))
    p_max = cfg.get("p_max") or float(np.percentile(p_vals, 99.5))
    if cfg.get("max_slew_rate") is None:
        diffs = np.abs(np.diff(p_vals))
        max_slew = float(np.percentile(diffs[~np.isnan(diffs)], 99))
    else:
        max_slew = cfg["max_slew_rate"]

    logger.info(f"Physics: P_min={p_min:.2f}, P_max={p_max:.2f}, "
                f"max_slew={max_slew:.2f} kW/step")

    # Create windows for each horizon
    results = {"regime_identifier": regime_id,
               "feature_cols": feature_cols,
               "regime_col_indices": regime_col_indices,
               "physics": {"p_min": p_min, "p_max": p_max,
                            "max_slew_rate": max_slew},
               "horizons": {}}

    for horizon in cfg["horizons"]:
        logger.info(f"\n--- Horizon = {horizon} steps ({horizon * 15 / 60:.0f}h) ---")

        trX, trY, tr_idx = create_windows(
            train_df, feature_cols, cfg["target_col"],
            cfg["lookback"], horizon, cfg["stride"])
        vaX, vaY, va_idx = create_windows(
            val_df, feature_cols, cfg["target_col"],
            cfg["lookback"], horizon, cfg["stride"])
        teX, teY, te_idx = create_windows(
            test_df, feature_cols, cfg["target_col"],
            cfg["lookback"], horizon, cfg["stride"])

        # Normalize
        trX_n, vaX_n, teX_n, trY_n, vaY_n, teY_n, norm = normalize(
            trX, vaX, teX, trY, vaY, teY)

        # Regime labels at prediction moment
        tr_regimes = train_df["regime_label"].values[tr_idx]
        va_regimes = val_df["regime_label"].values[va_idx]
        te_regimes = test_df["regime_label"].values[te_idx]

        results["horizons"][horizon] = {
            "train_X": trX_n, "train_y": trY_n, "train_regimes": tr_regimes,
            "val_X": vaX_n, "val_y": vaY_n, "val_regimes": va_regimes,
            "test_X": teX_n, "test_y": teY_n, "test_regimes": te_regimes,
            "norm_params": norm,
        }

    return results
