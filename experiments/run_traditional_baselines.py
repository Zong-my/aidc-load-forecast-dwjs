"""
Traditional Power Load Forecasting Baselines & M6 Replication.

This script implements four baseline methods to answer:
  1. "How well do traditional grid-operator methods forecast AI datacenter load?"
  2. "How well does the only prior AI-load forecasting work (M6) perform?"

Methods:
  - ARIMA(2,1,2): classical univariate time-series model
  - SVR-Calendar: support vector regression with lookback + calendar features
  - LSTM-Calendar: LSTM using only power + calendar (no scheduling features)
  - M6-Recursive: single-step LSTM with recursive multi-step rollout (M6 paper)

Usage:
    python run_traditional_baselines.py                # MIT only (default)
    python run_traditional_baselines.py --dataset Helios
    python run_traditional_baselines.py --dataset all   # all 4 datasets
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json
from common.config import DATA_PROCESSED

from evaluate_msra import compute_point_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger("trad_baselines")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CASE = "case10c_cluster_forecast"
HORIZONS = [16, 96]
LOOKBACK = 96  # 24h of 15-min steps
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

DATASET_CONFIGS = {
    "MIT": {
        "path": os.path.join(DATA_PROCESSED, "mit_supercloud_15min.csv"),
        "target": "cluster_power_kw",
    },
    "Helios": {
        "path": os.path.join(DATA_PROCESSED, "helios_saturn_15min.csv"),
        "target": "active_gpu_count",
    },
    "Alibaba": {
        "path": os.path.join(DATA_PROCESSED, "alibaba_v2026_spot_15min.csv"),
        "target": "active_gpu_count",
    },
    "Inference": {
        "path": os.path.join(DATA_PROCESSED, "inference_cluster_15min.csv"),
        "target": "cluster_power_kw",
    },
}


# ============================================================
# Data loading
# ============================================================

def load_data(ds_config):
    """Load CSV, add cyclic calendar features, chronological split.

    NaN rows in target are forward-filled then backward-filled
    so that sliding windows are always complete.
    """
    data_path = ds_config["path"]
    target = ds_config["target"]

    if not os.path.exists(data_path):
        logger.error(f"Dataset CSV not found: {data_path}. "
                     f"Build the processed datasets first (see data/README.md).")
        sys.exit(1)
    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    n_nan = df[target].isna().sum()
    logger.info(f"Loaded {len(df)} rows, {n_nan} NaN in {target} — interpolating")

    # Interpolate NaN (linear then ffill/bfill for edges)
    num_cols = [c for c in df.columns if c not in ("time", "hour", "minute", "dow", "month")]
    df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()

    # Cyclic calendar encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return df, train_df, val_df, test_df


# ============================================================
# Shared helpers
# ============================================================

def make_test_windows(test_df, lookback, horizon, target):
    """Create (X_lookback, y_horizon) windows from test data.

    Returns target values only (for ARIMA / M6 / Persistence).
    """
    power = test_df[target].values.astype(np.float32)
    n = len(power)
    X_list, y_list = [], []
    for i in range(n - lookback - horizon + 1):
        X_list.append(power[i:i + lookback])
        y_list.append(power[i + lookback:i + lookback + horizon])
    return np.array(X_list), np.array(y_list)


def run_persistence(test_df, horizon, target):
    """Persistence baseline: y_hat(t+h) = y(t) for all h."""
    logger.info(f"  Persistence: horizon={horizon}")
    t0 = time.time()
    power = test_df[target].values.astype(np.float32)
    n = len(power)
    preds, trues = [], []
    for i in range(n - LOOKBACK - horizon + 1):
        last_val = power[i + LOOKBACK - 1]
        preds.append(np.full(horizon, last_val))
        trues.append(power[i + LOOKBACK:i + LOOKBACK + horizon])
    preds = np.array(preds)
    trues = np.array(trues)
    elapsed = time.time() - t0
    metrics = compute_point_metrics(preds, trues)
    metrics["time_s"] = round(elapsed, 1)
    metrics["n_eval"] = len(preds)
    logger.info(f"  Persistence done: {metrics}")
    return metrics, preds, trues


# ============================================================
# 1. ARIMA
# ============================================================

def run_arima(train_df, test_df, horizon, target, max_test=500):
    """Rolling ARIMA(2,1,2) forecast.

    To keep runtime reasonable, we fit on a trailing window (2000 steps)
    and evaluate on at most `max_test` windows.
    """
    from statsmodels.tsa.arima.model import ARIMA

    logger.info(f"  ARIMA(2,1,2): horizon={horizon}, max_test={max_test}")
    t0 = time.time()

    power_all = np.concatenate([
        train_df[target].values,
        test_df[target].values,
    ]).astype(np.float64)
    train_len = len(train_df)
    test_len = len(test_df)

    # We only evaluate windows where the full horizon fits
    total_test_windows = test_len - horizon
    if total_test_windows <= 0:
        logger.warning("  Not enough test data for ARIMA")
        return None

    # Subsample test windows evenly for speed
    n_eval = min(max_test, total_test_windows)
    eval_indices = np.linspace(0, total_test_windows - 1, n_eval, dtype=int)

    preds, trues = [], []
    fit_window = 2000  # trailing context for each ARIMA fit

    for count, idx in enumerate(eval_indices):
        abs_idx = train_len + idx  # position in power_all
        start = max(0, abs_idx - fit_window)
        history = power_all[start:abs_idx]
        actual = power_all[abs_idx:abs_idx + horizon]

        if len(actual) < horizon:
            continue

        try:
            model = ARIMA(history, order=(2, 1, 2))
            fit = model.fit(method_kwargs={"maxiter": 200})
            fc = fit.forecast(steps=horizon)
            preds.append(fc)
            trues.append(actual)
        except Exception as e:
            # Fallback: persistence
            preds.append(np.full(horizon, history[-1]))
            trues.append(actual)

        if (count + 1) % 100 == 0:
            logger.info(f"    ARIMA: {count+1}/{n_eval} done")

    preds = np.array(preds)
    trues = np.array(trues)
    elapsed = time.time() - t0
    metrics = compute_point_metrics(preds, trues)
    metrics["time_s"] = round(elapsed, 1)
    metrics["n_eval"] = len(preds)
    logger.info(f"  ARIMA done: {metrics}, {elapsed:.0f}s")
    return metrics, preds, trues


# ============================================================
# 2. SVR-Calendar
# ============================================================

def run_svr_calendar(train_df, val_df, test_df, horizon, target, svr_lookback=24):
    """SVR with flattened lookback target + calendar features.

    svr_lookback: number of past target steps as features (default 24 = 6h).
    We keep it shorter than 96 to limit feature dimensionality for SVR.
    """
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor

    logger.info(f"  SVR-Calendar: horizon={horizon}, lookback={svr_lookback}")
    t0 = time.time()

    cal_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos"]

    def build_features(df, lookback, horizon):
        power = df[target].values.astype(np.float32)
        cal = df[cal_cols].values.astype(np.float32)
        n = len(df)
        X_list, y_list = [], []
        for i in range(lookback, n - horizon + 1):
            feat = np.concatenate([
                power[i - lookback:i],  # past target
                cal[i],                 # calendar at prediction moment
            ])
            X_list.append(feat)
            y_list.append(power[i:i + horizon])
        return np.array(X_list), np.array(y_list)

    trX, trY = build_features(train_df, svr_lookback, horizon)
    teX, teY = build_features(test_df, svr_lookback, horizon)
    logger.info(f"    SVR train samples={len(trX)}, test samples={len(teX)}")

    # Subsample training data if too large (SVR is O(n^2))
    max_train = 5000
    if len(trX) > max_train:
        idx = np.random.RandomState(SEED).choice(len(trX), max_train, replace=False)
        trX, trY = trX[idx], trY[idx]
        logger.info(f"    SVR subsampled train to {max_train}")

    scaler_X = StandardScaler().fit(trX)
    trX_s = scaler_X.transform(trX)
    teX_s = scaler_X.transform(teX)

    # For multi-step output, wrap SVR in MultiOutputRegressor
    svr = MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.01))
    svr.fit(trX_s, trY)

    preds = svr.predict(teX_s)
    elapsed = time.time() - t0
    metrics = compute_point_metrics(preds, teY)
    metrics["time_s"] = round(elapsed, 1)
    metrics["n_eval"] = len(preds)
    logger.info(f"  SVR-Calendar done: {metrics}, {elapsed:.0f}s")
    return metrics, preds, teY


# ============================================================
# 2b. LightGBM-Calendar
# ============================================================

def run_lgbm_calendar(train_df, val_df, test_df, horizon, target, lgbm_lookback=24):
    """LightGBM with same features as SVR-Calendar (past target + calendar)."""
    from lightgbm import LGBMRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor

    logger.info(f"  LightGBM-Calendar: horizon={horizon}, lookback={lgbm_lookback}")
    t0 = time.time()

    cal_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos"]

    def build_features(df, lookback, horizon):
        power = df[target].values.astype(np.float32)
        cal = df[cal_cols].values.astype(np.float32)
        n = len(df)
        X_list, y_list = [], []
        for i in range(lookback, n - horizon + 1):
            feat = np.concatenate([power[i - lookback:i], cal[i]])
            X_list.append(feat)
            y_list.append(power[i:i + horizon])
        return np.array(X_list), np.array(y_list)

    trX, trY = build_features(train_df, lgbm_lookback, horizon)
    teX, teY = build_features(test_df, lgbm_lookback, horizon)
    logger.info(f"    LightGBM train samples={len(trX)}, test samples={len(teX)}")

    lgbm = MultiOutputRegressor(
        LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8,
                      num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                      random_state=SEED, verbose=-1))
    lgbm.fit(trX, trY)

    preds = lgbm.predict(teX)
    elapsed = time.time() - t0
    metrics = compute_point_metrics(preds, teY)
    metrics["time_s"] = round(elapsed, 1)
    metrics["n_eval"] = len(preds)
    logger.info(f"  LightGBM-Calendar done: {metrics}, {elapsed:.0f}s")
    return metrics, preds, teY


# ============================================================
# 3. LSTM-Calendar
# ============================================================

class LSTMCalendar(nn.Module):
    """LSTM using only power + calendar features (no scheduling info)."""

    def __init__(self, input_dim, hidden=128, layers=2, horizon=16, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden, horizon)
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # (batch, horizon)


def run_lstm_calendar(train_df, val_df, test_df, horizon, target):
    """Train and evaluate LSTM with only target + calendar features."""
    logger.info(f"  LSTM-Calendar: horizon={horizon}")
    t0 = time.time()

    feature_cols = [target,
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                    "month_sin", "month_cos"]

    def build_windows(df, lookback, horizon):
        feats = df[feature_cols].values.astype(np.float32)
        tgt = df[target].values.astype(np.float32)
        n = len(df)
        X_list, y_list = [], []
        for i in range(n - lookback - horizon + 1):
            X_list.append(feats[i:i + lookback])
            y_list.append(tgt[i + lookback:i + lookback + horizon])
        return np.array(X_list), np.array(y_list)

    trX, trY = build_windows(train_df, LOOKBACK, horizon)
    vaX, vaY = build_windows(val_df, LOOKBACK, horizon)
    teX, teY = build_windows(test_df, LOOKBACK, horizon)

    # Normalize using train stats
    f_mean = trX.mean(axis=(0, 1))
    f_std = trX.std(axis=(0, 1))
    f_std = np.where(f_std < 1e-8, 1.0, f_std)
    t_mean = float(trY.mean())
    t_std = float(trY.std())
    if t_std < 1e-8:
        t_std = 1.0

    trX_n = (trX - f_mean) / f_std
    vaX_n = (vaX - f_mean) / f_std
    teX_n = (teX - f_mean) / f_std
    trY_n = (trY - t_mean) / t_std
    vaY_n = (vaY - t_mean) / t_std
    teY_n = (teY - t_mean) / t_std

    # DataLoaders
    bs = 128
    tr_dl = DataLoader(TensorDataset(
        torch.from_numpy(trX_n), torch.from_numpy(trY_n)),
        batch_size=bs, shuffle=True, pin_memory=True)
    va_dl = DataLoader(TensorDataset(
        torch.from_numpy(vaX_n), torch.from_numpy(vaY_n)),
        batch_size=bs, shuffle=False, pin_memory=True)

    n_features = len(feature_cols)
    model = LSTMCalendar(n_features, hidden=128, layers=2,
                         horizon=horizon, dropout=0.1).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_steps = len(tr_dl) * 50
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=n_steps, pct_start=0.2)
    loss_fn = nn.MSELoss()

    # Training with early stopping
    best_val = float("inf")
    best_state = None
    patience = 7
    no_improve = 0

    for epoch in range(50):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
                n_val += len(xb)
        val_loss /= max(n_val, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"    LSTM-Calendar early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test
    teX_t = torch.from_numpy(teX_n).float().to(DEVICE)
    with torch.no_grad():
        pred_norm = model(teX_t).cpu().numpy()
    pred_kw = pred_norm * t_std + t_mean
    true_kw = teY  # already in original scale

    elapsed = time.time() - t0
    metrics = compute_point_metrics(pred_kw, true_kw)
    metrics["time_s"] = round(elapsed, 1)
    metrics["n_eval"] = len(pred_kw)
    logger.info(f"  LSTM-Calendar done: {metrics}, {elapsed:.0f}s")
    return metrics, pred_kw, true_kw


# ============================================================
# 4. M6-Recursive (autoregressive single-step LSTM)
# ============================================================

class M6RecursiveLSTM(nn.Module):
    """Single-step LSTM mimicking the M6 paper approach.

    At inference: predict 1 step, feed back, repeat for H steps.
    At training: teacher-forcing on single-step prediction.
    """

    def __init__(self, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden, 1)

    def forward_one_step(self, x):
        """x: (batch, seq_len, 1) -> (batch, 1) next value."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # (batch, 1)

    def forward_recursive(self, x, horizon):
        """Recursive multi-step prediction.

        x: (batch, lookback, 1) -> (batch, horizon)
        """
        preds = []
        current = x.clone()
        for h in range(horizon):
            next_val = self.forward_one_step(current)  # (batch, 1)
            preds.append(next_val)
            # Shift window: drop oldest, append prediction
            current = torch.cat([current[:, 1:, :], next_val.unsqueeze(1)], dim=1)
        return torch.cat(preds, dim=1)  # (batch, horizon)

    def forward(self, x):
        """Training mode: single-step prediction (teacher forcing).
        x: (batch, lookback, 1) -> (batch, 1)
        """
        return self.forward_one_step(x)


def run_m6_recursive(train_df, val_df, test_df, horizon, target):
    """Train M6-style single-step LSTM and evaluate with recursive rollout."""
    logger.info(f"  M6-Recursive: horizon={horizon}")
    t0 = time.time()

    # Prepare univariate data
    def get_power(df):
        return df[target].values.astype(np.float32)

    train_p = get_power(train_df)
    val_p = get_power(val_df)
    test_p = get_power(test_df)

    # Normalize
    p_mean = float(train_p.mean())
    p_std = float(train_p.std())
    if p_std < 1e-8:
        p_std = 1.0

    train_pn = (train_p - p_mean) / p_std
    val_pn = (val_p - p_mean) / p_std
    test_pn = (test_p - p_mean) / p_std

    # Build single-step training windows: X=(lookback,), y=(1,)
    def build_single_step_windows(data, lookback):
        n = len(data)
        X_list, y_list = [], []
        for i in range(n - lookback):
            X_list.append(data[i:i + lookback])
            y_list.append(data[i + lookback])
        X = np.array(X_list, dtype=np.float32)[:, :, None]  # (N, L, 1)
        y = np.array(y_list, dtype=np.float32)[:, None]      # (N, 1)
        return X, y

    trX, trY = build_single_step_windows(train_pn, LOOKBACK)
    vaX, vaY = build_single_step_windows(val_pn, LOOKBACK)

    bs = 256
    tr_dl = DataLoader(TensorDataset(
        torch.from_numpy(trX), torch.from_numpy(trY)),
        batch_size=bs, shuffle=True, pin_memory=True)
    va_dl = DataLoader(TensorDataset(
        torch.from_numpy(vaX), torch.from_numpy(vaY)),
        batch_size=bs, shuffle=False, pin_memory=True)

    model = M6RecursiveLSTM(hidden=64, layers=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_steps = len(tr_dl) * 50
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=n_steps, pct_start=0.2)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 7
    no_improve = 0

    for epoch in range(50):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
                n_val += len(xb)
        val_loss /= max(n_val, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"    M6-Recursive early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Evaluate with recursive rollout on test windows
    def build_test_windows(data_norm, data_orig, lookback, horizon):
        n = len(data_norm)
        X_list, y_list = [], []
        for i in range(n - lookback - horizon + 1):
            X_list.append(data_norm[i:i + lookback])
            y_list.append(data_orig[i + lookback:i + lookback + horizon])
        X = np.array(X_list, dtype=np.float32)[:, :, None]
        y = np.array(y_list, dtype=np.float32)
        return X, y

    teX, teY = build_test_windows(test_pn, get_power(test_df), LOOKBACK, horizon)

    # Recursive prediction in batches
    preds_all = []
    eval_bs = 512
    with torch.no_grad():
        for start in range(0, len(teX), eval_bs):
            xb = torch.from_numpy(teX[start:start + eval_bs]).to(DEVICE)
            pred_n = model.forward_recursive(xb, horizon).cpu().numpy()
            pred_kw = pred_n * p_std + p_mean
            preds_all.append(pred_kw)

    preds = np.concatenate(preds_all, axis=0)
    elapsed = time.time() - t0
    metrics = compute_point_metrics(preds, teY)
    metrics["time_s"] = round(elapsed, 1)
    metrics["n_eval"] = len(preds)
    logger.info(f"  M6-Recursive done: {metrics}, {elapsed:.0f}s")
    return metrics, preds, teY


# ============================================================
# Main
# ============================================================

def main(ds_name="MIT"):
    ds_config = DATASET_CONFIGS[ds_name]
    target = ds_config["target"]

    logger.info("=" * 60)
    logger.info(f"Traditional Baselines & M6 — Dataset: {ds_name}")
    logger.info("=" * 60)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    df, train_df, val_df, test_df = load_data(ds_config)

    out_base = os.path.join(results_path(CASE), "data")
    pred_dir = os.path.join(out_base, "predictions", ds_name)
    os.makedirs(pred_dir, exist_ok=True)

    results = {}

    for horizon in HORIZONS:
        h_label = f"{horizon * 15 // 60}h"
        logger.info(f"\n{'#'*50}")
        logger.info(f"# {ds_name} | Horizon = {horizon} steps ({h_label})")
        logger.info(f"{'#'*50}")

        h_results = {}

        methods_run = [
            ("Persistence", lambda: run_persistence(test_df, horizon, target)),
            ("ARIMA", lambda: run_arima(train_df, test_df, horizon, target, max_test=500)),
            ("SVR-Calendar", lambda: run_svr_calendar(train_df, val_df, test_df, horizon, target)),
            ("LightGBM-Calendar", lambda: run_lgbm_calendar(train_df, val_df, test_df, horizon, target)),
            ("LSTM-Calendar", lambda: run_lstm_calendar(train_df, val_df, test_df, horizon, target)),
            ("M6-Recursive", lambda: run_m6_recursive(train_df, val_df, test_df, horizon, target)),
        ]

        for method_name, run_fn in methods_run:
            result = run_fn()
            if result is None:
                continue
            metrics, preds, trues = result

            # Add per-step RMSE
            from evaluate_msra import compute_step_metrics
            metrics["step_rmse"] = compute_step_metrics(preds, trues)

            h_results[method_name] = {k: v for k, v in metrics.items()
                                       if k not in ("time_s", "n_eval")}

            # Save predictions
            np.savez_compressed(
                os.path.join(pred_dir, f"H{horizon}_{method_name}.npz"),
                pred=preds, true=trues)

        results[str(horizon)] = h_results

    # Save results
    suffix = f"_{ds_name}" if ds_name != "MIT" else ""
    out_path = results_path(CASE, "data",
                            f"traditional_baseline_results{suffix}.json")
    save_json(results, out_path)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"SUMMARY — {ds_name}")
    logger.info("=" * 70)
    logger.info(f"{'Method':<20} {'Horizon':>8} {'RMSE':>10} {'MAE':>10} "
                f"{'R2':>8} {'MAPE%':>8}")
    logger.info("-" * 70)
    for h in HORIZONS:
        h_key = str(h)
        if h_key not in results:
            continue
        for method, m in results[h_key].items():
            logger.info(f"{method:<20} {h:>8} {m.get('rmse','N/A'):>10} "
                        f"{m.get('mae','N/A'):>10} {m.get('r2','N/A'):>8} "
                        f"{m.get('mape','N/A'):>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MIT",
                        choices=list(DATASET_CONFIGS.keys()) + ["all"])
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASET_CONFIGS:
            main(ds)
    else:
        main(args.dataset)
