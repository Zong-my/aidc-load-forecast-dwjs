"""
SOTA Time-Series Forecasting Baselines: DLinear, PatchTST, iTransformer.

Each model is trained with and without calendar features (6 variants total),
across 2 horizons (16, 96), yielding 12 experiments.

Output: quantile predictions (batch, horizon, 3) for quantiles [0.1, 0.5, 0.9].
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_logger, results_path, save_json

from config_msra import MSRA_CONFIG
from prepare_msra_data import build_msra_dataset
from evaluate_msra import compute_point_metrics
from msra.physics_constraints import pinball_loss
from train_msra import set_seed, make_loader

logger = get_logger("run_baselines")

# ============================================================
# Hyperparameters
# ============================================================

HPARAMS = {
    "batch_size": 128,
    "epochs": 50,
    "patience": 7,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # DLinear
    "dlinear_moving_avg": 25,
    # PatchTST
    "patch_size": 16,
    "patch_stride": 8,
    "ptst_d_model": 128,
    "ptst_nhead": 4,
    "ptst_layers": 2,
    "ptst_dropout": 0.1,
    # iTransformer
    "itrans_d_model": 128,
    "itrans_nhead": 4,
    "itrans_layers": 2,
    "itrans_dropout": 0.1,
}

QUANTILE_LEVELS = [0.1, 0.5, 0.9]
N_QUANTILES = len(QUANTILE_LEVELS)
CAL_FEATURE_INDICES = [12, 13, 14, 15]  # hour_sin, hour_cos, dow_sin, dow_cos


# ============================================================
# Model 1: DLinear
# ============================================================

class MovingAvg(nn.Module):
    """Moving average block for trend extraction."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        # AvgPool1d expects (batch, channels, seq_len)
        out = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class DLinear(nn.Module):
    """DLinear: Decomposition-Linear for time series forecasting.

    Decomposes input into trend (moving average) and remainder,
    applies separate linear layers to each component.
    Output: (batch, horizon, n_quantiles).
    """

    def __init__(self, lookback, horizon, n_features, n_quantiles=3,
                 moving_avg_kernel=25):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.decomp = MovingAvg(moving_avg_kernel)

        # Channel-independent: one linear per feature, then aggregate
        # Simpler approach: flatten all features and project
        self.trend_linear = nn.Linear(lookback * n_features,
                                      horizon * n_quantiles)
        self.remainder_linear = nn.Linear(lookback * n_features,
                                          horizon * n_quantiles)

    def forward(self, x):
        # x: (batch, lookback, n_features)
        trend = self.decomp(x)
        remainder = x - trend

        trend_flat = trend.reshape(x.shape[0], -1)
        remainder_flat = remainder.reshape(x.shape[0], -1)

        trend_out = self.trend_linear(trend_flat)
        remainder_out = self.remainder_linear(remainder_flat)

        out = trend_out + remainder_out
        return out.view(-1, self.horizon, self.n_quantiles)


# ============================================================
# Model 2: PatchTST
# ============================================================

class PatchTST(nn.Module):
    """PatchTST: Patch Time Series Transformer.

    Channel-independent: each feature channel is processed independently.
    Time series is split into patches, each patch becomes a token.
    Output: (batch, horizon, n_quantiles).
    """

    def __init__(self, lookback, horizon, n_features, n_quantiles=3,
                 patch_size=16, patch_stride=8, d_model=128, nhead=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_features = n_features
        self.n_quantiles = n_quantiles
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Number of patches per channel
        self.n_patches = (lookback - patch_size) // patch_stride + 1

        # Patch embedding: project each patch to d_model
        self.patch_proj = nn.Linear(patch_size, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        # Head: from all channels' CLS-like representation to output
        # Use mean-pooled patch representation per channel, then project
        self.head = nn.Linear(d_model * n_features, horizon * n_quantiles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, lookback, n_features)
        B, L, C = x.shape

        # Channel-independent patching
        # Reshape to (B*C, L) then create patches
        x_channels = x.permute(0, 2, 1).reshape(B * C, L)  # (B*C, L)

        # Extract patches: (B*C, n_patches, patch_size)
        patches = x_channels.unfold(1, self.patch_size, self.patch_stride)

        # Project patches: (B*C, n_patches, d_model)
        tokens = self.patch_proj(patches)
        tokens = tokens + self.pos_embed

        # Transformer encoding
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # Mean pool over patches: (B*C, d_model)
        pooled = tokens.mean(dim=1)

        # Reshape to (B, C*d_model)
        pooled = pooled.view(B, C, -1).reshape(B, -1)
        pooled = self.dropout(pooled)

        # Project to output
        out = self.head(pooled)
        return out.view(B, self.horizon, self.n_quantiles)


# ============================================================
# Model 3: iTransformer
# ============================================================

class iTransformer(nn.Module):
    """iTransformer: Inverted Transformer for time series forecasting.

    Each variable (channel) is treated as a token.
    Token embedding = linear projection of the full lookback for that variable.
    Attention captures cross-variable dependencies.
    Output: (batch, horizon, n_quantiles).
    """

    def __init__(self, lookback, horizon, n_features, n_quantiles=3,
                 d_model=128, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_features = n_features
        self.n_quantiles = n_quantiles

        # Embed each variable's lookback into d_model
        self.var_embed = nn.Linear(lookback, d_model)

        # Learnable variable positional encoding
        self.var_pos = nn.Parameter(
            torch.randn(1, n_features, d_model) * 0.02)

        # Transformer encoder (attention over variables)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        # Output projection: from first variable (target) token to predictions
        # Or use all variables pooled
        self.head = nn.Linear(d_model * n_features, horizon * n_quantiles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, lookback, n_features)
        B, L, C = x.shape

        # Transpose to (batch, n_features, lookback) — each var is a token
        x_var = x.permute(0, 2, 1)  # (B, C, L)

        # Embed: (B, C, d_model)
        tokens = self.var_embed(x_var)
        tokens = tokens + self.var_pos

        # Transformer: attention over variables
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # Flatten all variable tokens: (B, C*d_model)
        out = tokens.reshape(B, -1)
        out = self.dropout(out)

        # Project to output
        out = self.head(out)
        return out.view(B, self.horizon, self.n_quantiles)


# ============================================================
# Training utilities
# ============================================================

def train_one_model(model, train_X, train_y, val_X, val_y, hp, device="cuda"):
    """Train a single model with early stopping.

    Returns: (trained_model, best_val_loss, elapsed_time)
    """
    ql = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32)

    train_dl = make_loader(train_X, train_y, hp["batch_size"], shuffle=True)
    val_dl = make_loader(val_X, val_y, hp["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    n_steps = len(train_dl) * hp["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=hp["lr"], total_steps=n_steps, pct_start=0.2)

    def loss_fn(pred, target):
        return pinball_loss(pred, target, ql.to(device))

    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(hp["epochs"]):
        # Train
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * len(xb)
            n_samples += len(xb)
        train_loss = epoch_loss / max(n_samples, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
                n_val += len(xb)
        val_loss = val_loss / max(n_val, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hp["patience"]:
                logger.info(f"    Early stop at epoch {epoch+1}, "
                            f"best_val={best_val:.6f}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"    Epoch {epoch+1:3d}: train={train_loss:.6f}, "
                        f"val={val_loss:.6f}")

    elapsed = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val, elapsed


def evaluate_on_test(model, test_X, test_y, norm_params, device="cuda",
                     return_predictions=False):
    """Evaluate model on test set, return point metrics in original scale.

    If return_predictions=True, also returns (pred_quantiles, true) arrays.
    """
    from evaluate_msra import compute_step_metrics, compute_crps, compute_coverage

    model.eval()
    X_t = torch.from_numpy(test_X).float().to(device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()  # (N, H, Q)

    t_mean = norm_params["target_mean"]
    t_std = norm_params["target_std"]

    # De-normalize all quantiles
    pred_all = pred_norm * t_std + t_mean  # (N, H, Q)
    true_kw = test_y * t_std + t_mean      # (N, H)

    # Median quantile for point metrics
    pred_median = pred_all[:, :, 1]

    metrics = compute_point_metrics(pred_median, true_kw)
    metrics["step_rmse"] = compute_step_metrics(pred_median, true_kw)
    metrics["crps"] = compute_crps(pred_all, true_kw, [0.1, 0.5, 0.9])
    metrics.update(compute_coverage(pred_all, true_kw, [0.1, 0.5, 0.9]))

    if return_predictions:
        return metrics, pred_all, true_kw
    return metrics


def remove_calendar_features(X, cal_indices):
    """Remove calendar feature columns from input array.

    Args:
        X: (N, lookback, n_features)
        cal_indices: list of column indices to remove

    Returns:
        X_new: (N, lookback, n_features - len(cal_indices))
    """
    keep = [i for i in range(X.shape[2]) if i not in cal_indices]
    return X[:, :, keep]


# ============================================================
# Main
# ============================================================

def run_all_baselines():
    """Run all 12 experiments: 3 models × 2 horizons × 2 feature sets."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    logger.info("Loading MSRA dataset...")
    data = build_msra_dataset(MSRA_CONFIG)

    results = {}

    for horizon in MSRA_CONFIG["horizons"]:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# HORIZON = {horizon} steps ({horizon*15/60:.0f}h)")
        logger.info(f"{'#'*60}")

        hdata = data["horizons"][horizon]
        n_features_full = hdata["train_X"].shape[2]
        n_features_nocal = n_features_full - len(CAL_FEATURE_INDICES)
        lookback = hdata["train_X"].shape[1]
        norm = hdata["norm_params"]

        h_results = {}

        # Prepare NoCal data
        train_X_nocal = remove_calendar_features(hdata["train_X"],
                                                  CAL_FEATURE_INDICES)
        val_X_nocal = remove_calendar_features(hdata["val_X"],
                                                CAL_FEATURE_INDICES)
        test_X_nocal = remove_calendar_features(hdata["test_X"],
                                                 CAL_FEATURE_INDICES)

        # Model configurations: (name, model_factory, n_features, train_X, val_X, test_X)
        experiments = []

        # --- DLinear ---
        experiments.append((
            "DLinear",
            lambda nf: DLinear(
                lookback, horizon, nf, N_QUANTILES,
                HPARAMS["dlinear_moving_avg"]),
            n_features_full,
            hdata["train_X"], hdata["val_X"], hdata["test_X"]))
        experiments.append((
            "DLinear-NoCal",
            lambda nf: DLinear(
                lookback, horizon, nf, N_QUANTILES,
                HPARAMS["dlinear_moving_avg"]),
            n_features_nocal,
            train_X_nocal, val_X_nocal, test_X_nocal))

        # --- PatchTST ---
        experiments.append((
            "PatchTST",
            lambda nf: PatchTST(
                lookback, horizon, nf, N_QUANTILES,
                HPARAMS["patch_size"], HPARAMS["patch_stride"],
                HPARAMS["ptst_d_model"], HPARAMS["ptst_nhead"],
                HPARAMS["ptst_layers"], HPARAMS["ptst_dropout"]),
            n_features_full,
            hdata["train_X"], hdata["val_X"], hdata["test_X"]))
        experiments.append((
            "PatchTST-NoCal",
            lambda nf: PatchTST(
                lookback, horizon, nf, N_QUANTILES,
                HPARAMS["patch_size"], HPARAMS["patch_stride"],
                HPARAMS["ptst_d_model"], HPARAMS["ptst_nhead"],
                HPARAMS["ptst_layers"], HPARAMS["ptst_dropout"]),
            n_features_nocal,
            train_X_nocal, val_X_nocal, test_X_nocal))

        # --- iTransformer ---
        experiments.append((
            "iTransformer",
            lambda nf: iTransformer(
                lookback, horizon, nf, N_QUANTILES,
                HPARAMS["itrans_d_model"], HPARAMS["itrans_nhead"],
                HPARAMS["itrans_layers"], HPARAMS["itrans_dropout"]),
            n_features_full,
            hdata["train_X"], hdata["val_X"], hdata["test_X"]))
        experiments.append((
            "iTransformer-NoCal",
            lambda nf: iTransformer(
                lookback, horizon, nf, N_QUANTILES,
                HPARAMS["itrans_d_model"], HPARAMS["itrans_nhead"],
                HPARAMS["itrans_layers"], HPARAMS["itrans_dropout"]),
            n_features_nocal,
            train_X_nocal, val_X_nocal, test_X_nocal))

        for name, model_fn, nf, tr_X, va_X, te_X in experiments:
            logger.info(f"\n  --- {name} (H={horizon}, features={nf}) ---")

            model = model_fn(nf).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"    Parameters: {n_params:,}")

            model, best_val, elapsed = train_one_model(
                model, tr_X, hdata["train_y"],
                va_X, hdata["val_y"], HPARAMS, device)

            metrics = evaluate_on_test(model, te_X, hdata["test_y"],
                                       norm, device)
            metrics["val_loss"] = round(best_val, 6)
            metrics["train_time_s"] = round(elapsed, 1)
            metrics["n_params"] = n_params

            h_results[name] = metrics
            logger.info(f"    Result: RMSE={metrics['rmse']}, MAE={metrics['mae']}, "
                        f"R2={metrics['r2']}, MAPE={metrics['mape']}%")
            logger.info(f"    Time: {elapsed:.1f}s, Val loss: {best_val:.6f}")

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        results[str(horizon)] = h_results

    # Save results
    out_dir = os.path.join(results_path("case10c_cluster_forecast"), "data")
    out_path = os.path.join(out_dir, "baseline_results.json")
    save_json(results, out_path)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<22} {'Horizon':>7} {'RMSE':>8} {'MAE':>8} "
                f"{'R2':>7} {'MAPE%':>7} {'Time(s)':>8}")
    logger.info("-" * 80)
    for h_key in sorted(results.keys(), key=int):
        for name, m in results[h_key].items():
            logger.info(f"{name:<22} {h_key:>7} {m['rmse']:>8.3f} {m['mae']:>8.3f} "
                        f"{m['r2']:>7.4f} {m['mape']:>7.2f} "
                        f"{m.get('train_time_s', 0):>8.1f}")

    return results


if __name__ == "__main__":
    run_all_baselines()
