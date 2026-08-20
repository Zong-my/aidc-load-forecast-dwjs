"""
Multi-seed rerun for robustness evidence (paper resubmission, DWJS26-1067).

Purpose
-------
All original results were produced with SEED=42 and live in the (read-only)
results library -- they are treated as the "seed42" run and are NOT rerun.
This script reruns the full training grids with seeds {43, 44, 45, 46} so the
revision can report mean +/- std across 5 seeds.

Per seed, three stages:

  Stage A -- public-dataset baseline grid (72 configs):
      4 datasets (MIT / Helios / Alibaba / Inference)
      x 2 horizons (H16 / H96)
      x 3 backbones (DLinear / PatchTST / iTransformer)
      x 3 feature sets (Calendar / All / Workload)
      Plain supervised training (train_one_model), protocol identical to
      run_feature_comparison.py. Classical baselines (Persistence / ARIMA /
      SVR / LightGBM / LSTM) are NOT rerun here: they are deterministic or
      out of scope for this revision.

  Stage B -- public-dataset SSPM grid (72 configs):
      Same 72 configs; flow copied from run_sspm_all_backbones.run_one:
      masked pretraining (30 epochs, mask 0.3, once per config) + dataset-
      specific 6-value alpha search by validation loss + best-alpha test eval.

  Stage C -- field pipeline (2 scenarios x 2 horizons x 3 backbones,
      baseline + SSPM), flow identical to the paper's field pipeline
      (proprietary data; see field/README.md):
      Room302 (machine room, target room_power_kw) and BuildingC
      (building, target building_power_kw), Calendar feature set
      (no workload telemetry in the field), SSPM alpha grid
      [0.01, 0.05, 0.1, 0.2, 0.4] (mid-pi tier, pi ~ 0.207).
      Per-config quantile predictions are saved as npz (q10/q50/q90/true).

      DATA ENTRY:
      the two derived field CSVs are read from FIELD_DATA_DIR (see
      common/config.py):
          *room302_15min.csv   header: time,room_power_kw,hour,minute,dow,month
          *building_15min.csv  header: time,building_power_kw,hour,minute,dow,month
      They are located by suffix glob (any vendor-name prefix is accepted;
      the canonical field_room302_15min.csv / field_building_15min.csv match).
      The measured field CSVs are proprietary and NOT shipped with the repo:
      when they are absent, stage C is skipped automatically if it was only
      part of the default stage set, or aborts with a pointer to
      field/README.md if the user explicitly requested stage c.

Quantile rearrangement (applies to EVERY evaluation in all stages)
------------------------------------------------------------------
Before computing any metric, predicted quantiles (q10, q50, q90) are made
non-decreasing by sorting along the quantile axis (Chernozhukov et al.
rearrangement). The pre-fix crossing rate is recorded:
    crossing_rate = fraction of (sample, step) positions where the raw
                    quantile triple is not non-decreasing.
npz files store the SORTED quantiles; crossing_rate goes into the JSON.
sMAPE definition: mean over points with (|y|+|yhat|)/2 > 1e-8 of
    |yhat - y| / ((|y|+|yhat|)/2) * 100.

Seeding
-------
set_seed(seed) (same helper as all original runs: torch.manual_seed +
np.random.seed + cuda.manual_seed_all) is called before every training unit
-- baseline training, masked pretraining, and each alpha finetune -- exactly
at the call sites where the original scripts called set_seed(42).
Seed 42 itself is refused by the CLI (original results stand as seed42).

Outputs (incremental, one flush after every finished config; safe to Ctrl-C)
---------------------------------------------------------------------------
  {OUT_DIR} = <repo>/results/analysis/multiseed/
    seed{N}_baseline_results.json   {ds: {"16"|"96": {"Model-FS": rec}}}
    seed{N}_sspm_results.json       {ds: {"16"|"96": {"Model-FS": rec}}}
    seed{N}_field_results.json      {ds: {"pi","features","H16"|"H96": {...}}}
    seed{N}_field_npz/{ds}_H{H}_{model}_{baseline|sspm}.npz  (q10/q50/q90/true)
  Key layouts deliberately mirror the seed42 files
  (sspm_all_backbones_results.json / feature_comparison_results*.json use
  str(horizon); the field json uses "H16"/"H96") to ease later aggregation.
  Resume: on start each JSON is loaded and finished configs are skipped
  (stage C additionally requires the npz to exist).

Per-config record: r2 / rmse / mae / mape / smape / crps / coverage_80 /
interval_width / crossing_rate / val_loss / time_s (+ alpha,
alpha_val_losses, pretrain_time_s for SSPM; + n_features for stage A).

Hard constraints
----------------
  * Writes only under the analysis output dir (results/analysis/multiseed/);
    the seed42 results library and the reference tree are read-only.
  * Never modifies anything under experiments/ or field/ -- import / read only.
  * Single GPU, device fixed to cuda:0.

Time budget (from seed42 measurements)
--------------------------------------
  Stage B: 5259 s measured for the full 72-config SSPM grid (seed42).
  Stage A: ~2000-3000 s estimated (one supervised run per config,
           PatchTST dominates; no pretrain, no alpha search).
  Stage C: 758 s measured (12.6 min, both field scenarios, seed42);
           budget ~800-1500 s with the extra rearrangement eval.
  => per seed  ~ 8000-9500 s (~2.2-2.6 h);  4 seeds ~ 9-11 h total.

Suggested segmentation (any cwd; resume makes splits free):
  nohup python multiseed_rerun.py --seeds 43,44,45,46 > multiseed.log 2>&1 &
  # or split by stage / seed pair:
  python multiseed_rerun.py --seeds 43,44 --stage b     # longest stage first
  python multiseed_rerun.py --seeds 43,44 --stage ac
  python multiseed_rerun.py --seeds 45,46 --stage abc

Usage
-----
  python multiseed_rerun.py [--seeds 43,44,45,46] [--stage a|b|c|ab|abc|a,c]
"""

import os
import sys
import glob
import time
import argparse

import numpy as np
import torch

# ------------------------------------------------------------------
# Path setup: this script lives in analysis/, so both the repo root (for
# common.*) and experiments/ (for run_baselines, run_feature_comparison, ...)
# are put on sys.path. Derived from __file__ -- never hardcoded.
# ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.utils import get_logger, save_json, load_json          # noqa: E402
from common.config import FIELD_DATA_DIR                           # noqa: E402
from train_msra import set_seed                                    # noqa: E402
from run_baselines import train_one_model                          # noqa: E402
from evaluate_msra import (compute_point_metrics, compute_crps,    # noqa: E402
                           compute_coverage)
from run_feature_comparison import (DATASET_CONFIGS, HPARAMS,      # noqa: E402
                                    HORIZONS, QUANTILE_LEVELS,
                                    load_and_engineer,
                                    build_dataset_for_feature_set,
                                    make_model)
from run_sspm_all_backbones import (pretrain_masked, train_mixup,  # noqa: E402
                                    PRETRAIN_EPOCHS, MASK_RATIO,
                                    ALPHA_GRID as SSPM_ALPHA_GRIDS)

logger = get_logger("multiseed")

# ------------------------------------------------------------------
# Experiment definition
# ------------------------------------------------------------------
DEFAULT_SEEDS = [43, 44, 45, 46]
BACKBONES = ["DLinear", "PatchTST", "iTransformer"]
FEATURE_SET_NAMES = ["Calendar", "All", "Workload"]

# Field pipeline (values copied from the verified field script)
FIELD_ALPHA_GRID = [0.01, 0.05, 0.1, 0.2, 0.4]   # mid-pi tier (pi ~ 0.207)
FIELD_PI = {"Room302": 0.20742, "BuildingC": 0.20740}
FIELD_TARGETS = {"Room302": "room_power_kw", "BuildingC": "building_power_kw"}
FIELD_CSV_SUFFIX = {"Room302": "room302_15min.csv",
                    "BuildingC": "building_15min.csv"}

OUT_DIR = os.path.join(REPO_ROOT, "results", "analysis", "multiseed")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Evaluation with quantile rearrangement (used by ALL stages)
# ------------------------------------------------------------------
def _smape(pred, true):
    """Symmetric MAPE in %, guarded against near-zero denominators."""
    denom = (np.abs(pred) + np.abs(true)) / 2.0
    mask = denom > 1e-8
    if not mask.any():
        return float("nan")
    return round(float(np.mean(np.abs(pred[mask] - true[mask])
                               / denom[mask]) * 100), 3)


def evaluate_sorted(model, test_X, test_y, norm_params, device=DEVICE):
    """Test-set evaluation with quantile rearrangement.

    Mirrors run_baselines.evaluate_on_test (same denormalization, same metric
    helpers) but (1) records the raw quantile crossing rate and (2) sorts
    (q10, q50, q90) non-decreasingly along the quantile axis BEFORE any
    metric is computed. Returns (metrics, pred_sorted (N,H,3), true (N,H))
    in original scale.
    """
    model.eval()
    X_t = torch.from_numpy(test_X).float().to(device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()          # (N, H, Q)

    t_mean = norm_params["target_mean"]
    t_std = norm_params["target_std"]
    pred_all = pred_norm * t_std + t_mean             # (N, H, Q)
    true = test_y * t_std + t_mean                    # (N, H)

    # Crossing rate BEFORE rearrangement (affine denorm preserves order).
    crossing = (np.diff(pred_all, axis=2) < 0).any(axis=2)   # (N, H)
    crossing_rate = float(crossing.mean())

    # Rearrangement: enforce q10 <= q50 <= q90.
    pred_sorted = np.sort(pred_all, axis=2)
    pred_median = pred_sorted[:, :, 1]

    metrics = compute_point_metrics(pred_median, true)
    metrics["smape"] = _smape(pred_median, true)
    metrics["crps"] = compute_crps(pred_sorted, true, QUANTILE_LEVELS)
    metrics.update(compute_coverage(pred_sorted, true, QUANTILE_LEVELS))
    metrics["crossing_rate"] = round(crossing_rate, 6)
    return metrics, pred_sorted, true


def persistence_metrics(data):
    """Naive persistence reference (copied from the field script)."""
    te_X, te_y, norm = data["test_X"], data["test_y"], data["norm_params"]
    y_mu, y_sd = norm["target_mean"], norm["target_std"]
    H = te_y.shape[1]
    last = te_X[:, -1, 0]
    pred = np.repeat(last[:, None], H, axis=1)
    yt = te_y * y_sd + y_mu
    yp = pred * y_sd + y_mu
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return dict(r2=float(1 - ss_res / ss_tot),
                rmse=float(np.sqrt(((yt - yp) ** 2).mean())),
                mae=float(np.abs(yt - yp).mean()))


# ------------------------------------------------------------------
# Incremental persistence / resume
# ------------------------------------------------------------------
def load_results(path):
    """Load previous partial results so reruns merge instead of clobber."""
    if os.path.exists(path):
        try:
            prev = load_json(path)
            prev.pop("meta", None)
            logger.info(f"[resume] merging into existing {path}")
            return prev
        except Exception as e:
            logger.warning(f"[resume] cannot parse {path} ({e}); "
                           f"starting fresh")
    return {}


def flush(results, meta, path, t_start):
    meta["total_time_s"] = round(time.time() - t_start, 1)
    out = dict(results)
    out["meta"] = meta
    save_json(out, path)


def base_meta(seed, stage, extra=None):
    meta = {
        "seed": seed,
        "stage": stage,
        "script": "multiseed_rerun.py",
        "device": DEVICE,
        "quantile_rearrangement": True,
        "crossing_rate_def": ("fraction of (sample, step) positions with a "
                              "non-monotone raw (q10,q50,q90) triple, "
                              "measured before sorting"),
        "smape_def": ("mean |yhat-y| / ((|y|+|yhat|)/2) * 100 over points "
                      "with denominator > 1e-8"),
        "horizons": HORIZONS,
        "backbones": BACKBONES,
        "hparams": HPARAMS,
        "note": ("multi-seed robustness rerun for the resubmission; "
                 "seed42 = original results library (not rerun)"),
    }
    if extra:
        meta.update(extra)
    return meta


# ------------------------------------------------------------------
# Stage A: public-dataset baseline grid (72 configs / seed)
# ------------------------------------------------------------------
def run_stage_a(seed):
    out_path = os.path.join(OUT_DIR, f"seed{seed}_baseline_results.json")
    results = load_results(out_path)
    meta = base_meta(seed, "A_baseline_grid",
                     {"feature_sets": FEATURE_SET_NAMES,
                      "datasets": list(DATASET_CONFIGS.keys())})
    t_start = time.time()
    n_total = len(DATASET_CONFIGS) * len(HORIZONS) * \
        len(BACKBONES) * len(FEATURE_SET_NAMES)
    n_done = 0

    for ds in DATASET_CONFIGS:
        df, feature_sets, target = load_and_engineer(DATASET_CONFIGS[ds])
        results.setdefault(ds, {})

        for horizon in HORIZONS:
            hk = str(horizon)
            results[ds].setdefault(hk, {})
            pending = [(m, fs) for m in BACKBONES for fs in FEATURE_SET_NAMES
                       if f"{m}-{fs}" not in results[ds][hk]]
            n_done += (len(BACKBONES) * len(FEATURE_SET_NAMES) - len(pending))
            if not pending:
                logger.info(f"[A seed{seed}] {ds} H{horizon}: complete, skip")
                continue

            # Build each needed feature-set dataset once per (ds, horizon)
            datasets = {}
            for fs in {fs for _, fs in pending}:
                datasets[fs] = build_dataset_for_feature_set(
                    df, feature_sets[fs], horizon, target)

            for model_name, fs_name in pending:
                n_done += 1
                key = f"{model_name}-{fs_name}"
                data = datasets[fs_name]
                nf = data["train_X"].shape[2]
                t0 = time.time()

                set_seed(seed)
                model = make_model(model_name, nf, horizon).to(DEVICE)
                model, best_val, elapsed = train_one_model(
                    model, data["train_X"], data["train_y"],
                    data["val_X"], data["val_y"], HPARAMS, DEVICE)
                metrics, _, _ = evaluate_sorted(
                    model, data["test_X"], data["test_y"],
                    data["norm_params"])
                del model
                torch.cuda.empty_cache()

                metrics["val_loss"] = round(best_val, 6)
                metrics["n_features"] = nf
                metrics["train_time_s"] = round(elapsed, 1)
                metrics["time_s"] = round(time.time() - t0, 1)
                results[ds][hk][key] = metrics
                flush(results, meta, out_path, t_start)

                logger.info(
                    f"[A seed{seed}] ({n_done}/{n_total}) {ds} H{horizon} "
                    f"{key}: R2={metrics['r2']} RMSE={metrics['rmse']} "
                    f"cross={metrics['crossing_rate']} "
                    f"({metrics['time_s']}s)")
            del datasets

    logger.info(f"[A seed{seed}] stage done in "
                f"{time.time() - t_start:.1f}s -> {out_path}")


# ------------------------------------------------------------------
# Stage B: public-dataset SSPM grid (72 configs / seed)
# ------------------------------------------------------------------
def sspm_one(seed, ds, model_name, data, horizon, alphas):
    """Pretrain once + alpha search + best-alpha eval.

    Flow copied from run_sspm_all_backbones.run_one with SEED -> seed.
    Returns the metrics record.
    """
    nf = data["train_X"].shape[2]

    # Phase 1: masked pretraining (once, reused across the alpha search)
    set_seed(seed)
    model_pre = make_model(model_name, nf, horizon).to(DEVICE)
    t0 = time.time()
    model_pre = pretrain_masked(model_pre, data["train_X"], HPARAMS, DEVICE,
                                epochs=PRETRAIN_EPOCHS, mask_ratio=MASK_RATIO)
    pretrain_sd = {k: v.cpu().clone()
                   for k, v in model_pre.state_dict().items()}
    del model_pre
    torch.cuda.empty_cache()
    t_pre = time.time() - t0

    # Phase 2: alpha search by validation loss
    best_val = float("inf")
    best_alpha = None
    best_sd = None
    alpha_val_losses = {}
    for alpha in alphas:
        set_seed(seed)
        m = make_model(model_name, nf, horizon).to(DEVICE)
        m.load_state_dict(pretrain_sd)
        m, val_loss = train_mixup(m, data, HPARAMS, alpha, DEVICE)
        alpha_val_losses[str(alpha)] = round(val_loss, 6)
        if val_loss < best_val:
            best_val = val_loss
            best_alpha = alpha
            best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        del m
        torch.cuda.empty_cache()

    # Phase 3: evaluate the best model on the test set (rearranged quantiles)
    m = make_model(model_name, nf, horizon).to(DEVICE)
    m.load_state_dict(best_sd)
    metrics, _, _ = evaluate_sorted(m, data["test_X"], data["test_y"],
                                    data["norm_params"])
    del m
    torch.cuda.empty_cache()

    metrics["alpha"] = best_alpha
    metrics["val_loss"] = round(best_val, 6)
    metrics["alpha_val_losses"] = alpha_val_losses
    metrics["pretrain_time_s"] = round(t_pre, 1)
    return metrics


def run_stage_b(seed):
    out_path = os.path.join(OUT_DIR, f"seed{seed}_sspm_results.json")
    results = load_results(out_path)
    meta = base_meta(seed, "B_sspm_grid",
                     {"feature_sets": FEATURE_SET_NAMES,
                      "datasets": list(DATASET_CONFIGS.keys()),
                      "pretrain_epochs": PRETRAIN_EPOCHS,
                      "mask_ratio": MASK_RATIO,
                      "sspm_alpha_grids": SSPM_ALPHA_GRIDS})
    t_start = time.time()
    n_total = len(DATASET_CONFIGS) * len(HORIZONS) * \
        len(BACKBONES) * len(FEATURE_SET_NAMES)
    n_done = 0

    for ds in DATASET_CONFIGS:
        df, feature_sets, target = load_and_engineer(DATASET_CONFIGS[ds])
        results.setdefault(ds, {})
        alphas = SSPM_ALPHA_GRIDS[ds]

        for horizon in HORIZONS:
            hk = str(horizon)
            results[ds].setdefault(hk, {})
            pending = [(m, fs) for m in BACKBONES for fs in FEATURE_SET_NAMES
                       if f"{m}-{fs}" not in results[ds][hk]]
            n_done += (len(BACKBONES) * len(FEATURE_SET_NAMES) - len(pending))
            if not pending:
                logger.info(f"[B seed{seed}] {ds} H{horizon}: complete, skip")
                continue

            datasets = {}
            for fs in {fs for _, fs in pending}:
                datasets[fs] = build_dataset_for_feature_set(
                    df, feature_sets[fs], horizon, target)

            for model_name, fs_name in pending:
                n_done += 1
                key = f"{model_name}-{fs_name}"
                t0 = time.time()
                metrics = sspm_one(seed, ds, model_name,
                                   datasets[fs_name], horizon, alphas)
                metrics["time_s"] = round(time.time() - t0, 1)
                results[ds][hk][key] = metrics
                flush(results, meta, out_path, t_start)

                logger.info(
                    f"[B seed{seed}] ({n_done}/{n_total}) {ds} H{horizon} "
                    f"SSPM_{key}: R2={metrics['r2']} "
                    f"RMSE={metrics['rmse']} alpha={metrics['alpha']} "
                    f"cross={metrics['crossing_rate']} "
                    f"({metrics['time_s']}s)")
            del datasets

    logger.info(f"[B seed{seed}] stage done in "
                f"{time.time() - t_start:.1f}s -> {out_path}")


# ------------------------------------------------------------------
# Stage C: field pipeline (Room302 + BuildingC)
# ------------------------------------------------------------------
def field_csv_hits(ds_name):
    """Suffix-glob for a field CSV under FIELD_DATA_DIR (canonical name matches)."""
    return sorted(glob.glob(
        os.path.join(FIELD_DATA_DIR, "*" + FIELD_CSV_SUFFIX[ds_name])))


def field_data_available():
    """True iff both field CSVs are present under FIELD_DATA_DIR."""
    return all(field_csv_hits(ds) for ds in FIELD_CSV_SUFFIX)


def resolve_field_datasets():
    """Locate the derived field CSVs under FIELD_DATA_DIR by suffix glob.

    The measured field CSVs are proprietary and not shipped with the repo;
    place authorized copies under FIELD_DATA_DIR (see common/config.py and
    field/README.md). Any vendor-name prefix is accepted; the canonical
    field_room302_15min.csv / field_building_15min.csv also match.
    """
    cfgs = {}
    for ds_name, suffix in FIELD_CSV_SUFFIX.items():
        hits = field_csv_hits(ds_name)
        if not hits:
            raise FileNotFoundError(
                f"Field CSV *{suffix} not found in {FIELD_DATA_DIR}. "
                f"The measured field data are proprietary and not shipped; "
                f"see field/README.md for how to supply authorized copies.")
        if len(hits) > 1:
            logger.warning(f"Multiple CSVs match *{suffix}: {hits}; "
                           f"using {hits[0]}")
        cfgs[ds_name] = {
            "path": hits[0],
            "target": FIELD_TARGETS[ds_name],
            "workload_cols": [],          # no workload telemetry in the field
            "pi": FIELD_PI[ds_name],
        }
    return cfgs


def save_field_npz(npz_path, pred_sorted, true,
                   val_pred_sorted=None, val_true=None):
    """npz stores the SORTED quantiles (requirement 1).

    When val_pred_sorted/val_true are given, the validation-segment
    predictions are stored alongside (q10_val/...): they are the calibration
    inputs for the post-hoc CQR analysis (field-side conformal needs the
    latter half of the validation segment; without these arrays the analysis
    would require retraining, as stage C keeps no checkpoints).
    """
    arrs = dict(
        q10=pred_sorted[:, :, 0].astype(np.float32),
        q50=pred_sorted[:, :, 1].astype(np.float32),
        q90=pred_sorted[:, :, 2].astype(np.float32),
        true=true.astype(np.float32))
    if val_pred_sorted is not None:
        arrs.update(
            q10_val=val_pred_sorted[:, :, 0].astype(np.float32),
            q50_val=val_pred_sorted[:, :, 1].astype(np.float32),
            q90_val=val_pred_sorted[:, :, 2].astype(np.float32),
            true_val=val_true.astype(np.float32))
    np.savez_compressed(npz_path, **arrs)


def run_stage_c(seed):
    field_cfgs = resolve_field_datasets()
    out_path = os.path.join(OUT_DIR, f"seed{seed}_field_results.json")
    npz_dir = os.path.join(OUT_DIR, f"seed{seed}_field_npz")
    os.makedirs(npz_dir, exist_ok=True)

    results = load_results(out_path)
    meta = base_meta(seed, "C_field",
                     {"field_alpha_grid": FIELD_ALPHA_GRID,
                      "feature_set": "Calendar",
                      "datasets": {k: v["path"]
                                   for k, v in field_cfgs.items()},
                      "pretrain_epochs": PRETRAIN_EPOCHS,
                      "mask_ratio": MASK_RATIO,
                      "npz_dir": npz_dir})
    t_start = time.time()

    for ds_name, cfg in field_cfgs.items():
        df, fsets, target = load_and_engineer(cfg)
        fcols = fsets["Calendar"]
        results.setdefault(ds_name, {"pi": cfg["pi"], "features": fcols})

        for H in HORIZONS:
            hk = f"H{H}"
            entry = results[ds_name].setdefault(hk, {})

            def npz_path(m, kind):
                return os.path.join(npz_dir,
                                    f"{ds_name}_H{H}_{m}_{kind}.npz")

            pend_base = [m for m in BACKBONES
                         if f"{m}-Cal" not in entry
                         or not os.path.exists(npz_path(m, "baseline"))]
            pend_sspm = [m for m in BACKBONES
                         if f"SSPM-{m}-Cal" not in entry
                         or not os.path.exists(npz_path(m, "sspm"))]
            if not pend_base and not pend_sspm and "Persistence" in entry:
                logger.info(f"[C seed{seed}] {ds_name} {hk}: complete, skip")
                continue

            set_seed(seed)   # same call site as the field script
            data = build_dataset_for_feature_set(df, fcols, H, target)

            if "Persistence" not in entry:
                entry["Persistence"] = persistence_metrics(data)
                flush(results, meta, out_path, t_start)

            for m in BACKBONES:
                # ---------- baseline ----------
                if m in pend_base:
                    t0 = time.time()
                    set_seed(seed)
                    model = make_model(m, len(fcols), H).to(DEVICE)
                    model, base_val, _ = train_one_model(
                        model, data["train_X"], data["train_y"],
                        data["val_X"], data["val_y"], HPARAMS, device=DEVICE)
                    met, pred_sorted, true = evaluate_sorted(
                        model, data["test_X"], data["test_y"],
                        data["norm_params"])
                    _vm, val_sorted, val_true = evaluate_sorted(
                        model, data["val_X"], data["val_y"],
                        data["norm_params"])
                    del model
                    torch.cuda.empty_cache()

                    met["val_loss"] = round(float(base_val), 6)
                    met["time_s"] = round(time.time() - t0, 1)
                    entry[f"{m}-Cal"] = met
                    save_field_npz(npz_path(m, "baseline"), pred_sorted, true,
                                   val_sorted, val_true)
                    flush(results, meta, out_path, t_start)
                    logger.info(f"[C seed{seed}] {ds_name} {hk} {m} "
                                f"baseline R2={met['r2']} "
                                f"cross={met['crossing_rate']} "
                                f"({met['time_s']}s)")

                # ---------- SSPM ----------
                if m in pend_sspm:
                    t0 = time.time()
                    set_seed(seed)
                    model = make_model(m, len(fcols), H).to(DEVICE)
                    model = pretrain_masked(model, data["train_X"],
                                            HPARAMS, DEVICE)
                    pre_state = {k: v.clone()
                                 for k, v in model.state_dict().items()}
                    best = (None, float("inf"), None)
                    alpha_val_losses = {}
                    for a in FIELD_ALPHA_GRID:
                        model.load_state_dict(pre_state)
                        set_seed(seed)
                        model, val = train_mixup(model, data, HPARAMS, a,
                                                 DEVICE)
                        alpha_val_losses[str(a)] = round(float(val), 6)
                        if val < best[1]:
                            best = ({k: v.clone()
                                     for k, v in model.state_dict().items()},
                                    val, a)
                    model.load_state_dict(best[0])
                    met, pred_sorted, true = evaluate_sorted(
                        model, data["test_X"], data["test_y"],
                        data["norm_params"])
                    _vm, val_sorted, val_true = evaluate_sorted(
                        model, data["val_X"], data["val_y"],
                        data["norm_params"])
                    del model
                    torch.cuda.empty_cache()

                    met["alpha"] = best[2]
                    met["val_loss"] = round(float(best[1]), 6)
                    met["alpha_val_losses"] = alpha_val_losses
                    met["time_s"] = round(time.time() - t0, 1)
                    entry[f"SSPM-{m}-Cal"] = met
                    save_field_npz(npz_path(m, "sspm"), pred_sorted, true,
                                   val_sorted, val_true)
                    flush(results, meta, out_path, t_start)
                    logger.info(f"[C seed{seed}] {ds_name} {hk} {m} "
                                f"SSPM R2={met['r2']} alpha={best[2]} "
                                f"cross={met['crossing_rate']} "
                                f"({met['time_s']}s)")

    logger.info(f"[C seed{seed}] stage done in "
                f"{time.time() - t_start:.1f}s -> {out_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
STAGE_FUNCS = {"a": run_stage_a, "b": run_stage_b, "c": run_stage_c}


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed rerun (robustness evidence, seeds != 42)")
    p.add_argument("--seeds", type=str,
                   default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="comma-separated seeds, e.g. 43,44 "
                        f"(default {DEFAULT_SEEDS}; 42 is refused)")
    p.add_argument("--stage", type=str, default=None,
                   help="stages to run: a=baseline grid, b=SSPM grid, "
                        "c=field; combine freely, e.g. a | ab | a,c "
                        "(default abc)")
    args = p.parse_args()

    try:
        seeds = [int(s) for s in args.seeds.replace(" ", "").split(",") if s]
    except ValueError:
        p.error(f"--seeds must be comma-separated ints, got {args.seeds!r}")
    if not seeds:
        p.error("--seeds is empty")
    if 42 in seeds:
        p.error("seed 42 is the original results library and must not be "
                "rerun; pick seeds != 42")

    stage_explicit = args.stage is not None
    stage_str = args.stage if stage_explicit else "abc"
    stages = [ch for ch in stage_str.lower().replace(",", "")
              if ch.strip()]
    bad = [ch for ch in stages if ch not in STAGE_FUNCS]
    if bad or not stages:
        p.error(f"--stage must combine letters from a/b/c, got {stage_str!r}")
    # dedupe, keep canonical a -> b -> c order
    stages = [ch for ch in "abc" if ch in stages]
    return seeds, stages, stage_explicit


def main():
    seeds, stages, stage_explicit = parse_args()

    # Stage C needs the proprietary field CSVs. If they are absent: abort when
    # the user explicitly asked for stage c, otherwise silently drop it from
    # the default stage set with a one-line warning.
    if "c" in stages and not field_data_available():
        if stage_explicit:
            sys.exit("Stage c requested but the measured field CSVs are not "
                     f"present under {FIELD_DATA_DIR}. The field data are "
                     "proprietary and not shipped; see field/README.md.")
        stages = [st for st in stages if st != "c"]
        logger.warning("[stage c] field CSVs absent; dropping stage c from "
                       "the default set (see field/README.md)")

    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Seeds: {seeds}   Stages: {stages}")
    logger.info(f"Output dir: {OUT_DIR}")

    t_all = time.time()
    for seed in seeds:
        logger.info(f"\n{'=' * 64}\n SEED {seed}\n{'=' * 64}")
        t_seed = time.time()
        for st in stages:
            STAGE_FUNCS[st](seed)
        logger.info(f"[seed {seed}] all stages done in "
                    f"{(time.time() - t_seed) / 60:.1f} min")

    logger.info(f"\nALL DONE: seeds={seeds} stages={stages} in "
                f"{(time.time() - t_all) / 3600:.2f} h")


if __name__ == "__main__":
    main()
