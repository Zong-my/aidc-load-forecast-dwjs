"""
Clean-protocol CQR v2: in-protocol configuration selection
(audit fix 2 -- selection contamination, DWJS26-1067 revision).

v2 vs v1 (what "selection contamination" means and how it is fixed)
-------------------------------------------------------------------
clean_cqr_protocol.py (v1) fixed the calibration-segment leakage: it
introduced the four-segment split with a purge gap, early-stopped and
selected alpha on val1 only, calibrated CQR on non-overlapping val2
windows, and evaluated test exactly once. But v1 still INHERITED the
configuration identity from the old, full-validation-set protocol:
  * Stage P took the backbone x feature-set pair per (dataset, H) from
    the paper's table (STAGE_P_CONFIGS hardcode) -- those pairs had been
    selected on the FULL validation set of the old runs;
  * Stage F took the per-seed backbone from multiseed_aggregate.json --
    likewise selected on the FULL validation set.
Under v1, WHICH configuration got calibrated was therefore still a
function of data that includes what is now val2 -- a residual,
selection-level contamination. v2 removes it: every selection dimension
  Stage P: backbone x feature set x alpha   (9 configs x 6 alphas)
  Stage F: backbone x alpha                 (3 backbones x 5 alphas)
is chosen by val1 loss inside the four-segment protocol and frozen
together, before val2 calibration and the single test evaluation. No
fitting or selection decision ever sees val2 or test. The old choices
are kept ONLY as reference constants to report agreement
(matches_paper_v1 / matches_v1_ref); they influence nothing.

Four-segment protocol (unchanged from v1)
-----------------------------------------
build_dataset_for_feature_set gives the usual 70/15/15 train/val/test
window sets (stride-1, time-ordered). The val windows are then split:

    val1 = windows [0, mid)                mid = n_val // 2
           -> used ONLY for early stopping (train_mixup) and for the
              backbone / feature-set / alpha selection
    purge gap = windows [mid, mid + LOOKBACK + H - 1)   -> DISCARDED
    val2 = windows [mid + gap, n_val)
           -> used ONLY for calibration (forward inference; no fitting
              or selection decision may touch it)
    test  -> evaluated exactly once, after all decisions are frozen

Boundary handling (why the purge gap): window i covers raw rows
[i, i + L + H). The last val1 window covers rows up to mid-1 + L + H - 1,
so the first window sharing NO raw data row with any val1 window starts
at mid + L + H - 1. Dropping the first L + H - 1 windows of the second
half removes every cross-boundary overlap: the fitted model + selected
configuration are a function of (train, val1) only, and (train, val1)
shares no raw row with val2. The gap is taken from the val2 side so the
selection segment stays at full strength. Worst-case cost (H=96): val2
loses 191 windows, leaving n_h = 7 non-overlapping calibration windows
for Inference/H96 and Room302/H96 -- small but valid
(k = ceil(0.8*(7+1)) = 7 <= 7, guaranteed level 7/8).

CQR calibration (unchanged from v1)
-----------------------------------
From val2, take the NON-OVERLAPPING window subset idx = {0, H, 2H, ...}
(window starts differ by >= H, so target segments are disjoint).
Conformity score E = max(q10 - y, y - q90), computed on SORTED quantiles.
Per lead time h (each h has its own miscoverage profile):
    Qhat_h = the ceil(0.8 * (n_h + 1))-th smallest of {E[i, h]},
    n_h = number of non-overlapping val2 windows (same for all h).
Test interval: [q10_h - Qhat_h, q90_h + Qhat_h].
A POOLED variant (single Qhat over all (i, h) of the same non-overlapping
subset, n = n_h * H) is computed as a contrast, not as the primary
method. Guaranteed marginal coverage under exchangeability:
k / (n_h + 1) >= 0.8 -- this guaranteed level is computed per dataset x
horizon from its own n_h and stored AS-IS per record (never merged
across datasets into a single range).

Interval constructions compared on the test segment (unchanged)
---------------------------------------------------------------
    raw        : model (q10, q90) as-is
    w_rule     : q50 +/- (q - q50) * w  with the paper's cold-start rule
                 w(pi_pre) = max(1, 1.5 - 0.8 * pi_pre); pi_pre is the
                 PRE-TEST caliber (STL pi on the first 85% of the series,
                 pretest_pi_results.json):
                   MIT 0.016187  Helios 0.056307
                   Alibaba 0.208343  Inference 0.822543
                 Field scenarios use the precomputed w directly:
                   Room302 1.3398  BuildingC 1.3359
    cqr_perh   : per-lead-time CQR as above (primary)
    cqr_pooled : pooled-score CQR contrast
Every interval triple is made non-decreasing by per-point sorting before
evaluation; the pre-fix non-monotonicity rate is recorded. Test metrics
per method: PICP, MPIW, PINAW (normalized by test target range), Winkler
at alpha=0.2, CRPS (evaluate_msra.compute_crps, 2*mean-pinball), and the
per-lead-time coverage profile: the FULL H-length picp_h array is stored
in the npz for every method (new in v2), with min/p25/median/p75/max
summarized in the JSON.

Stage P -- public: 8 grid cells x 9 SSPM configurations (seed 42)
-----------------------------------------------------------------
For each (dataset, H) in {MIT, Helios, Alibaba, Inference} x {16, 96}:
enumerate ALL 9 configs (DLinear / PatchTST / iTransformer  x
Calendar / All / Workload -- the same 3x3 space and enumeration order as
run_sspm_all_backbones.main). Per config the training flow is exactly
the v1 single-config flow: masked pretrain (30 ep, mask 0.3) on train ->
the dataset's own 6-value ALPHA_GRID, each alpha finetuned with
train_mixup early-stopping on val1 ONLY, alpha by val1 loss. The winner
among the 9 configs is the joint argmin of val1 loss (backbone x
feature set x alpha frozen together; ties -> first in enumeration
order). ONLY the winner gets val2 calibration + test evaluation + npz.
The JSON records the full 9-config val1-loss table, the winner, and
whether it matches the old (paper-table) choice. 8 x 9 = 72 trainings.
Diagnostics (per-dataset w*_cal on val2, OLS on (pi_pre, w*_cal),
leave-one-out rule fit evaluated on the held-out TEST) are unchanged in
logic and rebuilt from the v2 winners' npz.

Stage F -- field: 16 cells x 3 backbones (4 seeds x 2 scen. x 2 H)
------------------------------------------------------------------
Calendar features fixed (the field data has no workload telemetry).
Per (seed, scenario, H): enumerate the 3 backbones (FIELD_ALPHA_GRID
[0.01, 0.05, 0.1, 0.2, 0.4] per backbone, alpha by val1 loss; call
sites mirror multiseed stage C), select backbone + alpha by val1 loss,
freeze, then val2 calibration + test evaluation.
multiseed_aggregate.json is NOT read; its per-seed backbone map is kept
only as a reference constant for the matches_v1_ref flag. The JSON
records the per-backbone val1-loss table and the winner. 16 x 3 = 48
trainings. Aggregated per (scenario, horizon): 4-seed mean +/- std.

Outputs (incremental flush after every training unit; Ctrl-C safe)
------------------------------------------------------------------
  {OUT_DIR} = <repo>/results/analysis/clean_cqr_v2/
              (v1 outputs in .../clean_cqr/ are left untouched, so the
              v1-vs-v2 audit diff stays available)
    clean_cqr_v2_results.json
        stage_p / stage_f: per-cell records (winner + full sweep tables)
        stage_p_sweeps / stage_f_sweeps: partial sweep caches of
              unfinished cells (resume bookkeeping; folded into the cell
              record and removed once the cell completes)
        stage_p_diagnostics / stage_f_aggregate: rebuilt on every flush
        meta: v2-vs-v1 note (selection-contamination fix), protocol,
              enumeration spaces, per-cell guarantee levels
    npz/P_{ds}_H{h}.npz, npz/F_{ds}_H{h}_s{seed}.npz   (winner only)
        q10/q50/q90/true for val2 and test (SORTED quantiles, original
        units), val2 non-overlap indices, per-h Qhat, split bookkeeping,
        and (new in v2) the full per-lead-time TEST coverage arrays
        picp_h_raw / picp_h_w_rule / picp_h_cqr_perh / picp_h_cqr_pooled.
  Resume: a cell is skipped iff its JSON record AND its npz both exist.
  Inside an unfinished cell, already-trained sweep units are reused from
  stage_*_sweeps (their val1 losses decide the winner); only the
  winner's weights are retrained once to re-materialize them -- every
  training unit is re-seeded identically, so the rerun is the same
  experiment.

Hard constraints (unchanged)
----------------------------
  * NEVER writes under the case10c results library
    (results/case10c_cluster_forecast/); outputs go to results/analysis/.
  * case10c and experiments trees are imported, never modified.
  * Single GPU, device fixed to cuda:0 (--device to override).

Time budget (RTX PRO 6000; per-unit costs from the v1 run logs)
---------------------------------------------------------------
  Stage P: 8 cells x 9 configs = 72 trainings x ~75 s
           (pretrain + 6 alpha finetunes)              ~= 90 min
  Stage F: 16 cells x 3 backbones = 48 trainings x ~50 s
           (pretrain + 5 alpha finetunes)              ~= 40 min
  Total ~= 2 h 15 min (+ inference / IO overhead).

Usage (any cwd)
---------------
  python clean_cqr_protocol_v2.py                    # both stages (pf)
  python clean_cqr_protocol_v2.py --stage p          # public 8 cells
  python clean_cqr_protocol_v2.py --stage f          # field 16 cells
  python clean_cqr_protocol_v2.py --config MIT_H16   # single P cell
  python clean_cqr_protocol_v2.py --config Room302_H96_s44  # single F cell
"""

import os
import sys
import glob
import math
import time
import argparse

import numpy as np
import torch

# ------------------------------------------------------------------
# Path setup: the repository root plus its experiments/ tree on sys.path,
# so common.* and the case10c pipeline modules import by name from any cwd.
# ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.utils import get_logger, save_json, load_json          # noqa: E402
from common.config import RESULTS_DIR, FIELD_DATA_DIR              # noqa: E402
from train_msra import set_seed                                    # noqa: E402
from evaluate_msra import (compute_point_metrics, compute_crps,    # noqa: E402
                           compute_coverage)
from run_feature_comparison import (DATASET_CONFIGS, HPARAMS,      # noqa: E402
                                    LOOKBACK, SEED, QUANTILE_LEVELS,
                                    load_and_engineer,
                                    build_dataset_for_feature_set,
                                    make_model)
from run_sspm_all_backbones import (pretrain_masked, train_mixup,  # noqa: E402
                                    PRETRAIN_EPOCHS, MASK_RATIO,
                                    ALPHA_GRID as SSPM_ALPHA_GRIDS)

logger = get_logger("clean_cqr_v2")

# ------------------------------------------------------------------
# Experiment definition
# ------------------------------------------------------------------
ALPHA = 0.2                     # 80% nominal interval == quantiles (0.1, 0.9)
NOMINAL = 1.0 - ALPHA

# pi in the PRE-TEST caliber: STL seasonal share on the first 85% of the
# series (train+val history only; pretest_pi_results.json, verified
# 2026-08-12). Fractions, not percent.
PI_PRETEST = {"MIT": 0.016187, "Helios": 0.056307,
              "Alibaba": 0.208343, "Inference": 0.822543}

# v2 enumeration spaces. Order copied from run_sspm_all_backbones.main
# (model_names / feat_set_names) so tie-breaks and tables follow the
# original SSPM search-space definition.
PUBLIC_DATASETS = ["MIT", "Helios", "Alibaba", "Inference"]
HORIZONS = [16, 96]
BACKBONES = ["DLinear", "PatchTST", "iTransformer"]
FEATURE_SETS = ["Calendar", "All", "Workload"]
STAGE_P_CONFIG_ORDER = [f"{b}-{f}" for b in BACKBONES for f in FEATURE_SETS]

# REFERENCE ONLY (never used for selection): the old protocol's choices.
# Stage P: the paper table's backbone-feature set per (dataset, H) --
# selected on the FULL validation set under the pre-audit protocol.
PAPER_TABLE_CONFIGS_V1 = {
    ("MIT",       16): "PatchTST-Workload",
    ("MIT",       96): "DLinear-Calendar",
    ("Helios",    16): "iTransformer-Calendar",
    ("Helios",    96): "iTransformer-Workload",
    ("Alibaba",   16): "PatchTST-Calendar",
    ("Alibaba",   96): "DLinear-All",
    ("Inference", 16): "iTransformer-All",
    ("Inference", 96): "iTransformer-All",
}

# Field: pre-test rule widths w(pi_pre) (pretest_pi_results.json) and the
# pre-test pi fractions they derive from (recorded in meta for the paper).
W_FIELD_PRETEST = {"Room302": 1.3398, "BuildingC": 1.3359}
FIELD_PI_PRETEST = {"Room302": 0.200191, "BuildingC": 0.205146}

# REFERENCE ONLY (never used for selection): seed -> backbone selected by
# the old full-val-set protocol (multiseed_aggregate.json field_selection).
STAGE_F_SEEDS = [43, 44, 45, 46]
STAGE_F_V1_BACKBONE_REF = {
    ("Room302", 16): {43: "iTransformer", 44: "iTransformer",
                      45: "iTransformer", 46: "iTransformer"},
    ("Room302", 96): {43: "iTransformer", 44: "PatchTST",
                      45: "iTransformer", 46: "iTransformer"},
    ("BuildingC", 16): {s: "PatchTST" for s in STAGE_F_SEEDS},
    ("BuildingC", 96): {s: "PatchTST" for s in STAGE_F_SEEDS},
}

FIELD_DATASETS = ["Room302", "BuildingC"]
FIELD_ALPHA_GRID = [0.01, 0.05, 0.1, 0.2, 0.4]   # mid-pi tier (pi ~ 0.207)
FIELD_TARGETS = {"Room302": "room_power_kw", "BuildingC": "building_power_kw"}
# Canonical field CSV names are field_room302_15min.csv /
# field_building_15min.csv under FIELD_DATA_DIR; the suffix glob keeps the
# tolerance for any prefix a local build script may add.
FIELD_CSV_SUFFIX = {"Room302": "room302_15min.csv",
                    "BuildingC": "building_15min.csv"}

METHODS = ["raw", "w_rule", "cqr_perh", "cqr_pooled"]

OUT_DIR = os.path.join(REPO_ROOT, "results", "analysis", "clean_cqr_v2")
NPZ_DIR = os.path.join(OUT_DIR, "npz")
OUT_PATH = os.path.join(OUT_DIR, "clean_cqr_v2_results.json")

# Outputs live under results/analysis/ (OUT_DIR above); the case10c results
# library under results/case10c_cluster_forecast/ is only ever read, never
# written by this script.

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Small numeric helpers
# ------------------------------------------------------------------
def ceil_int(x):
    """ceil with a tiny epsilon guard against float artifacts (0.8*(n+1))."""
    return int(math.ceil(x - 1e-9))


def _smape(pred, true):
    """Symmetric MAPE in %, guarded (copied from multiseed_rerun)."""
    denom = (np.abs(pred) + np.abs(true)) / 2.0
    mask = denom > 1e-8
    if not mask.any():
        return float("nan")
    return round(float(np.mean(np.abs(pred[mask] - true[mask])
                               / denom[mask]) * 100), 3)


def sort_triple(lo, mid, hi):
    """Per-point non-decreasing rearrangement of (lo, mid, hi).

    Returns (lo', mid', hi', nonmono_rate) with nonmono_rate =
    P(not (lo <= mid <= hi)) BEFORE the fix.
    """
    nonmono = float(np.mean(~((lo <= mid) & (mid <= hi))))
    stacked = np.sort(np.stack([lo, mid, hi], axis=-1), axis=-1)
    return stacked[..., 0], stacked[..., 1], stacked[..., 2], nonmono


# ------------------------------------------------------------------
# Four-segment split
# ------------------------------------------------------------------
def four_way_split(data, horizon):
    """Split the val windows into val1 / purge gap / val2 (time order).

    Windows from build_dataset_for_feature_set are stride-1 and
    time-ordered, so list position == temporal order. The purge gap
    (LOOKBACK + horizon - 1 windows, dropped from the val2 side) removes
    every window that shares a raw data row with any val1 window -- see
    module docstring, "Boundary handling".
    """
    n_val = len(data["val_y"])
    mid = n_val // 2
    gap = LOOKBACK + horizon - 1
    v2_start = min(mid + gap, n_val)
    n_val2 = n_val - v2_start
    nonoverlap_idx = np.arange(0, n_val2, horizon, dtype=np.int64)
    split = {
        "n_train": int(len(data["train_y"])),
        "n_val": int(n_val),
        "n_val1": int(mid),
        "purge_gap": int(v2_start - mid),
        "v2_start": int(v2_start),
        "n_val2": int(n_val2),
        "n_nonoverlap": int(len(nonoverlap_idx)),
        "n_test": int(len(data["test_y"])),
    }
    # ceil(0.8*(n+1)) <= n needs n_nonoverlap >= 4, i.e. n_val2 >= 3H + 1.
    assert n_val2 >= 3 * horizon + 1, (
        f"val2 too small after the purge gap ({n_val2} windows, H={horizon});"
        f" the CQR finite-sample rule needs n_nonoverlap >= 4 -- check the"
        f" split ratios")
    return (data["val_X"][:mid], data["val_y"][:mid],
            data["val_X"][v2_start:], data["val_y"][v2_start:],
            nonoverlap_idx, split)


def fit_data_view(data, val1_X, val1_y):
    """The dict handed to trainers: identical train, val1 as 'val'.

    train_mixup / train_one_model only read train_X/train_y/val_X/val_y, so
    early stopping and every selection loss can only ever see val1.
    """
    return {"train_X": data["train_X"], "train_y": data["train_y"],
            "val_X": val1_X, "val_y": val1_y}


# ------------------------------------------------------------------
# Inference + rearrangement (rearrangement applies to EVERY evaluation)
# ------------------------------------------------------------------
@torch.no_grad()
def predict_quantiles(model, X, device, batch_size=512):
    """Batched inference -> (N, H, 3) float32, normalized scale."""
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def infer_sorted(model, X, y, norm_params, device):
    """Forward pass -> de-normalized SORTED quantiles + truth.

    Returns (q_sorted (N,H,3), y_true (N,H), nonmono_rate) where
    nonmono_rate is the pre-fix fraction of (window, step) positions with a
    non-monotone raw (q10, q50, q90) triple (affine denorm preserves order,
    so the rate is measured after denorm, before sorting -- same definition
    as multiseed_rerun.evaluate_sorted).
    """
    t_mean = norm_params["target_mean"]
    t_std = norm_params["target_std"]
    q_raw = predict_quantiles(model, X, device) * t_std + t_mean
    y_true = y * t_std + t_mean
    nonmono = float((np.diff(q_raw, axis=2) < 0).any(axis=2).mean())
    return np.sort(q_raw, axis=2), y_true, nonmono


def point_metrics_sorted(q_sorted, y_true):
    """Point + probabilistic metrics on sorted quantiles (original units).

    Mirrors multiseed_rerun.evaluate_sorted's metric block: r2/rmse/mae/
    mape/smape/crps/coverage_80/interval_width.
    """
    med = q_sorted[:, :, 1]
    m = compute_point_metrics(med, y_true)
    m["smape"] = _smape(med, y_true)
    m["crps"] = compute_crps(q_sorted, y_true, QUANTILE_LEVELS)
    m.update(compute_coverage(q_sorted, y_true, QUANTILE_LEVELS))
    return m


# ------------------------------------------------------------------
# Conformal / interval machinery
# ------------------------------------------------------------------
def conformity_scores(q_sorted, y_true):
    """CQR score E = max(q10 - y, y - q90) on sorted triples. Shape (N, H)."""
    return np.maximum(q_sorted[..., 0] - y_true, y_true - q_sorted[..., 2])


def cqr_calibrate(q_val2, y_val2, nonoverlap_idx, horizon):
    """Per-lead-time + pooled conformal quantiles from the NON-OVERLAPPING
    val2 window subset.

    Qhat_h = ceil(0.8*(n_h+1))-th smallest of the n_h scores at lead h.
    Returns (qhat_h (H,), info dict).
    """
    E = conformity_scores(q_val2[nonoverlap_idx], y_val2[nonoverlap_idx])
    n_h = E.shape[0]
    k = ceil_int((1.0 - ALPHA) * (n_h + 1))
    info = {"n_h": int(n_h), "k_h": int(k),
            "guaranteed_level_h": round(k / (n_h + 1), 4)}
    if k > n_h:  # cannot certify 80% with so few windows
        qhat_h = np.full(horizon, np.inf)
        info["insufficient_calibration"] = True
        logger.warning(f"    CQR per-h: n_h={n_h} < needed {k}; Qhat_h=inf")
    else:
        qhat_h = np.sort(E, axis=0)[k - 1]              # (H,)
        info["insufficient_calibration"] = False
    info["qhat_h_min"] = round(float(np.min(qhat_h)), 4)
    info["qhat_h_median"] = round(float(np.median(qhat_h)), 4)
    info["qhat_h_max"] = round(float(np.max(qhat_h)), 4)

    pooled = np.sort(E.ravel())
    n_p = pooled.size
    k_p = ceil_int((1.0 - ALPHA) * (n_p + 1))
    qhat_pooled = float(pooled[k_p - 1]) if 0 < k_p <= n_p else float("inf")
    info.update({"n_pooled": int(n_p), "k_pooled": int(k_p),
                 "guaranteed_level_pooled": round(k_p / (n_p + 1), 6),
                 "qhat_pooled": round(qhat_pooled, 4)})
    return qhat_h, qhat_pooled, info


def build_intervals(method, q_test, w=None, qhat_h=None, qhat_pooled=None):
    """(lo, mid, hi) on the test segment for one interval construction."""
    q10, q50, q90 = q_test[..., 0], q_test[..., 1], q_test[..., 2]
    if method == "raw":
        return q10, q50, q90
    if method == "w_rule":
        return q50 + (q10 - q50) * w, q50, q50 + (q90 - q50) * w
    if method == "cqr_perh":
        return q10 - qhat_h[None, :], q50, q90 + qhat_h[None, :]
    if method == "cqr_pooled":
        return q10 - qhat_pooled, q50, q90 + qhat_pooled
    raise ValueError(f"unknown method: {method}")


def interval_metrics(lo, mid, hi, y, y_range):
    """Test metrics for one (already sorted) interval construction.

    Returns (metrics dict, picp_h array of length H). v2 adds the
    p25/p75 quantiles of picp_h to the dict and returns the full
    per-lead-time coverage array so callers can persist it in the npz.
    """
    inside = (y >= lo) & (y <= hi)
    picp = float(inside.mean())
    picp_h = inside.mean(axis=0)                       # per lead time (H,)
    width = hi - lo
    mpiw = float(width.mean())
    winkler = width \
        + (2.0 / ALPHA) * (lo - y) * (y < lo) \
        + (2.0 / ALPHA) * (y - hi) * (y > hi)
    crps = float(compute_crps(np.stack([lo, mid, hi], axis=-1), y,
                              QUANTILE_LEVELS))
    metrics = {
        "picp": round(picp, 4),
        "coverage_gap": round(picp - NOMINAL, 4),
        "picp_h_min": round(float(picp_h.min()), 4),
        "picp_h_p25": round(float(np.percentile(picp_h, 25)), 4),
        "picp_h_median": round(float(np.median(picp_h)), 4),
        "picp_h_p75": round(float(np.percentile(picp_h, 75)), 4),
        "picp_h_max": round(float(picp_h.max()), 4),
        "mpiw": round(mpiw, 3),
        "pinaw": round(mpiw / y_range, 4),
        "winkler": round(float(np.mean(winkler)), 3),
        "crps": round(crps, 4),
    }
    return metrics, picp_h


def evaluate_methods(q_val2, y_val2, nonoverlap_idx, q_test, y_test,
                     horizon, w_rule):
    """All four interval constructions on the test segment.

    Returns (methods dict, picp_h arrays per method, qhat_h, cqr_info,
    y_range). The finite-sample guaranteed level k/(n_h+1) is attached to
    the cqr method entries verbatim (per dataset x horizon; it is never
    aggregated across cells).
    """
    qhat_h, qhat_pooled, cqr_info = cqr_calibrate(
        q_val2, y_val2, nonoverlap_idx, horizon)
    y_range = float(y_test.max() - y_test.min())
    methods = {}
    picp_h_arrays = {}
    for method in METHODS:
        lo, mid, hi = build_intervals(method, q_test, w=w_rule,
                                      qhat_h=qhat_h, qhat_pooled=qhat_pooled)
        lo, mid, hi, nonmono = sort_triple(lo, mid, hi)
        m, picp_h = interval_metrics(lo, mid, hi, y_test, y_range)
        m["nonmono_rate_prefix"] = round(nonmono, 6)
        if method == "w_rule":
            m["w"] = round(float(w_rule), 4)
        elif method == "cqr_perh":
            m["guaranteed_level_h"] = cqr_info["guaranteed_level_h"]
        elif method == "cqr_pooled":
            m["guaranteed_level_pooled"] = cqr_info["guaranteed_level_pooled"]
        methods[method] = m
        picp_h_arrays[method] = picp_h
        logger.info(f"    {method:<11} PICP={m['picp']:.4f} "
                    f"(h: {m['picp_h_min']:.3f}/{m['picp_h_median']:.3f}/"
                    f"{m['picp_h_max']:.3f}) MPIW={m['mpiw']:>10.3f} "
                    f"Winkler={m['winkler']:>10.3f} CRPS={m['crps']:.4f}")
    return methods, picp_h_arrays, qhat_h, cqr_info, y_range


# ------------------------------------------------------------------
# SSPM training under the clean protocol (one training unit)
# ------------------------------------------------------------------
def sspm_train_clean(seed, model_name, data, val1_X, val1_y, horizon,
                     alphas, device, field_style=False):
    """Masked pretrain + alpha search; early stop / selection on val1 ONLY.

    field_style=False mirrors run_sspm_all_backbones.run_one call sites
    (fresh model per alpha); field_style=True mirrors multiseed_rerun
    stage C (one model object, state_dict reload per alpha). Every unit
    re-seeds at entry and per alpha, so results are independent of sweep
    order and a unit rerun reproduces the same experiment. Returns
    (best_state_dict, record_fields).
    """
    t_all = time.time()
    nf = data["train_X"].shape[2]
    data_fit = fit_data_view(data, val1_X, val1_y)

    set_seed(seed)
    model = make_model(model_name, nf, horizon).to(device)
    t0 = time.time()
    model = pretrain_masked(model, data["train_X"], HPARAMS, device,
                            epochs=PRETRAIN_EPOCHS, mask_ratio=MASK_RATIO)
    pre_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    t_pre = time.time() - t0
    if not field_style:
        del model
        torch.cuda.empty_cache()

    best_val = float("inf")
    best_alpha = None
    best_sd = None
    alpha_val1_losses = {}
    for alpha in alphas:
        if field_style:                       # multiseed stage C call sites
            model.load_state_dict(pre_sd)
            set_seed(seed)
            model, val_loss = train_mixup(model, data_fit, HPARAMS, alpha,
                                          device)
            m = model
        else:                                 # run_sspm_all_backbones sites
            set_seed(seed)
            m = make_model(model_name, nf, horizon).to(device)
            m.load_state_dict(pre_sd)
            m, val_loss = train_mixup(m, data_fit, HPARAMS, alpha, device)
        alpha_val1_losses[str(alpha)] = round(float(val_loss), 6)
        if val_loss < best_val:
            best_val = float(val_loss)
            best_alpha = alpha
            best_sd = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        if not field_style:
            del m
            torch.cuda.empty_cache()
    if field_style:
        del model
        torch.cuda.empty_cache()

    fields = {"alpha": best_alpha,
              "val1_loss": round(best_val, 6),
              "alpha_val1_losses": alpha_val1_losses,
              "pretrain_time_s": round(t_pre, 1),
              "train_time_s": round(time.time() - t_all, 1)}
    return best_sd, fields


# ------------------------------------------------------------------
# Evaluation of the SELECTED configuration (val2 calibration + test)
# ------------------------------------------------------------------
def eval_selected_config(key, seed, model_name, data, horizon, best_sd,
                         w_rule, npz_path, device):
    """val2/test inference + all interval constructions for the winner.

    Runs strictly AFTER every selection dimension is frozen. Saves the
    winner's npz (including the full per-lead-time test picp_h array of
    every method) and returns the core record dict.
    """
    val1_X, val1_y, val2_X, val2_y, nonoverlap_idx, split = \
        four_way_split(data, horizon)
    logger.info(f"  [{key}] split: train={split['n_train']} "
                f"val1={split['n_val1']} gap={split['purge_gap']} "
                f"val2={split['n_val2']} (nonoverlap={split['n_nonoverlap']}) "
                f"test={split['n_test']}")

    nf = data["train_X"].shape[2]
    model = make_model(model_name, nf, horizon).to(device)
    model.load_state_dict(best_sd)
    q_val2, y_val2, nm_val2 = infer_sorted(model, val2_X, val2_y,
                                           data["norm_params"], device)
    q_test, y_test, nm_test = infer_sorted(model, data["test_X"],
                                           data["test_y"],
                                           data["norm_params"], device)
    del model
    torch.cuda.empty_cache()

    methods, picp_h_arrays, qhat_h, cqr_info, y_range = evaluate_methods(
        q_val2, y_val2, nonoverlap_idx, q_test, y_test, horizon, w_rule)

    record = {
        "seed": seed,
        "backbone": model_name,
        "split": split,
        "val2_nonmono_rate": round(nm_val2, 6),
        "test_nonmono_rate": round(nm_test, 6),
        "test_target_range": round(y_range, 3),
        "test_point": point_metrics_sorted(q_test, y_test),
        "cqr": cqr_info,
        "methods": methods,
    }

    np.savez_compressed(
        npz_path,
        q10_val2=q_val2[:, :, 0].astype(np.float32),
        q50_val2=q_val2[:, :, 1].astype(np.float32),
        q90_val2=q_val2[:, :, 2].astype(np.float32),
        true_val2=y_val2.astype(np.float32),
        q10_test=q_test[:, :, 0].astype(np.float32),
        q50_test=q_test[:, :, 1].astype(np.float32),
        q90_test=q_test[:, :, 2].astype(np.float32),
        true_test=y_test.astype(np.float32),
        val2_nonoverlap_idx=nonoverlap_idx.astype(np.int64),
        cqr_qhat_perh=np.asarray(qhat_h, dtype=np.float64),
        n_val_windows=np.int64(split["n_val"]),
        val1_end=np.int64(split["n_val1"]),
        purge_gap=np.int64(split["purge_gap"]),
        val2_start=np.int64(split["v2_start"]),
        **{f"picp_h_{m}": np.asarray(picp_h_arrays[m], dtype=np.float64)
           for m in METHODS})
    return record


# ------------------------------------------------------------------
# Stage-P diagnostics: w*_cal on val2, OLS, leave-one-out on test
# (logic unchanged from v1; rebuilt from the v2 winners' npz)
# ------------------------------------------------------------------
def required_widening(q_sorted, y_true, eps=1e-8):
    """Per-point minimal w s.t. y in [q50+(q10-q50)w, q50+(q90-q50)w].

    Coverage is monotone in w, so the minimal w reaching 80% coverage is the
    ceil(0.8*n)-th smallest per-point requirement. Degenerate half-widths
    (<= eps) with y beyond them require w = inf.
    """
    q10, q50, q90 = q_sorted[..., 0], q_sorted[..., 1], q_sorted[..., 2]
    up = y_true >= q50
    hw_hi = q90 - q50
    hw_lo = q50 - q10
    with np.errstate(divide="ignore", invalid="ignore"):
        w_up = np.where(hw_hi > eps, (y_true - q50) / np.maximum(hw_hi, eps),
                        np.where(y_true - q50 <= eps, 0.0, np.inf))
        w_dn = np.where(hw_lo > eps, (q50 - y_true) / np.maximum(hw_lo, eps),
                        np.where(q50 - y_true <= eps, 0.0, np.inf))
    return np.where(up, w_up, w_dn).ravel()


def w_star_from_scores(w_req):
    """Minimal w with empirical coverage >= 0.8: ceil(0.8*n)-th smallest."""
    w = np.sort(np.asarray(w_req, dtype=np.float64))
    n = w.size
    if n == 0:
        return float("nan"), 0
    k = min(n, ceil_int((1.0 - ALPHA) * n))
    return float(w[k - 1]), n


def widened_metrics(npz, w):
    """PICP/MPIW/Winkler on a config's test arrays for rule width w."""
    q10, q50, q90 = npz["q10_test"], npz["q50_test"], npz["q90_test"]
    y = npz["true_test"]
    lo = q50 + (q10 - q50) * w
    hi = q50 + (q90 - q50) * w
    lo, mid, hi, _ = sort_triple(lo, q50, hi)
    y_range = float(y.max() - y.min())
    m, _ = interval_metrics(lo, mid, hi, y, y_range)
    return {k: m[k] for k in ("picp", "mpiw", "winkler")}


def p_npz_path(ds, horizon):
    return os.path.join(NPZ_DIR, f"P_{ds}_H{horizon}.npz")


def build_stage_p_diagnostics(stage_p):
    """w*_cal per dataset (calibration segment ONLY) + OLS + leave-one-out.

    Rebuilt from the stored npz on every flush; returns a pending stub until
    all 8 cells are finished. In v2 the npz belong to the val1-selected
    winners instead of the paper-table configs; the diagnostic logic is
    identical to v1.
    """
    cells = [(ds, h) for ds in PUBLIC_DATASETS for h in HORIZONS]
    datasets = sorted(PUBLIC_DATASETS)
    missing = [f"{ds}_H{h}" for ds, h in cells
               if f"{ds}_H{h}" not in stage_p
               or not os.path.exists(p_npz_path(ds, h))]
    if missing:
        return {"status": f"pending ({len(cells) - len(missing)}"
                          f"/{len(cells)})",
                "missing": missing}

    caliber = ("w*_cal solved on the val2 (calibration) segment only: "
               "per-point minimal widening pooled over ALL stride-1 val2 "
               "windows x lead times; w* = ceil(0.8*n)-th order statistic "
               "(empirical 80% coverage, diagnostic fit -- no finite-sample "
               "+1 correction). 'merged' pools both horizons of a dataset "
               "(primary, used for OLS/LOO); per-horizon and non-overlap-"
               "subset values recorded for robustness. v2: the underlying "
               "npz are the val1-selected winners, not the paper-table "
               "configs.")
    diag = {"status": "complete", "caliber": caliber, "w_star_cal": {},
            "ols": {}, "loo": {}}

    for ds in datasets:
        per = {}
        for h in HORIZONS:
            with np.load(p_npz_path(ds, h)) as z:
                q = np.stack([z["q10_val2"], z["q50_val2"], z["q90_val2"]],
                             axis=-1)
                y = z["true_val2"]
                idx = z["val2_nonoverlap_idx"]
                per[str(h)] = required_widening(q, y)
                per[f"no{h}"] = required_widening(q[idx], y[idx])
        w16, n16 = w_star_from_scores(per["16"])
        w96, n96 = w_star_from_scores(per["96"])
        wm, nm = w_star_from_scores(np.concatenate([per["16"], per["96"]]))
        wno, nno = w_star_from_scores(
            np.concatenate([per["no16"], per["no96"]]))
        diag["w_star_cal"][ds] = {
            "H16": round(w16, 4), "H96": round(w96, 4),
            "merged": round(wm, 4), "n_merged_points": int(nm),
            "merged_nonoverlap": round(wno, 4),
            "n_nonoverlap_points": int(nno),
        }

    # OLS on the 4 points (pi_pre, w*_cal merged).
    pis = np.array([PI_PRETEST[ds] for ds in datasets])
    ws = np.array([diag["w_star_cal"][ds]["merged"] for ds in datasets])
    slope, intercept = np.polyfit(pis, ws, 1)
    corr = float(np.corrcoef(pis, ws)[0, 1])
    diag["ols"] = {
        "points": {ds: [round(float(PI_PRETEST[ds]), 6),
                        diag["w_star_cal"][ds]["merged"]] for ds in datasets},
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "pearson_r": round(corr, 4),
        "paper_rule": "w(pi) = max(1, 1.5 - 0.8*pi)  (intercept 1.5, "
                      "slope -0.8, floored at 1)",
    }

    # Leave-one-out: fit on 3 datasets' calibration info, evaluate coverage
    # on the held-out dataset's TEST segment (test used ONLY to evaluate).
    for held in datasets:
        rest = [d for d in datasets if d != held]
        s, b = np.polyfit(np.array([PI_PRETEST[d] for d in rest]),
                          np.array([diag["w_star_cal"][d]["merged"]
                                    for d in rest]), 1)
        w_raw = float(b + s * PI_PRETEST[held])
        w_floor = max(1.0, w_raw)
        entry = {"fit_slope": round(float(s), 4),
                 "fit_intercept": round(float(b), 4),
                 "w_hat_raw": round(w_raw, 4),
                 "w_hat_floored": round(w_floor, 4),
                 "test_raw_w": {}, "test_floored_w": {}}
        for h in HORIZONS:
            with np.load(p_npz_path(held, h)) as z:
                entry["test_raw_w"][f"H{h}"] = widened_metrics(z, w_raw)
                entry["test_floored_w"][f"H{h}"] = widened_metrics(z, w_floor)
        diag["loo"][held] = entry
    return diag


# ------------------------------------------------------------------
# Stage-F aggregation (4-seed mean +/- std per scenario x horizon)
# ------------------------------------------------------------------
def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return [round(m, 4), round(s, 4)]


def build_stage_f_aggregate(stage_f):
    def _uniq_or_per_seed(d):
        """Scalar if identical across seeds (the split is seed-invariant),
        else the per-seed dict verbatim -- never an interval."""
        vals = sorted(set(d.values()))
        return vals[0] if len(vals) == 1 else d

    agg = {}
    for ds in FIELD_DATASETS:
        for h in HORIZONS:
            recs = {s: stage_f.get(f"{ds}_H{h}_s{s}") for s in STAGE_F_SEEDS}
            have = {s: r for s, r in recs.items() if r}
            if not have:
                continue
            entry = {
                "n_seeds": len(have),
                "seeds": sorted(have),
                "backbones": {str(s): r["backbone"]
                              for s, r in have.items()},
                "matches_v1_ref": {str(s): r["matches_v1_ref"]
                                   for s, r in have.items()},
                "alphas": {str(s): r["alpha"] for s, r in have.items()},
                "cqr_guarantee": {
                    "n_h": _uniq_or_per_seed(
                        {str(s): r["cqr"]["n_h"] for s, r in have.items()}),
                    "k_h": _uniq_or_per_seed(
                        {str(s): r["cqr"]["k_h"] for s, r in have.items()}),
                    "guaranteed_level_h": _uniq_or_per_seed(
                        {str(s): r["cqr"]["guaranteed_level_h"]
                         for s, r in have.items()}),
                },
                "test_r2": mean_std([r["test_point"]["r2"]
                                     for r in have.values()]),
                "test_crps": mean_std([r["test_point"]["crps"]
                                       for r in have.values()]),
                "methods": {},
            }
            for method in METHODS:
                entry["methods"][method] = {
                    metric: mean_std([r["methods"][method][metric]
                                      for r in have.values()])
                    for metric in ("picp", "mpiw", "pinaw", "winkler", "crps",
                                   "picp_h_min", "picp_h_p25",
                                   "picp_h_median", "picp_h_p75",
                                   "picp_h_max")}
            agg.setdefault(ds, {})[f"H{h}"] = entry
    return agg


# ------------------------------------------------------------------
# Persistence / resume
# ------------------------------------------------------------------
def load_results():
    """Load previous partial results so reruns merge instead of clobber."""
    empty = {"stage_p": {}, "stage_f": {},
             "stage_p_sweeps": {}, "stage_f_sweeps": {}}
    if os.path.exists(OUT_PATH):
        try:
            prev = load_json(OUT_PATH)
            out = {k: prev.get(k, {}) for k in empty}
            logger.info(f"[resume] merging into existing {OUT_PATH} "
                        f"(P: {sorted(out['stage_p'])}, "
                        f"F: {len(out['stage_f'])} cells; partial sweeps "
                        f"P={sorted(out['stage_p_sweeps'])}, "
                        f"F={sorted(out['stage_f_sweeps'])})")
            return out
        except Exception as e:
            logger.warning(f"[resume] cannot parse {OUT_PATH} ({e}); "
                           f"starting fresh")
    return empty


def build_meta(t_start):
    return {
        "script": "clean_cqr_protocol_v2.py",
        "device": DEVICE,
        "alpha": ALPHA,
        "nominal_coverage": NOMINAL,
        "quantile_levels": QUANTILE_LEVELS,
        "v2_vs_v1": (
            "v1 (clean_cqr_protocol.py, results/analysis/clean_cqr/) fixed "
            "the calibration-segment leakage but still inherited the "
            "configuration identity from the old FULL-validation-set "
            "protocol: stage P used the paper-table backbone x feature-set "
            "pairs, stage F used multiseed_aggregate's per-seed backbones. "
            "That is a residual selection contamination -- which config got "
            "calibrated was itself a decision fitted on data overlapping "
            "val2. v2 re-selects EVERYTHING inside val1: stage P jointly "
            "selects backbone x feature set x alpha among 9 configs per "
            "(dataset, H); stage F selects backbone x alpha among 3 "
            "backbones per (seed, scenario, H). The old choices are kept "
            "only as reference constants for the matches_paper_v1 / "
            "matches_v1_ref flags and influence no decision. Outputs are "
            "isolated from v1 (clean_cqr_v2/)."),
        "protocol_four_segments": (
            "train 70% | val1 = first floor(n_val/2) val windows (early "
            "stopping + backbone/feature-set/alpha selection ONLY) | purge "
            "gap = next LOOKBACK+H-1 val windows DISCARDED (no raw data row "
            "shared between the selection segment and the calibration "
            "segment) | val2 = remaining val windows (calibration ONLY, "
            "forward inference; no fitting decision touches it) | test 15% "
            "(evaluated exactly once, for the selected config only)"),
        "purge_gap_rationale": (
            "window i covers raw rows [i, i+L+H); dropping the first "
            "L+H-1 windows after the midpoint removes every cross-boundary "
            "window whose input or target rows were visible to the val1 "
            "selection loss, so the fitted model + selected configuration "
            "are independent of val2 at the raw-data-row level"),
        "cqr_calibration": (
            "non-overlapping val2 window subset (stride = H over the "
            "window list -> disjoint target segments); conformity score "
            "E = max(q10 - y, y - q90) on sorted quantiles; per lead time "
            "h: Qhat_h = ceil(0.8*(n_h+1))-th smallest of the n_h scores; "
            "test interval [q10_h - Qhat_h, q90_h + Qhat_h]; pooled "
            "variant (single Qhat over the same subset's (window, step) "
            "scores, n = n_h*H) reported as a contrast"),
        "finite_sample_formula": (
            "Qhat = k-th order statistic with k = ceil((1-alpha)(n+1)); "
            "guaranteed marginal coverage k/(n+1) >= 1-alpha under "
            "exchangeability of calibration and test scores (time-series "
            "caveat applies; non-overlapping windows remove the mechanical "
            "dependence the old stride-1 pooling induced)"),
        "picp_h_outputs": (
            "v2: for every winning config the FULL H-length per-lead-time "
            "TEST coverage array of each interval method is stored in its "
            "npz (picp_h_raw / picp_h_w_rule / picp_h_cqr_perh / "
            "picp_h_cqr_pooled); the JSON stores min/p25/median/p75/max of "
            "picp_h per method. The finite-sample guaranteed level "
            "k_h/(n_h+1) is computed per dataset x horizon from its own "
            "n_h and stored as-is per record (and as a per-cell scalar in "
            "stage_f_aggregate) -- deliberately never merged across "
            "datasets into a single range."),
        "quantile_rearrangement": (
            "every quantile triple (model outputs and constructed "
            "intervals) is sorted non-decreasingly per point before any "
            "metric; pre-fix non-monotonicity rates recorded"),
        "crps_definition": ("evaluate_msra.compute_crps: 2 * mean pinball "
                            "over levels [0.1, 0.5, 0.9]"),
        "winkler_definition": ("mean of (hi-lo) + (2/alpha)(lo-y)1[y<lo] + "
                               "(2/alpha)(y-hi)1[y>hi], alpha=0.2"),
        "pinaw_normalizer": "test-segment target range (max - min)",
        "pi_pretest_caliber": (
            "STL seasonal variance share on the first 85% of each series "
            "(train+val history, computed before any test data is seen; "
            "pretest_pi_results.json)"),
        "pi_pretest": PI_PRETEST,
        "field_pi_pretest": FIELD_PI_PRETEST,
        "w_rule": "w(pi_pre) = max(1, 1.5 - 0.8*pi_pre)",
        "w_field_pretest": W_FIELD_PRETEST,
        "stage_p": {
            "seed": SEED,
            "grid": [f"{ds}_H{h}" for ds in PUBLIC_DATASETS
                     for h in HORIZONS],
            "enumerated_backbones": BACKBONES,
            "enumerated_feature_sets": FEATURE_SETS,
            "n_configs_per_cell": len(STAGE_P_CONFIG_ORDER),
            "selection": ("joint argmin of val1 loss over backbone x "
                          "feature set x alpha; ties -> first in "
                          "enumeration order (BACKBONES x FEATURE_SETS)"),
            "paper_table_v1_reference": {
                f"{ds}_H{h}": bb
                for (ds, h), bb in PAPER_TABLE_CONFIGS_V1.items()},
            "alpha_grids": SSPM_ALPHA_GRIDS,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "mask_ratio": MASK_RATIO,
        },
        "stage_f": {
            "seeds": STAGE_F_SEEDS,
            "grid": [f"{ds}_H{h}" for ds in FIELD_DATASETS
                     for h in HORIZONS],
            "enumerated_backbones": BACKBONES,
            "selection": ("argmin of val1 loss over backbone x alpha; "
                          "ties -> first in enumeration order (BACKBONES); "
                          "multiseed_aggregate.json is NOT read"),
            "v1_reference_backbone_by_seed": {
                f"{ds}_H{h}": {str(s): m for s, m in bys.items()}
                for (ds, h), bys in STAGE_F_V1_BACKBONE_REF.items()},
            "alpha_grid": FIELD_ALPHA_GRID,
            "feature_set": ("Calendar (fixed; the field data has no "
                            "workload telemetry)"),
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "mask_ratio": MASK_RATIO,
        },
        "hparams": HPARAMS,
        "lookback": LOOKBACK,
        "resume": (
            "a cell is skipped iff its JSON record AND its npz exist; "
            "finished training units of an unfinished cell are reused from "
            "stage_p_sweeps / stage_f_sweeps (val1 losses decide the "
            "winner) and only the winner is retrained once to "
            "re-materialize its weights (every unit is re-seeded, so the "
            "rerun is the same experiment)"),
        "note": ("clean-protocol retrain: models here are NOT the results-"
                 "library checkpoints (those early-stopped on the full val "
                 "set); nothing under results/ is read or written"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_elapsed_s": round(time.time() - t_start, 1),
    }


def flush(results, t_start):
    out = {
        "stage_p": results["stage_p"],
        "stage_p_sweeps": results["stage_p_sweeps"],
        "stage_p_diagnostics": build_stage_p_diagnostics(results["stage_p"]),
        "stage_f": results["stage_f"],
        "stage_f_sweeps": results["stage_f_sweeps"],
        "stage_f_aggregate": build_stage_f_aggregate(results["stage_f"]),
    }
    meta = build_meta(t_start)
    meta["nonoverlap_counts"] = {
        **{k: v["split"]["n_nonoverlap"]
           for k, v in results["stage_p"].items()},
        **{k: v["split"]["n_nonoverlap"]
           for k, v in results["stage_f"].items()},
    }
    out["meta"] = meta
    save_json(out, OUT_PATH)


# ------------------------------------------------------------------
# Stage P: public 8 cells x 9 configs, winner by val1 loss
# ------------------------------------------------------------------
def run_stage_p(results, t_start, only_key=None, device=DEVICE):
    ds_cache = {}
    todo = [(ds, h) for ds in PUBLIC_DATASETS for h in HORIZONS
            if only_key in (None, f"{ds}_H{h}")]
    for i, (ds, horizon) in enumerate(todo, 1):
        key = f"{ds}_H{horizon}"
        npz_path = p_npz_path(ds, horizon)
        if key in results["stage_p"] and os.path.exists(npz_path):
            logger.info(f"[P {i}/{len(todo)}] {key}: complete, skip")
            continue
        pi = PI_PRETEST[ds]
        w_rule = max(1.0, 1.5 - 0.8 * pi)
        logger.info(f"\n{'=' * 60}\n[P {i}/{len(todo)}] {key}  sweep "
                    f"{len(STAGE_P_CONFIG_ORDER)} configs (3 backbones x "
                    f"3 feature sets)  pi_pre={pi}  w={w_rule:.4f}"
                    f"\n{'=' * 60}")

        if ds not in ds_cache:
            ds_cache[ds] = load_and_engineer(DATASET_CONFIGS[ds])
        df, feature_sets, target = ds_cache[ds]
        missing_fs = [fs for fs in FEATURE_SETS if fs not in feature_sets]
        assert not missing_fs, (
            f"{ds}: feature sets missing {missing_fs}; build_feature_sets "
            f"should always return Calendar/All/Workload")

        cell_t0 = time.time()
        cache = results["stage_p_sweeps"].setdefault(key, {})
        best = None
        for backbone in BACKBONES:
            for fs_name in FEATURE_SETS:
                cfg = f"{backbone}-{fs_name}"
                if cfg in cache:
                    fields, sd = cache[cfg], None
                    logger.info(f"  [{key}] {cfg}: cached "
                                f"val1_loss={fields['val1_loss']} "
                                f"alpha={fields['alpha']} (resume)")
                else:
                    set_seed(SEED)
                    data = build_dataset_for_feature_set(
                        df, feature_sets[fs_name], horizon, target)
                    val1_X, val1_y = four_way_split(data, horizon)[:2]
                    sd, fields = sspm_train_clean(
                        SEED, backbone, data, val1_X, val1_y, horizon,
                        SSPM_ALPHA_GRIDS[ds], device, field_style=False)
                    del data
                    torch.cuda.empty_cache()
                    cache[cfg] = fields
                    flush(results, t_start)
                    logger.info(f"  [{key}] {cfg}: "
                                f"val1_loss={fields['val1_loss']} "
                                f"alpha={fields['alpha']} "
                                f"({fields['train_time_s']}s)")
                # strict < : ties keep the first config in enumeration order
                if best is None or fields["val1_loss"] < best["val1_loss"]:
                    best = {"config": cfg, "backbone": backbone,
                            "feature_set": fs_name, "alpha": fields["alpha"],
                            "val1_loss": fields["val1_loss"], "sd": sd}
                else:
                    del sd

        # Winner frozen (backbone x feature set x alpha). Rebuild its data
        # (deterministic: seeded identically to the sweep-time build).
        set_seed(SEED)
        data = build_dataset_for_feature_set(
            df, feature_sets[best["feature_set"]], horizon, target)
        if best["sd"] is None:
            logger.info(f"  [{key}] winner {best['config']} came from the "
                        f"resume cache; retraining once to re-materialize "
                        f"its weights")
            val1_X, val1_y = four_way_split(data, horizon)[:2]
            best["sd"], refit = sspm_train_clean(
                SEED, best["backbone"], data, val1_X, val1_y, horizon,
                SSPM_ALPHA_GRIDS[ds], device, field_style=False)
            if (refit["alpha"] != best["alpha"]
                    or abs(refit["val1_loss"] - best["val1_loss"]) > 1e-5):
                logger.warning(
                    f"  [{key}] re-materialized winner drifted: cached "
                    f"(alpha={best['alpha']}, val1={best['val1_loss']}) vs "
                    f"rerun (alpha={refit['alpha']}, "
                    f"val1={refit['val1_loss']}); using the rerun weights, "
                    f"keeping the cached selection table")

        paper_cfg = PAPER_TABLE_CONFIGS_V1[(ds, horizon)]
        logger.info(f"  [{key}] SELECTED {best['config']} "
                    f"alpha={best['alpha']} val1_loss={best['val1_loss']} "
                    f"(paper-v1: {paper_cfg}, "
                    f"match={best['config'] == paper_cfg})")

        record = eval_selected_config(key, SEED, best["backbone"], data,
                                      horizon, best["sd"], w_rule, npz_path,
                                      device)
        record.update({
            "selected_config": best["config"],
            "backbone": best["backbone"],
            "feature_set": best["feature_set"],
            "alpha": best["alpha"],
            "val1_loss": best["val1_loss"],
            "config_val1_losses": {c: cache[c]["val1_loss"]
                                   for c in STAGE_P_CONFIG_ORDER},
            "config_search": {c: cache[c] for c in STAGE_P_CONFIG_ORDER},
            "paper_config_v1": paper_cfg,
            "matches_paper_v1": best["config"] == paper_cfg,
            "pi_pretest": pi,
            "w_rule": round(w_rule, 4),
            "time_s": round(time.time() - cell_t0, 1),
        })
        results["stage_p"][key] = record
        results["stage_p_sweeps"].pop(key, None)   # folded into the record
        flush(results, t_start)
        logger.info(f"  [{key}] {best['config']} alpha={record['alpha']} "
                    f"R2={record['test_point']['r2']} "
                    f"({record['time_s']}s) -> saved")
        del data
        torch.cuda.empty_cache()


# ------------------------------------------------------------------
# Stage F: field 16 cells x 3 backbones, winner by val1 loss
# ------------------------------------------------------------------
def field_data_available():
    """True iff every field scenario CSV resolves under FIELD_DATA_DIR."""
    return all(
        glob.glob(os.path.join(FIELD_DATA_DIR, "*" + suffix))
        for suffix in FIELD_CSV_SUFFIX.values())


def resolve_field_datasets():
    """Locate the field CSVs under FIELD_DATA_DIR by suffix glob (canonical
    names field_room302_15min.csv / field_building_15min.csv; any prefix is
    tolerated, the prefix is intentionally not hardcoded)."""
    cfgs = {}
    for ds_name, suffix in FIELD_CSV_SUFFIX.items():
        hits = sorted(glob.glob(os.path.join(FIELD_DATA_DIR, "*" + suffix)))
        if not hits:
            raise SystemExit(
                f"Field CSV *{suffix} not found in {FIELD_DATA_DIR}. "
                f"The proprietary field data is not shipped with the repo; "
                f"see field/README.md.")
        if len(hits) > 1:
            logger.warning(f"Multiple CSVs match *{suffix}: {hits}; "
                           f"using {hits[0]}")
        cfgs[ds_name] = {"path": hits[0],
                         "target": FIELD_TARGETS[ds_name],
                         "workload_cols": []}
    return cfgs


def run_stage_f(results, t_start, only_key=None, device=DEVICE):
    field_cfgs = resolve_field_datasets()
    ds_cache = {}
    todo = [(ds, h, s) for ds in FIELD_DATASETS for h in HORIZONS
            for s in STAGE_F_SEEDS
            if only_key in (None, f"{ds}_H{h}_s{s}")]
    for i, (ds, horizon, seed) in enumerate(todo, 1):
        key = f"{ds}_H{horizon}_s{seed}"
        npz_path = os.path.join(NPZ_DIR,
                                f"F_{ds}_H{horizon}_s{seed}.npz")
        if key in results["stage_f"] and os.path.exists(npz_path):
            logger.info(f"[F {i}/{len(todo)}] {key}: complete, skip")
            continue
        w_rule = W_FIELD_PRETEST[ds]
        logger.info(f"\n{'=' * 60}\n[F {i}/{len(todo)}] {key}  sweep "
                    f"{len(BACKBONES)} backbones (Calendar)  w={w_rule}"
                    f"\n{'=' * 60}")

        if ds not in ds_cache:
            ds_cache[ds] = load_and_engineer(field_cfgs[ds])
        df, fsets, target = ds_cache[ds]
        fcols = fsets["Calendar"]
        set_seed(seed)                 # same call site as multiseed stage C
        data = build_dataset_for_feature_set(df, fcols, horizon, target)
        # Calendar features are backbone-invariant: one build serves the
        # whole sweep (each unit re-seeds itself inside sspm_train_clean).
        val1_X, val1_y = four_way_split(data, horizon)[:2]

        cell_t0 = time.time()
        cache = results["stage_f_sweeps"].setdefault(key, {})
        best = None
        for backbone in BACKBONES:
            if backbone in cache:
                fields, sd = cache[backbone], None
                logger.info(f"  [{key}] {backbone}: cached "
                            f"val1_loss={fields['val1_loss']} "
                            f"alpha={fields['alpha']} (resume)")
            else:
                sd, fields = sspm_train_clean(
                    seed, backbone, data, val1_X, val1_y, horizon,
                    FIELD_ALPHA_GRID, device, field_style=True)
                cache[backbone] = fields
                flush(results, t_start)
                logger.info(f"  [{key}] {backbone}: "
                            f"val1_loss={fields['val1_loss']} "
                            f"alpha={fields['alpha']} "
                            f"({fields['train_time_s']}s)")
            # strict < : ties keep the first backbone in enumeration order
            if best is None or fields["val1_loss"] < best["val1_loss"]:
                best = {"backbone": backbone, "alpha": fields["alpha"],
                        "val1_loss": fields["val1_loss"], "sd": sd}
            else:
                del sd

        if best["sd"] is None:
            logger.info(f"  [{key}] winner {best['backbone']} came from the "
                        f"resume cache; retraining once to re-materialize "
                        f"its weights")
            best["sd"], refit = sspm_train_clean(
                seed, best["backbone"], data, val1_X, val1_y, horizon,
                FIELD_ALPHA_GRID, device, field_style=True)
            if (refit["alpha"] != best["alpha"]
                    or abs(refit["val1_loss"] - best["val1_loss"]) > 1e-5):
                logger.warning(
                    f"  [{key}] re-materialized winner drifted: cached "
                    f"(alpha={best['alpha']}, val1={best['val1_loss']}) vs "
                    f"rerun (alpha={refit['alpha']}, "
                    f"val1={refit['val1_loss']}); using the rerun weights, "
                    f"keeping the cached selection table")

        v1_bb = STAGE_F_V1_BACKBONE_REF[(ds, horizon)][seed]
        logger.info(f"  [{key}] SELECTED {best['backbone']} "
                    f"alpha={best['alpha']} val1_loss={best['val1_loss']} "
                    f"(v1-ref: {v1_bb}, match={best['backbone'] == v1_bb})")

        record = eval_selected_config(key, seed, best["backbone"], data,
                                      horizon, best["sd"], w_rule, npz_path,
                                      device)
        record.update({
            "backbone": best["backbone"],
            "feature_set": "Calendar",
            "alpha": best["alpha"],
            "val1_loss": best["val1_loss"],
            "backbone_val1_losses": {b: cache[b]["val1_loss"]
                                     for b in BACKBONES},
            "backbone_search": {b: cache[b] for b in BACKBONES},
            "v1_ref_backbone": v1_bb,
            "matches_v1_ref": best["backbone"] == v1_bb,
            "pi_pretest": FIELD_PI_PRETEST[ds],
            "w_rule": w_rule,
            "time_s": round(time.time() - cell_t0, 1),
        })
        results["stage_f"][key] = record
        results["stage_f_sweeps"].pop(key, None)   # folded into the record
        flush(results, t_start)
        logger.info(f"  [{key}] {best['backbone']} alpha={record['alpha']} "
                    f"R2={record['test_point']['r2']} "
                    f"({record['time_s']}s) -> saved")
        del data
        torch.cuda.empty_cache()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def all_config_keys():
    p_keys = [f"{ds}_H{h}" for ds in PUBLIC_DATASETS for h in HORIZONS]
    f_keys = [f"{ds}_H{h}_s{s}" for ds in FIELD_DATASETS for h in HORIZONS
              for s in STAGE_F_SEEDS]
    return p_keys, f_keys


def parse_args():
    p_keys, f_keys = all_config_keys()
    ap = argparse.ArgumentParser(
        description="Clean-protocol CQR v2: in-protocol backbone x "
                    "feature-set selection (four-segment split, "
                    "non-overlapping per-h conformal)")
    ap.add_argument("--stage", type=str, default="pf",
                    choices=["p", "f", "pf"],
                    help="p = public 8 cells (9 configs each), f = field "
                         "16 cells (3 backbones each), pf = both (default)")
    ap.add_argument("--config", type=str, default=None,
                    choices=p_keys + f_keys,
                    help="run a single cell (stage inferred from the key; "
                         "P keys {ds}_H{h}, F keys {ds}_H{h}_s{seed})")
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()
    global DEVICE
    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    p_keys, f_keys = all_config_keys()

    stages = list(args.stage)
    only_p = only_f = None
    if args.config:
        if args.config in p_keys:
            stages, only_p = ["p"], args.config
        else:
            stages, only_f = ["f"], args.config

    # Field stage needs the proprietary field CSVs (not shipped). If absent:
    #  - default run (no --stage on the CLI, no field --config): drop stage F
    #    with a warning and run public stage P standalone;
    #  - an explicit field request (--stage f/pf given, or a field --config):
    #    exit with the field/README.md pointer.
    if "f" in stages and not field_data_available():
        field_explicit = only_f is not None or any(
            a == "--stage" or a.startswith("--stage=") for a in sys.argv[1:])
        if field_explicit:
            raise SystemExit(
                "Field stage requested but the proprietary field CSVs "
                f"(*{' / *'.join(FIELD_CSV_SUFFIX.values())}) are not "
                f"present in {FIELD_DATA_DIR}; see field/README.md.")
        logger.warning(
            f"Field CSVs not found in {FIELD_DATA_DIR}; running public "
            f"stage P only (field data is proprietary, see field/README.md).")
        stages = [s for s in stages if s != "f"]

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(NPZ_DIR, exist_ok=True)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Stages: {stages}  config={args.config or 'all'}")
    logger.info(f"Output: {OUT_PATH}")
    logger.info(f"Results library (read-only): {RESULTS_DIR}")

    results = load_results()
    t_start = time.time()
    if "p" in stages:
        run_stage_p(results, t_start, only_key=only_p, device=DEVICE)
    if "f" in stages:
        run_stage_f(results, t_start, only_key=only_f, device=DEVICE)

    flush(results, t_start)
    diag = build_stage_p_diagnostics(results["stage_p"])
    logger.info(f"\nDONE in {(time.time() - t_start) / 60:.1f} min. "
                f"stage_p={len(results['stage_p'])}/8 cells (x9 configs), "
                f"stage_f={len(results['stage_f'])}/16 cells (x3 backbones), "
                f"diagnostics={diag.get('status')}")


if __name__ == "__main__":
    main()
