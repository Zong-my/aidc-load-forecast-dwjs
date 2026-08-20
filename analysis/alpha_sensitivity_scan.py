"""
Mixup alpha sensitivity scan for SSPM (R2-6 revision experiment).

Motivation: reviewer comment R2-6 pointed out that the original claim
("low-pi datasets need small alpha, high-SNR datasets need large alpha")
contradicts the experimental tables. The claim is removed in the revision;
this controlled experiment supports the new statement instead:
"the optimal alpha couples with backbone / feature set / horizon, and the
sensitivity of SSPM to alpha over a wide range must be assessed empirically."

Design (differences from run_sspm_all_backbones.py are intentional):
  - Single horizon H=96 (24h), 4 datasets, 2 backbones {DLinear, iTransformer}.
  - Feature set fixed per dataset to the one used by that dataset's best
    SSPM variant (paper Table 2 note):
      MIT -> Calendar, Helios -> Workload, Alibaba -> All, Inference -> Calendar
  - A UNIFIED log-spaced alpha grid, identical for all datasets (the original
    script used dataset-specific grids -- this uniformity is the point):
      {0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}
  - Per (dataset, backbone): masked pretrain ONCE (30 epochs, mask 0.3),
    keep the pretrained state_dict in memory, then for every alpha reload the
    pretrained weights into a fresh model and finetune with mixup.
    set_seed(SEED) is called before every finetune so alphas are comparable.
  - ALL alphas are evaluated on the test set (no selection) -- we report the
    full sensitivity curve: val_loss, r2, rmse, coverage_80, interval_width,
    crps for every alpha.

All training / evaluation code is reused verbatim from experiments/:
  pretrain_masked, train_mixup            (run_sspm_all_backbones.py)
  evaluate_on_test                        (run_baselines.py)
  DATASET_CONFIGS, HPARAMS, SEED,
  load_and_engineer,
  build_dataset_for_feature_set,
  make_model                              (run_feature_comparison.py)
  set_seed                                (train_msra.py)

Output (incremental -- rewritten after each dataset x backbone finishes):
  results/analysis/alpha_sensitivity_results.json
  Structure: {dataset: {backbone: {alpha_str: {val_loss, r2, rmse,
              coverage_80, interval_width, crps, finetune_time_s}}},
              "meta": {...}}

Hard constraints respected:
  - Never modifies anything in the experiments/ directory (import only).
  - Single GPU (cuda:0).

Usage (any cwd):
  python analysis/alpha_sensitivity_scan.py                   # all 4 datasets
  python analysis/alpha_sensitivity_scan.py --dataset Helios  # single dataset
"""

import os
import sys
import time
import argparse

import torch

# ------------------------------------------------------------------
# Path setup: repo root (common.*) and experiments/ (training modules).
# ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.utils import get_logger, save_json, load_json          # noqa: E402
from train_msra import set_seed                                    # noqa: E402
from run_baselines import evaluate_on_test                         # noqa: E402
from run_feature_comparison import (DATASET_CONFIGS, HPARAMS, SEED,  # noqa: E402
                                    load_and_engineer,
                                    build_dataset_for_feature_set,
                                    make_model)
from run_sspm_all_backbones import (pretrain_masked, train_mixup,  # noqa: E402
                                    PRETRAIN_EPOCHS, MASK_RATIO)

logger = get_logger("alpha_sensitivity")

# ------------------------------------------------------------------
# Experiment definition
# ------------------------------------------------------------------
HORIZON = 96  # 24h at 15-min resolution; single horizon by design

# Unified log-spaced grid, identical for every dataset (key difference
# from the dataset-specific ALPHA_GRID in run_sspm_all_backbones.py).
ALPHA_GRID = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

BACKBONES = ["DLinear", "iTransformer"]

# Feature set of each dataset's best SSPM variant (paper Table 2 note).
FEATURE_SET_MAP = {
    "MIT": "Calendar",
    "Helios": "Workload",
    "Alibaba": "All",
    "Inference": "Calendar",
}

OUT_DIR = os.path.join(REPO_ROOT, "results", "analysis")
OUT_PATH = os.path.join(OUT_DIR, "alpha_sensitivity_results.json")

METRIC_KEYS = ["r2", "rmse", "coverage_80", "interval_width", "crps"]


# ------------------------------------------------------------------
# Incremental persistence
# ------------------------------------------------------------------
def load_existing_results():
    """Load previous partial results so --dataset reruns merge, not clobber."""
    if os.path.exists(OUT_PATH):
        try:
            prev = load_json(OUT_PATH)
            prev.pop("meta", None)
            logger.info(f"Merging into existing results: "
                        f"{sorted(prev.keys())} @ {OUT_PATH}")
            return prev
        except Exception as e:  # corrupt / partial file -> start fresh
            logger.warning(f"Could not parse existing {OUT_PATH} ({e}); "
                           f"starting fresh")
    return {}


def save_results(results, meta, t_start):
    meta["total_time_s"] = round(time.time() - t_start, 1)
    out = dict(results)
    out["meta"] = meta
    save_json(out, OUT_PATH)
    logger.info(f"  [saved] {OUT_PATH} ({meta['total_time_s']}s elapsed)")


# ------------------------------------------------------------------
# Core scan
# ------------------------------------------------------------------
def scan_one_backbone(ds_name, backbone, data, device):
    """Pretrain once, then finetune + test-evaluate at every alpha.

    Returns ({alpha_str: record}, pretrain_time_s).
    """
    nf = data["train_X"].shape[2]

    # Phase 1: masked pretraining, done once and reused for all alphas
    # (same protocol as run_sspm_all_backbones.run_one: 30 epochs, mask 0.3)
    set_seed(SEED)
    model_pre = make_model(backbone, nf, HORIZON).to(device)
    t0 = time.time()
    model_pre = pretrain_masked(model_pre, data["train_X"], HPARAMS, device,
                                epochs=PRETRAIN_EPOCHS, mask_ratio=MASK_RATIO)
    pretrain_sd = {k: v.cpu().clone()
                   for k, v in model_pre.state_dict().items()}
    del model_pre
    torch.cuda.empty_cache()
    t_pre = time.time() - t0
    logger.info(f"  [{ds_name}/{backbone}] pretrain done "
                f"({PRETRAIN_EPOCHS} ep, mask={MASK_RATIO}) in {t_pre:.1f}s")

    # Phase 2: alpha scan -- every alpha gets identical starting conditions
    bb_results = {}
    for i, alpha in enumerate(ALPHA_GRID):
        t_a = time.time()
        set_seed(SEED)  # reseed per alpha: identical data order / init noise
        m = make_model(backbone, nf, HORIZON).to(device)
        m.load_state_dict(pretrain_sd)  # fresh copy of pretrained weights
        m, val_loss = train_mixup(m, data, HPARAMS, alpha, device)
        metrics = evaluate_on_test(m, data["test_X"], data["test_y"],
                                   data["norm_params"], device)
        del m
        torch.cuda.empty_cache()

        record = {"val_loss": round(val_loss, 6)}
        for k in METRIC_KEYS:
            record[k] = metrics.get(k)
        record["finetune_time_s"] = round(time.time() - t_a, 1)
        bb_results[str(alpha)] = record

        logger.info(f"  [{ds_name}/{backbone}] alpha={alpha:<6} "
                    f"({i + 1}/{len(ALPHA_GRID)})  "
                    f"val={record['val_loss']:.6f}  R2={record['r2']}  "
                    f"RMSE={record['rmse']}  CRPS={record['crps']}  "
                    f"({record['finetune_time_s']}s)")

    return bb_results, t_pre


def main(ds_name="all"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Alpha grid (unified): {ALPHA_GRID}")
    logger.info(f"Horizon: {HORIZON}  Backbones: {BACKBONES}")
    logger.info(f"Feature sets: {FEATURE_SET_MAP}")

    ds_list = [ds_name] if ds_name != "all" else list(DATASET_CONFIGS.keys())

    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_existing_results()
    t_start = time.time()
    meta = {
        "seed": SEED,
        "alpha_grid": ALPHA_GRID,
        "feature_set_map": FEATURE_SET_MAP,
        "horizon": HORIZON,
        "backbones": BACKBONES,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "mask_ratio": MASK_RATIO,
        "hparams": HPARAMS,
        "timing": {},
        "note": ("R2-6 controlled alpha sensitivity scan: unified log grid, "
                 "one pretrain per (dataset, backbone), per-alpha reseeded "
                 "finetune, all alphas evaluated on test set."),
    }

    n_total = len(ds_list) * len(BACKBONES)
    n_done = 0

    for ds in ds_list:
        cfg = DATASET_CONFIGS[ds]
        fs_name = FEATURE_SET_MAP[ds]
        logger.info(f"\n{'=' * 60}\nDataset: {ds}  "
                    f"(feature set: {fs_name}, H={HORIZON})\n{'=' * 60}")

        df, feature_sets, target = load_and_engineer(cfg)
        data = build_dataset_for_feature_set(
            df, feature_sets[fs_name], HORIZON, target)

        results.setdefault(ds, {})

        for backbone in BACKBONES:
            n_done += 1
            logger.info(f"\n--- [{n_done}/{n_total}] {ds} x {backbone} ---")
            t_bb = time.time()

            bb_results, t_pre = scan_one_backbone(ds, backbone, data, device)

            results[ds][backbone] = bb_results
            meta["timing"][f"{ds}/{backbone}"] = {
                "pretrain_s": round(t_pre, 1),
                "total_s": round(time.time() - t_bb, 1),
            }
            # Incremental flush after every dataset x backbone
            save_results(results, meta, t_start)

    logger.info(f"\nAll done: {n_done} (dataset x backbone) x "
                f"{len(ALPHA_GRID)} alphas in "
                f"{time.time() - t_start:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSPM mixup alpha sensitivity scan (R2-6)")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASET_CONFIGS.keys()) + ["all"],
                        help="Dataset to run (default: all)")
    args = parser.parse_args()
    main(args.dataset)
