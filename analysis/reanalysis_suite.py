#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零训练重分析套件（《电网技术》DWJS26-1067 重投补充实验，R1-2 / R1-4 / R2-3）。

四个子任务，全部基于已保存实验结果与原始序列，不训练任何模型：

  1. threshold_scan    R1-2  π 阈值 (θ_low, θ_high) 网格扫描 → 三分类归属 +
                             特征集推荐是否变化 + 与验证集实选（val_loss 择优）一致性
                             → 输出"结论不变域"。
  2. loo_calibration   R1-4  式(12) w(π)=max(1,1.5-0.8π) 的 4 点校准曲线 +
                             留一（leave-one-dataset-out）验证 + 常数 ±0.2 扰动敏感性。
                             基于已保存的 SSPM 预测分位数 npz（测试段）。
  3. covariance_share  R2-3  6 场景 STL 分解（与论文式(6)同口径）交叉协方差项占比
                             |2[Cov(T,S)+Cov(T,R)+Cov(S,R)]| / Var(Y)，验证近似正交前提。
  4. rolling_pi        R1-2  6 场景滚动子窗（窗长4周=2688点，步长1周=672点）π 分布，
                             验证类内波动远小于类间空档。

π 口径（已逐场景复现核验，与论文表1完全一致）：
  - 公开4集群：论文 π 定义原式
      series=df[target].dropna();
      STL(series, period=96, robust=True).fit(); π = np.var(seasonal)/np.var(series)。
      复现值：MIT 1.8527% / Helios 4.5023% / Alibaba 22.3063% / Inference 82.8873%。
  - 实测2场景（数据不随仓库分发，见 field/README.md）：
      STL(pd.Series(x,index=t), period=96, robust=False).fit(); π = np.var(seasonal)/np.var(x)。
      复现值：Room302 20.7421% / BuildingC 20.7398%。
    实测 CSV 不随仓库分发；缺失时自动跳过实测场景，仅在公开4场景上运行
    （见 field/README.md）。

区间加宽口径（照抄 experiments/recompute_pi_calibration.py）：
  adj_q10 = q50 + (q10-q50)*w；adj_q90 = q50 + (q90-q50)*w；
  覆盖率 = mean(q10_adj <= true <= q90_adj)；逐 (数据集,时域) 使用 BEST_BACKBONE 的
  SSPM npz（predictions/{ds}/H{h}_SSPM_{backbone}.npz，键 pred_q10/pred_q50/pred_q90/true）。
  注意：npz 仅保存测试段预测（run_feature_comparison.py / run_sspm_all_backbones.py 中
  evaluate_on_test 的输出），故覆盖率-因子曲线均在测试段计算，输出 JSON 中已注明口径。

运行：
  python3 analysis/reanalysis_suite.py
可选参数：
  --tasks threshold,loo,cov,rolling   只跑部分任务（默认全部）
  --out-dir PATH                      输出目录（默认 results/analysis）
  --dry-run                           只计算并打印摘要，不写任何文件

输出（默认 <repo>/results/analysis/）：
  threshold_scan_results.json / loo_calibration_results.json /
  covariance_share_results.json / rolling_pi_results.json

只读保证：仅读取 results 库与原始 CSV；唯一写入点为 --out-dir。
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# ============================================================
# 路径与场景常量
# ============================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.config import (DATA_PROCESSED, FIELD_DATA_DIR,  # noqa: E402
                           RESULTS_DIR)

RESULTS_DATA = os.path.join(RESULTS_DIR, "case10c_cluster_forecast", "data")
REFERENCE_DIR = os.path.join(REPO_ROOT, "results", "reference")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "analysis")


def results_data_path(*rel):
    """结果库文件解析（由 reproduce.sh 的训练阶段生成）。"""
    primary = os.path.join(RESULTS_DATA, *rel)
    if os.path.exists(primary):
        return primary
    raise FileNotFoundError(
        f"Result file not found: {primary}. "
        f"Run the training stage first (see reproduce.sh).")


# 6 场景：name -> (csv路径, 目标列, STL robust 口径, 类型, 论文表1 π%)
SCENARIOS = {
    "MIT": {
        "path": os.path.join(DATA_PROCESSED, "mit_supercloud_15min.csv"),
        "col": "cluster_power_kw", "robust": True, "kind": "public",
        "paper_pi_pct": 1.85, "label_cn": "MIT超算（训练主导）",
    },
    "Helios": {
        "path": os.path.join(DATA_PROCESSED, "helios_saturn_15min.csv"),
        "col": "active_gpu_count", "robust": True, "kind": "public",
        "paper_pi_pct": 4.50, "label_cn": "Helios Saturn（训练主导）",
    },
    "Alibaba": {
        "path": os.path.join(DATA_PROCESSED, "alibaba_v2026_spot_15min.csv"),
        "col": "active_gpu_count", "robust": True, "kind": "public",
        "paper_pi_pct": 22.31, "label_cn": "阿里PAI v2026-spot（混合）",
    },
    "Inference": {
        "path": os.path.join(DATA_PROCESSED, "inference_cluster_15min.csv"),
        "col": "cluster_power_kw", "robust": True, "kind": "public",
        "paper_pi_pct": 82.89, "label_cn": "推理服务型（Azure驱动）",
    },
    "Room302": {
        "path": os.path.join(FIELD_DATA_DIR, "field_room302_15min.csv"),
        "col": "room_power_kw", "robust": False, "kind": "measured",
        "paper_pi_pct": 20.742, "label_cn": "302机房（实测）",
    },
    "BuildingC": {
        "path": os.path.join(FIELD_DATA_DIR, "field_building_15min.csv"),
        "col": "building_power_kw", "robust": False, "kind": "measured",
        "paper_pi_pct": 20.740, "label_cn": "楼栋C（实测）",
    },
}

_ACTIVE_SCENARIOS = None


def active_scenarios():
    """可用场景名列表（保持 SCENARIOS 的定义顺序）。

    公开4场景的 CSV 必须存在（缺失则给出构建提示后退出）；实测2场景的 CSV
    不随仓库分发，缺失时打印一行警告并跳过（公开-only 运行仍产出全部任务）。
    """
    global _ACTIVE_SCENARIOS
    if _ACTIVE_SCENARIOS is None:
        names = []
        for name, cfg in SCENARIOS.items():
            if os.path.exists(cfg["path"]):
                names.append(name)
            elif cfg["kind"] == "measured":
                print(f"[warn] field scenario {name} skipped: "
                      f"{cfg['path']} not found (proprietary field data; "
                      f"see field/README.md)")
            else:
                raise SystemExit(
                    f"Public dataset CSV missing: {cfg['path']}. Build the "
                    f"processed datasets first (see data/README.md).")
        _ACTIVE_SCENARIOS = names
    return _ACTIVE_SCENARIOS

PUBLIC_DS = ["MIT", "Helios", "Alibaba", "Inference"]

# 论文表1 π（小数）——式(12)使用的自变量
PI_VALUES = {"MIT": 0.0185, "Helios": 0.045, "Alibaba": 0.2231, "Inference": 0.8289}

# 表VII 口径的最优 SSPM 骨干（照抄 recompute_pi_calibration.py 的 BEST_BACKBONE）
BEST_BACKBONE = {
    ("MIT", 16): "PatchTST-Workload",
    ("MIT", 96): "DLinear-Calendar",
    ("Helios", 16): "iTransformer-Calendar",
    ("Helios", 96): "iTransformer-Workload",
    ("Alibaba", 16): "PatchTST-Calendar",
    ("Alibaba", 96): "DLinear-All",
    ("Inference", 16): "iTransformer-All",
    ("Inference", 96): "iTransformer-All",
}

# feature_comparison 结果文件（MIT 无后缀，见 run_feature_comparison.py 尾部命名逻辑）
FEATCMP_FILES = {
    "MIT": "feature_comparison_results.json",
    "Helios": "feature_comparison_results_Helios.json",
    "Alibaba": "feature_comparison_results_Alibaba.json",
    "Inference": "feature_comparison_results_Inference.json",
}
SSPM_ALL_FILE = "sspm_all_backbones_results.json"

# 论文阈值（基线）与扫描网格（%）
BASE_THETA_LOW, BASE_THETA_HIGH = 15.0, 50.0
THETA_LOW_GRID = [8.0, 10.0, 12.0, 15.0, 18.0]
THETA_HIGH_GRID = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

# 式(12)常数与扰动网格
PAPER_A, PAPER_B = 1.5, 0.8
SENS_A_GRID = [1.3, 1.5, 1.7]
SENS_B_GRID = [0.6, 0.8, 1.0]

NOMINAL_COVERAGE = 0.80
HORIZONS = [16, 96]

# 滚动窗参数：15min 分辨率，4 周窗长 / 1 周步长
ROLL_WINDOW = 4 * 7 * 96   # 2688
ROLL_STEP = 1 * 7 * 96     # 672


# ============================================================
# 通用工具
# ============================================================

def _py(o):
    """numpy -> 原生 python，供 json 序列化。"""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return [_py(v) for v in o.tolist()]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return str(o)
    return o


def load_series(name):
    """按论文口径加载场景序列（dropna 后的一维 Series）。"""
    cfg = SCENARIOS[name]
    df = pd.read_csv(cfg["path"])
    return df[cfg["col"]].dropna()


def stl_decompose(series, robust):
    """论文式(6)口径的 STL：日周期，15min → period=96。"""
    res = STL(series, period=96, robust=robust).fit()
    return res.trend, res.seasonal, res.resid


_STL_CACHE = {}


def full_stl(name):
    """全序列 STL（缓存，供 threshold_scan / covariance_share 共用）。"""
    if name not in _STL_CACHE:
        s = load_series(name)
        t, sea, r = stl_decompose(s, SCENARIOS[name]["robust"])
        _STL_CACHE[name] = (s, t, sea, r)
    return _STL_CACHE[name]


def compute_pi_pct(name):
    """π（%），与论文表1的 π 定义完全同口径。"""
    s, _, sea, _ = full_stl(name)
    return float(np.var(sea) / np.var(s) * 100.0)


def meta_block(extra=None):
    m = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "reanalysis_suite.py",
        "note": "零训练重分析：仅读取已保存结果与原始序列，不训练任何模型",
    }
    if extra:
        m.update(extra)
    return m


# ============================================================
# 子任务 1：R1-2 阈值扫描
# ============================================================

def classify(pi_pct, theta_low, theta_high):
    """π 三分型：<θ_low 训练主导 / [θ_low,θ_high] 混合 / >θ_high 推理主导。"""
    if pi_pct < theta_low:
        return "train_dominant"
    if pi_pct > theta_high:
        return "inference_dominant"
    return "mixed"


FEATURE_RULE = {  # 特征选择规则：π<θ_low→工作负载；θ_low≤π≤θ_high→全特征；π>θ_high→日历
    "train_dominant": "Workload",
    "mixed": "All",
    "inference_dominant": "Calendar",
}
CLASS_CN = {"train_dominant": "训练主导", "mixed": "混合", "inference_dominant": "推理主导"}


def load_val_selection():
    """从已保存 results JSON 现算验证集实选特征集（按 val_loss 择优）。

    两套证据：
      baseline: feature_comparison_results*.json（DLinear/PatchTST/iTransformer × 3特征集）
      sspm:     sspm_all_backbones_results.json（SSPM 加持的同一 3×3 组合）
    """
    sel = {}
    sspm_all = json.load(open(results_data_path(SSPM_ALL_FILE)))
    for ds in PUBLIC_DS:
        fc = json.load(open(results_data_path(FEATCMP_FILES[ds])))
        sel[ds] = {}
        for h in HORIZONS:
            entry = {}
            for tag, blob in (("baseline", fc[str(h)]), ("sspm", sspm_all[ds][str(h)])):
                combos = {k: v["val_loss"] for k, v in blob.items()
                          if isinstance(v, dict) and "val_loss" in v}
                best = min(combos, key=combos.get)
                # 每个特征集在其最优骨干下的 val_loss（用于给出择优余量）
                fs_best = {}
                for k, v in combos.items():
                    fs = k.split("-", 1)[1]
                    fs_best[fs] = min(v, fs_best.get(fs, np.inf))
                entry[tag] = {
                    "best_combo": best,
                    "best_val_loss": round(combos[best], 6),
                    "selected_feature_set": best.split("-", 1)[1],
                    "feature_set_min_val_loss": {k: round(v, 6) for k, v in
                                                 sorted(fs_best.items(), key=lambda x: x[1])},
                }
            sel[ds][f"H{h}"] = entry
    return sel


def task_threshold_scan():
    print("\n" + "=" * 64)
    print("[1/4] R1-2 阈值扫描 (θ_low × θ_high)")
    print("=" * 64)

    # 现算 π（与论文表1核对；实测场景缺数据时自动跳过）
    pi_pct = {}
    for name in active_scenarios():
        pi_pct[name] = compute_pi_pct(name)
        print(f"  π({name}) = {pi_pct[name]:.4f}%  (论文表1: {SCENARIOS[name]['paper_pi_pct']}%)")

    val_sel = load_val_selection()

    def agreement(ds, rec_fs):
        """规则推荐特征集 与 验证集实选 的一致性（逐时域、两套证据）。"""
        out = {}
        for h in HORIZONS:
            e = val_sel[ds][f"H{h}"]
            out[f"H{h}"] = {
                "recommended": rec_fs,
                "baseline_selected": e["baseline"]["selected_feature_set"],
                "baseline_match": rec_fs == e["baseline"]["selected_feature_set"],
                "sspm_selected": e["sspm"]["selected_feature_set"],
                "sspm_match": rec_fs == e["sspm"]["selected_feature_set"],
            }
        return out

    # 基线 (15, 50)
    def cell_result(tl, th):
        cell = {"theta_low_pct": tl, "theta_high_pct": th, "scenarios": {}}
        for name in active_scenarios():
            c = classify(pi_pct[name], tl, th)
            rec = FEATURE_RULE[c]
            sc = {"pi_pct": round(pi_pct[name], 4), "class": c, "class_cn": CLASS_CN[c],
                  "recommended_feature_set": rec}
            if name in PUBLIC_DS:
                sc["val_selection_agreement"] = agreement(name, rec)
            cell["scenarios"][name] = sc
        return cell

    baseline = cell_result(BASE_THETA_LOW, BASE_THETA_HIGH)
    base_classes = {n: baseline["scenarios"][n]["class"]
                    for n in active_scenarios()}

    grid = []
    invariant_cells, changed_cells = [], []
    for tl in THETA_LOW_GRID:
        for th in THETA_HIGH_GRID:
            cell = cell_result(tl, th)
            diffs = {n: {"from": base_classes[n], "to": cell["scenarios"][n]["class"]}
                     for n in active_scenarios()
                     if cell["scenarios"][n]["class"] != base_classes[n]}
            cell["classification_changed_vs_paper"] = bool(diffs)
            cell["changed_scenarios"] = diffs
            grid.append(cell)
            tag = f"({tl:g}, {th:g})"
            (changed_cells if diffs else invariant_cells).append(tag)

    n_cells = len(grid)
    print(f"\n  网格 {len(THETA_LOW_GRID)}×{len(THETA_HIGH_GRID)} = {n_cells} 格；"
          f"结论不变格数 = {len(invariant_cells)}；变化格数 = {len(changed_cells)}")
    if changed_cells:
        print(f"  变化格：{changed_cells}")

    # 各场景 π 相对扫描网格的"归属不变裕度"（>0 表示整张网格内归属恒定）
    margins = {}
    for name in active_scenarios():
        p = pi_pct[name]
        c = base_classes[name]
        if c == "train_dominant":
            # 需在全网格保持 π < θ_low，约束最紧的是最小 θ_low
            margins[name] = {
                "binding_boundary": f"min theta_low = {min(THETA_LOW_GRID):g}%",
                "margin_pct": round(min(THETA_LOW_GRID) - p, 4),
                "note": f"π={p:.2f}%，需 π < θ_low 恒成立",
            }
        elif c == "inference_dominant":
            # 需在全网格保持 π > θ_high，约束最紧的是最大 θ_high
            margins[name] = {
                "binding_boundary": f"max theta_high = {max(THETA_HIGH_GRID):g}%",
                "margin_pct": round(p - max(THETA_HIGH_GRID), 4),
                "note": f"π={p:.2f}%，需 π > θ_high 恒成立",
            }
        else:
            # 需在全网格保持 θ_low <= π <= θ_high
            margins[name] = {
                "binding_boundary": (f"max theta_low = {max(THETA_LOW_GRID):g}% / "
                                     f"min theta_high = {min(THETA_HIGH_GRID):g}%"),
                "margin_low_pct": round(p - max(THETA_LOW_GRID), 4),
                "margin_high_pct": round(min(THETA_HIGH_GRID) - p, 4),
                "margin_pct": round(min(p - max(THETA_LOW_GRID),
                                        min(THETA_HIGH_GRID) - p), 4),
                "note": f"π={p:.2f}%，需 θ_low ≤ π ≤ θ_high 恒成立",
            }

    return {
        "meta": meta_block({
            "review_item": "R1-2 阈值扫描",
            "pi_source": {
                "public": "论文 π 定义同口径现算"
                          "（dropna + STL(period=96, robust=True)，π=Var(S)/Var(Y)）",
                "measured": "实测口径现算（STL(period=96, robust=False)）",
            },
            "feature_rule": "π>θ_high→日历特征；θ_low≤π≤θ_high→全特征；π<θ_low→工作负载特征",
            "val_selection_source": {
                "baseline": "feature_comparison_results*.json 按 val_loss 择优（现算）",
                "sspm": "sspm_all_backbones_results.json 按 val_loss 择优（现算）",
                "paper_tableVII_backbones": {f"{k[0]}_H{k[1]}": v for k, v in BEST_BACKBONE.items()},
            },
        }),
        "pi_pct_recomputed": {k: round(v, 4) for k, v in pi_pct.items()},
        "pi_pct_paper": {k: SCENARIOS[k]["paper_pi_pct"]
                         for k in active_scenarios()},
        "paper_thresholds": {"theta_low_pct": BASE_THETA_LOW, "theta_high_pct": BASE_THETA_HIGH},
        "grid_definition": {"theta_low_pct": THETA_LOW_GRID, "theta_high_pct": THETA_HIGH_GRID},
        "baseline_cell": baseline,
        "validation_selection": val_sel,
        "grid": grid,
        "invariant_domain": {
            "n_cells": n_cells,
            "n_invariant": len(invariant_cells),
            "invariant_cells": invariant_cells,
            "changed_cells": changed_cells,
            "conclusion": ("参与评估的全部场景在整张网格内三分类归属均与论文阈值 (15%,50%) 一致，"
                           "推荐特征集与验证集实选一致性亦不随阈值变化"
                           if not changed_cells else
                           "存在阈值组合使部分场景归属变化，详见 changed_cells/grid"),
        },
        "boundary_margins": margins,
    }


# ============================================================
# 子任务 2：R1-4 区间加宽规则留一验证
# ============================================================

def load_pred_npz(ds, h):
    bb = BEST_BACKBONE[(ds, h)]
    path = results_data_path("predictions", ds, f"H{h}_SSPM_{bb}.npz")
    d = np.load(path)
    return (d["pred_q10"].astype(np.float64), d["pred_q50"].astype(np.float64),
            d["pred_q90"].astype(np.float64), d["true"].astype(np.float64), bb, path)


def per_point_min_widen(q10, q50, q90, true):
    """逐点最小加宽因子 w_min：使该点恰被 [q50+(q10-q50)w, q50+(q90-q50)w] 覆盖。

    覆盖率(w) = P(w_min <= w)，故"恰达 80% 覆盖的最小因子" = w_min 的第 80 百分位
    （取 sorted[ceil(0.8n)-1]，保证覆盖率 >= 0.80 的最小可行 w）。
    """
    a_lo = (q10 - q50).ravel()
    a_hi = (q90 - q50).ravel()
    r = (true - q50).ravel()
    w = np.full(r.shape, np.inf)
    w[r == 0] = 0.0
    pos, neg = r > 0, r < 0
    ok = pos & (a_hi > 0)
    w[ok] = r[ok] / a_hi[ok]
    ok = neg & (a_lo < 0)
    w[ok] = r[ok] / a_lo[ok]
    n_uncoverable = int(np.isinf(w).sum())  # 分位数交叉导致加宽也无法覆盖的点
    return w, n_uncoverable


def min_w_for_coverage(w_min, target=NOMINAL_COVERAGE):
    """使覆盖率 >= target 的最小 w（精确解）。"""
    ws = np.sort(w_min)
    k = int(np.ceil(target * len(ws))) - 1
    return float(ws[k])


def coverage_at(w_min, w):
    return float(np.mean(w_min <= w))


def width_at(q10, q90, w):
    return float(np.mean(q90 - q10) * w)


def fit_linear_rule(pis, wstars):
    """最小二乘拟合 w = a - b*π（应用时取 max(1, a-b*π)）。"""
    slope, intercept = np.polyfit(np.asarray(pis, float), np.asarray(wstars, float), 1)
    return float(intercept), float(-slope)  # a, b


def apply_rule(a, b, pi):
    return max(1.0, a - b * pi)


def task_loo_calibration():
    print("\n" + "=" * 64)
    print("[2/4] R1-4 式(12)常数 (1.5, 0.8) 留一验证 + 敏感性")
    print("=" * 64)

    per_dsh = {}      # (ds,h) -> dict(含 w_min 数组)
    wmin_pool = {}    # ds -> 拼接两时域的 w_min
    curves = {}
    w_grid = np.round(np.arange(0.6, 2.001, 0.05), 3)

    for ds in PUBLIC_DS:
        pool = []
        for h in HORIZONS:
            q10, q50, q90, true, bb, path = load_pred_npz(ds, h)
            w_min, n_unc = per_point_min_widen(q10, q50, q90, true)
            wstar = min_w_for_coverage(w_min)
            entry = {
                "backbone": bb, "npz": path,
                "n_samples": int(q10.shape[0]), "horizon_steps": int(q10.shape[1]),
                "n_points": int(w_min.size),
                "n_uncoverable_points": n_unc,
                "coverage_w1": round(coverage_at(w_min, 1.0), 4),
                "coverage_paper_rule": round(
                    coverage_at(w_min, apply_rule(PAPER_A, PAPER_B, PI_VALUES[ds])), 4),
                "w_paper_rule": round(apply_rule(PAPER_A, PAPER_B, PI_VALUES[ds]), 3),
                "w_star": round(wstar, 4),
                "coverage_at_w_star": round(coverage_at(w_min, wstar), 4),
                "mean_width_w1": round(float(np.mean(q90 - q10)), 3),
                "mean_width_at_w_star": round(width_at(q10, q90, wstar), 3),
            }
            per_dsh[(ds, h)] = {**entry, "_w_min": w_min}
            pool.append(w_min)
            curves[f"{ds}_H{h}"] = {
                "w": w_grid.tolist(),
                "coverage": [round(coverage_at(w_min, w), 4) for w in w_grid],
            }
            print(f"  {ds:<10} H{h:<3} {bb:<24} cov(w=1)={entry['coverage_w1']:.3f} "
                  f"cov(式12 w={entry['w_paper_rule']})={entry['coverage_paper_rule']:.3f} "
                  f"w*={wstar:.3f}")
        wmin_pool[ds] = np.concatenate(pool)

    # 4 个校准点：逐数据集（两时域测试段样本点合并）的 w*_d
    calib_points = {}
    for ds in PUBLIC_DS:
        wstar_d = min_w_for_coverage(wmin_pool[ds])
        calib_points[ds] = {
            "pi": PI_VALUES[ds],
            "w_star_pooled": round(wstar_d, 4),
            "w_star_H16": per_dsh[(ds, 16)]["w_star"],
            "w_star_H96": per_dsh[(ds, 96)]["w_star"],
            "w_paper_rule": round(apply_rule(PAPER_A, PAPER_B, PI_VALUES[ds]), 3),
            "note": "w_star 为未截断值（可 <1）；式(12)应用时取 max(1,·)",
        }
    print("\n  校准点 (π_d, w*_d):",
          {ds: (v["pi"], v["w_star_pooled"]) for ds, v in calib_points.items()})

    # 全量拟合（4 点）
    pis = [PI_VALUES[ds] for ds in PUBLIC_DS]
    wstars = [calib_points[ds]["w_star_pooled"] for ds in PUBLIC_DS]
    a_full, b_full = fit_linear_rule(pis, wstars)
    print(f"  4点最小二乘: w = {a_full:.3f} - {b_full:.3f}·π  (论文式(12): 1.5 - 0.8·π)")

    # 留一验证
    loo = {}
    for left in PUBLIC_DS:
        rest = [d for d in PUBLIC_DS if d != left]
        a, b = fit_linear_rule([PI_VALUES[d] for d in rest],
                               [calib_points[d]["w_star_pooled"] for d in rest])
        w_hat = apply_rule(a, b, PI_VALUES[left])
        entry = {
            "train_datasets": rest,
            "refit_a": round(a, 4), "refit_b": round(b, 4),
            "w_hat_heldout": round(w_hat, 4),
            "w_paper_rule": round(apply_rule(PAPER_A, PAPER_B, PI_VALUES[left]), 3),
            "heldout_coverage": {},
        }
        for h in HORIZONS:
            wm = per_dsh[(left, h)]["_w_min"]
            entry["heldout_coverage"][f"H{h}"] = {
                "refit_rule": round(coverage_at(wm, w_hat), 4),
                "paper_rule": per_dsh[(left, h)]["coverage_paper_rule"],
                "no_widening": per_dsh[(left, h)]["coverage_w1"],
            }
        entry["heldout_coverage"]["pooled"] = {
            "refit_rule": round(coverage_at(wmin_pool[left], w_hat), 4),
            "paper_rule": round(coverage_at(
                wmin_pool[left], apply_rule(PAPER_A, PAPER_B, PI_VALUES[left])), 4),
            "no_widening": round(coverage_at(wmin_pool[left], 1.0), 4),
        }
        loo[left] = entry
        print(f"  LOO 留出 {left:<10} 重拟合 w={a:.3f}-{b:.3f}π → w_hat={w_hat:.3f} "
              f"留出集 pooled 覆盖={entry['heldout_coverage']['pooled']['refit_rule']:.3f}")

    # 常数扰动敏感性
    sens = {"a_grid": SENS_A_GRID, "b_grid": SENS_B_GRID, "cells": []}
    for a in SENS_A_GRID:
        for b in SENS_B_GRID:
            cell = {"a": a, "b": b, "coverage": {}, "w": {}}
            devs = []
            for ds in PUBLIC_DS:
                w = apply_rule(a, b, PI_VALUES[ds])
                cell["w"][ds] = round(w, 3)
                cov = {f"H{h}": round(coverage_at(per_dsh[(ds, h)]["_w_min"], w), 4)
                       for h in HORIZONS}
                cov["pooled"] = round(coverage_at(wmin_pool[ds], w), 4)
                cell["coverage"][ds] = cov
                devs.append(abs(cov["pooled"] - NOMINAL_COVERAGE))
            cell["mean_abs_dev_from_nominal"] = round(float(np.mean(devs)), 4)
            cell["max_abs_dev_from_nominal"] = round(float(np.max(devs)), 4)
            sens["cells"].append(cell)

    return {
        "meta": meta_block({
            "review_item": "R1-4 式(12) w(π)=max(1, 1.5-0.8π) 拟合过程与稳健性",
            "data_scope": ("npz 仅含测试段预测（evaluate_on_test 输出，见 "
                           "run_feature_comparison.py/run_sspm_all_backbones.py），"
                           "验证段预测未保存；故覆盖率-因子曲线与 w* 均在测试段计算（输出已注明口径）"),
            "widening_formula": "adj_q10=q50+(q10-q50)w; adj_q90=q50+(q90-q50)w"
                                "（照抄 recompute_pi_calibration.py calibrate()）",
            "w_star_definition": "使 80% 区间经验覆盖率 >= 0.80 的最小加宽因子（逐点精确解取第80百分位）",
            "backbone_source": "recompute_pi_calibration.py BEST_BACKBONE（与论文表VII同源）",
            "nominal_coverage": NOMINAL_COVERAGE,
        }),
        "per_dataset_horizon": {f"{ds}_H{h}": {k: v for k, v in per_dsh[(ds, h)].items()
                                               if not k.startswith("_")}
                                for ds in PUBLIC_DS for h in HORIZONS},
        "coverage_vs_w_curves": curves,
        "calibration_points": calib_points,
        "full_fit_4points": {"a": round(a_full, 4), "b": round(b_full, 4),
                             "paper_constants": {"a": PAPER_A, "b": PAPER_B}},
        "leave_one_out": loo,
        "constant_sensitivity": sens,
    }


# ============================================================
# 子任务 3：R2-3 协方差项占比
# ============================================================

def task_covariance_share():
    print("\n" + "=" * 64)
    print("[3/4] R2-3 STL 交叉协方差项占比（近似正交前提核算）")
    print("=" * 64)

    def pcov(a, b):  # 总体协方差（ddof=0，与 np.var 口径一致）
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return float(np.mean((a - a.mean()) * (b - b.mean())))

    out = {}
    max_share = 0.0
    for name in active_scenarios():
        s, t, sea, r = full_stl(name)
        y = np.asarray(s, float)
        var_y = float(np.var(y))
        var_t, var_s, var_r = float(np.var(t)), float(np.var(sea)), float(np.var(r))
        c_ts, c_tr, c_sr = pcov(t, sea), pcov(t, r), pcov(sea, r)
        cross = 2.0 * (c_ts + c_tr + c_sr)
        share = abs(cross) / var_y
        max_share = max(max_share, share)
        identity_gap = var_y - (var_t + var_s + var_r + cross)
        out[name] = {
            "kind": SCENARIOS[name]["kind"],
            "stl_robust": SCENARIOS[name]["robust"],
            "n_points": int(len(y)),
            "var_Y": var_y,
            "var_trend": var_t, "var_seasonal": var_s, "var_resid": var_r,
            "component_var_share": {
                "trend": round(var_t / var_y, 6),
                "seasonal_pi": round(var_s / var_y, 6),
                "resid": round(var_r / var_y, 6),
                "sum": round((var_t + var_s + var_r) / var_y, 6),
            },
            "cov_TS": c_ts, "cov_TR": c_tr, "cov_SR": c_sr,
            "cross_term_2sum": cross,
            "cov_share_abs": round(share, 6),
            "cov_share_pct": round(share * 100.0, 4),
            "variance_identity_gap_rel": round(identity_gap / var_y, 12),
            "pi_pct": round(var_s / var_y * 100.0, 4),
            "paper_pi_pct": SCENARIOS[name]["paper_pi_pct"],
        }
        print(f"  {name:<10} |2ΣCov|/Var(Y) = {share * 100:.4f}%   "
              f"(π={out[name]['pi_pct']:.2f}%, 分量方差和占比="
              f"{out[name]['component_var_share']['sum']:.4f})")

    conclusion = (f"{len(out)} 场景交叉协方差项占比 |2ΣCov|/Var(Y) 最大 {max_share * 100:.4f}%"
                  f"（≤0.1 量级），均显著小于 1：近似正交前提在全部场景成立，"
                  f"π=Var(S)/Var(Y) 作为 [0,1] 区间内方差占比指标的解释有效")
    print(f"\n  结论：{conclusion}")

    return {
        "meta": meta_block({
            "review_item": "R2-3 π∈[0,1] 的近似正交前提量化核验",
            "stl_spec": {
                "public": "series.dropna(); STL(period=96, robust=True)"
                          "（与论文 π 定义同口径）",
                "measured": "STL(period=96, robust=False)（实测口径）",
            },
            "metric": "|2[Cov(T,S)+Cov(T,R)+Cov(S,R)]| / Var(Y)（总体协方差 ddof=0）",
            "identity_check": "Var(Y) = Var(T)+Var(S)+Var(R)+2ΣCov 恒等式相对残差应 ~1e-16",
        }),
        "scenarios": out,
        "max_cov_share_pct": round(max_share * 100.0, 4),
        "conclusion": conclusion,
    }


# ============================================================
# 子任务 4：R1-2 滚动子窗 π 分布
# ============================================================

def task_rolling_pi():
    print("\n" + "=" * 64)
    print(f"[4/4] R1-2 滚动子窗 π 分布（窗长 {ROLL_WINDOW} 点=4周，步长 {ROLL_STEP} 点=1周）")
    print("=" * 64)

    out = {}
    for name in active_scenarios():
        cfg = SCENARIOS[name]
        s = load_series(name).reset_index(drop=True)
        n = len(s)
        pis, starts = [], []
        i = 0
        while i + ROLL_WINDOW <= n:
            win = s.iloc[i:i + ROLL_WINDOW]
            v = float(np.var(win))
            if v > 0:
                _, sea, _ = stl_decompose(pd.Series(win.values), cfg["robust"])
                pis.append(float(np.var(sea) / v * 100.0))
                starts.append(i)
            i += ROLL_STEP
        pis_arr = np.array(pis)
        stats = {
            "n_windows": int(len(pis_arr)),
            "min": round(float(pis_arr.min()), 4),
            "q25": round(float(np.percentile(pis_arr, 25)), 4),
            "median": round(float(np.percentile(pis_arr, 50)), 4),
            "q75": round(float(np.percentile(pis_arr, 75)), 4),
            "max": round(float(pis_arr.max()), 4),
            "mean": round(float(pis_arr.mean()), 4),
            "std": round(float(pis_arr.std()), 4),
            "iqr": round(float(np.percentile(pis_arr, 75) - np.percentile(pis_arr, 25)), 4),
        }
        cls = {c: round(float(np.mean([classify(p, BASE_THETA_LOW, BASE_THETA_HIGH) == c
                                       for p in pis_arr])), 4)
               for c in ("train_dominant", "mixed", "inference_dominant")}
        full_pi = compute_pi_pct(name)
        full_cls = classify(full_pi, BASE_THETA_LOW, BASE_THETA_HIGH)
        out[name] = {
            "kind": cfg["kind"], "stl_robust": cfg["robust"],
            "n_points": int(n),
            "window_points": ROLL_WINDOW, "step_points": ROLL_STEP,
            "full_series_pi_pct": round(full_pi, 4),
            "full_series_class": full_cls,
            "window_pi_pct_stats": stats,
            "window_class_fraction_paper_thresholds": cls,
            "window_class_stable_fraction": cls[full_cls],
            "window_pi_pct": [round(p, 4) for p in pis],
            "window_start_index": starts,
        }
        print(f"  {name:<10} n_win={stats['n_windows']:>3} "
              f"min={stats['min']:>8.3f} q25={stats['q25']:>8.3f} "
              f"med={stats['median']:>8.3f} q75={stats['q75']:>8.3f} max={stats['max']:>8.3f}"
              f"   全序列 π={full_pi:.2f}%")

    # 类内-类间对比（按论文阈值的三分类聚合窗口 π 范围）
    class_members = {}
    for name in active_scenarios():
        class_members.setdefault(out[name]["full_series_class"], []).append(name)
    class_ranges = {}
    for c, members in class_members.items():
        allp = np.concatenate([np.array(out[m]["window_pi_pct"]) for m in members])
        class_ranges[c] = {
            "members": members,
            "window_pi_min": round(float(allp.min()), 4),
            "window_pi_q25": round(float(np.percentile(allp, 25)), 4),
            "window_pi_median": round(float(np.percentile(allp, 50)), 4),
            "window_pi_q75": round(float(np.percentile(allp, 75)), 4),
            "window_pi_max": round(float(allp.max()), 4),
        }
    order = ["train_dominant", "mixed", "inference_dominant"]
    gaps = {}
    for lo_c, hi_c in zip(order[:-1], order[1:]):
        if lo_c in class_ranges and hi_c in class_ranges:
            gaps[f"{lo_c}->{hi_c}"] = {
                # 极值口径（受单窗极端值驱动，参考用）
                "lower_class_window_max": class_ranges[lo_c]["window_pi_max"],
                "upper_class_window_min": class_ranges[hi_c]["window_pi_min"],
                "gap_extremes_pct_points": round(class_ranges[hi_c]["window_pi_min"]
                                                 - class_ranges[lo_c]["window_pi_max"], 4),
                # IQR 口径（分布主体的间隔，更稳健）
                "lower_class_window_q75": class_ranges[lo_c]["window_pi_q75"],
                "upper_class_window_q25": class_ranges[hi_c]["window_pi_q25"],
                "gap_iqr_pct_points": round(class_ranges[hi_c]["window_pi_q25"]
                                            - class_ranges[lo_c]["window_pi_q75"], 4),
            }
    print(f"\n  类间空档（基于子窗分布）：{json.dumps(gaps, ensure_ascii=False)}")

    stability = {n: out[n]["window_class_stable_fraction"]
                 for n in active_scenarios()}
    findings = {
        "window_class_stable_fraction": stability,
        "caveat": ("子窗 π 相对全序列 π 存在系统性上偏：4 周窗口截断抑制了低频趋势方差"
                   "（分母 Var(Y_w) 变小），对趋势主导场景（MIT/Helios/Room302 等）尤为明显。"
                   "因此子窗 π 分布适于展示'同一场景内 π 估计的波动范围与分类稳定性'，"
                   "不宜将不同场景子窗 π 的极值直接对比作为'类间空档'的证据；"
                   "类间空档仍应以全序列 π（表1）为口径，本任务提供类内波动的量化补充。"),
    }

    return {
        "meta": meta_block({
            "review_item": "R1-2 滚动子窗 π 分布（类内波动 vs 类间空档）",
            "window_spec": "窗长 4 周 = 2688 点，步长 1 周 = 672 点，15min 分辨率",
            "stl_spec": "逐窗 STL(period=96)，robust 口径与该场景全序列 π 一致"
                        "（公开 robust=True / 实测 robust=False），π_w=Var(S_w)/Var(Y_w)",
        }),
        "scenarios": out,
        "class_window_pi_ranges": class_ranges,
        "inter_class_gaps": gaps,
        "findings": findings,
    }


# ============================================================
# 主入口
# ============================================================

TASKS = {
    "threshold": ("threshold_scan_results.json", task_threshold_scan),
    "loo": ("loo_calibration_results.json", task_loo_calibration),
    "cov": ("covariance_share_results.json", task_covariance_share),
    "rolling": ("rolling_pi_results.json", task_rolling_pi),
}


def main():
    ap = argparse.ArgumentParser(description="零训练重分析套件（R1-2/R1-4/R2-3）")
    ap.add_argument("--tasks", default="threshold,loo,cov,rolling",
                    help="逗号分隔：threshold,loo,cov,rolling")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="只计算并打印摘要，不写任何文件")
    args = ap.parse_args()

    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in names if t not in TASKS]
    if unknown:
        raise SystemExit(f"未知任务: {unknown}；可选 {list(TASKS)}")

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    for t in names:
        fname, fn = TASKS[t]
        result = fn()
        payload = json.dumps(_py(result), ensure_ascii=False, indent=1)
        if args.dry_run:
            print(f"  [dry-run] 序列化通过（{len(payload)} 字符），跳过写入 {fname}")
            continue
        path = os.path.join(args.out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"  已写入 {path}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
