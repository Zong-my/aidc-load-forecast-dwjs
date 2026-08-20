#!/usr/bin/env python3
"""前测历史 π 重算（阻断项3修复）。

口径：π 在测试期开始前的全部可用历史上计算，即时间序列前 85%
（训练 70% + 验证 15%，与实验划分一致），STL 规格与全序列口径完全相同
（公开 robust=True / 实测 robust=False，period=96）。

输出 pretest_pi_results.json：逐场景 π_pretest 与 π_full 对照、三档分类
（阈值 15%/50%）在两口径下是否一致、到阈值的裕度、实测场景的
w(π)=max(1, 1.5-0.8π) 新旧对照。只读数据，仅写本 JSON。
实测 CSV 不随仓库分发；缺失时自动跳过实测场景（见 field/README.md）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reanalysis_suite import (load_series, stl_decompose,  # noqa: E402
                              meta_block, _py, active_scenarios)
import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "analysis")
OUT = os.path.join(OUT_DIR, "pretest_pi_results.json")
SCEN = [("MIT", True), ("Helios", True), ("Alibaba", True),
        ("Inference", True), ("Room302", False), ("BuildingC", False)]
SPLIT = 0.85          # 训练70% + 验证15%
TL, TH = 15.0, 50.0   # 论文阈值（%）


def classify(pi_pct):
    if pi_pct > TH:
        return "inference_dominant"
    if pi_pct >= TL:
        return "mixed"
    return "train_dominant"


def pi_pct_of(series, robust):
    _t, seasonal, _r = stl_decompose(series, robust)
    return float(np.var(seasonal) / np.var(series.values) * 100)


def main():
    out = {"meta": meta_block({
        "definition": "π_pretest = STL π on first 85% of series "
                      "(train+val, pre-test history); STL spec identical "
                      "to full-series caliber",
        "thresholds_pct": [TL, TH]})}
    available = active_scenarios()   # skips missing field CSVs with a warning
    rows = {}
    for name, robust in SCEN:
        if name not in available:
            continue
        s = load_series(name)
        n = len(s)
        cut = int(n * SPLIT)
        pre = s.iloc[:cut]
        pi_pre = pi_pct_of(pre, robust)
        pi_full = pi_pct_of(s, robust)
        cls_pre, cls_full = classify(pi_pre), classify(pi_full)
        margin = min(abs(pi_pre - TL), abs(pi_pre - TH))
        rec = {"n_points": n, "n_pretest": cut,
               "pi_pretest_pct": round(pi_pre, 4),
               "pi_full_pct": round(pi_full, 4),
               "class_pretest": cls_pre, "class_full": cls_full,
               "class_unchanged": cls_pre == cls_full,
               "margin_to_nearest_threshold_pp": round(margin, 4)}
        if name in ("Room302", "BuildingC"):
            w_new = max(1.0, 1.5 - 0.8 * pi_pre / 100)
            w_old = max(1.0, 1.5 - 0.8 * pi_full / 100)
            rec["w_pretest"] = round(w_new, 4)
            rec["w_full"] = round(w_old, 4)
        rows[name] = rec
        print(f"{name:12s} pre={pi_pre:7.4f}%  full={pi_full:7.4f}%  "
              f"{cls_pre:18s} unchanged={cls_pre == cls_full}  "
              f"margin={margin:.2f}pp" +
              (f"  w={rec.get('w_pretest')}" if "w_pretest" in rec else ""))
    out["scenarios"] = rows
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(json.loads(json.dumps(out, default=_py)), f,
                  ensure_ascii=False, indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
