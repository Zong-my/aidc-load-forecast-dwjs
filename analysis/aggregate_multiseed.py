#!/usr/bin/env python3
"""聚合 5 种子（42=既有结果库 + 43..46=multiseed 重跑）并计算实测侧 CQR。

产出 results/analysis/multiseed_aggregate.json 与终端摘要，供修订稿数字更新：
  A. 公开数据集：逐种子按验证损失选择的最优基线（现代骨干9配置内）与最优SSPM
     的测试 R²、Δ，5种子 mean±std 与符号一致性。
  B. 同配置无退化核验：逐种子 72 个（ds×H×config）SSPM−baseline 精确 ΔR² 的
     提高/持平/回退计数与最小值。
  C. 实测：逐种子（43..46）基线族/SSPM族按验证损失选骨干的测试 R²、Δ、被选骨
     干；302机房24h回退是否跨种子复现；固定 PatchTST 的同骨干 Δ。
  D. 实测 CQR（新 npz 含 q*_val）：SSPM 选中骨干的 npz，校准=验证段后半
     （窗×步池化，有限样本修正 ceil(0.8*(n+1))），对比 raw / w(π)=1.334 / CQR
     的测试段 PICP 与平均宽度，按种子 mean±std。
只读 results 库与 multiseed 输出；仅写 results/analysis/multiseed_aggregate.json。
"""
import glob
import json
import math
import os
import sys

import numpy as np

# ------------------------------------------------------------------
# Path setup: this script lives in analysis/, so both the repo root (for
# common.*) and experiments/ are put on sys.path, plus the analysis dir
# itself so reanalysis_suite (a sibling module) is importable. Derived from
# __file__ -- never hardcoded.
# ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "experiments"),
           os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.config import RESULTS_DIR                       # noqa: E402
from reanalysis_suite import results_data_path              # noqa: E402

# Seed42 results library (produced by the training stage of reproduce.sh)
RESULTS_DATA = os.path.join(RESULTS_DIR, "case10c_cluster_forecast", "data")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "results", "analysis")
MS = os.path.join(ANALYSIS_DIR, "multiseed")
OUT = os.path.join(ANALYSIS_DIR, "multiseed_aggregate.json")

DATASETS = ["MIT", "Helios", "Alibaba", "Inference"]
HORIZONS = ["16", "96"]
SEEDS_NEW = [43, 44, 45, 46]
W_FIELD = 1.334  # w(pi=0.207)


def ms_public_path(fname):
    """Seed 43-46 public JSON produced by multiseed_rerun.py."""
    return os.path.join(MS, fname)


def ms_field_path(fname):
    """Per-seed field JSON (only present when the proprietary field pipeline ran)."""
    return os.path.join(MS, fname)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def lib_suffix(ds):
    return "" if ds == "MIT" else f"_{ds}"


def load_public(seed):
    """返回 (baseline, sspm)：{ds:{H:{config:rec}}}。"""
    if seed == 42:
        base = {ds: jload(results_data_path(
            f"feature_comparison_results{lib_suffix(ds)}.json"))
            for ds in DATASETS}
        sspm_all = jload(results_data_path("sspm_all_backbones_results.json"))
        sspm = {ds: sspm_all[ds] for ds in DATASETS}
        return base, sspm
    base_f = jload(ms_public_path(f"seed{seed}_baseline_results.json"))
    sspm_f = jload(ms_public_path(f"seed{seed}_sspm_results.json"))
    base_f.pop("meta", None)
    sspm_f.pop("meta", None)
    return base_f, sspm_f


def best_by_val(rows):
    k = min((k for k, v in rows.items() if v.get("val_loss") is not None),
            key=lambda k: rows[k]["val_loss"])
    return k, rows[k]


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return round(m, 4), round(s, 4)


def part_a_b():
    sel = {ds: {H: {"base": [], "sspm": [], "delta": [],
                    "base_cfg": [], "sspm_cfg": []}
                for H in HORIZONS} for ds in DATASETS}
    counts = {}
    for seed in [42] + SEEDS_NEW:
        base, sspm = load_public(seed)
        imp = tie = reg = 0
        min_d = math.inf
        for ds in DATASETS:
            for H in HORIZONS:
                bt = base[ds][H]
                st = sspm[ds][H]
                bk, bv = best_by_val(bt)
                sk, sv = best_by_val(st)
                sel[ds][H]["base"].append(bv["r2"])
                sel[ds][H]["sspm"].append(sv["r2"])
                sel[ds][H]["delta"].append(round(sv["r2"] - bv["r2"], 4))
                sel[ds][H]["base_cfg"].append(f"s{seed}:{bk}")
                sel[ds][H]["sspm_cfg"].append(f"s{seed}:{sk}")
                for cfg, srec in st.items():
                    if cfg not in bt:
                        continue
                    d = srec["r2"] - bt[cfg]["r2"]
                    min_d = min(min_d, d)
                    if d > 0:
                        imp += 1
                    elif d < 0:
                        reg += 1
                    else:
                        tie += 1
        counts[f"seed{seed}"] = {"improved": imp, "tied": tie,
                                 "regressed": reg,
                                 "min_delta": round(min_d, 4)}
    agg = {}
    for ds in DATASETS:
        agg[ds] = {}
        for H in HORIZONS:
            e = sel[ds][H]
            bm, bs = mean_std(e["base"])
            sm, ss = mean_std(e["sspm"])
            dm, dsd = mean_std(e["delta"])
            agg[ds][H] = {
                "base_r2_seeds": e["base"], "sspm_r2_seeds": e["sspm"],
                "delta_seeds": e["delta"],
                "base_mean_std": [bm, bs], "sspm_mean_std": [sm, ss],
                "delta_mean_std": [dm, dsd],
                "delta_pos_seeds": sum(1 for d in e["delta"] if d > 0),
                "base_cfg": e["base_cfg"], "sspm_cfg": e["sspm_cfg"]}
    return agg, counts


def part_c():
    out = {}
    for seed in SEEDS_NEW:
        f = jload(ms_field_path(f"seed{seed}_field_results.json"))
        f.pop("meta", None)
        for ds, drec in f.items():
            for hk in ["H16", "H96"]:
                e = drec.get(hk, {})
                bases = {k: v for k, v in e.items()
                         if k.endswith("-Cal") and not k.startswith("SSPM")}
                sspms = {k: v for k, v in e.items() if k.startswith("SSPM-")}
                if not bases or not sspms:
                    continue
                bk, bv = best_by_val(bases)
                sk, sv = best_by_val(sspms)
                d = out.setdefault(ds, {}).setdefault(hk, {
                    "base": [], "sspm": [], "delta": [], "base_sel": [],
                    "sspm_sel": [], "patchtst_delta": []})
                d["base"].append(bv["r2"])
                d["sspm"].append(sv["r2"])
                d["delta"].append(round(sv["r2"] - bv["r2"], 4))
                d["base_sel"].append(bk.replace("-Cal", ""))
                d["sspm_sel"].append(sk.replace("SSPM-", "")
                                     .replace("-Cal", ""))
                pb = e.get("PatchTST-Cal")
                ps = e.get("SSPM-PatchTST-Cal")
                if pb and ps:
                    d["patchtst_delta"].append(round(ps["r2"] - pb["r2"], 4))
    for ds in out:
        for hk in out[ds]:
            e = out[ds][hk]
            e["delta_mean_std"] = mean_std(e["delta"])
            e["n_regress"] = sum(1 for x in e["delta"] if x < 0)
            e["patchtst_delta_mean_std"] = mean_std(e["patchtst_delta"])
    return out


def cqr_field_one(npz_path):
    d = np.load(npz_path)
    need = ["q10", "q50", "q90", "true", "q10_val", "q90_val", "true_val"]
    if any(k not in d for k in need):
        return None
    # 校准段 = 验证段后半（窗口时间有序）
    n_val = d["true_val"].shape[0]
    lo = n_val // 2
    q10c, q90c = d["q10_val"][lo:], d["q90_val"][lo:]
    yc = d["true_val"][lo:]
    scores = np.maximum(q10c - yc, yc - q90c).ravel()
    n = scores.size
    k = min(n, int(math.ceil(0.8 * (n + 1))))
    Q = float(np.sort(scores)[k - 1])
    q10, q50, q90, y = d["q10"], d["q50"], d["q90"], d["true"]

    def stat(a, b):
        return (float(((y >= a) & (y <= b)).mean()),
                float((b - a).mean()))

    raw_c, raw_w = stat(q10, q90)
    wl, wh = q50 + (q10 - q50) * W_FIELD, q50 + (q90 - q50) * W_FIELD
    rule_c, rule_w = stat(wl, wh)
    cq_c, cq_w = stat(q10 - Q, q90 + Q)
    return {"Q": round(Q, 4), "n_cal": int(n),
            "raw": [round(raw_c, 4), round(raw_w, 3)],
            "rule": [round(rule_c, 4), round(rule_w, 3)],
            "cqr": [round(cq_c, 4), round(cq_w, 3)]}


def part_d(field_sel):
    out = {}
    for seed in SEEDS_NEW:
        npz_dir = os.path.join(MS, f"seed{seed}_field_npz")
        f = jload(ms_field_path(f"seed{seed}_field_results.json"))
        f.pop("meta", None)
        for ds in f:
            for hk in ["H16", "H96"]:
                sels = field_sel.get(ds, {}).get(hk, {}).get("sspm_sel", [])
                idx = seed - 43
                if idx >= len(sels):
                    continue
                m = sels[idx]
                H = hk[1:]
                p = os.path.join(npz_dir, f"{ds}_H{H}_{m}_sspm.npz")
                if not os.path.exists(p):
                    continue
                r = cqr_field_one(p)
                if r is None:
                    continue
                d = out.setdefault(ds, {}).setdefault(hk, {
                    "raw_c": [], "rule_c": [], "cqr_c": [],
                    "raw_w": [], "rule_w": [], "cqr_w": [], "Q": []})
                d["raw_c"].append(r["raw"][0])
                d["rule_c"].append(r["rule"][0])
                d["cqr_c"].append(r["cqr"][0])
                d["raw_w"].append(r["raw"][1])
                d["rule_w"].append(r["rule"][1])
                d["cqr_w"].append(r["cqr"][1])
                d["Q"].append(r["Q"])
    for ds in out:
        for hk in out[ds]:
            e = out[ds][hk]
            for k in ["raw_c", "rule_c", "cqr_c", "raw_w", "rule_w", "cqr_w"]:
                e[k + "_ms"] = mean_std(e[k])
    return out


def field_npz_available():
    """True iff at least one seed4x_field_npz dir is present under MS."""
    return any(os.path.isdir(os.path.join(MS, f"seed{seed}_field_npz"))
               for seed in SEEDS_NEW)


def field_json_available():
    """True iff every per-seed field results JSON is resolvable."""
    return all(os.path.exists(ms_field_path(f"seed{seed}_field_results.json"))
               for seed in SEEDS_NEW)


def main():
    agg, counts = part_a_b()

    # The paper's field-validation part uses proprietary measured data that is
    # not distributed with this repository; parts C/D run only when a user
    # with authorized data has produced the per-seed field results.
    field = cqr = None
    field_npz_recomputed = False
    if field_json_available():
        field = part_c()
        if field_npz_available():
            cqr = part_d(field)
            field_npz_recomputed = True
    else:
        print("[field] per-seed field results absent — the proprietary field "
              "part is outside the open-source reproduction scope "
              "(see field/README.md); aggregating public sections only")

    res = {"public_selection": agg, "same_config_counts": counts,
           "meta": {"seeds": [42] + SEEDS_NEW,
                    "field_seeds": SEEDS_NEW,
                    "w_field": W_FIELD,
                    "cal": "val latter half, pooled scores, "
                           "ceil(0.8*(n+1)) order stat"}}
    if field is not None:
        res["field_selection"] = field
    if cqr is not None:
        res["field_cqr"] = cqr
        res["field_npz_recomputed"] = field_npz_recomputed
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("== B 同配置计数（逐种子）==")
    for k, v in counts.items():
        print(f"  {k}: +{v['improved']} ={v['tied']} -{v['regressed']} "
              f"min_d={v['min_delta']}")
    print("== A 24h(H96) 验证集选择（5种子）==")
    for ds in DATASETS:
        e = agg[ds]["96"]
        print(f"  {ds:10s} base={e['base_mean_std']} "
              f"sspm={e['sspm_mean_std']} d={e['delta_mean_std']} "
              f"d>0:{e['delta_pos_seeds']}/5")
    if field is not None:
        print("== C 实测（4种子）==")
        for ds in field:
            for hk in field[ds]:
                e = field[ds][hk]
                print(f"  {ds} {hk}: d={e['delta_mean_std']} "
                      f"回退{e['n_regress']}/4 sel={e['sspm_sel']} "
                      f"PatchTST_d={e['patchtst_delta_mean_std']}")
    if cqr is not None:
        print("== D 实测 CQR（4种子均值：覆盖率/宽度）==")
        for ds in cqr:
            for hk in cqr[ds]:
                e = cqr[ds][hk]
                print(f"  {ds} {hk}: raw={e['raw_c_ms'][0]:.3f}/"
                      f"{e['raw_w_ms'][0]:.2f} rule={e['rule_c_ms'][0]:.3f}/"
                      f"{e['rule_w_ms'][0]:.2f} cqr={e['cqr_c_ms'][0]:.3f}/"
                      f"{e['cqr_w_ms'][0]:.2f}")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
