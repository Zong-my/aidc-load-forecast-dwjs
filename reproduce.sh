#!/usr/bin/env bash
# =============================================================================
# One-click reproduction of the paper's public-data results.
#
#   bash reproduce.sh              # everything: build -> theory -> train ->
#                                  # analysis -> figures
#   bash reproduce.sh <stage>      # one stage: env|build|theory|train|
#                                  # analysis|figures
#
# Prerequisite: download the four public datasets into data/ first
# (see data/README.md).
#
# Stages are idempotent: a stage whose key output already exists is skipped.
# Delete the corresponding file under results/ (or data_processed/) to force
# a rerun. Full run from raw data: several CPU-hours (MIT build) plus roughly
# 15-20 GPU-hours on a single modern GPU.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
RES=results/case10c_cluster_forecast/data
ANA=results/analysis
LOGDIR=results/logs
mkdir -p "$LOGDIR"

say()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
have() { [ -e "$1" ]; }

# run <key-output> <logname> <cmd...>: skip if key output exists, else run.
run() {
  local out="$1" log="$2"; shift 2
  if have "$out"; then echo "   [skip] $out exists"; return 0; fi
  echo "   [run ] $* (log: $LOGDIR/$log.log)"
  "$@" 2>&1 | tee "$LOGDIR/$log.log"
  have "$out" || { echo "ERROR: expected output $out was not produced"; exit 1; }
}

stage_env() {
  say "stage 0: environment check"
  $PY - <<'EOF'
import importlib.util
import sys
need = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "sklearn", "lightgbm", "torch"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing packages: {missing} — pip install -r requirements.txt")
import torch
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"cuda available: {torch.cuda.is_available()}")
EOF
}

stage_build() {
  say "stage 1: build 15-min datasets from raw downloads (CPU)"
  run data_processed/mit_supercloud_15min.csv      build_mit       $PY experiments/build_dataset.py
  run data_processed/helios_saturn_15min.csv       build_helios    $PY experiments/build_multi_datasets.py
  run data_processed/alibaba_v2026_spot_15min.csv  build_alibaba   $PY experiments/build_alibaba_v2026.py
  run data_processed/inference_cluster_15min.csv   build_inference $PY experiments/build_inference_cluster.py
  run data_processed/azure_lmm_15min.csv           build_azure_lmm $PY experiments/build_azure_lmm.py
}

stage_theory() {
  say "stage 2: queueing-theory analysis (CPU)"
  run "$RES/queueing_params_mit.json"          queueing_mit    $PY experiments/queueing_theory_analysis.py
  run "$RES/queueing_params_helios.json"       queueing_helios $PY experiments/queueing_theory_helios.py
  run "$RES/queueing_theory_validation.json"   queueing_valid  $PY experiments/queueing_theory_validation.py
}

stage_train() {
  say "stage 3: model training, seed 42 (GPU)"
  run "$RES/feature_comparison_results_Inference.json" feat_comp \
      $PY experiments/run_feature_comparison.py --dataset all
  run "$RES/traditional_baseline_results_Inference.json" trad_base \
      $PY experiments/run_traditional_baselines.py --dataset all
  run "$RES/sspm_all_backbones_results.json" sspm_all \
      $PY experiments/run_sspm_all_backbones.py
  run "$RES/sspm_ablation_aligned_results.json" sspm_ablation \
      $PY experiments/run_sspm_ablation.py
  say "stage 3b: pi-guided interval calibration (CPU, post-hoc)"
  run "$RES/s2_fixed_results.json" pi_calibration \
      $PY experiments/recompute_pi_calibration.py
}

stage_analysis() {
  say "stage 4: revision analyses (CPU parts)"
  run "$ANA/pretest_pi_results.json"               pretest_pi   $PY analysis/pretest_pi.py
  run "$ANA/pretest_stl_audit_results.json"        pretest_stl  $PY analysis/pretest_stl_audit.py
  run "$ANA/pretest_robustness_audit_results.json" pretest_rob  $PY analysis/pretest_robustness_audit.py
  run "$ANA/threshold_scan_results.json"           reanalysis \
      $PY analysis/reanalysis_suite.py --tasks threshold,cov,rolling
  say "stage 4b: revision analyses (GPU parts)"
  run "$ANA/alpha_sensitivity_results.json"        alpha_scan   $PY analysis/alpha_sensitivity_scan.py
  run "$ANA/clean_cqr_v2/clean_cqr_v2_results.json" clean_cqr   $PY analysis/clean_cqr_protocol_v2.py
  run "$ANA/multiseed/seed46_sspm_results.json"    multiseed    $PY analysis/multiseed_rerun.py
  run "$ANA/multiseed_aggregate.json"              aggregate    $PY analysis/aggregate_multiseed.py
}

stage_figures() {
  say "stage 5: paper figures"
  $PY figures/make_figs_cn.py 2>&1 | tee "$LOGDIR/figures.log"
}

STAGE="${1:-all}"
case "$STAGE" in
  env)      stage_env ;;
  build)    stage_env; stage_build ;;
  theory)   stage_theory ;;
  train)    stage_train ;;
  analysis) stage_analysis ;;
  figures)  stage_figures ;;
  all)      stage_env; stage_build; stage_theory; stage_train; stage_analysis; stage_figures ;;
  *) echo "unknown stage: $STAGE (use env|build|theory|train|analysis|figures|all)"; exit 2 ;;
esac
say "done: $STAGE"
