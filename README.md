# aidc-load-forecast-hve

This repository ships **code only** — no datasets and no precomputed results.
It covers everything needed to reproduce the paper's public-data results:
the periodicity index π and its queueing-theory grounding, the two-stage SSPM
training strategy (masked self-supervised pretraining + temporal-Mixup
fine-tuning) on three backbones (DLinear / PatchTST / iTransformer), the
π-guided cold-start interval-widening rule, the four-segment CQR protocol,
multi-seed statistics, and the public-data figures. The paper's field
validation uses proprietary measured data and is outside the reproduction
scope — see [field/README.md](field/README.md).

## Reproducing the paper

```bash
# 1. environment (Python >= 3.10, one CUDA GPU)
pip install -r requirements.txt

# 2. download the 4 public datasets into data/  (links & layout: data/README.md)

# 3. one-click pipeline
bash reproduce.sh
```

`reproduce.sh` runs, in order (each stage skips work whose outputs exist):

| stage | what | compute | ~time* |
|---|---|---|---|
| build | raw traces → four 15-min datasets (`data_processed/`) | CPU | minutes–hours (MIT: ~1.5 TB scan) |
| theory | M/G/k spectral-attenuation validation | CPU | minutes |
| train | 3 backbones × 3 feature sets × 4 datasets × 2 horizons: baselines, SSPM, ablation (seed 42) | GPU | ~4–6 h |
| analysis | pre-test π audits, threshold/rolling scans, α-sensitivity, four-segment CQR, multi-seed rerun (seeds 43–46) | GPU | ~9–10 h |
| figures | paper figures 2, 4–9 | CPU | minutes |

\* measured on one NVIDIA RTX PRO 6000; a 12-GB consumer GPU works with
similar wall-times. All experiments are seeded; small cross-GPU numeric
drift in trained metrics is normal.

## Code map

| paper item | script |
|---|---|
| dataset construction (§4.1) | `experiments/build_dataset.py` (MIT), `build_multi_datasets.py` (Helios), `build_alibaba_v2026.py`, `build_inference_cluster.py`, `build_azure_lmm.py` |
| queueing-theory validation (Fig 6) | `experiments/queueing_theory_{analysis,helios,validation}.py` |
| backbones + training loop | `experiments/run_baselines.py` (DLinear/PatchTST/iTransformer), `train_msra.py` (seeding/loaders) |
| baseline matrix (Table 2, Figs 7/9) | `experiments/run_feature_comparison.py`, `run_traditional_baselines.py` |
| SSPM two-stage training (§3.2, Table 2, Fig 7) | `experiments/run_sspm_all_backbones.py`, `run_sspm_ablation.py` |
| π-guided interval widening (Fig 8) | `experiments/recompute_pi_calibration.py` |
| pre-test π & threshold audits (Table 1, §2.2) | `analysis/pretest_pi.py`, `pretest_stl_audit.py`, `pretest_robustness_audit.py`, `reanalysis_suite.py` |
| α-sensitivity (§3.2/§4.3) | `analysis/alpha_sensitivity_scan.py` |
| four-segment CQR protocol (Table 3) | `analysis/clean_cqr_protocol_v2.py` |
| multi-seed statistics (Table 2) | `analysis/multiseed_rerun.py`, `aggregate_multiseed.py` |
| figures 2, 4–9 | `figures/make_figs_cn.py` |
| power-model constants derivation (Zeus / GreenSKU) | `tools/power_chain_zeus.py`, `tools/power_chain_greensku.py` (optional; constants are already embedded in `build_inference_cluster.py`) |

Figures 1 and 3 are TikZ drawings inside the manuscript (no data). Figures
10–11 and the field-validation tables belong to the proprietary field part
(excluded; see `field/README.md`); Figure 9 regenerates with its four public
points.

## Repository layout

```
common/         path configuration (repo-relative, env-overridable) + utilities
data/           raw public datasets — user-downloaded, see data/README.md
experiments/    dataset builders, queueing theory, backbones, SSPM training
analysis/       revision-stage analyses (CQR protocol, multi-seed, audits)
field/          README documenting the excluded proprietary field part
figures/        figure generator
tools/          optional power-model constant derivations
```

Paths are configured centrally in `common/config.py` and can be relocated
with `AIDC_DATA_RAW`, `AIDC_DATA_PROCESSED`, and `AIDC_RESULTS`.

## Data licensing

Code: MIT (see LICENSE). The raw datasets are downloaded by the user from
their original hosts and remain under their own terms — MIT SuperCloud
(CC BY-NC-ND 4.0), Azure traces (CC-BY 4.0), Helios (CC-BY 4.0), Alibaba per
its repository; none are redistributed here.

## Citation

The paper is under review; a citation entry will be added upon publication.
