# Raw public datasets

This directory is intentionally empty in the repository. Download the four
public datasets below and arrange them exactly as shown; the build scripts in
`experiments/` then derive the 15-min series used throughout the paper into
`data_processed/`.

## Expected layout

```
data/
├── mit_supercloud/
│   └── datacenter-challenge/202201/
│       ├── gpu/                    # ~120k per-GPU CSV files (≈1.5 TB)
│       └── slurm-log.csv
├── helios/
│   └── data/Saturn/
│       ├── cluster_log.csv
│       └── cluster_gpu_number.csv
├── alibaba/
│   └── cluster-trace-v2026-spot-gpu/
│       ├── job_info_df.csv
│       └── node_info_df.csv
├── azure_llm/
│   └── data/
│       ├── AzureLLMInferenceTrace_conv.csv
│       ├── AzureLLMInferenceTrace_code.csv
│       └── AzureLMMInferenceTrace_multimodal.csv.gz
└── field/                          # proprietary — see field/README.md
```

The root can be relocated with the environment variable `AIDC_DATA_RAW`.

## 1. MIT SuperCloud Datacenter Challenge (training cluster)

- Home: <https://dcc.mit.edu> (download instructions: <https://dcc.mit.edu/data/>)
- Hosted on AWS Open Data; download the `202201` release, e.g.:

  ```bash
  aws s3 sync --no-sign-request \
      s3://mit-supercloud-dataset/datacenter-challenge/202201/ \
      data/mit_supercloud/datacenter-challenge/202201/
  ```

  Only `gpu/` and `slurm-log.csv` are required (≈1.5 TB of the ~2 TB release).
- License: see the LICENSE file in the bucket (CC BY-NC-ND 4.0 at the time of
  writing) — which is why this repository ships the *builder script* rather
  than the derived CSV. Cite the SuperCloud paper (Samsi et al., 2021) per
  the dataset's terms.

## 2. Helios / SenseTime Saturn (hybrid training cluster)

- Repository: <https://github.com/S-Lab-System-Group/HeliosData>
- Clone the repository into `data/helios/` and unzip its `data.zip`
  (36 MB compressed) so that `data/helios/data/Saturn/cluster_log.csv` and
  `cluster_gpu_number.csv` exist. (`experiments/queueing_theory_helios.py`
  can also auto-extract `data.zip` if it is present.)
- License: CC-BY-4.0; cite the SC'21 Helios paper (doi:10.1145/3458817.3476223).

## 3. Alibaba spot-GPU trace (mixed training/inference cluster)

- Repository: <https://github.com/alibaba/clusterdata>,
  subdirectory `cluster-trace-v2026-spot-gpu`
- Needed files: `job_info_df.csv`, `node_info_df.csv`
- Follow the repository's usage terms.

## 4. Azure LLM/LMM inference traces (inference service)

- Repository: <https://github.com/Azure/AzurePublicDataset>
- Needed files (see `AzureLLMInferenceDataset2023.md` and
  `AzureLMMInferenceDataset2025.md` in that repo):
  - `data/AzureLLMInferenceTrace_conv.csv`
  - `data/AzureLLMInferenceTrace_code.csv`
  - `data/AzureLMMInferenceTrace_multimodal.csv.gz`
- License: CC-BY 4.0.

The inference-cluster dataset used in the paper is **semi-synthetic**: real
Azure arrival patterns are mapped through the Zeus (NSDI'23) V100 power law
and the GreenSKU (ISCA'24) server power chain to a 256-GPU cluster kW series
(`experiments/build_inference_cluster.py`; the fitted constants are embedded
in the script, and `tools/power_chain_*.py` documents how they were derived).

## Integrity quick-check

After building (`bash reproduce.sh build`), `data_processed/` should contain:

| file | rows | target column |
|---|---|---|
| mit_supercloud_15min.csv | 22,509 | cluster_power_kw |
| helios_saturn_15min.csv | 19,607 | active_gpu_count |
| alibaba_v2026_spot_15min.csv | 17,673 | active_gpu_count |
| inference_cluster_15min.csv | 11,520 | cluster_power_kw |
| azure_lmm_15min.csv | 672 | request_count (figure input only) |

