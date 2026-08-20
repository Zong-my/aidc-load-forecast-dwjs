"""Global configuration: repository-relative paths, dataset locations, plotting defaults.

All paths are derived from the repository root and can be overridden with
environment variables so the pipeline runs on any machine:

    AIDC_DATA_RAW        root of the raw public datasets   (default: <repo>/data)
    AIDC_DATA_PROCESSED  built 15-min CSV datasets         (default: <repo>/data_processed)
    AIDC_RESULTS         experiment outputs                (default: <repo>/results)
"""

import os

# ── Repository-relative roots ─────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Kept as an alias because many scripts import PROJECT_ROOT.
PROJECT_ROOT = REPO_ROOT

DATA_RAW       = os.environ.get("AIDC_DATA_RAW",       os.path.join(REPO_ROOT, "data"))
DATA_PROCESSED = os.environ.get("AIDC_DATA_PROCESSED", os.path.join(REPO_ROOT, "data_processed"))
RESULTS_DIR    = os.environ.get("AIDC_RESULTS",        os.path.join(REPO_ROOT, "results"))

# ── Raw public dataset locations (see data/README.md for download links) ──
# MIT SuperCloud Datacenter Challenge, 202201 release
SUPERCLOUD_DIR = os.path.join(DATA_RAW, "mit_supercloud", "datacenter-challenge", "202201")
# Helios (SenseTime) GPU cluster traces — HeliosData repository checkout
HELIOS_TRACE_DIR = os.path.join(DATA_RAW, "helios")
# Alibaba cluster traces — clusterdata repository checkout
ALIBABA_TRACE_DIR = os.path.join(DATA_RAW, "alibaba")
# Azure LLM inference traces — AzurePublicDataset files
AZURE_LLM_DIR = os.path.join(DATA_RAW, "azure_llm")

# ── Field (proprietary) data placeholder ──────────────────────────────────
# The measured AIDC-campus data used in the paper's field validation are NOT
# redistributable (see field/README.md).  Users with authorized access can
# place the derived CSVs here to re-enable the field stages; otherwise those
# stages are skipped automatically.
FIELD_DATA_DIR = os.path.join(DATA_RAW, "field")

# ── Plotting defaults ─────────────────────────────────────────────────────
FIGURE_DPI    = 1200
FIGURE_FORMAT = "svg"
FIGURE_FORMATS = ("svg", "png")   # vector + raster
PNG_DPI       = 300

# Colorblind-friendly palette
COLORS = {
    "primary":    "#1f77b4",  # blue
    "secondary":  "#ff7f0e",  # orange
    "tertiary":   "#2ca02c",  # green
    "quaternary": "#d62728",  # red
    "gray":       "#7f7f7f",
    "light_gray": "#c7c7c7",
    "mamba":      "#b0b0b0",
    "transformer":"#4a90d9",
}

# Training-phase shading (used by common/plotting.py)
PHASE_COLORS = {
    "init":       "#e0e0e0",
    "early":      "#cce5ff",
    "late":       "#fff3cd",
    "shutdown":   "#f8d7da",
}
