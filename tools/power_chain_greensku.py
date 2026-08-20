#!/usr/bin/env python3
"""
Extract and validate multi-layer power chain parameters from GreenSKU framework.

GreenSKU (ISCA 2024) is Microsoft's server carbon model that provides detailed
component-level power breakdowns for Azure cloud servers. We extract Layer 2
(server-level) and Layer 3 (rack/DC-level) parameters for our multi-layer
power chain model in Section III.

Data source: Azure/AzurePublicDataset GreenSKU-Framework
Paper: "Designing Cloud Servers for Lower Carbon" (ISCA 2024)

Output:
  - greensku_power_params.json: Structured parameter table
  - greensku_power_model.svg: Fan power model and server power breakdown
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.utils import get_logger, results_path, save_json
from common.config import DATA_RAW
from common.plotting import setup_journal_style, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = get_logger("power_chain_greensku")

# ── Paths ──────────────────────────────────────────────────────────────
# GreenSKU ISCA'24 artifact: clone https://github.com/Azure/GreenSKU-Framework
# into data/greensku/GreenSKU-Framework.
GREENSKU_ROOT = os.path.join(DATA_RAW, "greensku", "GreenSKU-Framework")
CARBON_DATA = os.path.join(GREENSKU_ROOT, "data/carbon_data")
OTHER_DATA = os.path.join(GREENSKU_ROOT, "data/other_data")
SERVER_CONFIGS = os.path.join(GREENSKU_ROOT, "server_configs/Eval-Configs")

CASE_NAME = "case10c_cluster_forecast"


def load_yaml(path):
    """Load YAML file using ruamel.yaml or PyYAML."""
    try:
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        with open(path) as f:
            return yaml.load(f)
    except ImportError:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def extract_params():
    """Extract all power chain parameters from GreenSKU data files."""
    logger.info("Loading GreenSKU data files...")

    # Load all data sources
    params = load_yaml(os.path.join(CARBON_DATA, "params.yaml"))
    cpu_data = load_yaml(os.path.join(CARBON_DATA, "CPU.yaml"))
    dram_data = load_yaml(os.path.join(CARBON_DATA, "DRAM.yaml"))
    ssd_data = load_yaml(os.path.join(CARBON_DATA, "SSD.yaml"))
    nic_data = load_yaml(os.path.join(CARBON_DATA, "NIC.yaml"))
    server_data = load_yaml(os.path.join(CARBON_DATA, "server.yaml"))
    rack_data = load_yaml(os.path.join(CARBON_DATA, "rack.yaml"))
    dc_data = load_yaml(os.path.join(CARBON_DATA, "data_center.yaml"))
    fan_power_df = pd.read_csv(os.path.join(OTHER_DATA, "server_fan_power.csv"))

    # Load server configs
    baseline_config = load_yaml(os.path.join(SERVER_CONFIGS, "Baseline.yaml"))
    greensku_config = load_yaml(os.path.join(SERVER_CONFIGS, "GreenSKU-Full.yaml"))

    # ── Layer 2: Server-Level Parameters ────────────────────────────────

    # CPU power model
    # GreenSKU uses spec_derates: a cubic-fit derate curve mapping SPECint% -> power fraction
    # At spec=40 (typical Azure allocation), derate ~0.44
    amd_genoa = cpu_data[0]  # AMD vendor
    spec_derates = amd_genoa['spec_derates']
    spec_points = sorted(spec_derates.keys())
    derate_values = [spec_derates[k] for k in spec_points]

    # Fit cubic to spec_derates for CPU utilization -> power mapping
    spec_x = np.array([float(x) for x in spec_points])
    derate_y = np.array([float(y) for y in derate_values])
    cpu_cubic_coeffs = np.polyfit(spec_x, derate_y, 3)

    # CPU component powers
    genoa_80c = amd_genoa['types'][0]['core_counts'][0]  # Genoa 80-core
    bergamo_128c = amd_genoa['types'][1]['core_counts'][0]  # Bergamo 128-core

    # DRAM power
    ddr5_4800_64gb = dram_data['DDR5']['frequencies'][0]['sizes'][0]
    ddr5_4800_96gb = dram_data['DDR5']['frequencies'][0]['sizes'][1]
    ddr4_2400_32gb = dram_data['DDR4']['frequencies'][0]['sizes'][0]

    # SSD power
    ssd_2tb = ssd_data[0]['sizes'][0]
    ssd_4tb = ssd_data[0]['sizes'][1]

    # NIC power
    nic_100g = nic_data['bandwidths'][0]

    # Server chassis power (motherboard, PSU standby, etc.)
    server_1u = server_data[0]['items']
    server_2u = server_data[1]['items']

    # Fan power model: P_fan = P_fan_base + fan_slope * (P_server - P_server_base)
    fan_slope = float(params['fan_slope'])
    server_base_1u = float(str(params['1U_server_base']).rstrip('W'))
    server_base_2u = float(str(params['2U_server_base']).rstrip('W'))

    def strip_unit(val):
        """Strip unit suffixes (W, kW, MW, etc.) and convert to float."""
        s = str(val)
        for suffix in ['kW', 'MW', 'GW', 'W', 'MHz', 'GB', 'TB']:
            if s.endswith(suffix):
                multiplier = {'kW': 1000, 'MW': 1e6, 'GW': 1e9}.get(suffix, 1)
                return float(s[:-len(suffix)]) * multiplier
        return float(s)

    # Verify fan slope from empirical data
    logger.info("Verifying fan slope from empirical data...")
    fan_df = fan_power_df.copy()
    fan_df.columns = [c.strip() for c in fan_df.columns]
    if 'server power' in fan_df.columns and 'fan power' in fan_df.columns:
        x_fan = fan_df['server power'].values
        y_fan = fan_df['fan power'].values
        fan_fit = np.polyfit(x_fan, y_fan, 1)
        logger.info(f"  Empirical fan model: P_fan = {fan_fit[1]:.2f} + {fan_fit[0]:.4f} * P_server")
        logger.info(f"  GreenSKU fan_slope parameter: {fan_slope:.5f}")
        fan_r2 = 1 - np.sum((y_fan - np.polyval(fan_fit, x_fan))**2) / np.sum((y_fan - np.mean(y_fan))**2)
        logger.info(f"  Linear fit R^2: {fan_r2:.4f}")
    else:
        fan_fit = [fan_slope, 0]
        fan_r2 = None

    # VRM overhead
    vrm_overhead = float(params['voltage_regulator_overhead'])

    # PSU efficiency
    psu_efficiency = float(params['PSU_efficiency'])

    # Power factor (fraction of allocated power actually used)
    power_factor = float(params['power_factor'])

    # ── Layer 3: Rack/DC-Level Parameters ───────────────────────────────

    # PUE
    pue = float(params['PUE'])

    # Rack infrastructure
    rack_info = rack_data[0]['items']
    rack_power = strip_unit(rack_info['rack']['power'])

    # Rack configuration from server config
    baseline_rack = baseline_config['server']['rack']
    rack_capacity_kw = strip_unit(baseline_rack['power'])
    rack_capacity_u = int(str(baseline_rack['capacity']).rstrip('U'))

    # DC configuration
    dc_info = dc_data[0]
    dc_rack_capacity = int(dc_info['rack_capacity'])
    dc_power_capacity_w = strip_unit(dc_info['power_capacity'])
    dc_power_capacity_mw = dc_power_capacity_w / 1e6  # convert W to MW

    # ── Compute Baseline Server Power Breakdown ─────────────────────────

    logger.info("Computing Baseline (Genoa 80C) server power breakdown at spec=40...")
    spec = 40
    # Derate at spec=40 from the data
    derate_at_40 = float(spec_derates[40])

    cpu_power_rated = strip_unit(genoa_80c['power'])
    cpu_power_derated = cpu_power_rated * derate_at_40
    cpu_power_with_vrm = cpu_power_derated * vrm_overhead

    n_dimm = int(baseline_config['server']['memory']['number'])
    dram_power_per_dimm = strip_unit(ddr5_4800_64gb['power'])
    dram_power_total = n_dimm * dram_power_per_dimm * derate_at_40

    n_ssd = int(baseline_config['server']['ssd']['number'])
    ssd_power_per_drive = strip_unit(ssd_2tb['power'])
    ssd_power_total = n_ssd * ssd_power_per_drive * derate_at_40

    n_nic = int(baseline_config['server']['nic']['number'])
    nic_power_per_card = strip_unit(nic_100g['power'])
    nic_power_total = n_nic * nic_power_per_card * derate_at_40

    chassis_power = strip_unit(server_1u['server']['power'])  # Motherboard etc.
    fan_power_base = strip_unit(server_1u['fan']['power'])
    fan_power_derated = fan_power_base * derate_at_40

    # Total IT power before PSU
    it_power_before_fan = cpu_power_with_vrm + dram_power_total + ssd_power_total + nic_power_total + chassis_power
    # Fan power adjusted by fan slope model
    fan_power_adjusted = fan_power_base + fan_slope * (it_power_before_fan - server_base_1u)
    fan_power_adjusted_derated = fan_power_adjusted * derate_at_40

    server_power_it = it_power_before_fan + fan_power_adjusted_derated
    server_power_wall = server_power_it * (1 + (1 - psu_efficiency))

    logger.info(f"  CPU (Genoa 80C, derated@40): {cpu_power_with_vrm:.1f} W (incl. VRM)")
    logger.info(f"  DRAM ({n_dimm}x DDR5-4800 64GB): {dram_power_total:.1f} W")
    logger.info(f"  SSD ({n_ssd}x E1.S 2TB): {ssd_power_total:.1f} W")
    logger.info(f"  NIC ({n_nic}x 100G): {nic_power_total:.1f} W")
    logger.info(f"  Chassis (MB, etc.): {chassis_power:.1f} W")
    logger.info(f"  Fan (slope-adjusted): {fan_power_adjusted_derated:.1f} W")
    logger.info(f"  Server IT total: {server_power_it:.1f} W")
    logger.info(f"  Server wall (incl. PSU loss): {server_power_wall:.1f} W")

    # ── Build Parameter JSON ────────────────────────────────────────────

    result = {
        "metadata": {
            "source": "GreenSKU Carbon Model (ISCA 2024)",
            "paper": "Designing Cloud Servers for Lower Carbon",
            "authors": "Microsoft Azure",
            "repository": "https://github.com/Azure/AzurePublicDataset",
            "data_path": GREENSKU_ROOT,
            "description": "Multi-layer power chain parameters extracted from GreenSKU framework for Layer 2 (server) and Layer 3 (rack/DC) modeling"
        },
        "layer2_server": {
            "cpu_power_model": {
                "description": "CPU power scales with SPECint allocation via cubic derate curve: P_CPU(spec) = P_TDP * derate(spec). The derate curve maps spec% [10-100] to power fraction [0.2-0.95].",
                "parameters": {
                    "genoa_80c_tdp": {
                        "value": strip_unit(genoa_80c['power']),
                        "unit": "W",
                        "source": "Tom's Hardware AMD EPYC 9654 review",
                        "note": "AMD EPYC Genoa, 80 cores, typical Azure baseline SKU"
                    },
                    "bergamo_128c_tdp": {
                        "value": strip_unit(bergamo_128c['power']),
                        "unit": "W",
                        "source": "Phoronix AMD EPYC 9754 review",
                        "note": "AMD EPYC Bergamo, 128 cores, GreenSKU cloud-native SKU"
                    },
                    "spec_derate_curve": {
                        "type": "cubic_polynomial",
                        "coefficients": cpu_cubic_coeffs.tolist(),
                        "note": "Coefficients [a3, a2, a1, a0] for derate(spec) = a3*spec^3 + a2*spec^2 + a1*spec + a0",
                        "source": "SPECpower_ssj2008 benchmark data (Kasture & Sanchez, ISCA 2016)"
                    },
                    "spec_derate_table": {k: float(v) for k, v in spec_derates.items()},
                    "typical_allocation_spec": {
                        "value": 40,
                        "unit": "SPECint%",
                        "note": "Typical Azure server allocation level"
                    },
                    "derate_at_spec40": {
                        "value": derate_at_40,
                        "unit": "dimensionless",
                        "note": "At spec=40, power is 44% of TDP"
                    }
                }
            },
            "vrm_model": {
                "description": "Voltage Regulator Module adds overhead to CPU power delivery. P_CPU_actual = P_CPU * eta_VRM",
                "parameters": {
                    "vrm_overhead_factor": {
                        "value": vrm_overhead,
                        "unit": "dimensionless",
                        "source": "DigiKey VRM efficiency article",
                        "note": "1.05 means 5% power loss in VRM, applied to CPU power only"
                    }
                }
            },
            "dram_power": {
                "description": "Per-DIMM power for DDR5/DDR4 memory. Total DRAM power = N_DIMM * P_per_DIMM * derate(spec).",
                "parameters": {
                    "ddr5_4800_64gb": {
                        "value": strip_unit(ddr5_4800_64gb['power']),
                        "unit": "W/DIMM",
                        "source": "Tom's Hardware Intel i7-5960X review (DRAM power measurement)",
                        "note": "Active power per 64GB DDR5-4800 DIMM"
                    },
                    "ddr5_4800_96gb": {
                        "value": strip_unit(ddr5_4800_96gb['power']),
                        "unit": "W/DIMM",
                        "source": "Tom's Hardware (same methodology)",
                        "note": "Active power per 96GB DDR5-4800 DIMM"
                    },
                    "ddr4_2400_32gb": {
                        "value": strip_unit(ddr4_2400_32gb['power']),
                        "unit": "W/DIMM",
                        "source": "Tom's Hardware",
                        "note": "Used for CXL-attached memory in GreenSKU-CXL config"
                    },
                    "baseline_dimm_count": {
                        "value": n_dimm,
                        "unit": "DIMMs",
                        "note": "12x DDR5-4800 64GB in Baseline config = 768 GB total"
                    },
                    "dram_derate_curve": {
                        "note": "Same cubic derate as CPU: maps SPECint% to power fraction",
                        "source": "SPECpower_ssj2008"
                    }
                }
            },
            "ssd_power": {
                "description": "Per-drive SSD power. Total SSD power = N_SSD * P_per_SSD * derate(spec).",
                "parameters": {
                    "e1s_2tb": {
                        "value": strip_unit(ssd_2tb['power']),
                        "unit": "W/drive",
                        "source": "Seagate Nytro 3530 sustainability report"
                    },
                    "e1s_4tb": {
                        "value": strip_unit(ssd_4tb['power']),
                        "unit": "W/drive",
                        "source": "Seagate Nytro 3530 sustainability report"
                    },
                    "baseline_ssd_count": {
                        "value": n_ssd,
                        "unit": "drives",
                        "note": "6x E1.S 2TB in Baseline config"
                    }
                }
            },
            "nic_power": {
                "description": "Network interface card power.",
                "parameters": {
                    "nic_100g": {
                        "value": strip_unit(nic_100g['power']),
                        "unit": "W/NIC",
                        "source": "Microsoft Research, Azure Accelerated Networking"
                    },
                    "baseline_nic_count": {
                        "value": n_nic,
                        "unit": "NICs"
                    }
                }
            },
            "chassis_power": {
                "description": "Server chassis overhead (motherboard, BMC, PSU standby). Lump sum estimate.",
                "parameters": {
                    "server_1u_chassis": {
                        "value": strip_unit(server_1u['server']['power']),
                        "unit": "W",
                        "source": "Azure internal estimate",
                        "note": "Motherboard, BMC, PSU standby for 1U server"
                    },
                    "server_2u_chassis": {
                        "value": strip_unit(server_2u['server']['power']),
                        "unit": "W",
                        "source": "Azure internal estimate",
                        "note": "Motherboard, BMC, PSU standby for 2U server"
                    }
                }
            },
            "fan_power_model": {
                "description": "Fan power is linear in total server IT power: P_fan = P_fan_base + fan_slope * (P_IT - P_server_base). The slope captures thermal coupling.",
                "parameters": {
                    "fan_slope": {
                        "value": fan_slope,
                        "unit": "W_fan/W_IT",
                        "source": "ServeTheHome server power deep dive data",
                        "note": "Derived from 8-point empirical fit of server vs fan power"
                    },
                    "fan_linear_fit_intercept": {
                        "value": float(fan_fit[1]),
                        "unit": "W",
                        "source": "Linear regression on server_fan_power.csv",
                        "note": "Direct linear fit: P_fan = intercept + slope * P_server"
                    },
                    "fan_linear_fit_slope": {
                        "value": float(fan_fit[0]),
                        "unit": "W_fan/W_server",
                        "source": "Linear regression on server_fan_power.csv"
                    },
                    "fan_linear_fit_r2": {
                        "value": float(fan_r2) if fan_r2 is not None else None,
                        "unit": "dimensionless"
                    },
                    "fan_power_1u_base": {
                        "value": strip_unit(server_1u['fan']['power']),
                        "unit": "W",
                        "source": "TechTarget server efficiency article",
                        "note": "Base fan power for 1U server (4 fans)"
                    },
                    "fan_power_2u_base": {
                        "value": strip_unit(server_2u['fan']['power']),
                        "unit": "W",
                        "source": "Dell LCA (1.5x of 1U)",
                        "note": "Base fan power for 2U server (6 fans)"
                    },
                    "server_base_power_1u": {
                        "value": server_base_1u,
                        "unit": "W",
                        "note": "Reference server power for fan slope calculation (1U)"
                    },
                    "server_base_power_2u": {
                        "value": server_base_2u,
                        "unit": "W",
                        "note": "Reference server power for fan slope calculation (2U)"
                    },
                    "empirical_data_points": {
                        "server_power_W": fan_df['server power'].tolist() if 'server power' in fan_df.columns else [],
                        "fan_power_W": fan_df['fan power'].tolist() if 'fan power' in fan_df.columns else [],
                        "fan_fraction": fan_df['fan/server'].tolist() if 'fan/server' in fan_df.columns else [],
                        "source": "server_fan_power.csv from ServeTheHome"
                    }
                }
            },
            "psu_model": {
                "description": "Power Supply Unit efficiency. Server wall power = P_IT / eta_PSU (GreenSKU uses P_wall = P_IT * (1 + (1 - eta_PSU))).",
                "parameters": {
                    "psu_efficiency": {
                        "value": psu_efficiency,
                        "unit": "dimensionless",
                        "source": "Corsair PSU efficiency ratings (80+ Titanium class)",
                        "note": "95% efficiency at typical load. Equivalent to 5% power loss."
                    },
                    "psu_loss_formula": {
                        "description": "P_wall = P_IT * (1 + (1 - eta_PSU)) = P_IT * 1.05",
                        "note": "GreenSKU applies PSU loss as multiplicative overhead on total server IT power"
                    }
                }
            },
            "power_factor": {
                "description": "Fraction of allocated power actually consumed (accounts for stranded capacity).",
                "parameters": {
                    "power_factor": {
                        "value": power_factor,
                        "unit": "dimensionless",
                        "source": "Microsoft Research, SmartOClock (ISCA)",
                        "note": "0.66 means servers typically use 66% of their allocated power budget"
                    }
                }
            },
            "server_power_breakdown_baseline": {
                "description": "Example power breakdown for Baseline config (AMD Genoa 80C, 1U, spec=40)",
                "cpu_power_W": round(cpu_power_with_vrm, 1),
                "dram_power_W": round(dram_power_total, 1),
                "ssd_power_W": round(ssd_power_total, 1),
                "nic_power_W": round(nic_power_total, 1),
                "chassis_power_W": round(chassis_power, 1),
                "fan_power_W": round(fan_power_adjusted_derated, 1),
                "server_it_total_W": round(server_power_it, 1),
                "server_wall_total_W": round(server_power_wall, 1)
            }
        },
        "layer3_rack_dc": {
            "pue_model": {
                "description": "Power Usage Effectiveness: P_facility = P_IT * PUE. PUE includes cooling, lighting, UPS losses, etc.",
                "parameters": {
                    "pue": {
                        "value": pue,
                        "unit": "dimensionless",
                        "source": "Microsoft Azure sustainability blog",
                        "note": "Azure fleet-wide average PUE = 1.12 (industry-leading). Cooling overhead = 12%."
                    },
                    "pue_breakdown": {
                        "cooling_overhead": round(pue - 1.0, 2),
                        "unit": "fraction of IT power",
                        "note": "PUE = 1.12 implies 12% overhead for cooling, UPS, lighting, etc."
                    }
                }
            },
            "rack_infrastructure": {
                "description": "Per-rack infrastructure power (ToR switch, PDU, cabling).",
                "parameters": {
                    "rack_infra_power": {
                        "value": rack_power,
                        "unit": "W",
                        "source": "Azure internal estimate",
                        "note": "Top-of-rack switch, power distribution, monitoring"
                    },
                    "rack_power_capacity": {
                        "value": rack_capacity_kw,
                        "unit": "W",
                        "source": "GreenSKU Baseline config",
                        "note": "15 kW per rack is typical Azure rack power budget"
                    },
                    "rack_space_capacity": {
                        "value": rack_capacity_u,
                        "unit": "U",
                        "source": "GreenSKU Baseline config",
                        "note": "42U standard rack"
                    }
                }
            },
            "dc_configuration": {
                "description": "Data center scale parameters.",
                "parameters": {
                    "dc_rack_capacity": {
                        "value": dc_rack_capacity,
                        "unit": "racks",
                        "source": "Azure estimate",
                        "note": "2500 racks per data center"
                    },
                    "dc_power_capacity": {
                        "value": dc_power_capacity_mw,
                        "unit": "MW",
                        "source": "Azure estimate",
                        "note": "50 MW per data center"
                    }
                }
            },
            "carbon_intensity_by_region": {
                "description": "Grid carbon intensity varies by Azure region, affecting operational emissions.",
                "note": "See azure_dc_data.csv for per-region values. Range: 0.026-0.208 kgCO2e/kWh.",
                "default_emissions_factor": {
                    "value": float(params['emissions_factor']),
                    "unit": "kgCO2e/kWh",
                    "note": "Default for calculations; overridden per-region in practice"
                }
            }
        },
        "multi_layer_power_equations": {
            "description": "Summary of power chain equations using GreenSKU parameters",
            "layer1_gpu": "P_GPU = a * f^b (from Case 9 DVFS measurements, not in GreenSKU)",
            "layer2_server": {
                "equation": "P_server = [N_CPU * P_CPU(spec) * eta_VRM + N_DIMM * P_DRAM(spec) + N_SSD * P_SSD(spec) + N_NIC * P_NIC(spec) + P_chassis + P_fan(P_IT)] * (1 + (1 - eta_PSU))",
                "where": {
                    "P_CPU(spec)": "P_TDP * derate(spec), derate is cubic polynomial of SPECint%",
                    "P_fan(P_IT)": "P_fan_base + fan_slope * (P_IT_no_fan - P_server_base)",
                    "eta_VRM": f"{vrm_overhead} (5% overhead on CPU only)",
                    "eta_PSU": f"{psu_efficiency} (5% loss on total IT power)"
                }
            },
            "layer3_rack": {
                "equation": "P_rack = N_servers * P_server + P_rack_infra",
                "note": f"Rack infra = {rack_power} W (ToR switch, PDU)"
            },
            "layer3_dc": {
                "equation": "P_DC = (N_racks * P_rack + P_DC_infra) * PUE",
                "note": f"PUE = {pue}, includes cooling overhead"
            }
        }
    }

    return result, fan_df, spec_x, derate_y, cpu_cubic_coeffs


def plot_greensku_models(fan_df, spec_x, derate_y, cpu_cubic_coeffs):
    """Generate validation figures for GreenSKU power models."""
    setup_journal_style()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    # ── (a) Fan power vs server power ──────────────────────────────────
    ax = axes[0]
    x_fan = fan_df['server power'].values
    y_fan = fan_df['fan power'].values

    ax.scatter(x_fan, y_fan, s=30, color='#2196F3', zorder=3, label='Empirical data')

    # Linear fit
    fit_coeffs = np.polyfit(x_fan, y_fan, 1)
    x_fit = np.linspace(min(x_fan) * 0.9, max(x_fan) * 1.05, 100)
    y_fit = np.polyval(fit_coeffs, x_fit)
    r2 = 1 - np.sum((y_fan - np.polyval(fit_coeffs, x_fan))**2) / np.sum((y_fan - np.mean(y_fan))**2)

    ax.plot(x_fit, y_fit, 'r-', linewidth=1.2, label=f'Linear fit ($R^2$={r2:.3f})')
    ax.set_xlabel('Server Power (W)')
    ax.set_ylabel('Fan Power (W)')
    ax.set_title('(a) Fan Power Model')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Annotate equation
    ax.text(0.95, 0.15,
            f'$P_{{fan}}$ = {fit_coeffs[1]:.1f} + {fit_coeffs[0]:.3f}$P_{{server}}$',
            transform=ax.transAxes, fontsize=7, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # ── (b) SPECint derate curve ───────────────────────────────────────
    ax = axes[1]
    ax.scatter(spec_x, derate_y, s=30, color='#4CAF50', zorder=3, label='GreenSKU data')

    x_smooth = np.linspace(5, 105, 200)
    y_smooth = np.polyval(cpu_cubic_coeffs, x_smooth)
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=1.2, label='Cubic fit')

    ax.axvline(x=40, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0.44, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.scatter([40], [0.44], s=60, color='red', marker='*', zorder=4, label='Typical (spec=40)')

    ax.set_xlabel('SPECint Allocation (%)')
    ax.set_ylabel('Power Derate Factor')
    ax.set_title('(b) CPU Power Derate Curve')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 1.05)

    # ── (c) Server power breakdown (stacked bar) ──────────────────────
    ax = axes[2]

    # Compute breakdown for spec = 20, 40, 60, 80, 100
    spec_levels = [20, 40, 60, 80, 100]
    spec_derate_map = {10: 0.2, 20: 0.28, 30: 0.36, 40: 0.44, 50: 0.53,
                       60: 0.62, 70: 0.7, 80: 0.79, 90: 0.88, 100: 0.95}

    cpu_vals, dram_vals, ssd_vals, nic_vals, chassis_vals, fan_vals, psu_vals = [], [], [], [], [], [], []
    for spec in spec_levels:
        d = spec_derate_map[spec]
        cpu_p = 290 * d * 1.05  # Genoa 80C * derate * VRM
        dram_p = 12 * 23.68 * d
        ssd_p = 6 * 11.2 * d
        nic_p = 1 * 19 * d
        chassis_p = 35.0
        it_no_fan = cpu_p + dram_p + ssd_p + nic_p + chassis_p
        fan_p = (75 + 0.179 * (it_no_fan - 250)) * d
        it_total = it_no_fan + fan_p
        psu_loss = it_total * 0.05

        cpu_vals.append(cpu_p)
        dram_vals.append(dram_p)
        ssd_vals.append(ssd_p)
        nic_vals.append(nic_p)
        chassis_vals.append(chassis_p)
        fan_vals.append(fan_p)
        psu_vals.append(psu_loss)

    x_pos = np.arange(len(spec_levels))
    width = 0.55
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1', '#757575']
    labels = ['CPU+VRM', 'DRAM', 'SSD', 'NIC', 'Chassis', 'Fan', 'PSU loss']

    bottom = np.zeros(len(spec_levels))
    for vals, color, label in zip(
        [cpu_vals, dram_vals, ssd_vals, nic_vals, chassis_vals, fan_vals, psu_vals],
        colors, labels
    ):
        ax.bar(x_pos, vals, width, bottom=bottom, color=color, label=label, edgecolor='white', linewidth=0.3)
        bottom += np.array(vals)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{s}%' for s in spec_levels])
    ax.set_xlabel('SPECint Allocation')
    ax.set_ylabel('Power (W)')
    ax.set_title('(c) Server Power Breakdown')
    ax.legend(fontsize=6, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = results_path(CASE_NAME, "figures", "greensku_power_model")
    save_figure(fig, output_path)
    logger.info(f"Saved figure to {output_path}.svg")


def main():
    logger.info("=" * 60)
    logger.info("GreenSKU Power Chain Parameter Extraction")
    logger.info("=" * 60)

    # Extract parameters
    result, fan_df, spec_x, derate_y, cpu_cubic_coeffs = extract_params()

    # Save JSON
    json_path = results_path(CASE_NAME, "data", "greensku_power_params.json")
    save_json(result, json_path)
    logger.info(f"Saved parameters to {json_path}")

    # Generate figures
    plot_greensku_models(fan_df, spec_x, derate_y, cpu_cubic_coeffs)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Key parameters for multi-layer power chain")
    logger.info("=" * 60)
    logger.info("Layer 2 (Server):")
    logger.info(f"  VRM overhead:     {result['layer2_server']['vrm_model']['parameters']['vrm_overhead_factor']['value']} (5% on CPU)")
    logger.info(f"  PSU efficiency:   {result['layer2_server']['psu_model']['parameters']['psu_efficiency']['value']} (5% loss on total)")
    logger.info(f"  Fan slope:        {result['layer2_server']['fan_power_model']['parameters']['fan_slope']['value']:.5f} W_fan/W_IT")
    logger.info(f"  Power factor:     {result['layer2_server']['power_factor']['parameters']['power_factor']['value']} (66% utilization)")
    logger.info("Layer 3 (Rack/DC):")
    logger.info(f"  PUE:              {result['layer3_rack_dc']['pue_model']['parameters']['pue']['value']}")
    logger.info(f"  Rack infra:       {result['layer3_rack_dc']['rack_infrastructure']['parameters']['rack_infra_power']['value']} W")
    logger.info(f"  Rack capacity:    {result['layer3_rack_dc']['rack_infrastructure']['parameters']['rack_power_capacity']['value']} W")
    logger.info(f"  DC capacity:      {result['layer3_rack_dc']['dc_configuration']['parameters']['dc_power_capacity']['value']} MW")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
