"""
顶刊级绘图模块 —— Nature / IEEE TPWRS / TSG / TII / Applied Energy 风格。

输出: 1200 DPI SVG 矢量图 + 300 DPI PNG 预览。
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from .config import (
    FIGURE_DPI, FIGURE_FORMAT, FIGURE_FORMATS, PNG_DPI,
    COLORS, PHASE_COLORS,
)


# ── 全局样式设置 ─────────────────────────────────────────

def setup_journal_style():
    """设置顶刊级 matplotlib 全局样式。"""
    # 优先使用 Times New Roman，回退到 serif
    plt.rcParams.update({
        # 大数据集渲染
        "agg.path.chunksize": 10000,
        "path.simplify_threshold": 0.1,
        # 字体
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # 线条
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        # 刻度
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        # 图例
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        # 布局
        "figure.dpi": 150,  # 屏幕预览
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # 数学文本
        "mathtext.fontset": "stix",
    })


# 模块导入时自动设置样式
setup_journal_style()


# ── 保存工具 ─────────────────────────────────────────────

def save_figure(fig, filepath: str, dpi: int = FIGURE_DPI, close: bool = True):
    """
    保存图表为 SVG + PNG 双格式。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : 输出路径 (不含扩展名，或含 .svg/.png)
    dpi : SVG 等效分辨率
    close : 保存后是否关闭 figure
    """
    base = os.path.splitext(filepath)[0]
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    # SVG (矢量，DPI 对矢量图无实际影响，但设置元数据)
    svg_path = base + ".svg"
    fig.savefig(svg_path, format="svg", dpi=dpi, bbox_inches="tight")

    # PNG (栅格预览)
    png_path = base + ".png"
    fig.savefig(png_path, format="png", dpi=PNG_DPI, bbox_inches="tight")

    if close:
        plt.close(fig)

    return svg_path


# ── 功耗时序图 ───────────────────────────────────────────

def plot_power_timeseries(
    t, P,
    title: str = "GPU Power Consumption",
    xlabel: str = "Elapsed Time (s)",
    ylabel: str = "Power (W)",
    color: str = None,
    phases: list = None,
    annotations: list = None,
    stats_box: bool = True,
    figsize: tuple = (7, 2.8),
    output_path: str = None,
):
    """
    绘制功耗时序图。适用于 Fig6/9/11/13/14/15。

    Parameters
    ----------
    t : 时间序列
    P : 功率序列
    phases : 阶段标注列表，每项 {"start": float, "end": float, "label": str, "color": str}
    annotations : 标注箭头列表，每项 {"x": float, "y": float, "text": str}
    stats_box : 是否显示统计量文本框
    output_path : 输出路径 (不含扩展名)
    """
    t = np.asarray(t)
    P = np.asarray(P)
    color = color or COLORS["primary"]

    fig, ax = plt.subplots(figsize=figsize)

    # 阶段彩色背景 (如 Figure 13)
    if phases:
        for phase in phases:
            ax.axvspan(
                phase["start"], phase["end"],
                alpha=0.3, color=phase.get("color", "#e0e0e0"),
                label=phase.get("label"),
            )

    # 大数据集降采样以避免渲染溢出 (>200k点)
    if len(t) > 200000:
        step = max(1, len(t) // 200000)
        t_plot, P_plot = t[::step], P[::step]
    else:
        t_plot, P_plot = t, P
    ax.plot(t_plot, P_plot, color=color, linewidth=0.6, rasterized=len(t_plot) > 50000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True)

    # 统计量文本框
    if stats_box and len(P) > 0:
        valid = P[np.isfinite(P)]
        stats_text = (
            f"Peak: {np.max(valid):.1f} W\n"
            f"Mean: {np.mean(valid):.1f} W\n"
            f"Std: {np.std(valid):.1f} W"
        )
        ax.text(
            0.98, 0.97, stats_text,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9),
        )

    # 标注箭头 (如 Power Surge)
    if annotations:
        for ann in annotations:
            ax.annotate(
                ann["text"],
                xy=(ann["x"], ann["y"]),
                xytext=(ann.get("tx", ann["x"]), ann.get("ty", ann["y"] * 1.1)),
                fontsize=7,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.6),
                ha="center",
            )

    if phases:
        ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig, ax


# ── CDF 图 ───────────────────────────────────────────────

def plot_power_cdf(
    sorted_P, cdf_values,
    title: str = "Power Consumption CDF",
    xlabel: str = "Power (W)",
    ylabel: str = "CDF",
    figsize: tuple = (4.5, 3.2),
    output_path: str = None,
):
    """CDF 图。对应 Figure 7。"""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(sorted_P, cdf_values, color=COLORS["primary"], linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True)
    fig.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig, ax


# ── 变化率 PDF 图 ────────────────────────────────────────

def plot_rate_pdf(
    x, pdf,
    title: str = "GPU Power Ramping Rate (PDF)",
    xlabel: str = "Ramping Rate (W/s)",
    ylabel: str = "Probability Density",
    figsize: tuple = (4.5, 3.2),
    output_path: str = None,
):
    """变化率 PDF 图。对应 Figure 8。"""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, pdf, color=COLORS["primary"], linewidth=1.0)
    ax.fill_between(x, pdf, alpha=0.15, color=COLORS["primary"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig, ax


# ── 瞬态放大对比图 ───────────────────────────────────────

def plot_transient_zoom(
    t, P,
    max_decline,
    max_ramp,
    title: str = "Power Transients",
    window_s: float = 30.0,
    figsize: tuple = (7, 2.8),
    output_path: str = None,
    power_unit: str = "W",
):
    """
    左右双图放大展示最大下降和最大上升瞬态。对应 Fig10/12。

    Parameters
    ----------
    max_decline, max_ramp : TransientEvent 对象 (from analysis.detect_transients)
    window_s : 放大窗口宽度 (秒，瞬态事件前后各取一半)
    power_unit : 功率单位 ("W" 或 "kW")，用于标注和Y轴标签
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, event, label, color in [
        (ax1, max_decline, "Max Decline", COLORS["quaternary"]),
        (ax2, max_ramp, "Max Ramp", COLORS["tertiary"]),
    ]:
        if event is None:
            ax.text(0.5, 0.5, "No event detected",
                    transform=ax.transAxes, ha="center")
            ax.set_title(label)
            continue

        center = (event.start_time + event.end_time) / 2
        mask = (t >= center - window_s / 2) & (t <= center + window_s / 2)
        t_win = t[mask]
        P_win = P[mask]

        if len(P_win) == 0:
            # 如果窗口内无数据，扩大到事件前后索引范围
            idx_center = (event.start_idx + event.end_idx) // 2
            margin = max(50, (event.end_idx - event.start_idx) * 3)
            idx_start = max(0, idx_center - margin)
            idx_end = min(len(t), idx_center + margin)
            t_win = t[idx_start:idx_end]
            P_win = P[idx_start:idx_end]

        if len(P_win) == 0:
            ax.text(0.5, 0.5, "No data in window",
                    transform=ax.transAxes, ha="center")
            ax.set_title(label)
            continue

        ax.plot(t_win, P_win, color=color, linewidth=0.8)
        ax.axvspan(event.start_time, event.end_time, alpha=0.15, color=color)
        ax.set_xlabel("Elapsed Time (s)")
        ax.set_title(f"{label}: {event.delta_power_w:+.1f} {power_unit}")
        ax.grid(True)

        # 标注变化量
        dp = event.delta_power_w
        rate = event.rate_w_s
        rate_unit = f"{power_unit}/s"
        label_str = f"\u0394{dp:+.1f} {power_unit}\n({rate:+.1f} {rate_unit})"
        ax.annotate(
            label_str,
            xy=(center, (P_win.max() + P_win.min()) / 2),
            fontsize=7, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="0.7", alpha=0.9),
        )

    ax1.set_ylabel(f"Power ({power_unit})")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig, (ax1, ax2)


# ── 批量推理对比图 ───────────────────────────────────────

def plot_batch_comparison(
    segments: list,
    title: str = "Mamba-2.8B vs GPT-Neo-2.7B Inference Power",
    ylabel: str = "Power (W)",
    figsize: tuple = (8, 3.0),
    output_path: str = None,
):
    """
    交替色块对比图。对应 Figure 16。

    Parameters
    ----------
    segments : 列表，每项:
        {"t": array, "P": array, "model": "mamba"|"gptneo",
         "batch_size": int, "oom": bool}
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_colors = {
        "mamba": COLORS["mamba"],
        "gptneo": COLORS["transformer"],
    }
    model_labels = {"mamba": "Mamba-2.8B", "gptneo": "GPT-Neo-2.7B"}

    # 拼接所有段到连续时间轴
    offset = 0
    bs_positions = []  # 用于标注 batch_size
    drawn_labels = set()

    for seg in segments:
        t_local = np.asarray(seg["t"]) - seg["t"][0] + offset
        P = np.asarray(seg["P"])
        model = seg["model"]
        bs = seg["batch_size"]
        color = model_colors.get(model, "#888888")
        label = model_labels.get(model) if model not in drawn_labels else None
        drawn_labels.add(model)

        # 背景色块
        ax.axvspan(t_local[0], t_local[-1], alpha=0.2, color=color)
        # 功耗曲线
        ax.plot(t_local, P, color=color, linewidth=0.5, label=label)

        if seg.get("oom"):
            mid_t = (t_local[0] + t_local[-1]) / 2
            ax.text(mid_t, ax.get_ylim()[1] * 0.5, "OOM",
                    ha="center", fontsize=7, color="red", fontweight="bold")

        # 记录 batch_size 标注位置
        bs_positions.append({
            "x": (t_local[0] + t_local[-1]) / 2,
            "bs": bs,
            "model_char": "M" if model == "mamba" else "T",
        })

        offset = t_local[-1] + 1  # 1秒间隔

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, axis="y")

    # 底部标注 batch_size（每个 batch size 只标一次，取同 bs 段的中点）
    from collections import OrderedDict
    bs_groups = OrderedDict()
    for pos in bs_positions:
        bs = pos["bs"]
        if bs not in bs_groups:
            bs_groups[bs] = []
        bs_groups[bs].append(pos["x"])
    for bs, xs in bs_groups.items():
        mid_x = (min(xs) + max(xs)) / 2
        ax.text(mid_x, ax.get_ylim()[0] - 5, f"bs={bs}",
                ha="center", fontsize=6, va="top", color="0.4")

    fig.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig, ax


# ── 多面板系统仪表盘 ─────────────────────────────────────

def plot_system_dashboard(
    df: pd.DataFrame,
    title: str = "System Monitoring Dashboard",
    figsize: tuple = (8, 8),
    output_path: str = None,
):
    """
    6面板仪表盘: GPU功率/GPU温度/GPU利用率+显存/CPU利用率/内存/磁盘IO。

    Parameters
    ----------
    df : SystemMonitor 输出的 DataFrame (含 elapsed_s 列)
    """
    t = df["elapsed_s"].values

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)

    panels = [
        # (ax_idx, y_col, ylabel, color, title)
        ((0, 0), "gpu_power_w", "Power (W)", COLORS["primary"], "GPU Power"),
        ((0, 1), "gpu_temp_c", "Temp (°C)", COLORS["quaternary"], "GPU Temperature"),
        ((1, 0), "gpu_util_pct", "Util (%)", COLORS["tertiary"], "GPU Utilization"),
        ((1, 1), "gpu_mem_used_mb", "Memory (MB)", COLORS["secondary"], "GPU Memory Used"),
        ((2, 0), "cpu_util_pct", "Util (%)", COLORS["tertiary"], "CPU Utilization"),
        ((2, 1), "mem_used_gb", "Memory (GB)", COLORS["secondary"], "System Memory Used"),
    ]

    for (r, c), col, ylabel, color, panel_title in panels:
        ax = axes[r][c]
        if col in df.columns:
            y = df[col].values
            ax.plot(t, y, color=color, linewidth=0.5)
            ax.set_ylabel(ylabel)
        ax.set_title(panel_title, fontsize=9)
        ax.grid(True)

    axes[2][0].set_xlabel("Elapsed Time (s)")
    axes[2][1].set_xlabel("Elapsed Time (s)")

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig, axes
