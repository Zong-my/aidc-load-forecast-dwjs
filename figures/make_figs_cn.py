#!/usr/bin/env python3
"""（中文标签，PDF+PNG）。

图1  研究架构(framework, tikz/main.tex)  图2  STL分解对比(低π vs 高π)
图3  预测模型I/O与结构(tikz/main.tex)     图4  四类集群日内曲线与ACF
图5  STL方差分解                          图6  排队论衰减指标vs实测π
图7  SSPM同骨干ΔR²热力图                  图8  π引导区间校准效果
图9  π预测适用性分区(含实测场景)          图10 实测场景滚动预测(需训练npz)
图11 制冷负荷-气象耦合(实测)
注：本脚本生成图2、图4~图11；图1(framework)与图3(模型结构)为 tikz、在 main.tex 内，
    勿删 fig1_framework.*。函数名(fig1/fig2...)为历史命名、与图号不对应，以 save() 输出名为准。

用法: python3 figures/make_figs_cn.py [--figs all|fig2,fig7,...]（图号为输出图号）
"""
import argparse
import glob
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

warnings.filterwarnings('ignore')
# 中文宋体(AR PL SungtiL GB) + 英文/数字/数学 Times New Roman。
# 默认在下列系统目录中查找；也可用环境变量 AIDC_FONT_DIR 指定含
# gbsn00lp.ttf / times*.ttf 的目录。缺字体时仍可出图(退回 matplotlib 默认字体)。
_FONT_FILES = ('gbsn00lp.ttf', 'times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf')
_FONT_DIRS = [os.environ.get('AIDC_FONT_DIR'),
              '/usr/share/fonts/truetype/arphic-gbsn00lp',
              '/usr/share/fonts/truetype/msttcorefonts']
_fonts_loaded = 0
for _f in _FONT_FILES:
    for _d in _FONT_DIRS:
        if not _d:
            continue
        _p = os.path.join(_d, _f)
        if os.path.exists(_p):
            try:
                fm.fontManager.addfont(_p)
                _fonts_loaded += 1
                break
            except Exception:
                pass
if _fonts_loaded < len(_FONT_FILES):
    print('[warn] some fonts (gbsn00lp.ttf/times*.ttf) not found; set AIDC_FONT_DIR '
          'to locate them — figures still render with default fonts')
plt.rcParams.update({
    'font.family': ['Times New Roman', 'AR PL SungtiL GB'],  # 逐字回退: 拉丁/数字->Times New Roman, 中文->宋体(AR PL SungtiL GB). 切勿放 .ttc 字体集(会破坏回退)
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'mathtext.cal': 'Times New Roman:italic',
    # 单栏图统一字号(图6热力图除外): 标题/坐标轴名=8, 刻度/图例/图内标注=7
    'font.size': 7,  # ax.text/annotate 默认随此 → 标注统一 7
    'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7, 'figure.dpi': 200,
    'axes.linewidth': 0.5,   # 坐标框线 0.5 磅(刊物要求: 图中线 0.5pt)
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'svg.fonttype': 'path',  # SVG 文本转路径，自含、跨机渲染一致
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.4,
})

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
from common.config import DATA_PROCESSED, RESULTS_DIR, FIELD_DATA_DIR  # noqa: E402


def _find_result(name, producer):
    """定位结果 JSON（由复现管线生成，见 reproduce.sh）。"""
    for d in (os.path.join(RESULTS_DIR, 'case10c_cluster_forecast', 'data'),
              os.path.join(RESULTS_DIR, 'analysis')):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"result file '{name}' not found — generate it with {producer} "
        "(run the pipeline stages in reproduce.sh first)")


def _find_field(name):
    """定位实测(field)结果：论文实测部分不在开源复现范围内（见 field/README.md）。"""
    p = os.path.join(RESULTS_DIR, 'field', name)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(
        f"field artifact '{name}' not found — the proprietary field-validation "
        "part of the paper is outside the open-source reproduction scope "
        "(see field/README.md)")

COLORS = {'MIT': '#1f77b4', 'Helios': '#ff7f0e', 'Alibaba': '#2ca02c', 'Inference': '#d62728'}
CN = {'MIT': 'MIT超算(学术训练型)', 'Helios': 'Helios(工业训练型)',
      'Alibaba': '阿里Spot-GPU(混合负载型)', 'Inference': '推理服务型'}
MARK = {'MIT': 'o', 'Helios': 's', 'Alibaba': '^', 'Inference': 'D'}


def save(fig, stem):
    fig.savefig(os.path.join(HERE, f'{stem}.pdf'))
    fig.savefig(os.path.join(HERE, f'{stem}.png'), dpi=300)
    fig.savefig(os.path.join(HERE, f'{stem}.svg'))
    plt.close(fig)
    print(' saved', stem)


def align_stack(fig, axs):
    """强制竖排子图坐标框左右边界严格对齐(取左界最大、右界最小, 不裁标签)。"""
    fig.canvas.draw()
    ps = [a.get_position() for a in axs]
    L, R = max(p.x0 for p in ps), min(p.x1 for p in ps)
    for a, p in zip(axs, ps):
        a.set_position([L, p.y0, R - L, p.height])


def load_ds():
    out = {}
    for k, f, c in [('MIT', 'mit_supercloud_15min.csv', 'cluster_power_kw'),
                    ('Helios', 'helios_saturn_15min.csv', 'active_gpu_count'),
                    ('Alibaba', 'alibaba_v2026_spot_15min.csv', 'active_gpu_count'),
                    ('Inference', 'inference_cluster_15min.csv', 'cluster_power_kw')]:
        d = pd.read_csv(os.path.join(DATA_PROCESSED, f))
        d['time'] = pd.to_datetime(d['time'], utc=True)
        d['value'] = d[c]
        out[k] = d
    return out


# ================= 图2 STL分解对比(低π vs 高π) =================
def fig_stl_decomp():
    # 标注与分解均采用表1的测试期前历史口径；窗口仅作展示
    stl_audit = json.load(open(_find_result(
        'pretest_stl_audit_results.json', 'analysis/pretest_stl_audit.py')))['scenarios']
    spec = [('MIT', 'mit_supercloud_15min.csv', 'cluster_power_kw', COLORS['MIT'],
             '(a) MIT超算（训练主导，低π）', stl_audit['MIT']['pi_pretest_pct']),
            ('Inference', 'inference_cluster_15min.csv', 'cluster_power_kw',
             COLORS['Inference'], '(b) 推理服务（推理型，高π）',
             stl_audit['Inference']['pi_pretest_pct'])]
    ndays = 10
    t = np.arange(ndays * 96) / 96
    # 各栏按测试期前完整历史的 σ(Y) 归一，且跨栏共用纵轴，
    # 使两栏日季节"高度"可直接比较，避免展示窗波动改变归一化口径。
    # 低π日季节塌成贴零细带(√0.0162≈0.13σ), 高π日季节高耸(√0.823≈0.91σ)
    rows = []
    for k, f, c, col, ttl, pival in spec:
        d = pd.read_csv(os.path.join(DATA_PROCESSED, f))
        y = pd.to_numeric(d[c], errors='coerce').interpolate().dropna().values
        y = y[:int(len(y) * 0.85)]              # 训练+验证段，避免测试信息进入 π
        r = STL(y, period=96, robust=True).fit()
        s0 = len(y) // 3                       # 取中段稳定窗口
        sl = slice(s0, s0 + ndays * 96)
        m = y[sl] - y[sl].mean()
        sd = y.std()                           # 按测试期前完整历史 σ(Y) 归一
        rows.append((col, ttl, pival, m / sd, r.seasonal[sl] / sd, r.resid[sl] / sd))
    amp = 1.1 * max(max(np.abs(a).max() for a in row[3:]) for row in rows)
    fig, axes = plt.subplots(3, 2, figsize=(3.4, 2.8), sharex='col', sharey=True)
    for j, (col, ttl, pival, OBS, SEA, RES) in enumerate(rows):
        a0, a1, a2 = axes[0, j], axes[1, j], axes[2, j]
        a0.set_ylim(-amp, amp)                 # sharey=True → 三行两栏同纵轴
        a1.set_facecolor('#fff3cc')            # 浅黄高亮中间行: 周期性看此行(题注说明)
        for ax in (a0, a1, a2):
            ax.axhline(0, color='0.8', lw=0.4, zorder=0)
        a0.plot(t, OBS, color='0.45', lw=0.5)
        a1.plot(t, SEA, color=col, lw=0.9)
        a1.fill_between(t, SEA, 0, color=col, alpha=0.25)
        # 图例移到面板底部空白区(MIT曲线居中、推理尖峰朝上, 底部均空), 去掉白底框
        a1.text(0.035, 0.06, f'π={pival:.2f}%', transform=a1.transAxes,
                color=col, va='bottom', fontweight='bold')
        a1.text(0.97, 0.06, r'$S_{\mathrm{daily}}$', transform=a1.transAxes,
                color=col, va='bottom', ha='right')
        a2.plot(t, RES, color='0.45', lw=0.4)
        a2.set_xlabel('时间/d')
        a2.set_xlim(0, ndays)
        # (a)/(b) 子图标题置于各列正下方(在 x 轴名之下)
        a2.annotate(ttl, xy=(0.5, 0), xycoords='axes fraction', xytext=(0, -28),
                    textcoords='offset points', ha='center', va='top', fontsize=8)
        if j == 0:
            a0.set_ylabel('观测 /σ')
            a1.set_ylabel('日季节 /σ')
            a2.set_ylabel('残差 /σ')
    fig.tight_layout(h_pad=0.3, w_pad=0.8)
    save(fig, 'fig2_stl_decomp')


# ================= 图3 日内曲线 + ACF =================
def fig1(dfs):
    az = pd.read_csv(os.path.join(DATA_PROCESSED, 'azure_lmm_15min.csv'))
    az['time'] = pd.to_datetime(az['time'], utc=True)
    az['value'] = az['request_count']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.5))   # 单栏(上下两面板)
    for k in COLORS:
        d = az if k == 'Inference' else dfs[k]
        h = d.groupby(d['time'].dt.hour)['value'].mean()
        a1.plot(h.index, h / h.mean(), color=COLORS[k], lw=1.4, label=CN[k])
    a1.axhline(1.0, color='gray', ls='--', lw=0.7, alpha=0.6)
    a1.set_xlabel('时刻/h')
    a1.set_ylabel('负荷/日均值')
    a1.set_xlim(0, 23)
    a1.set_xticks([0, 4, 8, 12, 16, 20])
    a1.annotate('(a) 日内归一化负荷曲线', xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=8)
    for k in COLORS:
        v = dfs[k]['value'].dropna().values
        a2.plot(np.arange(201), acf(v, nlags=200, fft=True), color=COLORS[k], lw=1.1)
    a2.axvline(96, color='gray', ls='--', lw=0.8)
    a2.annotate('24 h', xy=(96, 0.93), xytext=(110, 0.85), color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
    a2.set_xlabel('滞后步数/(×15 min)')
    a2.set_ylabel('自相关系数')
    a2.set_xlim(0, 200)
    a2.annotate('(b) 自相关函数', xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=8)
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    fig.tight_layout(rect=[0, 0.09, 1, 1], h_pad=2.0)   # 面板间距缩小一点
    align_stack(fig, [a1, a2])                          # 上下子图坐标框严格对齐
    save(fig, 'fig4_diurnal_acf')


# ================= 图2 STL方差分解 =================
def fig2():
    names = ['MIT超算', 'Helios', '阿里', '推理服务']
    # 测试期前历史（前85%）的 Var(component)/Var(Y)；不强制归一到100%，
    # 分量之和与100%的偏差由STL交叉协方差项造成。
    audit = json.load(open(_find_result(
        'pretest_stl_audit_results.json', 'analysis/pretest_stl_audit.py')))['scenarios']
    order = ['MIT', 'Helios', 'Alibaba', 'Inference']
    trend = [audit[name]['trend_share_pct'] for name in order]
    seas = [audit[name]['pi_pretest_pct'] for name in order]
    resid = [audit[name]['residual_share_pct'] for name in order]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(4)
    ax.bar(x, trend, 0.52, label='趋势项', color='#4e79a7')
    ax.bar(x, seas, 0.52, bottom=trend, label='日季节项(π)', color='#f28e2b')
    ax.bar(x, resid, 0.52, bottom=[t + s for t, s in zip(trend, seas)],
           label='残差项', color='#76b7b2')
    for i in range(4):
        ax.text(i, trend[i] / 2, f'{trend[i]:.1f}', ha='center', va='center',
                color='white', fontweight='bold')
        if seas[i] > 5:
            ax.text(i, trend[i] + seas[i] / 2, f'{seas[i]:.1f}', ha='center',
                    va='center', color='white', fontweight='bold')
        if resid[i] > 5:
            ax.text(i, trend[i] + seas[i] + resid[i] / 2, f'{resid[i]:.1f}',
                    ha='center', va='center', color='white', fontweight='bold')
    ax.set_ylabel('方差占比/%')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.axhline(100, color='0.35', ls='--', lw=0.6)
    ax.set_ylim(0, 112)
    ax.legend(loc='upper right', frameon=False, ncol=3,
              handlelength=1.0, columnspacing=0.7, handletextpad=0.3)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    save(fig, 'fig5_stl')


# ================= 图3 理论 vs 实测 =================
def fig3():
    data = json.load(open(_find_result(
        'queueing_theory_validation.json', 'experiments/queueing_theory_validation.py')))
    nm = {'MIT Supercloud': 'MIT', 'Helios Saturn': 'Helios',
          'Alibaba PAI': 'Alibaba', 'Inference': 'Inference'}
    short_cn = {'MIT': 'MIT超算', 'Helios': 'Helios', 'Alibaba': '阿里',
                'Inference': '推理服务'}
    ann = {'MIT': (10, -9, 'left', 'top'), 'Helios': (-4, 11, 'center', 'bottom'),
           'Alibaba': (10, 0, 'left', 'center'), 'Inference': (0, -13, 'center', 'top')}
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    # r2 修订：横轴改为无唯象修正的严格导出指标 alpha_d^2/(1+(omega_d*E[S])^2)，
    # 与修订稿式(4)一致；假定排队参数的场景（assumed_params）用空心标记区分。
    omega_d = data['omega_d_rad_per_h']
    # r2 阻断项3修复：实测 π 改为前测历史口径（pretest_pi_results.json）
    PRETEST_PI = {'MIT': 0.016187, 'Helios': 0.056307,
                  'Alibaba': 0.208343, 'Inference': 0.822543}
    for s in data['scenarios']:
        k = nm[s['scenario']]
        x = s['alpha_d'] ** 2 / (1.0 + (omega_d * s['E_S_h']) ** 2)
        y = PRETEST_PI[k]
        assumed = bool(s.get('assumed_params', False))
        ax.scatter(x, y, s=55, marker=MARK[k],
                   facecolors=('none' if assumed else COLORS[k]),
                   edgecolors=(COLORS[k] if assumed else 'k'),
                   linewidths=(1.2 if assumed else 0.5), zorder=5)
        dx, dy, ha, va = ann[k]
        ax.annotate(short_cn[k], (x, y), textcoords='offset points', xytext=(dx, dy),
                    ha=ha, va=va, color=COLORS[k])
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel('理论频谱衰减指标(式(4))')
    ax.set_ylabel('实测周期性指数 π')
    fig.tight_layout()
    save(fig, 'fig6_theory')


# ================= 图4 SSPM ΔR² 热力图 =================
def fig4():
    fcs, sspm = {}, json.load(open(_find_result(
        'sspm_all_backbones_results.json', 'experiments/run_sspm_all_backbones.py')))
    for ds in COLORS:
        suf = '' if ds == 'MIT' else f'_{ds}'
        fcs[ds] = json.load(open(_find_result(
            f'feature_comparison_results{suf}.json', 'experiments/run_feature_comparison.py')))
    models = ['DLinear', 'PatchTST', 'iTransformer']
    fsets = ['Calendar', 'All', 'Workload']
    fs_cn = {'Calendar': '日历', 'All': '全', 'Workload': '负载'}
    rows = [f'{m}-{f}' for m in models for f in fsets]
    row_cn = [f'{m}-{fs_cn[f]}' for m in models for f in fsets]
    cols, mat = [], []
    for ds in COLORS:
        for h in ['16', '96']:
            cols.append(f'{ds.replace("Inference","推理").replace("Alibaba","阿里").replace("Helios","Hel")}\nH={h}')
            mat.append([sspm[ds][h][r]['r2'] - fcs[ds][h][r]['r2'] for r in rows])
    M = np.array(mat).T * 1000  # ×10^-3
    # 单栏: 原生≈栏宽出图(不被缩小→字体不再变小), 纵向压缩(不狭长) (图6不纳入全局统一字号)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    vcap = 60.0  # 色阶截断：个别极大增益（>60）单独标注，避免压扁常规增益
    im = ax.imshow(np.clip(M, 0, vcap), cmap='Greens', vmin=0, vmax=vcap,
                   aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f'{v:.0f}' if abs(v) >= 0.5 else '0', ha='center',
                    va='center', fontsize=7,
                    color='white' if v > 0.6 * vcap else 'black')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6)   # 8 窄列, 略小避免 H=16/H=96 相邻相撞
    ax.set_yticks(range(9))
    ax.set_yticklabels(row_cn, fontsize=6)   # 与横坐标刻度值字号一致
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, extend='max')
    cb.set_label(r'$\Delta R^2\,(\times 10^{-3})$', fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    save(fig, 'fig7_heatmap')


# ================= 图5 π引导区间校准 =================
def fig5():
    # 论文表VII数值（80%名义覆盖）
    rows = [  # (数据集, π, 4h前, 4h后, 24h前, 24h后)
        ('MIT超算', 0.016, 0.744, 0.877, 0.718, 0.874),
        ('Helios', 0.056, 0.797, 0.929, 0.864, 0.951),
        ('阿里', 0.208, 0.776, 0.869, 0.740, 0.854),
        ('推理服务', 0.822, 0.783, 0.783, 0.788, 0.788)]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))   # 恢复原尺寸
    x = np.arange(4)
    w = 0.2
    ax.bar(x - 1.5 * w, [r[4] for r in rows], w, label='24h 调整前', color='#aec7e8')
    ax.bar(x - 0.5 * w, [r[5] for r in rows], w, label='24h 调整后', color='#1f77b4')
    ax.bar(x + 0.5 * w, [r[2] for r in rows], w, label='4h 调整前', color='#ffbb78')
    ax.bar(x + 1.5 * w, [r[3] for r in rows], w, label='4h 调整后', color='#ff7f0e')
    ax.axhline(0.80, color='k', ls='--', lw=1.0)
    ax.text(3.0, 0.82, '名义80%', ha='center')   # 移到推理组上方(其各柱均<0.80, 空白处), 不溢出
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r[0]}\nπ={r[1]:.3f}' for r in rows])
    ax.set_ylabel('经验覆盖率')
    ax.set_ylim(0.55, 1.02)
    ax.legend(frameon=False, ncol=2, loc='upper right',
              handlelength=1.0, columnspacing=0.8)
    fig.tight_layout()
    save(fig, 'fig8_calibration')


# ================= 图6 适用性分区（含实测场景） =================
def fig6():
    # 公开数据集现代骨干基线 R²：骨干×特征配置全局按验证损失选择，
    # 与实测点采用相同协议；不混入按测试 R² 事后选出的传统基线。
    def selected_public(ds, h):
        suffix = '' if ds == 'MIT' else f'_{ds}'
        result = json.load(open(_find_result(
            f'feature_comparison_results{suffix}.json',
            'experiments/run_feature_comparison.py')))[str(h)]
        selected = min(result, key=lambda key: result[key]['val_loss'])
        return result[selected]['r2']

    pub = [  # (名, π, R2_4h, R2_24h)
        (label, pi, selected_public(ds, 16), selected_public(ds, 96))
        for ds, label, pi in [
            ('MIT', 'MIT超算', 0.0162), ('Helios', 'Helios', 0.0563),
            ('Alibaba', '阿里', 0.2083), ('Inference', '推理服务', 0.8225)]]
    # 实测落位点为可选叠加：实测数据/结果不随仓库分发（论文实测部分
    # 不在公开复现范围内），缺失时仅绘制公开数据集分区图。
    field_pts = []
    try:
        vres = json.load(open(_find_field('field_sspm_results.json')))

        def best_base(ds, h):
            d = vres[ds][f'H{h}']
            candidates = {k: v for k, v in d.items()
                          if isinstance(v, dict) and 'val_loss' in v
                          and not k.startswith('SSPM')}
            selected = min(candidates, key=lambda k: candidates[k]['val_loss'])
            return candidates[selected]['r2']

        field_pts = [('Room302(实测)', 0.20019, best_base('Room302', 16),
                      best_base('Room302', 96)),
                     ('楼栋C(实测)', 0.20515, best_base('BuildingC', 16),
                      best_base('BuildingC', 96))]
    except FileNotFoundError:
        print('fig9: field results unavailable -> public-only strategy map '
              '(the paper overlays two proprietary field points here)')
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.set_xscale('log')
    ax.set_xlim(0.008, 1.1)
    ax.set_ylim(0, 1.24)
    ax.axvspan(0.008, 0.15, facecolor='0.95')
    ax.axvspan(0.15, 0.50, facecolor='0.86')
    ax.axvspan(0.50, 1.1, facecolor='0.74')
    for v in (0.15, 0.50):
        ax.axvline(v, color='k', ls='--', lw=0.8)
    ax.axhline(1.04, color='k', lw=0.6)
    for xx, t in [(0.035, '区间I'), (0.27, '区间II'), (0.74, '区间III')]:
        ax.text(xx, 1.20, t, ha='center', va='top')
    for (n, p, r4, r24), k in zip(pub, COLORS):
        ax.plot(p, r24, marker=MARK[k], color=COLORS[k], mfc='white', ms=7, ls='')
        ax.plot(p, r4, marker=MARK[k], color=COLORS[k], ms=4.5, ls='')
    # 标注（公开）
    offs = {'MIT超算': (1.15, -0.09), 'Helios': (1.18, -0.115),
            '阿里': (1.45, -0.04), '推理服务': (0.5, -0.13)}
    for n, p, r4, r24 in pub:
        dx, dy = offs[n]
        ax.annotate(n, (p, r24), xytext=(p * dx, r24 + dy))
    vn_off = {'Room302(实测)': (1.13, 0.035), '楼栋C(实测)': (0.42, -0.085)}
    for n, p, r4, r24 in field_pts:
        ax.plot(p, r24, marker='*', color='crimson', mfc='gold', ms=12, ls='', zorder=6)
        ax.plot(p, r4, marker='*', color='crimson', mfc='white', ms=8, ls='', zorder=6)
        dx, dy = vn_off[n]
        ax.annotate(n, (p, r24), xytext=(p * dx, r24 + dy),
                    color='crimson')
    ax.plot([], [], 'o', color='k', mfc='white', ms=7, label='24h(空心)')
    ax.plot([], [], 'o', color='k', ms=4.5, label='4h(实心)')
    if field_pts:
        ax.plot([], [], '*', color='crimson', mfc='gold', ms=10, label='实测场景')
    ax.set_xlabel('周期性指数 π(对数轴)')
    ax.set_ylabel('验证集选定的现代骨干基线 R²')
    ax.legend(frameon=False, loc='lower right', handlelength=1.2)
    fig.tight_layout()
    save(fig, 'fig9_strategy_map')


# ================= 图7 制冷-气象耦合 =================
def fig7():
    # 原始输入为专有实测数据(不随仓库分发)：
    #   FIELD_DATA_DIR/cooling/  下的楼栋C小时级用电量 Excel(sheet '总表')
    #   FIELD_DATA_DIR/weather_hourly.csv  站点小时级气象
    cool_dir = os.path.join(FIELD_DATA_DIR, 'cooling')
    wx_csv = os.path.join(FIELD_DATA_DIR, 'weather_hourly.csv')
    xls = sorted(glob.glob(os.path.join(cool_dir, '**', '*.xlsx'), recursive=True))
    hourly = [p for p in xls if '1小时' in os.path.basename(p)] or xls
    if not hourly or not os.path.exists(wx_csv):
        print('fig11_cooling skipped: this figure belongs to the proprietary '
              'field-validation part of the paper, which is outside the '
              'open-source reproduction scope (see field/README.md)')
        return
    ylc = pd.read_excel(hourly[0], sheet_name='总表').drop_duplicates(subset='时间') \
        .set_index('时间').sort_index()
    w = pd.read_csv(wx_csv, parse_dates=['time']) \
        .set_index('time')
    cool = ylc['制冷系统总电量(kWh)']
    cool = cool[cool > 0].resample('D').mean()
    wb = w['wet_bulb_temperature_2m'].resample('D').mean().reindex(cool.index)
    m = pd.concat([cool, wb], axis=1).dropna()
    m.columns = ['cool', 'wb']

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.9))   # 单栏(上下两面板), 同图3
    sc = a1.scatter(m['wb'], m['cool'], c=m.index.month, cmap='viridis', s=10,
                    alpha=0.85)
    cb = fig.colorbar(sc, ax=a1, fraction=0.045, pad=0.02)
    cb.set_label('月份')
    cb.ax.tick_params(labelsize=7)
    z = np.polyfit(m['wb'], m['cool'], 2)
    xs = np.linspace(m['wb'].min(), m['wb'].max(), 100)
    a1.plot(xs, np.polyval(z, xs), 'r-', lw=1.4)
    r = np.corrcoef(np.polyval(z, m['wb']), m['cool'])[0, 1]
    a1.text(0.04, 0.92, f'二次拟合 R²={r**2:.3f}', transform=a1.transAxes)
    a1.set_xlabel('室外日均湿球温度/℃')
    a1.set_ylabel('日均制冷电量/(kW·h/h)')
    a1.annotate('(a) 楼栋C：制冷负荷-湿球温度', xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=8)

    cfg = [('楼栋C', [0.018, 0.714, 0.837]), ('楼栋BD', [0.812, 0.918, 0.942])]
    x = np.arange(3)
    wd = 0.32
    for i, (n, v) in enumerate(cfg):
        b = a2.bar(x + (i - 0.5) * wd, v, wd, label=n,
                   color=['#1f77b4', '#d62728'][i])
        for xi, vi in zip(x + (i - 0.5) * wd, v):
            a2.text(xi, vi + 0.015, f'{vi:.2f}', ha='center')
    a2.set_xticks(x)
    a2.set_xticklabels(['仅IT负荷', '+干球温度', '+湿球温度'])
    a2.set_ylabel('制冷电量回归 R²')
    a2.set_ylim(0, 1.22)
    a2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a2.legend(frameon=False, loc='upper left', ncol=2,
              columnspacing=0.8, handlelength=1.0)
    a2.annotate('(b) 制冷电量可解释度递进', xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1], h_pad=2.0)   # 面板间距缩小一点
    align_stack(fig, [a1, a2])                          # 上下子图坐标框严格对齐
    save(fig, 'fig11_cooling')


# ================= 图8 实测场景预测示例 =================
def fig8():
    # 实测场景滚动预测(3面板): (a) Room302 4h滚动, 红点=真实值落入SSPM区间内/基线区间外
    #   (基线在波谷欠覆盖、SSPM罩住, 覆盖率优势); (b) 楼栋C 24h日前滚动, 外推至高平稳整楼
    #   (尺度约为Room302的27倍)、SSPM区间窄于基线; (c) 覆盖率-宽度曲线(原版逐元素保留, 仅 a2->a3)
    #   ——同宽度SSPM覆盖更高, 即区间优势源于校准而非加宽.
    npz_dir = _find_field('predictions_field')
    need = [os.path.join(npz_dir, f) for f in (
        'Room302_H16_PatchTST_baseline.npz', 'Room302_H16_PatchTST_sspm.npz',
        'BuildingC_H96_PatchTST_baseline.npz', 'BuildingC_H96_PatchTST_sspm.npz')]
    if not all(os.path.exists(f) for f in need):
        print('fig10_field_pred skipped: this figure belongs to the proprietary '
              'field-validation part of the paper, which is outside the '
              'open-source reproduction scope (see field/README.md)')
        return
    B, G, K, R = '#3b6fb0', '#2ca02c', '0.12', '#d62728'
    BL = '#a9c7ea'  # 浅蓝: 基线区间界(类比SSPM浅色区间带), 与深蓝基线中位数B拉开
    SPD = 96  # 15min steps/day

    def roll(tag, H):  # non-overlap concat -> continuous rolling H-step series
        zs = np.load(os.path.join(npz_dir, f'{tag}_H{H}_PatchTST_sspm.npz'))
        zb = np.load(os.path.join(npz_dir, f'{tag}_H{H}_PatchTST_baseline.npz'))
        t = zs['true']
        ps = np.sort(zs['pred'], axis=2)
        pb = np.sort(zb['pred'], axis=2)
        idx = np.arange(t.shape[0] // H) * H
        return dict(true=t[idx].reshape(-1),
                    slo=ps[idx, :, 0].reshape(-1), smd=ps[idx, :, 1].reshape(-1), shi=ps[idx, :, 2].reshape(-1),
                    blo=pb[idx, :, 0].reshape(-1), bmd=pb[idx, :, 1].reshape(-1), bhi=pb[idx, :, 2].reshape(-1))

    def setyl(ax, *arrs):
        lo = min(a.min() for a in arrs); hi = max(a.max() for a in arrs)
        p = (hi - lo) * 0.08; ax.set_ylim(lo - p, hi + p)

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(3.4, 6.2))
    # ---- (a) 302 room 4h rolling: red dots = baseline misses / SSPM covers ----
    d = roll('Room302', 16); a, b = int(4 * SPD), int(7 * SPD); x = np.arange(a, b) / SPD
    tr, blo, bhi, slo, shi = (d[k][a:b] for k in ('true', 'blo', 'bhi', 'slo', 'shi'))
    a1.fill_between(x, slo, shi, color=G, alpha=0.18, lw=0, label='SSPM 80%区间')
    a1.plot(x, blo, color=BL, lw=0.9, ls='--', label='基线 80%区间'); a1.plot(x, bhi, color=BL, lw=0.9, ls='--')
    a1.plot(x, tr, color=K, lw=0.9, label='真实值')
    adv = ((tr > bhi) | (tr < blo)) & (tr >= slo) & (tr <= shi)
    a1.plot(x[adv], tr[adv], 'o', color=R, ms=3.0, mec='none', zorder=6, label='基线漏')
    setyl(a1, blo, bhi, tr); a1.set_xlabel('时间/天'); a1.set_ylabel('Room302 IT负荷/kW')
    a1.legend(loc='lower right', ncol=2, fontsize=6.0, handlelength=1.4, columnspacing=1.0,
              labelspacing=0.32, frameon=True, facecolor='white', framealpha=0.8,
              edgecolor='none', borderpad=0.3)
    # ---- (b) BuildingC 24h day-ahead rolling: extrapolate to stationary building ----
    d = roll('BuildingC', 96); a, b = int(38 * SPD), int(45 * SPD); x = np.arange(a, b) / SPD
    tr, blo, bhi, slo, shi, smd, bmd = (d[k][a:b] for k in ('true', 'blo', 'bhi', 'slo', 'shi', 'smd', 'bmd'))
    a2.fill_between(x, slo, shi, color=G, alpha=0.18, lw=0, label='SSPM 80%区间')
    a2.plot(x, blo, color=BL, lw=0.9, ls='--', label='基线 80%区间'); a2.plot(x, bhi, color=BL, lw=0.9, ls='--')
    a2.plot(x, tr, color=K, lw=0.9, label='真实值')
    a2.plot(x, smd, color=G, lw=0.8, label='SSPM中位数')
    a2.plot(x, bmd, color=B, lw=0.8, label='基线中位数')
    setyl(a2, blo, bhi, tr); a2.set_xlabel('时间/天'); a2.set_ylabel('楼栋C IT负荷/kW')
    a2.legend(loc='lower right', ncol=3, fontsize=5.6, columnspacing=0.7, handlelength=1.0,
              frameon=True, facecolor='white', framealpha=0.8, edgecolor='none', borderpad=0.3)
    # ---- (c) coverage-width curve (original kept verbatim, only a2->a3): Room302 4h ----
    base, ss = np.load(need[0]), np.load(need[1]); true = base['true']
    def cw_curve(pred):
        pred = np.sort(pred, axis=2)
        med, lo, hi = pred[:, :, 1], pred[:, :, 0], pred[:, :, 2]
        W, C = [], []
        for k in np.arange(0.6, 2.41, 0.04):
            l = med + (lo - med) * k; h = med + (hi - med) * k
            W.append((h - l).mean()); C.append(((true >= l) & (true <= h)).mean())
        return np.array(W), np.array(C)
    def op(pred):
        pred = np.sort(pred, axis=2)
        lo, hi = pred[:, :, 0], pred[:, :, 2]
        return (hi - lo).mean(), ((true >= lo) & (true <= hi)).mean()
    Wb, Cb = cw_curve(base['pred']); Ws, Cs = cw_curve(ss['pred'])
    wb, cb = op(base['pred']); ws, cs = op(ss['pred']); baw = np.interp(ws, Wb, Cb)
    a3.plot(Wb, Cb, '-', color=B, lw=1.4, label='基线（区间按比例缩放）')
    a3.plot(Ws, Cs, '-', color=G, lw=1.4, label='SSPM（区间按比例缩放）')
    a3.axhline(0.80, color='0.4', ls='--', lw=0.8)
    a3.text(Wb.max(), 0.81, '名义80%', ha='right', va='bottom', fontsize=7, color='0.3')
    a3.plot([ws, ws], [baw, cs], color='#d62728', lw=1.0, zorder=4)
    a3.plot(ws, cs, 'o', color=G, ms=5, zorder=5, label='SSPM实际部署区间')
    a3.plot(ws, baw, 'o', mfc='white', mec=B, mew=1.3, ms=5, zorder=6, label='同宽度基线')
    a3.annotate('同宽度下高约\n%.0f个百分点' % ((cs - baw) * 100), xy=(ws, (cs + baw) / 2),
                xytext=(ws - 1.05, 0.51), fontsize=7, color='#d62728', ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8,
                                connectionstyle='arc3,rad=0.5'))
    a3.set_xlabel('平均区间宽度/kW'); a3.set_ylabel('经验覆盖率'); a3.set_ylim(0.42, 1.0)
    a3.legend(frameon=False, loc='lower right')
    # ---- subpanel captions + frame alignment ----
    a1.annotate('(a) Room302 4 h滚动预测', xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=8)
    a2.annotate('(b) 楼栋C 24 h日前滚动预测', xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=8)
    a3.annotate('(c) 覆盖率-宽度曲线', xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97], h_pad=0.6); align_stack(fig, [a1, a2, a3])
    save(fig, 'fig10_field_pred')


# 输出图号 → 生成函数(函数名为历史命名, 与图号不对应)
FIG_BUILDERS = {
    'fig2': fig_stl_decomp,   # fig2_stl_decomp
    'fig4': None,             # fig4_diurnal_acf (fig1, 需先 load_ds)
    'fig5': fig2,             # fig5_stl
    'fig6': fig3,             # fig6_theory
    'fig7': fig4,             # fig7_heatmap
    'fig8': fig5,             # fig8_calibration
    'fig9': fig6,             # fig9_strategy_map
    'fig10': fig8,            # fig10_field_pred (需实测预测npz)
    'fig11': fig7,            # fig11_cooling (需专有实测原始数据)
}
PUBLIC_FIGS = ('fig2', 'fig4', 'fig5', 'fig6', 'fig7', 'fig8')


def main():
    ap = argparse.ArgumentParser(
        description='Generate the paper figures (fig2, fig4-fig11) into figures/.')
    ap.add_argument('--figs', default='all',
                    help="'all' (default) or a comma list of output figure ids, "
                         "e.g. fig2,fig7")
    args = ap.parse_args()
    if args.figs.strip().lower() == 'all':
        selected = list(FIG_BUILDERS)
    else:
        selected = [s.strip() for s in args.figs.split(',') if s.strip()]
        unknown = [s for s in selected if s not in FIG_BUILDERS]
        if unknown:
            ap.error(f"unknown figure id(s) {unknown}; choose from {list(FIG_BUILDERS)}")
    done = set()
    for name in FIG_BUILDERS:
        if name not in selected:
            continue
        try:
            if name == 'fig4':
                fig1(load_ds())
            else:
                FIG_BUILDERS[name]()
            done.add(name)
        except Exception as e:
            print(f'{name} SKIPPED: {e}')
    failed = [n for n in selected if n not in done]
    failed_public = [n for n in failed if n in PUBLIC_FIGS]
    print('done —', f'{len(done)}/{len(selected)} figures built',
          (f'(failed: {", ".join(failed)})' if failed else ''))
    if failed_public:
        sys.exit(1)
    if not any(n in PUBLIC_FIGS for n in selected) and failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
