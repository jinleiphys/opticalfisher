#!/usr/bin/env python3
"""
PRL Figure: Step-by-step constraint analysis.

Shows how D_eff increases as we add observables and systematics:
  Step 1: Elastic dσ/dΩ
  Step 2: + σ_R
  Step 3: + Ay (+ σ_T)
  Step 4: Multi-energy
  Step 5: KD02 global systematics

Panel (a): D_eff progression with eigenvalue spectrum inset
Panel (b): Information per parameter subgroup at each step

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COLORS = {
    'pink': '#FFD1DC',
    'purple': '#E6D5F2',
    'green': '#D5F5E3',
    'lavender': '#E6E6FA',
    'mint': '#C8F7DC',
    'blush': '#FFE4E8',
    'dark_pink': '#D63384',
    'dark_purple': '#6F42C1',
    'dark_green': '#198754',
    'dark_orange': '#E67E22',
    'gray': '#6B6B6B',
}

STEP_COLORS = ['#198754', '#2ECC71', '#D63384', '#6F42C1', '#E67E22']
STEP_LABELS_SHORT = [
    r'$d\sigma/d\Omega$',
    r'+ $\sigma_R$',
    r'+ $A_y$',
    r'Multi-$E$',
    r'KD02',
]
GROUP_COLORS = {
    'Real Volume': '#198754',
    'Imaginary': '#D63384',
    'Spin-Orbit': '#E67E22',
    'Geometry': '#198754',
    'Real Depth': '#2ECC71',
    'Imag Depth': '#D63384',
    'SO Depth': '#E67E22',
}


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 18,
        'axes.linewidth': 1.5,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_stepwise(data, save_path, case_idx=0):
    """
    Panel (a): D_eff bar chart showing progressive improvement
    Panel (b): Eigenvalue spectrum at each step (log scale)
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    case = data['cases'][case_idx]
    proj = case['projectile']
    nuc = case['nucleus']
    steps = case['steps']

    step_keys = ['1_elastic', '2_elastic_sigR', '3_all_single_E',
                 '4_multi_energy', '5_kd02_global']
    step_labels = []
    step_deffs = []
    step_eigenvals = []
    step_n_params = []

    for key in step_keys:
        if key not in steps:
            continue
        s = steps[key]
        step_labels.append(s['label'].split('(')[0].strip())
        step_deffs.append(s['D_eff'])
        step_eigenvals.append(np.array(s['eigenvalues']))
        step_n_params.append(s.get('N_global_params', 11))

    n_steps = len(step_labels)

    # === Panel (a): D_eff progression ===
    ax1 = axes[0]

    x = np.arange(n_steps)
    bars = ax1.bar(x, step_deffs, width=0.6,
                   color=STEP_COLORS[:n_steps],
                   edgecolor='white', linewidth=2, alpha=0.85)

    # Value labels on bars
    for i, (d, n_p) in enumerate(zip(step_deffs, step_n_params)):
        ax1.text(i, d + 0.15, f'{d:.2f}', ha='center', va='bottom',
                 fontsize=18, fontweight='bold')
        if n_p != 11:
            ax1.text(i, d + 0.45, f'({n_p}p)', ha='center', va='bottom',
                     fontsize=12, color=COLORS['gray'])

    # Arrows showing increments
    for i in range(1, n_steps):
        delta = step_deffs[i] - step_deffs[i-1]
        if delta > 0.05:
            mid_y = (step_deffs[i-1] + step_deffs[i]) / 2
            ax1.annotate('', xy=(i, step_deffs[i] - 0.05),
                         xytext=(i-0.6, step_deffs[i-1] + 0.05),
                         arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                                         lw=1.5, connectionstyle='arc3,rad=0.2'))
            ax1.text(i - 0.35, mid_y, f'+{delta:.2f}', fontsize=12,
                     color=COLORS['gray'], ha='center', rotation=45)

    ax1.set_xticks(x)
    ax1.set_xticklabels(STEP_LABELS_SHORT[:n_steps], fontsize=14)
    ax1.set_ylabel('$D_\\mathrm{eff}$')
    ax1.set_title(f'(a) Progressive Constraints: {proj}+{nuc}', pad=10)
    ax1.set_ylim(0, max(step_deffs) + 1.2)
    ax1.grid(True, alpha=0.3, axis='y')

    # Reference lines
    ax1.axhline(1, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(2, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.5)

    # === Panel (b): Eigenvalue spectrum ===
    ax2 = axes[1]

    for i, (ev, label) in enumerate(zip(step_eigenvals, STEP_LABELS_SHORT)):
        ev_sorted = np.sort(ev)[::-1]
        # Normalize by max eigenvalue for comparison
        ev_norm = ev_sorted / (ev_sorted[0] + 1e-30)
        n_ev = len(ev_sorted)
        ax2.semilogy(range(1, n_ev + 1), ev_norm, 'o-',
                     color=STEP_COLORS[i], linewidth=2, markersize=6,
                     label=f'{label} ($D_{{eff}}$={step_deffs[i]:.1f})')

    # Mark the "threshold" for 1% of max
    ax2.axhline(0.01, color=COLORS['gray'], linestyle='--', linewidth=1.5,
                label='1% threshold')

    ax2.set_xlabel('Eigenvalue Index')
    ax2.set_ylabel('Normalized Eigenvalue $\\lambda_i / \\lambda_1$')
    ax2.set_title('(b) Eigenvalue Spectrum', pad=10)
    ax2.legend(loc='lower left', fontsize=12, framealpha=0.9)
    ax2.set_xlim(0.5, 17.5)
    ax2.set_ylim(1e-10, 5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_subgroup_heatmap(data, save_path, case_idx=0):
    """
    Heatmap showing D_eff per parameter subgroup at each step.
    Rows: steps, Columns: parameter groups.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    case = data['cases'][case_idx]
    steps = case['steps']

    step_keys = ['1_elastic', '2_elastic_sigR', '3_all_single_E',
                 '4a_multi_energy_elastic', '4_multi_energy']
    group_names = ['Real Volume', 'Imaginary', 'Spin-Orbit']

    step_labels = []
    matrix = []
    deffs = []

    for key in step_keys:
        if key not in steps:
            continue
        s = steps[key]
        step_labels.append(s['label'].split('(')[0].strip())
        deffs.append(s['D_eff'])
        row = []
        for gname in group_names:
            info = s['subgroups'].get(gname, {})
            row.append(info.get('trace', 0))
        matrix.append(row)

    matrix = np.array(matrix)
    # Log-normalize for visualization
    matrix_log = np.log10(matrix + 1e-10)

    im = ax.imshow(matrix_log, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')

    # Annotations
    for i in range(len(step_labels)):
        for j in range(len(group_names)):
            val = matrix[i, j]
            color = 'white' if matrix_log[i, j] > np.median(matrix_log) else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, fontsize=14)
    ax.set_yticks(range(len(step_labels)))
    deff_labels = [f'{l} (D={d:.2f})' for l, d in zip(step_labels, deffs)]
    ax.set_yticklabels(deff_labels, fontsize=12)

    ax.set_title(f'Fisher Information by Parameter Group\n'
                 f'{case["projectile"]}+{case["nucleus"]}', pad=10)
    plt.colorbar(im, ax=ax, label='log₁₀(tr F)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("PRL Figure: Step-by-Step Constraint Analysis")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'data', 'deff_stepwise.json')

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        print("Run: python analysis/deff_stepwise.py")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data['cases'])} cases")

    # Main figure: D_eff progression + eigenvalue spectrum
    save_path = os.path.join(base_dir, 'fig_stepwise.png')
    plot_stepwise(data, save_path, case_idx=0)

    # Subgroup heatmap
    save_path2 = os.path.join(base_dir, 'fig_subgroup_info.png')
    plot_subgroup_heatmap(data, save_path2, case_idx=0)

    print("\nDone!")


if __name__ == '__main__':
    main()
