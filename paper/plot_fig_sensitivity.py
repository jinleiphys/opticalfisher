#!/usr/bin/env python3
"""
PRL Figure: Angle-Resolved Sensitivity Analysis

Panel (a): S_i(theta) curves for all 11 parameters
Panel (b): Cumulative Fisher information and D_eff(theta_max)

Shows that diffuseness parameters gain information at backward angles,
and that angular coverage beyond ~140 deg provides diminishing returns.

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    'gray': '#6B6B6B',
}

# Parameter colors: real volume (green), imaginary (pink), surface (purple), SO (orange)
PARAM_COLORS = {
    'V': '#198754',   'rv': '#2ECC71',  'av': '#82E0AA',
    'W': '#D63384',   'rw': '#FF6B9D',  'aw': '#FFB3C6',
    'Wd': '#6F42C1',  'rvd': '#9B59B6', 'avd': '#D2B4DE',
    'Vso': '#E67E22',  'Wso': '#F0B27A',
}

PARAM_STYLES = {
    'V': '-',  'rv': '-',  'av': '--',
    'W': '-',  'rw': '-',  'aw': '--',
    'Wd': '-', 'rvd': '-', 'avd': '--',
    'Vso': '-', 'Wso': '--',
}

PARAM_WIDTHS = {
    'V': 3.0,  'rv': 2.5,  'av': 2.0,
    'W': 2.5,  'rw': 2.0,  'aw': 1.5,
    'Wd': 2.5, 'rvd': 2.0, 'avd': 1.5,
    'Vso': 2.0, 'Wso': 1.5,
}

PARAM_LABELS = {
    'V': '$V$', 'rv': '$r_v$', 'av': '$a_v$',
    'W': '$W$', 'rw': '$r_w$', 'aw': '$a_w$',
    'Wd': '$W_d$', 'rvd': '$r_d$', 'avd': '$a_d$',
    'Vso': '$V_{so}$', 'Wso': '$W_{so}$',
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
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_fig_sensitivity(data, save_path, case_idx=0):
    """
    Plot angle-resolved sensitivity.
    (a) |S_i(theta)| for all parameters
    (b) Cumulative D_eff(theta_max)
    """
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    case = data['cases'][case_idx]
    theta = np.array(data['theta_deg'])
    S_dcs = np.array(case['S_dcs'])
    C_i = np.array(case['cumulative_info'])
    D_eff_cumul = np.array(case['D_eff_cumulative'])
    param_names = data['param_names']

    proj = case['projectile']
    nuc = case['nucleus']
    E = case['E']

    # === Panel (a): S_i(theta) ===
    ax1 = axes[0]

    # Shade backward angle region
    ax1.axvspan(130, 175, alpha=0.15, color=COLORS['mint'],
                label='Backward angles')

    for i, pname in enumerate(param_names):
        ax1.plot(theta, np.abs(S_dcs[i]),
                 color=PARAM_COLORS[pname],
                 linestyle=PARAM_STYLES[pname],
                 linewidth=PARAM_WIDTHS[pname],
                 label=PARAM_LABELS[pname])

    ax1.set_xlabel(r'Scattering Angle $\theta$ (deg)')
    ax1.set_ylabel(r'$|S_i(\theta)| = |d\log\sigma / d\log p_i|$')
    # Extract element name from nucleus string (e.g. '40Ca' -> 'Ca')
    elem = nuc[len(str(case["A"])):]
    ax1.set_title(f'(a) Parameter Sensitivity: '
                  f'${proj}+{{}}^{{{case["A"]}}}${elem} '
                  f'at {E} MeV', pad=10)
    ax1.set_xlim(5, 175)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-3, None)
    ax1.legend(ncol=3, loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # === Panel (b): Cumulative D_eff ===
    ax2 = axes[1]

    ax2.plot(theta, D_eff_cumul, '-', color=COLORS['dark_purple'],
             linewidth=3, label=r'$D_\mathrm{eff}(\theta_\mathrm{max})$')

    # Mark 90% level
    D_final = D_eff_cumul[-1]
    ax2.axhline(D_final, color=COLORS['gray'], linestyle=':', linewidth=1.5,
                alpha=0.7)
    ax2.axhline(0.9 * D_final, color=COLORS['dark_pink'], linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'90% of final')

    target90 = 0.9 * D_final
    idx90 = np.searchsorted(D_eff_cumul, target90)
    if idx90 < len(theta):
        ax2.axvline(theta[idx90], color=COLORS['dark_pink'], linestyle='--',
                    linewidth=1, alpha=0.5)
        ax2.annotate(f'{theta[idx90]:.0f}°',
                     xy=(theta[idx90], target90),
                     xytext=(theta[idx90] + 10, target90 - 0.1),
                     fontsize=14, color=COLORS['dark_pink'])

    ax2.text(0.95, 0.15, f'$D_{{\\mathrm{{eff}}}}$ = {D_final:.2f}',
             transform=ax2.transAxes, fontsize=18, ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['lavender'],
                       alpha=0.9))

    # Also plot cumulative info for a few key parameters
    ax2_twin = ax2.twinx()
    for i, pname in enumerate(['V', 'rv', 'av', 'Wd']):
        idx = param_names.index(pname)
        C_norm = C_i[idx] / (C_i[idx, -1] + 1e-30)
        ax2_twin.plot(theta, C_norm, '--',
                      color=PARAM_COLORS[pname], linewidth=1.5, alpha=0.6,
                      label=f'{PARAM_LABELS[pname]} (norm.)')
    ax2_twin.set_ylabel('Cumulative Info (normalized)', fontsize=16)
    ax2_twin.set_ylim(0, 1.3)
    ax2_twin.legend(loc='center right', fontsize=11, framealpha=0.8)

    ax2.set_xlabel(r'Maximum Angle $\theta_\mathrm{max}$ (deg)')
    ax2.set_ylabel(r'$D_\mathrm{eff}(\theta_\mathrm{max})$')
    ax2.set_title('(b) Cumulative Information Content', pad=10)
    ax2.set_xlim(5, 175)
    ax2.legend(loc='upper left', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("PRL Figure: Angle-Resolved Sensitivity")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'data', 'angle_sensitivity.json')

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        print("Run: python analysis/angle_resolved_sensitivity.py")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Plot for first case (n+40Ca@50MeV)
    save_path = os.path.join(base_dir, 'fig_sensitivity.png')
    plot_fig_sensitivity(data, save_path, case_idx=0)

    print("\nDone!")


if __name__ == '__main__':
    main()
