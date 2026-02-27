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
    'V': 1.5,  'rv': 1.2,  'av': 1.0,
    'W': 1.2,  'rw': 1.0,  'aw': 0.8,
    'Wd': 1.2, 'rvd': 1.0, 'avd': 0.8,
    'Vso': 1.0, 'Wso': 0.8,
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
        'font.size': 8,
        'axes.linewidth': 0.8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
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
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.5))

    case = data['cases'][case_idx]
    theta = np.array(data['theta_deg'])
    S_dcs = np.array(case['S_dcs'])
    C_i = np.array(case['cumulative_info'])
    D_eff_cumul = np.array(case['D_eff_cumulative'])
    param_names = data['param_names']

    proj = case['projectile']
    nuc = case['nucleus']
    E = case['E']

    # === Panel (a): Raw derivatives on log scale — Igo degeneracy ===
    ax1 = axes[0]

    # Convert log-sensitivities S_i to raw derivatives: dsigma/dp_i = S_i * sigma / p_i
    sigma = np.array(case['elastic_dcs'])
    params_val = np.array(case['params'])

    idx_V = param_names.index('V')
    idx_rv = param_names.index('rv')
    idx_Wd = param_names.index('Wd')

    raw_V = np.abs(S_dcs[idx_V] * sigma / params_val[idx_V])
    raw_rv = np.abs(S_dcs[idx_rv] * sigma / params_val[idx_rv])
    raw_Wd = np.abs(S_dcs[idx_Wd] * sigma / params_val[idx_Wd])

    # Plot raw derivatives on log scale
    ax1.semilogy(theta, raw_V, '-', color='#198754', marker='o',
                 linewidth=1.5, markersize=3, markevery=4,
                 label=r'$|\partial\sigma/\partial V|$')
    ax1.semilogy(theta, raw_rv, '--', color='#D63384', marker='s',
                 linewidth=1.5, markersize=3, markevery=4,
                 label=r'$|\partial\sigma/\partial r_v|$')
    ax1.semilogy(theta, raw_Wd, ':', color='#6F42C1', marker='D',
                 linewidth=1.5, markersize=3, markevery=4,
                 label=r'$|\partial\sigma/\partial W_d|$')

    ax1.set_xlabel(r'Scattering Angle $\theta$ (deg)')
    ax1.set_ylabel(r'$|\partial(d\sigma/d\Omega)/\partial p_i|$')
    ax1.set_xlim(5, 175)
    ax1.legend(loc='upper right', fontsize=6.5, framealpha=0.9,
               handlelength=2.0)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=10,
             fontweight='bold', va='top', ha='left')

    # === Panel (b): Cumulative D_eff ===
    ax2 = axes[1]

    ax2.plot(theta, D_eff_cumul, '-', color=COLORS['dark_purple'],
             linewidth=1.5, label=r'$D_\mathrm{eff}(\theta_\mathrm{max})$')

    # Mark final D_eff level
    D_final = D_eff_cumul[-1]
    ax2.axhline(D_final, color=COLORS['gray'], linestyle=':', linewidth=1.5,
                alpha=0.7)

    # Mark peak
    idx_peak = np.argmax(D_eff_cumul)
    ax2.annotate(f'peak: {D_eff_cumul[idx_peak]:.1f} at {theta[idx_peak]:.0f}°',
                 xy=(theta[idx_peak], D_eff_cumul[idx_peak]),
                 xytext=(theta[idx_peak] + 15, D_eff_cumul[idx_peak] - 0.05),
                 fontsize=7, color=COLORS['dark_purple'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['dark_purple'],
                                 lw=1.0))

    ax2.text(0.95, 0.15, f'$D_{{\\mathrm{{eff}}}}$ = {D_final:.2f}',
             transform=ax2.transAxes, fontsize=8, ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['lavender'],
                       alpha=0.9))

    ax2.set_xlabel(r'Maximum Angle $\theta_\mathrm{max}$ (deg)')
    ax2.set_ylabel(r'$D_\mathrm{eff}(\theta_\mathrm{max})$')
    ax2.set_xlim(5, 175)
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.97, 0.55, '(b)', transform=ax2.transAxes, fontsize=10,
             fontweight='bold', va='top', ha='right')

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

    # Plot for n+40Ca@50MeV (case 0)
    save_path = os.path.join(base_dir, 'fig_sensitivity_v2.png')
    plot_fig_sensitivity(data, save_path, case_idx=0)

    print("\nDone!")


if __name__ == '__main__':
    main()
