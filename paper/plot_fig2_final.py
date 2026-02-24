#!/usr/bin/env python3
"""
PRL Figure 2: Combined D_eff Analysis (Final Version)

Layout: Two-column width (figure*), 3 panels
(a) D_eff vs Nuclear Mass
(b) D_eff vs Energy
(c) Condition Number

Author: Jin Lei
Date: December 2024
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Pastel Color Palette (Pink, Purple, Green)
# =============================================================================
COLORS = {
    # Light pastel colors (main)
    'pink': '#FFD1DC',        # light pink
    'purple': '#E6D5F2',      # soft lavender
    'green': '#D5F5E3',       # soft mint green
    'lavender': '#E6E6FA',    # lavender
    'mint': '#C8F7DC',        # mint
    'blush': '#FFE4E8',       # blush pink
    # Dark versions (for text/lines)
    'dark_pink': '#D63384',
    'dark_purple': '#6F42C1',
    'dark_green': '#198754',
    'gray': '#6B6B6B',
}


def setup_style():
    """Set up matplotlib style for two-column figures - all text same size."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 22,
        'axes.linewidth': 2.0,
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 18,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_fig2(data, save_path):
    """
    PRL Figure 2: Combined D_eff analysis
    (a) D_eff vs Nuclear Mass
    (b) D_eff vs Energy
    (c) Condition Number
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    # === (a) D_eff vs Nuclear Mass ===
    ax1 = axes[0]
    A_vals = np.array(data['nuclei_scan']['A'])
    D_eff_vals = np.array(data['nuclei_scan']['D_eff'])
    names = data['nuclei_scan']['name']

    ax1.bar(range(len(A_vals)), D_eff_vals, color=COLORS['purple'],
            edgecolor='white', linewidth=1.5)

    ax1.axhline(1.0, color=COLORS['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axhline(2.0, color=COLORS['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    mean_d = np.mean(D_eff_vals)
    ax1.axhline(mean_d, color=COLORS['dark_purple'], linestyle='--', linewidth=2,
               label=f'Mean = {mean_d:.2f}')

    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=16)
    ax1.set_xlabel('Target Nucleus')
    ax1.set_ylabel('$D_{eff}$')
    ax1.set_title('(a) $D_{eff}$ vs Mass ($E$=50 MeV)', pad=15)
    ax1.set_ylim(0, 2.5)
    ax1.legend(loc='upper left', fontsize=16)
    ax1.grid(True, alpha=0.3, axis='y')

    # === (b) D_eff vs Energy ===
    ax2 = axes[1]
    E_vals = np.array(data['energy_scan']['E'])
    D_eff_E = np.array(data['energy_scan']['D_eff'])

    ax2.plot(E_vals, D_eff_E, 'o-', color=COLORS['dark_green'],
             linewidth=2, markersize=10, markerfacecolor=COLORS['green'],
             markeredgecolor='white', markeredgewidth=2)

    ax2.axhline(1.0, color=COLORS['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(2.0, color=COLORS['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.fill_between(E_vals, 1.0, D_eff_E, alpha=0.3, color=COLORS['mint'])

    ax2.set_xlabel('$E_{lab}$ (MeV)')
    ax2.set_ylabel('$D_{eff}$')
    ax2.set_title('(b) $D_{eff}$ vs Energy ($^{40}$Ca)', pad=15)
    ax2.set_xlim(0, 210)
    ax2.set_xticks(np.arange(0, 250, 50))
    ax2.set_ylim(0.8, 2.5)
    ax2.grid(True, alpha=0.3)

    # === (c) Condition Number ===
    ax3 = axes[2]
    cond_vals = np.array(data['nuclei_scan']['cond'])

    ax3.bar(range(len(A_vals)), np.log10(cond_vals), color=COLORS['lavender'],
            edgecolor='white', linewidth=1.5)

    ax3.axhline(6, color=COLORS['dark_pink'], linestyle='--', linewidth=1.5,
               label='Severely ill-posed')

    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=16)
    ax3.set_xlabel('Target Nucleus')
    ax3.set_ylabel('$\\log_{10}$(Condition Number)')
    ax3.set_title('(c) Inverse Problem Ill-posedness', pad=15)
    ax3.legend(loc='upper left', fontsize=16)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("PRL Figure 2: Combined D_eff Analysis (Final)")
    print("="*60)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'deff_scan_data.json')

    print(f"\nLoading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    save_path = os.path.join(base_dir, 'fig2_deff_combined.png')
    plot_fig2(data, save_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
