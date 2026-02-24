#!/usr/bin/env python3
"""
PRL Figure 1: Universal D_eff Heatmap

Single panel showing D_eff for n+A (green) and p+A (pink)

Author: Jin Lei
Date: December 2024
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

# Colormaps - green for neutrons, pink for protons
CMAP_GREEN = LinearSegmentedColormap.from_list('pastel_green',
    ['#F0FFF4', '#D5F5E3', '#A3E4BC', '#6FCF97'], N=256)
CMAP_PINK = LinearSegmentedColormap.from_list('pastel_pink',
    ['#FFF5F7', '#FFD1DC', '#FFB3C6', '#FF8FA3'], N=256)


def setup_style():
    """Set up matplotlib style for Figure 1 (single column, x1.5 fonts)."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'axes.linewidth': 1.2,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_fig1(data, save_path):
    """
    PRL Figure 1: Universal D_eff Heatmap
    Single panel showing D_eff for n+A (green) and p+A (pink)
    """
    setup_style()

    # Create figure - single panel
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    n_data = data.get('full_scan_n', [])
    p_data = data.get('full_scan_p', [])

    if not n_data:
        n_data = [r for r in data['full_scan'] if r.get('projectile', 'n') == 'n']
    if not p_data:
        p_data = [r for r in data['full_scan'] if r.get('projectile') == 'p']

    A_unique = sorted(list(set([r['A'] for r in n_data])))
    E_unique = sorted(list(set([r['E'] for r in n_data])))
    name_map = {r['A']: r['name'] for r in n_data}

    n_E = len(E_unique)
    n_A = len(A_unique)

    D_eff_n = np.zeros((n_A, n_E))
    D_eff_p = np.zeros((n_A, n_E))

    for r in n_data:
        i = A_unique.index(r['A'])
        j = E_unique.index(r['E'])
        D_eff_n[i, j] = r['D_eff']

    for r in p_data:
        i = A_unique.index(r['A'])
        j = E_unique.index(r['E'])
        D_eff_p[i, j] = r['D_eff']

    # Plot heatmaps side by side
    extent_n = [0, n_E, -0.5, n_A - 0.5]
    extent_p = [n_E, 2*n_E, -0.5, n_A - 0.5]

    im_n = ax1.imshow(D_eff_n, aspect='auto', cmap=CMAP_GREEN, vmin=1.0, vmax=2.5,
                      origin='lower', extent=extent_n, interpolation='nearest')
    im_p = ax1.imshow(D_eff_p, aspect='auto', cmap=CMAP_PINK, vmin=1.0, vmax=2.5,
                      origin='lower', extent=extent_p, interpolation='nearest')

    # Separator line
    ax1.axvline(x=n_E, color=COLORS['gray'], linewidth=2.5)
    ax1.spines['left'].set_visible(False)
    ax1.axvline(x=0, color='black', linewidth=0.8)

    # Ticks
    x_ticks = [i + 0.5 for i in range(n_E)] + [n_E + i + 0.5 for i in range(n_E)]
    x_labels = [str(e) for e in E_unique] + [str(e) for e in E_unique]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, fontsize=14)
    ax1.set_yticks(range(n_A))
    ax1.set_yticklabels([name_map[a] for a in A_unique], fontsize=14)

    # Value annotations
    for i in range(n_A):
        for j in range(n_E):
            ax1.text(j + 0.5, i, f'{D_eff_n[i,j]:.1f}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=COLORS['dark_green'])
            ax1.text(n_E + j + 0.5, i, f'{D_eff_p[i,j]:.1f}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=COLORS['dark_pink'])

    ax1.set_xlabel('$E_{lab}$ (MeV)', fontsize=18)
    ax1.set_ylabel('Target Nucleus', fontsize=18)
    ax1.set_title('$D_{eff}$:  $n+A$ (green)  |  $p+A$ (pink)', fontsize=18, pad=10)

    # Statistics annotations
    mean_n, std_n = np.mean(D_eff_n), np.std(D_eff_n)
    mean_p, std_p = np.mean(D_eff_p), np.std(D_eff_p)

    ax1.text(0.25, -0.12, f'$n$: {mean_n:.2f}±{std_n:.2f}',
             transform=ax1.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='center', color=COLORS['dark_green'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['green'], alpha=0.9, edgecolor='#6FCF97'))
    ax1.text(0.75, -0.12, f'$p$: {mean_p:.2f}±{std_p:.2f}',
             transform=ax1.transAxes, fontsize=16, fontweight='bold',
             va='top', ha='center', color=COLORS['dark_pink'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['pink'], alpha=0.9, edgecolor='#FFB3C6'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("PRL Figure 1: Universal D_eff (Final)")
    print("="*60)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'deff_scan_data.json')

    print(f"\nLoading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    save_path = os.path.join(base_dir, 'fig1_deff_universal.png')
    plot_fig1(data, save_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
