#!/usr/bin/env python3
"""
PRL Figure Plotting Script (Morandi Light Colors)

Reads pre-computed data from JSON and generates publication figures.
Separated from computation for faster iteration.

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
# Morandi Light Color Palette
# =============================================================================
MORANDI = {
    'dusty_rose': '#EFD9D4',
    'sage': '#D9E2D5',
    'dusty_blue': '#C5D5E0',
    'mauve': '#E5D5D8',
    'clay': '#E8DCD0',
    'terracotta': '#DECCBC',
    'lavender': '#E0D8E8',
    'mint': '#D5E8E0',
    'peach': '#F5E0D8',
    'sky': '#D8E8F0',
    # Darker accents for text/lines
    'dark_blue': '#2A4A6A',
    'dark_rose': '#6A3A3A',
    'dark_sage': '#3A5A4A',
    'gray': '#6B6B6B',
}

# Colormaps
CMAP_BLUE = LinearSegmentedColormap.from_list('morandi_blue',
    ['#F0F4F8', '#C5D5E0', '#9BB5C9', '#7095B0'], N=256)
CMAP_ROSE = LinearSegmentedColormap.from_list('morandi_rose',
    ['#FDF5F3', '#EFD9D4', '#DFB8AD', '#CF9686'], N=256)
CMAP_SAGE = LinearSegmentedColormap.from_list('morandi_sage',
    ['#F5F8F5', '#D9E2D5', '#B8C9B0', '#98B088'], N=256)


def setup_style():
    """Set up matplotlib style for publication."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# Figure 1: D_eff Universal (Core PRL Result)
# =============================================================================
def plot_deff_universal(data, save_path):
    """
    PRL Figure 1: Universal D_eff
    (a) Heatmap: n+A (blue) | p+A (rose)
    (b) V0-r0 correlation vs A
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # === Panel (a): Combined heatmap ===
    ax1 = axes[0]

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

    # Plot heatmaps
    extent_n = [0, n_E, -0.5, n_A - 0.5]
    extent_p = [n_E, 2*n_E, -0.5, n_A - 0.5]

    im_n = ax1.imshow(D_eff_n, aspect='auto', cmap=CMAP_BLUE, vmin=1.0, vmax=2.1,
                      origin='lower', extent=extent_n, interpolation='nearest')
    im_p = ax1.imshow(D_eff_p, aspect='auto', cmap=CMAP_ROSE, vmin=1.0, vmax=2.1,
                      origin='lower', extent=extent_p, interpolation='nearest')

    # Separator line
    ax1.axvline(x=n_E, color=MORANDI['gray'], linewidth=2)
    ax1.spines['left'].set_visible(False)
    ax1.axvline(x=0, color='black', linewidth=0.8)

    # Ticks
    x_ticks = [i + 0.5 for i in range(n_E)] + [n_E + i + 0.5 for i in range(n_E)]
    x_labels = [str(e) for e in E_unique] + [str(e) for e in E_unique]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, fontsize=8)
    ax1.set_yticks(range(n_A))
    ax1.set_yticklabels([name_map[a] for a in A_unique], fontsize=8)

    # Value annotations
    for i in range(n_A):
        for j in range(n_E):
            ax1.text(j + 0.5, i, f'{D_eff_n[i,j]:.1f}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color=MORANDI['dark_blue'])
            ax1.text(n_E + j + 0.5, i, f'{D_eff_p[i,j]:.1f}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color=MORANDI['dark_rose'])

    ax1.set_xlabel('$E_{lab}$ (MeV)', fontsize=11)
    ax1.set_ylabel('Target Nucleus', fontsize=11)
    ax1.set_title('(a) $D_{eff}$: $n+A$ (blue) | $p+A$ (rose)', fontsize=11)

    # Statistics annotations
    mean_n, std_n = np.mean(D_eff_n), np.std(D_eff_n)
    mean_p, std_p = np.mean(D_eff_p), np.std(D_eff_p)

    ax1.text(0.25, -0.12, f'$n$: {mean_n:.2f}$\\pm${std_n:.2f}',
             transform=ax1.transAxes, fontsize=9, fontweight='bold',
             va='top', ha='center', color=MORANDI['dark_blue'],
             bbox=dict(boxstyle='round', facecolor='#E0EBF5', alpha=0.9, edgecolor='#8AAAC8'))
    ax1.text(0.75, -0.12, f'$p$: {mean_p:.2f}$\\pm${std_p:.2f}',
             transform=ax1.transAxes, fontsize=9, fontweight='bold',
             va='top', ha='center', color=MORANDI['dark_rose'],
             bbox=dict(boxstyle='round', facecolor='#F5E8E5', alpha=0.9, edgecolor='#C8A098'))

    # === Panel (b): V0-r0 correlation ===
    ax2 = axes[1]

    A_vals = np.array(data['nuclei_scan']['A'])
    corr_vals = np.array(data['nuclei_scan']['corr'])
    names = data['nuclei_scan']['name']

    ax2.scatter(A_vals, -corr_vals, s=100, c=MORANDI['dusty_blue'],
                edgecolors='white', linewidths=2, zorder=5)
    ax2.axhline(1.0, color=MORANDI['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(0.9, color=MORANDI['mauve'], linestyle='--', linewidth=2,
                label='Strong degeneracy')
    ax2.fill_between([0, 220], 0.9, 1.05, alpha=0.3, color=MORANDI['mauve'])

    # Label offsets
    offsets = {
        '12C': (25, 20), '16O': (20, -20), '27Al': (-30, -20), '28Si': (30, -25),
        '40Ca': (-45, 20), '48Ti': (0, 25), '56Fe': (40, 18), '58Ni': (45, -12),
        '90Zr': (20, -20), '120Sn': (0, 18), '197Au': (20, -20), '208Pb': (20, 12),
    }

    for i, name in enumerate(names):
        offset = offsets.get(name, (0, 10))
        ax2.annotate(name, (A_vals[i], -corr_vals[i]),
                    textcoords="offset points", xytext=offset,
                    ha='center', fontsize=8,
                    arrowprops=dict(arrowstyle='-', color='#888888', lw=0.5))

    ax2.set_xlabel('Mass Number $A$', fontsize=11)
    ax2.set_ylabel('$|r_{V_0-r_0}|$', fontsize=11)
    ax2.set_title('(b) Igo ambiguity', fontsize=11)
    ax2.set_xlim(-5, 225)
    ax2.set_ylim(0.5, 1.12)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.08, bottom=0.15, wspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Figure 2: Combined D_eff analysis (3 panels)
# =============================================================================
def plot_deff_combined(data, save_path):
    """
    Combined figure: D_eff vs A, D_eff vs E, and condition number.
    All tell the same story: ~1-2 effective parameters.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # === (a) D_eff vs Nuclear Mass ===
    ax1 = axes[0]
    A_vals = np.array(data['nuclei_scan']['A'])
    D_eff_vals = np.array(data['nuclei_scan']['D_eff'])
    names = data['nuclei_scan']['name']

    colors = [MORANDI['dusty_blue'] if d < 1.5 else MORANDI['dusty_rose'] for d in D_eff_vals]
    ax1.bar(range(len(A_vals)), D_eff_vals, color=colors,
            edgecolor='white', linewidth=1.5)

    ax1.axhline(1.0, color=MORANDI['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axhline(2.0, color=MORANDI['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    mean_d = np.mean(D_eff_vals)
    ax1.axhline(mean_d, color=MORANDI['dark_blue'], linestyle='--', linewidth=2,
               label=f'Mean = {mean_d:.2f}')

    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel('Target Nucleus')
    ax1.set_ylabel('$D_{eff}$')
    ax1.set_title('(a) $D_{eff}$ vs Mass ($E$=50 MeV)')
    ax1.set_ylim(0, 2.5)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # === (b) D_eff vs Energy ===
    ax2 = axes[1]
    E_vals = np.array(data['energy_scan']['E'])
    D_eff_E = np.array(data['energy_scan']['D_eff'])

    ax2.plot(E_vals, D_eff_E, 'o-', color=MORANDI['dark_sage'],
             linewidth=2, markersize=10, markerfacecolor=MORANDI['sage'],
             markeredgecolor='white', markeredgewidth=2)

    ax2.axhline(1.0, color=MORANDI['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(2.0, color=MORANDI['gray'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.fill_between(E_vals, 1.0, D_eff_E, alpha=0.3, color=MORANDI['sage'])

    ax2.set_xlabel('$E_{lab}$ (MeV)')
    ax2.set_ylabel('$D_{eff}$')
    ax2.set_title('(b) $D_{eff}$ vs Energy ($^{40}$Ca)')
    ax2.set_xlim(0, 210)
    ax2.set_ylim(0.8, 2.0)
    ax2.grid(True, alpha=0.3)

    # === (c) Condition Number ===
    ax3 = axes[2]
    cond_vals = np.array(data['nuclei_scan']['cond'])

    ax3.bar(range(len(A_vals)), np.log10(cond_vals), color=MORANDI['lavender'],
            edgecolor='white', linewidth=1.5)

    ax3.axhline(6, color=MORANDI['dark_rose'], linestyle='--', linewidth=1.5,
               label='Severely ill-posed')

    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax3.set_xlabel('Target Nucleus')
    ax3.set_ylabel('$\\log_{10}$(Condition Number)')
    ax3.set_title('(c) Inverse Problem Ill-posedness')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Combined Summary Figure
# =============================================================================
def plot_prl_summary(data, save_path):
    """
    Combined 2x2 summary figure for PRL.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # === (a) D_eff vs A ===
    ax1 = axes[0, 0]
    A_vals = np.array(data['nuclei_scan']['A'])
    D_eff_vals = np.array(data['nuclei_scan']['D_eff'])
    names = data['nuclei_scan']['name']

    ax1.scatter(A_vals, D_eff_vals, s=100, c=MORANDI['dusty_blue'],
                edgecolors='white', linewidths=2, zorder=5)

    for i, name in enumerate(names):
        ax1.annotate(name, (A_vals[i], D_eff_vals[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    mean_d = np.mean(D_eff_vals)
    ax1.axhline(mean_d, color=MORANDI['dark_blue'], linestyle='--',
                linewidth=2, label=f'Mean = {mean_d:.2f}')
    ax1.axhline(1.0, color=MORANDI['gray'], linestyle=':', linewidth=1)
    ax1.axhline(2.0, color=MORANDI['gray'], linestyle=':', linewidth=1)

    ax1.set_xlabel('Mass Number $A$')
    ax1.set_ylabel('$D_{eff}$')
    ax1.set_title('(a) $D_{eff}$ vs Nuclear Mass')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 220)
    ax1.set_ylim(0.8, 2.3)

    # === (b) D_eff vs E ===
    ax2 = axes[0, 1]
    E_vals = np.array(data['energy_scan']['E'])
    D_eff_E = np.array(data['energy_scan']['D_eff'])

    ax2.plot(E_vals, D_eff_E, 'o-', color=MORANDI['dark_sage'],
             linewidth=2, markersize=10, markerfacecolor=MORANDI['sage'],
             markeredgecolor='white', markeredgewidth=2)

    ax2.axhline(1.0, color=MORANDI['gray'], linestyle=':', linewidth=1)
    ax2.axhline(2.0, color=MORANDI['gray'], linestyle=':', linewidth=1)
    ax2.fill_between(E_vals, 1.0, D_eff_E, alpha=0.3, color=MORANDI['sage'])

    ax2.set_xlabel('$E_{lab}$ (MeV)')
    ax2.set_ylabel('$D_{eff}$')
    ax2.set_title('(b) $D_{eff}$ vs Energy ($^{40}$Ca)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 210)
    ax2.set_ylim(0.8, 2.0)

    # === (c) V0-r0 correlation ===
    ax3 = axes[1, 0]
    corr_vals = np.array(data['nuclei_scan']['corr'])

    ax3.scatter(A_vals, -corr_vals, s=100, c=MORANDI['dusty_rose'],
                edgecolors='white', linewidths=2, zorder=5)
    ax3.axhline(0.9, color=MORANDI['mauve'], linestyle='--', linewidth=2)
    ax3.axhline(1.0, color=MORANDI['gray'], linestyle=':', linewidth=1)
    ax3.fill_between([0, 220], 0.9, 1.05, alpha=0.3, color=MORANDI['mauve'])

    for i, name in enumerate(names):
        ax3.annotate(name, (A_vals[i], -corr_vals[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    ax3.set_xlabel('Mass Number $A$')
    ax3.set_ylabel('$|r_{V_0-r_0}|$')
    ax3.set_title('(c) Igo Ambiguity: $V_0$-$r_0$ Correlation')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 220)
    ax3.set_ylim(0.2, 1.1)

    # === (d) Condition number ===
    ax4 = axes[1, 1]
    cond_vals = np.array(data['nuclei_scan']['cond'])

    ax4.scatter(A_vals, np.log10(cond_vals), s=100, c=MORANDI['lavender'],
                edgecolors='white', linewidths=2, zorder=5)
    ax4.axhline(6, color=MORANDI['dark_rose'], linestyle='--', linewidth=1.5,
               label='Ill-posed threshold')

    for i, name in enumerate(names):
        ax4.annotate(name, (A_vals[i], np.log10(cond_vals[i])),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    ax4.set_xlabel('Mass Number $A$')
    ax4.set_ylabel('$\\log_{10}$(Condition Number)')
    ax4.set_title('(d) Inverse Problem Ill-posedness')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 220)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("="*60)
    print("PRL Figure Generation (Morandi Light Colors)")
    print("="*60)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'deff_scan_data.json')
    print(f"\nLoading data from: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  - {len(data['nuclei_scan']['A'])} nuclei")
    print(f"  - {len(data['energy_scan']['E'])} energies")
    print(f"  - {len(data.get('full_scan', []))} total scan points")

    # Generate figures
    print("\nGenerating figures...")

    base_dir = os.path.dirname(__file__)

    # Figure 1: Core result (heatmap + Igo ambiguity)
    plot_deff_universal(data, os.path.join(base_dir, 'prl_fig1_deff_universal.png'))

    # Figure 2: Combined D_eff analysis (A, E, condition number)
    plot_deff_combined(data, os.path.join(base_dir, 'prl_fig2_deff_combined.png'))

    # Summary figure (optional, 2x2)
    plot_prl_summary(data, os.path.join(base_dir, 'prl_summary.png'))

    print("\nDone!")


if __name__ == '__main__':
    main()
