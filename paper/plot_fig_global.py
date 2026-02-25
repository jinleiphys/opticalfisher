#!/usr/bin/env python3
"""
PRL Figure: Global KD02 Fisher Information Analysis

Three-panel figure showing the constraints from a worldwide dataset on
the 45 universal KD02 parameters:

  (a) D_eff growth as (nucleus, projectile) systems are accumulated.
  (b) Eigenvalue spectrum of the global Fisher matrix with cumulative
      information on a secondary axis.
  (c) Eigenvector composition heatmap for the top stiff directions,
      grouped by parameter category.

Data source: data/deff_global_kd02.json

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# =============================================================================
# Pastel Color Palette (consistent with other paper figures)
# =============================================================================
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
    'light_gray': '#D0D0D0',
}

# Category colors for parameter groups
GROUP_COLORS = {
    'Shared Geometry': '#198754',
    'Shared Depth': '#6F42C1',
    'Neutron-specific': '#2E86C1',
    'Proton-specific': '#D63384',
}

GROUP_COLORS_LIGHT = {
    'Shared Geometry': '#D5F5E3',
    'Shared Depth': '#E6D5F2',
    'Neutron-specific': '#D6EAF8',
    'Proton-specific': '#FFD1DC',
}


def setup_style():
    """Set up matplotlib style for double-column PRL figure."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'legend.fontsize': 7.5,
        'legend.framealpha': 0.85,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'text.usetex': False,
        'mathtext.fontset': 'cm',
    })


def load_data(data_path):
    """Load the global KD02 Fisher analysis data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def panel_a_growth(ax, data):
    """
    Panel (a): D_eff growth as (nucleus, projectile) systems are added.

    Uses nuclei_growth which has 24 entries (12 neutron + 12 proton systems).
    """
    growth = data['nuclei_growth']
    n_nuclei = [g['n_nuclei'] for g in growth]
    deff_vals = [g['D_eff'] for g in growth]

    N_params = data['N_universal_params']  # 45

    # Identify neutron vs proton boundary
    n_neutron = sum(1 for g in growth if g['last_added'].startswith('n+'))

    # Split into neutron and proton series
    n_n = [n for n, g in zip(n_nuclei, growth) if g['last_added'].startswith('n+')]
    d_n = [d for d, g in zip(deff_vals, growth) if g['last_added'].startswith('n+')]
    n_p = [n for n, g in zip(n_nuclei, growth) if g['last_added'].startswith('p+')]
    d_p = [d for d, g in zip(deff_vals, growth) if g['last_added'].startswith('p+')]

    ax.plot(n_n, d_n, 'o-', color=COLORS['dark_green'], markersize=4,
            markerfacecolor=COLORS['mint'], markeredgecolor=COLORS['dark_green'],
            markeredgewidth=0.8, linewidth=1.2, label='Neutron systems', zorder=5)
    ax.plot(n_p, d_p, 's-', color=COLORS['dark_pink'], markersize=4,
            markerfacecolor=COLORS['pink'], markeredgecolor=COLORS['dark_pink'],
            markeredgewidth=0.8, linewidth=1.2, label='Proton systems', zorder=5)

    # Final D_eff reference line
    final_deff = data['D_eff_global']
    ax.axhline(y=final_deff, color=COLORS['dark_purple'], linestyle='--',
               linewidth=0.8, alpha=0.7)
    ax.annotate(f'$D_{{\\mathrm{{eff}}}} \\approx {final_deff:.0f}$',
                xy=(18, final_deff), xytext=(18, 2.85),
                fontsize=8, color=COLORS['dark_purple'], fontweight='bold',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark_purple'],
                                lw=0.7, shrinkA=0, shrinkB=2))

    # N_param annotation
    ax.text(0.97, 0.95,
            f'$N_{{\\mathrm{{param}}}} = {N_params}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            color=COLORS['gray'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=COLORS['light_gray'], alpha=0.9))

    # Shade neutron-only and proton regions
    ax.axvspan(0.5, n_neutron + 0.5, alpha=0.06, color=COLORS['dark_green'],
               zorder=0)
    ax.axvspan(n_neutron + 0.5, 24.5, alpha=0.06, color=COLORS['dark_pink'],
               zorder=0)

    # Boundary marker
    ax.axvline(x=n_neutron + 0.5, color=COLORS['gray'], linestyle='-',
               linewidth=0.5, alpha=0.4)

    # Region labels
    ax.text(n_neutron / 2 + 0.5, 0.22, 'neutrons only',
            ha='center', va='bottom', fontsize=7, color=COLORS['dark_green'],
            fontstyle='italic', alpha=0.8)
    ax.text(n_neutron + (24 - n_neutron) / 2 + 0.5, 0.22, '+ protons',
            ha='center', va='bottom', fontsize=7, color=COLORS['dark_pink'],
            fontstyle='italic', alpha=0.8)

    ax.set_xlabel('Number of (nucleus, projectile) systems')
    ax.set_ylabel('$D_{\\mathrm{eff}}$')
    ax.set_title('(a) $D_{\\mathrm{eff}}$ saturation', fontweight='bold',
                 loc='left')
    ax.set_xlim(0.3, 25.0)
    ax.set_ylim(0, 3.5)
    ax.legend(loc='upper right', frameon=True, edgecolor=COLORS['light_gray'])
    ax.grid(True, alpha=0.15, linewidth=0.5)


def panel_b_eigenvalues(ax, data):
    """
    Panel (b): Eigenvalue spectrum (log scale) with cumulative information
    on a secondary y-axis.
    """
    eigenvalues = np.array(data['eigenvalues_global'])
    N = len(eigenvalues)
    indices = np.arange(1, N + 1)

    # Cumulative fraction of information (trace)
    total = np.sum(eigenvalues)
    cumulative = np.cumsum(eigenvalues) / total * 100

    # Color bars by magnitude category
    colors = []
    for i in range(N):
        if i < 2:
            colors.append(COLORS['dark_green'])
        elif i < 6:
            colors.append(COLORS['dark_purple'])
        elif i < 15:
            colors.append('#5DADE2')
        else:
            colors.append(COLORS['light_gray'])

    # Bar chart of eigenvalues
    ax.bar(indices, eigenvalues, width=0.75, color=colors, alpha=0.85,
           edgecolor='none', zorder=3)

    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue index $i$')
    ax.set_ylabel('Eigenvalue $\\lambda_i$')
    ax.set_title('(b) Eigenvalue spectrum', fontweight='bold', loc='left')

    ax.set_ylim(eigenvalues[-1] * 0.3, eigenvalues[0] * 3)
    ax.set_xlim(0, N + 1)

    # Secondary axis: cumulative information
    ax2 = ax.twinx()
    ax2.plot(indices, cumulative, 'k-', linewidth=1.0, alpha=0.7, zorder=4)
    ax2.plot(indices, cumulative, 'ko', markersize=1.5, alpha=0.5, zorder=4)
    ax2.set_ylabel('Cumulative information (%)', fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='y', labelsize=8)

    # Mark where 90% and 99% of information is reached
    for threshold, ls, y_offset in [(90, '--', -10), (99, ':', -8)]:
        idx_th = np.searchsorted(cumulative, threshold)
        if idx_th < N:
            ax2.axhline(y=threshold, color='gray', linestyle=ls,
                        linewidth=0.5, alpha=0.4)
            text_x = min(idx_th + 8, N - 3)
            ax2.annotate(f'{threshold}% ({idx_th + 1} modes)',
                         xy=(idx_th + 1, threshold),
                         xytext=(text_x, threshold + y_offset),
                         fontsize=6, color=COLORS['gray'], ha='left',
                         arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                                         lw=0.5))

    # Dynamic range annotation
    dyn_range = eigenvalues[0] / eigenvalues[-1]
    ax.text(0.97, 0.50,
            f'$\\lambda_1/\\lambda_{{45}}$\n$= {dyn_range:.0e}$',
            transform=ax.transAxes, ha='right', va='center', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['light_gray'], alpha=0.9))

    # Legend for color categories
    legend_elements = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=COLORS['dark_green'], markersize=5,
               label='Top 2 (stiff)'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=COLORS['dark_purple'], markersize=5,
               label='3--6'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor='#5DADE2', markersize=5,
               label='7--15'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=COLORS['light_gray'], markersize=5,
               label='16--45 (sloppy)'),
        Line2D([0], [0], color='k', linewidth=0.8,
               label='Cumulative'),
    ]
    ax.legend(handles=legend_elements, loc='center right', frameon=True,
              edgecolor=COLORS['light_gray'], fontsize=6.5)

    ax.grid(True, alpha=0.15, which='major', linewidth=0.5)


def panel_c_eigenvectors(ax, data):
    """
    Panel (c): Eigenvector composition heatmap for the top eigenvectors.

    Rows = top eigenvectors (stiff directions), columns = universal parameters
    grouped by category.
    """
    # Diagonalize the global Fisher matrix
    F = np.array(data['fisher_matrix_global'])
    eigenvalues_all, eigenvectors = np.linalg.eigh(F)
    idx = np.argsort(eigenvalues_all)[::-1]
    eigenvalues_sorted = eigenvalues_all[idx]
    eigenvectors = eigenvectors[:, idx]

    param_names = data['universal_param_names']
    groups = data['groups']

    # Number of top eigenvectors to display
    n_evecs = 6

    # Build ordered parameter indices grouped by category
    group_order = ['Shared Geometry', 'Shared Depth',
                   'Neutron-specific', 'Proton-specific']
    ordered_indices = []
    group_boundaries = []
    for g in group_order:
        start = len(ordered_indices)
        ordered_indices.extend(groups[g])
        group_boundaries.append((start, len(ordered_indices)))

    N_cols = len(ordered_indices)

    # Build the eigenvector component matrix (reordered by group)
    evec_matrix = np.zeros((n_evecs, N_cols))
    for i in range(n_evecs):
        for j, param_idx in enumerate(ordered_indices):
            evec_matrix[i, j] = eigenvectors[param_idx, i]

    # Normalize each eigenvector row so max |component| = 1
    for i in range(n_evecs):
        max_abs = np.max(np.abs(evec_matrix[i, :]))
        if max_abs > 0:
            evec_matrix[i, :] /= max_abs

    # Custom diverging colormap: pink-white-green
    cmap_div = LinearSegmentedColormap.from_list(
        'pink_green_div',
        ['#D63384', '#FFD1DC', '#FFFFFF', '#D5F5E3', '#198754'],
        N=256)

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    im = ax.imshow(evec_matrix, aspect='auto', cmap=cmap_div, norm=norm,
                   interpolation='nearest')

    # Y-axis: eigenvector labels with info fractions
    total_info = np.sum(eigenvalues_sorted[eigenvalues_sorted > 0])
    fractions = eigenvalues_sorted / total_info * 100
    y_labels = []
    for i in range(n_evecs):
        y_labels.append(f'$\\mathbf{{e}}_{{{i+1}}}$ ({fractions[i]:.0f}%)')

    ax.set_yticks(np.arange(n_evecs))
    ax.set_yticklabels(y_labels, fontsize=7.5)

    # X-axis: parameter names
    reordered_names = [param_names[i] for i in ordered_indices]
    ax.set_xticks(np.arange(N_cols))
    ax.set_xticklabels(reordered_names, rotation=90, fontsize=5.5,
                       ha='center')

    # Group boundary lines
    for (start, end) in group_boundaries:
        if start > 0:
            ax.axvline(x=start - 0.5, color='k', linewidth=0.8, alpha=0.6)

    # Group labels above the heatmap using axes transform for positioning
    for (start, end), gname in zip(group_boundaries, group_order):
        mid = (start + end - 1) / 2
        # Place in data x-coords but use axes-fraction for y (above plot)
        ax.annotate(gname,
                    xy=(mid, 0), xycoords=('data', 'axes fraction'),
                    xytext=(mid, 1.04), textcoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=6,
                    color=GROUP_COLORS[gname], fontweight='bold',
                    annotation_clip=False)
        # Colored bracket above heatmap
        ax.plot([start - 0.4, end - 0.6], [1.015, 1.015],
                color=GROUP_COLORS[gname], linewidth=2.5,
                transform=ax.get_xaxis_transform(), clip_on=False,
                solid_capstyle='butt')

    ax.set_title('(c) Eigenvector composition', fontweight='bold',
                 loc='left', pad=22)
    ax.set_ylabel('Eigenvector (info fraction)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.015, aspect=20)
    cbar.set_label('Normalized component', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Annotate dominant components
    for i in range(n_evecs):
        for j in range(N_cols):
            val = evec_matrix[i, j]
            if abs(val) > 0.5:
                text_color = 'white' if abs(val) > 0.75 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=4.5, color=text_color, fontweight='bold')


def plot_fig_global(data, save_path):
    """
    Create the three-panel global Fisher analysis figure.

    Layout: 2 rows.
      Top row: panels (a) and (b) side by side.
      Bottom row: panel (c) heatmap spanning full width.
    """
    setup_style()

    fig = plt.figure(figsize=(7.0, 7.0))

    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          hspace=0.50, wspace=0.55,
                          left=0.10, right=0.93, top=0.97, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_a_growth(ax_a, data)
    panel_b_eigenvalues(ax_b, data)
    panel_c_eigenvectors(ax_c, data)

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    plt.close()


def main():
    print("=" * 60)
    print("PRL Figure: Global KD02 Fisher Information Analysis")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    data_path = os.path.join(data_dir, 'deff_global_kd02.json')

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("Run analysis/deff_global_kd02.py first.")
        return

    data = load_data(data_path)

    print(f"N_universal_params: {data['N_universal_params']}")
    print(f"N_systems: {data['N_systems']}")
    print(f"D_eff_global: {data['D_eff_global']:.4f}")
    print(f"D_eff_neutron: {data['D_eff_neutron']:.4f}")
    print(f"D_eff_proton: {data['D_eff_proton']:.4f}")
    print(f"Local D_eff mean: {data['local_deff_mean']:.4f} +/- {data['local_deff_std']:.4f}")

    eigenvalues = np.array(data['eigenvalues_global'])
    print(f"\nEigenvalue range: {eigenvalues[0]:.3e} to {eigenvalues[-1]:.3e}")
    print(f"Dynamic range: {eigenvalues[0] / eigenvalues[-1]:.3e}")
    print(f"Condition number: {eigenvalues[0] / eigenvalues[-1]:.1e}")

    save_path = os.path.join(base_dir, 'fig_global_fisher.png')
    plot_fig_global(data, save_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
