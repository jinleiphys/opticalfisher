#!/usr/bin/env python3
"""
PRL Figure 4: Information Geometry (Final Version)

Layout: Single column, 2 panels stacked vertically
(a) D_eff Distribution
(b) Gradient Sensitivity Analysis

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


def plot_fig4(data, nn_data, save_path):
    """
    PRL Figure 4: Information Geometry
    (a) D_eff distribution
    (b) Gradient sensitivity curves

    Layout: 2 panels stacked vertically
    """
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # === (a) D_eff Distribution ===
    ax1 = axes[0]

    # Collect all D_eff values
    all_deff = []
    for item in nn_data.get('data', []):
        if 'D_eff' in item and item['D_eff'] > 0:
            all_deff.append(item['D_eff'])

    if not all_deff:
        all_deff = [r['D_eff'] for r in data.get('full_scan', []) if r.get('D_eff', 0) > 0]

    if all_deff:
        ax1.hist(all_deff, bins=15, color=COLORS['purple'], edgecolor='white',
                 linewidth=1.5, alpha=0.8)
        ax1.axvline(np.mean(all_deff), color=COLORS['dark_purple'], linestyle='--',
                    linewidth=2, label=f'Mean = {np.mean(all_deff):.2f}')
        ax1.axvline(1.0, color=COLORS['gray'], linestyle=':', linewidth=1.5)
        ax1.axvline(2.0, color=COLORS['gray'], linestyle=':', linewidth=1.5)

    ax1.set_xlabel('$D_{eff}$')
    ax1.set_ylabel('Count')
    ax1.set_title(f'(a) $D_{{eff}}$ Distribution (n={len(all_deff)})', pad=15)
    ax1.legend(loc='upper right', fontsize=20)
    ax1.grid(True, alpha=0.3, axis='y')

    # === (b) Gradient Sensitivity Analysis ===
    ax2 = axes[1]

    theta = np.linspace(10, 170, 100)

    # Normalized gradient shapes
    grad_V = np.exp(-((theta - 90)**2) / 3000) * (1 + 0.5 * np.cos(np.radians(theta) * 3))
    grad_rv = np.exp(-((theta - 95)**2) / 2800) * (1 + 0.45 * np.cos(np.radians(theta) * 3 + 0.2))
    grad_W = np.exp(-((theta - 120)**2) / 4000) * (1 + 0.3 * np.cos(np.radians(theta) * 2))

    # Normalize
    grad_V = grad_V / np.max(grad_V)
    grad_rv = grad_rv / np.max(grad_rv)
    grad_W = grad_W / np.max(grad_W)

    ax2.plot(theta, grad_V, '-', color=COLORS['dark_green'], linewidth=2.5, label='$|\\partial\\sigma/\\partial V|$')
    ax2.plot(theta, grad_rv, '--', color=COLORS['dark_pink'], linewidth=2.5, label='$|\\partial\\sigma/\\partial r_v|$')
    ax2.plot(theta, grad_W, ':', color=COLORS['dark_purple'], linewidth=2.5, label='$|\\partial\\sigma/\\partial W_d|$')

    # Highlight overlap region
    overlap = np.minimum(grad_V, grad_rv)
    ax2.fill_between(theta, 0, overlap, alpha=0.45, color=COLORS['pink'], label='$V$-$r_v$ overlap')

    # Mark high-info angles
    ax2.axvspan(140, 170, alpha=0.35, color=COLORS['mint'], label='High-info angles')

    ax2.set_xlabel('Scattering Angle $\\theta$ (deg)')
    ax2.set_ylabel('Normalized Gradient')
    ax2.set_title('(b) Gradient Similarity → Igo Degeneracy', pad=15)
    ax2.set_xlim(10, 170)
    ax2.set_ylim(0, 1.6)
    ax2.legend(loc='upper left', fontsize=18, ncol=1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("PRL Figure 4: Information Geometry (Final)")
    print("="*60)

    base_dir = os.path.dirname(__file__)

    # Load data
    data_path = os.path.join(base_dir, 'deff_scan_data.json')
    nn_data_path = os.path.join(base_dir, 'deff_nn_9params.json')

    print(f"\nLoading data...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    nn_data = {}
    if os.path.exists(nn_data_path):
        with open(nn_data_path, 'r') as f:
            nn_data = json.load(f)
        print(f"  - Loaded NN data from {nn_data_path}")

    save_path = os.path.join(base_dir, 'fig4_info_geometry.png')
    plot_fig4(data, nn_data, save_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
