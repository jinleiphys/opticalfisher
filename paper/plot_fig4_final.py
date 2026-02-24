#!/usr/bin/env python3
"""
PRL Figure 4: Information Geometry (Numerov-based)

Layout: Single column, 2 panels stacked vertically
(a) D_eff Distribution across all 168 configurations
(b) Real gradient sensitivity curves from Numerov computation

Replaces previous version that used fake Gaussian curves with actual
computed sensitivities.

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import sys
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


def setup_style():
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


def plot_fig4(ext_data, sens_data, save_path):
    """
    PRL Figure 4: Information Geometry
    (a) D_eff distribution
    (b) Real gradient sensitivity curves
    """
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # === (a) D_eff Distribution ===
    ax1 = axes[0]

    # Collect D_eff values from extended scan (all_11p)
    all_deff = []
    if ext_data:
        for item in ext_data.get('data', []):
            if 'deff_results' in item and 'all_11p' in item['deff_results']:
                all_deff.append(item['deff_results']['all_11p']['D_eff'])

    # Also show elastic_9p for comparison
    elastic_9p_deff = []
    if ext_data:
        for item in ext_data.get('data', []):
            if 'deff_results' in item and 'elastic_9p' in item['deff_results']:
                elastic_9p_deff.append(item['deff_results']['elastic_9p']['D_eff'])

    if elastic_9p_deff:
        ax1.hist(elastic_9p_deff, bins=15, color=COLORS['green'],
                 edgecolor='white', linewidth=1.5, alpha=0.6,
                 label=f'Elastic only (9p): {np.mean(elastic_9p_deff):.2f}')

    if all_deff:
        ax1.hist(all_deff, bins=15, color=COLORS['purple'],
                 edgecolor='white', linewidth=1.5, alpha=0.7,
                 label=f'All obs. (11p): {np.mean(all_deff):.2f}')

    if elastic_9p_deff:
        ax1.axvline(np.mean(elastic_9p_deff), color=COLORS['dark_green'],
                    linestyle='--', linewidth=2)
    if all_deff:
        ax1.axvline(np.mean(all_deff), color=COLORS['dark_purple'],
                    linestyle='--', linewidth=2)

    ax1.axvline(1.0, color=COLORS['gray'], linestyle=':', linewidth=1.5)
    ax1.axvline(2.0, color=COLORS['gray'], linestyle=':', linewidth=1.5)

    n_total = max(len(all_deff), len(elastic_9p_deff))
    ax1.set_xlabel('$D_{\\mathrm{eff}}$')
    ax1.set_ylabel('Count')
    ax1.set_title(f'(a) $D_{{\\mathrm{{eff}}}}$ Distribution (n={n_total})', pad=15)
    ax1.legend(loc='upper right', fontsize=18)
    ax1.grid(True, alpha=0.3, axis='y')

    # === (b) Gradient Sensitivity Analysis (REAL DATA) ===
    ax2 = axes[1]

    if sens_data and 'cases' in sens_data and len(sens_data['cases']) > 0:
        case = sens_data['cases'][0]  # n+40Ca@50MeV
        theta = np.array(sens_data['theta_deg'])
        S_dcs = np.array(case['S_dcs'])
        param_names_raw = sens_data['param_names']

        # Plot key parameters
        key_params = [
            ('V', COLORS['dark_green'], '-', 3.0),
            ('rv', '#2ECC71', '-', 2.5),
            ('av', '#82E0AA', '--', 2.0),
            ('Wd', COLORS['dark_purple'], '-', 2.5),
            ('W', COLORS['dark_pink'], '-', 2.0),
        ]

        for pname, color, ls, lw in key_params:
            if pname in param_names_raw:
                idx = param_names_raw.index(pname)
                label_map = {
                    'V': '$|S_V|$', 'rv': '$|S_{r_v}|$', 'av': '$|S_{a_v}|$',
                    'Wd': '$|S_{W_d}|$', 'W': '$|S_W|$',
                }
                ax2.plot(theta, np.abs(S_dcs[idx]), ls, color=color,
                         linewidth=lw, label=label_map.get(pname, pname))

        # Highlight overlap region between V and rv
        idx_V = param_names_raw.index('V')
        idx_rv = param_names_raw.index('rv')
        S_V_norm = np.abs(S_dcs[idx_V]) / (np.max(np.abs(S_dcs[idx_V])) + 1e-30)
        S_rv_norm = np.abs(S_dcs[idx_rv]) / (np.max(np.abs(S_dcs[idx_rv])) + 1e-30)
        overlap = np.minimum(S_V_norm, S_rv_norm) * np.max(np.abs(S_dcs[idx_V]))
        ax2.fill_between(theta, 0, overlap, alpha=0.3, color=COLORS['pink'],
                         label='$V$-$r_v$ overlap')

        # Mark backward angles
        ax2.axvspan(130, 175, alpha=0.2, color=COLORS['mint'],
                    label='Backward angles')

        ax2.set_yscale('log')
        ax2.set_ylim(1e-3, None)
    else:
        # Fallback: compute on the fly
        ax2.text(0.5, 0.5, 'Sensitivity data not available.\n'
                 'Run: python analysis/angle_resolved_sensitivity.py',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=14, color=COLORS['gray'])

    ax2.set_xlabel('Scattering Angle $\\theta$ (deg)')
    ax2.set_ylabel('$|S_i(\\theta)|$')
    ax2.set_title('(b) Gradient Similarity $\\rightarrow$ Igo Degeneracy', pad=15)
    ax2.set_xlim(5, 175)
    ax2.legend(loc='upper left', fontsize=16, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("PRL Figure 4: Information Geometry (Numerov)")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', 'data')

    # Load extended scan data
    ext_data = None
    ext_path = os.path.join(data_dir, 'deff_scan_extended.json')
    if os.path.exists(ext_path):
        with open(ext_path, 'r') as f:
            ext_data = json.load(f)
        print(f"Loaded extended scan data")
    else:
        # Fall back to older data format
        old_path = os.path.join(base_dir, 'deff_scan_data.json')
        if os.path.exists(old_path):
            with open(old_path, 'r') as f:
                old_data = json.load(f)
            # Convert to ext_data format
            ext_data = {'data': []}
            for r in old_data.get('full_scan', []):
                ext_data['data'].append({
                    'deff_results': {
                        'elastic_9p': {'D_eff': r.get('D_eff', 0)},
                    }
                })
            print("Loaded old scan data (fallback)")

    # Load angle sensitivity data
    sens_data = None
    sens_path = os.path.join(data_dir, 'angle_sensitivity.json')
    if os.path.exists(sens_path):
        with open(sens_path, 'r') as f:
            sens_data = json.load(f)
        print("Loaded angle sensitivity data")
    else:
        print(f"WARNING: {sens_path} not found. Panel (b) will show fallback.")

    save_path = os.path.join(base_dir, 'fig4_info_geometry.png')
    plot_fig4(ext_data, sens_data, save_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
