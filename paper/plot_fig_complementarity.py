#!/usr/bin/env python3
"""
PRL Figure: Observable Complementarity and Multi-Energy Analysis

Panel (a): Bar chart of D_eff for different observable combinations
           (averaged over 168 configurations)
Panel (b): D_eff vs number of energies combined

Addresses referee request: shows how Ay, sigma_R, sigma_T increase D_eff
beyond elastic-only, and how multi-energy analysis further improves it.

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


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 18,
        'axes.linewidth': 1.5,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_fig_complementarity(ext_data, multi_data, save_path):
    """
    Panel (a): Observable complementarity bar chart
    Panel (b): D_eff vs number of combined energies
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # === Panel (a): Observable combinations ===
    ax1 = axes[0]

    combos = [
        ('elastic_9p', r'$d\sigma/d\Omega$ (9p)'),
        ('elastic_11p', r'$d\sigma/d\Omega$ (11p)'),
        ('elastic_Ay_11p', r'$d\sigma/d\Omega$ + $A_y$'),
        ('elastic_sigR_11p', r'$d\sigma/d\Omega$ + $\sigma_R$'),
        ('elastic_Ay_sigR_11p', r'$d\sigma/d\Omega$ + $A_y$ + $\sigma_R$'),
        ('all_11p', r'All obs. (11p)'),
        ('all_9p', r'All obs. (9p)'),
    ]

    entries = [d for d in ext_data['data'] if 'deff_results' in d]

    means = []
    stds = []
    labels = []
    for key, label in combos:
        vals = [d['deff_results'][key]['D_eff'] for d in entries
                if key in d['deff_results']]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            labels.append(label)

    x = np.arange(len(means))
    bar_colors = [
        COLORS['green'],     # elastic 9p
        '#A3E4BC',           # elastic 11p
        COLORS['pink'],      # +Ay
        COLORS['lavender'],  # +sigR
        COLORS['purple'],    # +Ay+sigR
        COLORS['dark_purple'],  # all 11p
        COLORS['dark_green'],   # all 9p
    ][:len(means)]

    bars = ax1.barh(x, means, xerr=stds, height=0.6,
                    color=bar_colors, edgecolor='white', linewidth=1.5,
                    capsize=4, error_kw={'linewidth': 1.5})

    # Value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(m + s + 0.05, i, f'{m:.2f}', va='center', fontsize=14,
                 fontweight='bold')

    ax1.set_yticks(x)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('$D_\\mathrm{eff}$')
    ax1.set_title('(a) Observable Complementarity', pad=10)
    ax1.set_xlim(0, max(means) + max(stds) + 0.8)
    ax1.axvline(means[0], color=COLORS['gray'], linestyle=':', linewidth=1.5,
                alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')

    # === Panel (b): Multi-energy D_eff ===
    ax2 = axes[1]

    if multi_data and 'combinations' in multi_data:
        # Collect D_eff vs N for all systems
        max_N = max(len(c['multi_energy_deff']) for c in multi_data['combinations'])

        # Group by projectile
        for proj, color, marker, label in [
                ('n', COLORS['dark_green'], 'o', 'Neutron'),
                ('p', COLORS['dark_pink'], 's', 'Proton')]:
            proj_combos = [c for c in multi_data['combinations']
                           if c['projectile'] == proj]
            if not proj_combos:
                continue

            # Average across nuclei
            all_deff_vs_N = []
            for c in proj_combos:
                deff_N = [d['D_eff'] for d in c['multi_energy_deff']]
                all_deff_vs_N.append(deff_N)

            # Pad shorter sequences
            max_len = max(len(d) for d in all_deff_vs_N)
            padded = np.full((len(all_deff_vs_N), max_len), np.nan)
            for i, d in enumerate(all_deff_vs_N):
                padded[i, :len(d)] = d

            mean_deff = np.nanmean(padded, axis=0)
            std_deff = np.nanstd(padded, axis=0)
            N_vals = np.arange(1, max_len + 1)

            ax2.fill_between(N_vals, mean_deff - std_deff,
                             mean_deff + std_deff,
                             alpha=0.2, color=color)
            ax2.plot(N_vals, mean_deff, f'-{marker}', color=color,
                     linewidth=2.5, markersize=8, markerfacecolor='white',
                     markeredgewidth=2, label=label)

            # Also show individual systems as faint lines
            for c in proj_combos:
                deff_N = [d['D_eff'] for d in c['multi_energy_deff']]
                ax2.plot(range(1, len(deff_N) + 1), deff_N,
                         '-', color=color, alpha=0.1, linewidth=0.8)

        # Reference line for single-energy elastic-only
        if ext_data:
            elastic_9p_vals = [d['deff_results']['elastic_9p']['D_eff']
                               for d in ext_data['data']
                               if 'deff_results' in d
                               and 'elastic_9p' in d['deff_results']]
            if elastic_9p_vals:
                ax2.axhline(np.mean(elastic_9p_vals), color=COLORS['gray'],
                            linestyle=':', linewidth=1.5,
                            label=f'Single-E elastic ({np.mean(elastic_9p_vals):.1f})')

        ax2.set_xlabel('Number of Energies Combined')
        ax2.set_ylabel('$D_\\mathrm{eff}$')
        ax2.set_title('(b) Multi-Energy Fisher Analysis', pad=10)
        ax2.legend(loc='lower right', fontsize=14)
        ax2.set_xlim(0.5, 7.5)
        ax2.set_xticks(range(1, 8))
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Multi-energy data\nnot available',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=16, color=COLORS['gray'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("PRL Figure: Observable Complementarity + Multi-Energy")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', 'data')

    # Load extended scan data
    ext_path = os.path.join(data_dir, 'deff_scan_extended.json')
    ext_data = None
    if os.path.exists(ext_path):
        with open(ext_path, 'r') as f:
            ext_data = json.load(f)
        print(f"Loaded extended scan: {len(ext_data['data'])} entries")
    else:
        print(f"WARNING: {ext_path} not found")

    # Load multi-energy data
    multi_path = os.path.join(data_dir, 'deff_multi_energy.json')
    multi_data = None
    if os.path.exists(multi_path):
        with open(multi_path, 'r') as f:
            multi_data = json.load(f)
        print(f"Loaded multi-energy: {len(multi_data.get('combinations', []))} systems")
    else:
        print(f"WARNING: {multi_path} not found")

    if ext_data is None:
        print("ERROR: Need at least deff_scan_extended.json")
        return

    save_path = os.path.join(base_dir, 'fig_complementarity.png')
    plot_fig_complementarity(ext_data, multi_data, save_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
