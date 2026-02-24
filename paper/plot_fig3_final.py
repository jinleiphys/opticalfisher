#!/usr/bin/env python3
"""
PRL Figure 3: Eigenvector Analysis (Numerov-based)

Shows the composition of the dominant eigenvectors of the Fisher Information
Matrix, revealing what physics each constrained direction corresponds to.

Uses Numerov solver with finite-difference gradients (no NN dependencies).

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
    """Set up matplotlib style for single-column figure."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 22,
        'axes.linewidth': 1.8,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 22,
        'legend.fontsize': 18,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def compute_fisher_numerov(proj, A, Z, E_lab, param_names, n_params=11):
    """
    Compute Fisher matrix and eigenvectors using Numerov solver.

    Returns F, eigenvalues (descending), eigenvectors, D_eff.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
    from deff_scan_extended import (compute_fisher_extended, get_kd02_params_11)

    theta_deg = np.linspace(10, 170, 17)
    params, rvso, avso = get_kd02_params_11(proj, A, Z, E_lab)

    if n_params == 9:
        # Use only central potential parameters
        params_use = params[:9]
        F_full, gradients, obs_0, n_data = compute_fisher_extended(
            proj, A, Z, E_lab, theta_deg, params, rvso, avso)
        # Extract 9x9 sub-block
        F = F_full[:9, :9]
    else:
        F, gradients, obs_0, n_data = compute_fisher_extended(
            proj, A, Z, E_lab, theta_deg, params, rvso, avso)

    eigenvalues, eigenvectors = np.linalg.eigh(F)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    sum_lambda = np.sum(eigenvalues[eigenvalues > 0])
    sum_lambda2 = np.sum(eigenvalues[eigenvalues > 0]**2)
    D_eff = sum_lambda**2 / sum_lambda2 if sum_lambda2 > 0 else 0

    return F, eigenvalues, eigenvectors, D_eff, params


def load_elastic_fisher_from_gradients(grad_path, proj, nucleus, E, n_params=11):
    """
    Load gradient data and build elastic-only Fisher matrix.

    This gives the Fisher matrix for dsigma/dOmega only (not Ay, sigma_R, sigma_T),
    which is what the paper discusses for the eigenvector analysis.
    """
    if not os.path.exists(grad_path):
        return None, None, None, None

    with open(grad_path, 'r') as f:
        data = json.load(f)

    for case in data['cases']:
        if (case.get('projectile') == proj
                and case.get('nucleus') == nucleus
                and case.get('E') == E):
            grads = np.array(case['gradients'])
            n_dcs = case['n_data']['dcs']

            # Extract elastic-only gradients for n_params parameters
            g = grads[:n_params, :n_dcs]
            F = g @ g.T

            eigenvalues, eigenvectors = np.linalg.eigh(F)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            ev_pos = eigenvalues[eigenvalues > 0]
            D_eff = np.sum(ev_pos)**2 / np.sum(ev_pos**2) if len(ev_pos) > 0 else 0
            return F, eigenvalues, eigenvectors, D_eff

    return None, None, None, None


def load_fisher_from_json(data_path, proj, nucleus, E):
    """Try to load pre-computed full Fisher matrix from extended scan data."""
    if not os.path.exists(data_path):
        return None, None, None, None

    with open(data_path, 'r') as f:
        data = json.load(f)

    for entry in data['data']:
        if (entry.get('projectile') == proj
                and entry.get('nucleus') == nucleus
                and entry.get('E') == E
                and 'fisher_matrix' in entry):
            F = np.array(entry['fisher_matrix'])
            eigenvalues, eigenvectors = np.linalg.eigh(F)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            ev_pos = eigenvalues[eigenvalues > 0]
            D_eff = np.sum(ev_pos)**2 / np.sum(ev_pos**2) if len(ev_pos) > 0 else 0
            return F, eigenvalues, eigenvectors, D_eff

    return None, None, None, None


def plot_fig3(save_path, F=None, eigenvalues=None, eigenvectors=None,
              D_eff=None, n_params=11):
    """
    PRL Figure 3: Eigenvector composition.
    Extended to 11 parameters (with spin-orbit).
    """
    setup_style()

    if n_params == 11:
        param_names = ['$V$', '$r_v$', '$a_v$', '$W$', '$r_w$', '$a_w$',
                        '$W_d$', '$r_d$', '$a_d$', '$V_{so}$', '$W_{so}$']
        param_names_raw = ['V', 'rv', 'av', 'W', 'rw', 'aw',
                           'Wd', 'rvd', 'avd', 'Vso', 'Wso']
    else:
        param_names = ['$V$', '$r_v$', '$a_v$', '$W$', '$r_w$', '$a_w$',
                        '$W_d$', '$r_d$', '$a_d$']
        param_names_raw = ['V', 'rv', 'av', 'W', 'rw', 'aw',
                           'Wd', 'rvd', 'avd']

    # Eigenvalue fractions
    total_info = np.sum(eigenvalues[eigenvalues > 0])
    fractions = eigenvalues / total_info * 100

    e1 = eigenvectors[:, 0]
    e2 = eigenvectors[:, 1]

    print(f"D_eff = {D_eff:.2f}")
    print(f"Eigenvalue fractions: {fractions[:5]}")

    print("\nEigenvector e1 (largest eigenvalue):")
    for i, name in enumerate(param_names_raw):
        contrib = e1[i]**2 / np.sum(e1**2) * 100
        print(f"  {name}: {e1[i]:.4f} (contrib: {contrib:.1f}%)")

    print("\nEigenvector e2 (second largest):")
    for i, name in enumerate(param_names_raw):
        contrib = e2[i]**2 / np.sum(e2**2) * 100
        print(f"  {name}: {e2[i]:.4f} (contrib: {contrib:.1f}%)")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    e1_contrib = e1**2 / np.sum(e1**2) * 100
    e2_contrib = e2**2 / np.sum(e2**2) * 100

    x = np.arange(len(param_names))
    width = 0.35

    # Background shading for parameter groups
    ax.axvspan(-0.5, 2.5, alpha=0.15, color=COLORS['mint'], zorder=0)
    ax.axvspan(2.5, 8.5, alpha=0.08, color=COLORS['lavender'], zorder=0)
    if n_params == 11:
        ax.axvspan(8.5, 10.5, alpha=0.10, color=COLORS['blush'], zorder=0)
    ax.axvline(x=2.5, color=COLORS['gray'], linestyle='--', linewidth=1.5, alpha=0.7)
    if n_params == 11:
        ax.axvline(x=8.5, color=COLORS['gray'], linestyle='--', linewidth=1.5, alpha=0.7)

    # Region labels
    ax.text(1.0, 95, 'Real Volume', ha='center', fontsize=16, fontweight='bold',
            color=COLORS['dark_green'])
    ax.text(5.5, 95, 'Imaginary', ha='center', fontsize=16, fontweight='bold',
            color=COLORS['dark_purple'])
    if n_params == 11:
        ax.text(9.5, 95, 'Spin-Orbit', ha='center', fontsize=16, fontweight='bold',
                color=COLORS['dark_pink'])

    # Physical interpretation labels
    e1_dominant = param_names_raw[np.argmax(e1_contrib)]
    e1_interp = "Potential Scale" if e1_dominant == 'V' else "Volume"
    e2_dominant = param_names_raw[np.argmax(e2_contrib)]
    e2_interp = "Surface" if 'd' in e2_dominant.lower() else "Volume Radius"

    e1_label = f'$\\mathbf{{e}}_1$ ({fractions[0]:.0f}% info): {e1_interp}'
    e2_label = f'$\\mathbf{{e}}_2$ ({fractions[1]:.0f}% info): {e2_interp}'

    bars1 = ax.bar(x - width/2, e1_contrib, width,
                   label=e1_label,
                   color=COLORS['green'], alpha=0.9,
                   edgecolor=COLORS['dark_green'], linewidth=1.5)
    bars2 = ax.bar(x + width/2, e2_contrib, width,
                   label=e2_label,
                   color=COLORS['pink'], alpha=0.9,
                   edgecolor=COLORS['dark_pink'], linewidth=1.5, hatch='//')

    ax.set_xlabel('Parameter')
    ax.set_ylabel('Contribution to Eigenvector (%)')
    ax.set_title(f'$n+^{{40}}$Ca, 50 MeV ($D_{{\\mathrm{{eff}}}}$ = {D_eff:.2f})',
                 pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, fontsize=18)
    ax.legend(loc='center right', fontsize=16)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotations for dominant components
    for i, (c1, c2) in enumerate(zip(e1_contrib, e2_contrib)):
        if c1 > 15:
            ax.annotate(f'{c1:.0f}%', (i - width/2, c1 + 2), ha='center',
                        fontsize=18, fontweight='bold')
        if c2 > 15:
            ax.annotate(f'{c2:.0f}%', (i + width/2, c2 + 2), ha='center',
                        fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("PRL Figure 3: Eigenvector Analysis (Numerov)")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', 'data')
    grad_path = os.path.join(data_dir, 'deff_gradients_representative.json')
    ext_path = os.path.join(data_dir, 'deff_scan_extended.json')

    # Use 11-parameter elastic-only Fisher (consistent with paper discussion)
    n_params = 11

    # Try loading elastic-only Fisher from gradient data (preferred)
    F, eigenvalues, eigenvectors, D_eff = load_elastic_fisher_from_gradients(
        grad_path, 'n', '40Ca', 50, n_params)

    if F is not None:
        print(f"Loaded elastic-only Fisher ({n_params}p) from gradient data")
    else:
        # Fall back to full Fisher from scan
        F, eigenvalues, eigenvectors, D_eff = load_fisher_from_json(
            ext_path, 'n', '40Ca', 50)
        if F is not None:
            print("WARNING: Using full Fisher (all observables), not elastic-only")
            n_params = F.shape[0]
        else:
            print("Computing Fisher matrix with Numerov solver...")
            param_names_raw = ['V', 'rv', 'av', 'W', 'rw', 'aw',
                               'Wd', 'rvd', 'avd', 'Vso', 'Wso']
            F, eigenvalues, eigenvectors, D_eff, _ = compute_fisher_numerov(
                'n', 40, 20, 50, param_names_raw, n_params)

    save_path = os.path.join(base_dir, 'fig3_eigenvectors.png')
    plot_fig3(save_path, F, eigenvalues, eigenvectors, D_eff, n_params)

    print("\nDone!")


if __name__ == '__main__':
    main()
