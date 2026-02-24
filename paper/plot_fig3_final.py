#!/usr/bin/env python3
"""
PRL Figure 3: Eigenvector Analysis (Final Version)

Shows the composition of the dominant eigenvectors of the Fisher Information Matrix,
revealing what physics each constrained direction corresponds to.

Uses the PINN model for consistent results with the paper's analysis.

Author: Jin Lei
Date: December 2024
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/jinlei/Desktop/code/PINN_CFC')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Physical constants
HBARC = 197.327  # MeV·fm
AMU = 931.494    # MeV/c²
E2 = 1.44        # e²/(4πε₀) in MeV·fm

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
    """Set up matplotlib style for single-column figure (x1.25 fonts)."""
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


# =============================================================================
# Import PINN model components from deff_autograd.py
# =============================================================================
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class BidirectionalCfC(nn.Module):
    """Bidirectional CfC for wave function prediction."""

    def __init__(self, input_size=9, hidden_size=256, n_units=64):
        super().__init__()
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        cfc_output = min(32, n_units - 4)
        wiring_fwd = AutoNCP(n_units, cfc_output)
        self.cfc_forward = CfC(hidden_size, wiring_fwd, batch_first=True)

        wiring_bwd = AutoNCP(n_units, cfc_output)
        self.cfc_backward = CfC(hidden_size, wiring_bwd, batch_first=True)

        self.combiner = nn.Sequential(
            nn.Linear(cfc_output * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        h = self.encoder(x)
        h_fwd, _ = self.cfc_forward(h)
        h_flipped = h.flip(dims=[1])
        h_bwd, _ = self.cfc_backward(h_flipped)
        h_bwd = h_bwd.flip(dims=[1])
        h_combined = torch.cat([h_fwd, h_bwd], dim=-1)
        h_combined = self.combiner(h_combined)
        psi = self.decoder(h_combined)
        return psi[:, :, 0], psi[:, :, 1]


def woods_saxon(r, R, a):
    """Woods-Saxon form factor."""
    return 1.0 / (1.0 + np.exp((r - R) / a))


def derivative_ws(r, R, a):
    """Derivative Woods-Saxon: 4*f*(1-f)."""
    f = woods_saxon(r, R, a)
    return 4.0 * f * (1.0 - f)


def coulomb_potential(r, Z1, Z2, Rc):
    """Coulomb potential."""
    V_C = np.zeros_like(r)
    if Z1 * Z2 < 1e-6:
        return V_C
    e2 = 1.44
    inside = r < Rc
    V_C[inside] = Z1 * Z2 * e2 / (2 * Rc) * (3 - r[inside]**2 / Rc**2)
    V_C[~inside] = Z1 * Z2 * e2 / r[~inside]
    return V_C


def kd02_potential_9params(r, A, Z_proj, Z, params):
    """KD02 potential with 9 parameters."""
    V, rv, av, W, rw, aw, Wd, rvd, avd = params
    A_third = A ** (1./3.)

    V_real = -V * woods_saxon(r, rv * A_third, av)
    if Z_proj > 0:
        rc = 1.198 + 0.697 * A**(-2./3.) + 12.994 * A**(-5./3.)
        Rc = rc * A_third
        V_real = V_real + coulomb_potential(r, Z_proj, Z, Rc)

    V_imag = -W * woods_saxon(r, rw * A_third, aw)
    V_imag = V_imag - Wd * derivative_ws(r, rvd * A_third, avd)

    return V_real, V_imag


def get_kd02_params(projectile, A, Z, E_lab):
    """Get KD02 parameters."""
    from potentials import KD02Potential
    pot = KD02Potential(projectile, A, Z, E_lab)
    return np.array([
        pot.V, pot.rv, pot.av,
        pot.W, pot.rw, pot.aw,
        pot.Wd, pot.rvd, pot.avd
    ])


def prepare_features_batch(r_mesh, V_real, V_imag, E_cm, k, eta, l_values, A, l_max, r_max):
    """Prepare features for multiple l values at once."""
    n_points = len(r_mesh)
    n_l = len(l_values)

    rho = k * r_mesh
    rho_max = k * r_max
    rho_norm = rho / rho_max

    V_real_rel = V_real / (E_cm + 1e-6)
    V_imag_rel = V_imag / (E_cm + 1e-6)
    V_real_norm = np.clip(V_real_rel / 3.0, -1, 1)
    V_imag_norm = np.clip(V_imag_rel / 0.5, -1, 1)

    eta_norm = eta / 20.0
    A_norm = (A ** (1./3.)) / 6.0

    dr = r_mesh[1] - r_mesh[0]
    k_local = np.sqrt(np.maximum(E_cm - V_real, 0.01))
    phase = np.cumsum(k_local * dr)
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    decay = np.cumsum(-V_imag / (2 * k_local + 1e-6) * dr)
    decay_norm = decay / (np.abs(decay).max() + 1e-8)

    features = np.zeros((n_l, n_points, 9), dtype=np.float32)
    for i, l in enumerate(l_values):
        l_norm = l / l_max
        features[i, :, 0] = rho_norm
        features[i, :, 1] = V_real_norm
        features[i, :, 2] = V_imag_norm
        features[i, :, 3] = eta_norm
        features[i, :, 4] = l_norm
        features[i, :, 5] = A_norm
        features[i, :, 6] = sin_phase
        features[i, :, 7] = cos_phase
        features[i, :, 8] = decay_norm

    return features


def compute_smatrix_from_psi(psi_real, psi_imag, r, k, eta, l, V_real, V_imag):
    """Extract S-matrix using amplitude integral."""
    dr = r[1] - r[0]
    psi = psi_real + 1j * psi_imag
    V = V_real + 1j * V_imag

    kr = k * r
    phase = kr - l * np.pi / 2
    if eta > 0.01:
        phase = phase - eta * np.log(2 * kr + 1e-10)
    F_l = np.sin(phase)

    integrand = V * psi * F_l
    integral = np.sum(integrand) * dr
    S = 1.0 + 2j * k * integral

    return S


def compute_cross_section_from_smatrix(S_list, k, theta):
    """Compute differential cross section from S-matrices."""
    cos_theta = np.cos(theta)
    l_max = len(S_list) - 1

    P = np.zeros((l_max + 1, len(theta)))
    P[0] = 1.0
    if l_max >= 1:
        P[1] = cos_theta
    for l in range(2, l_max + 1):
        P[l] = ((2*l - 1) * cos_theta * P[l-1] - (l - 1) * P[l-2]) / l

    f = np.zeros(len(theta), dtype=complex)
    for l in range(l_max + 1):
        f += (2*l + 1) * (S_list[l] - 1) / (2j * k) * P[l]

    return np.abs(f) ** 2


def compute_cross_section_nn(model, params, projectile, A, Z, E_lab, r_mesh, theta_deg, l_max, device):
    """Compute cross section using neural network."""
    Z_proj = 1 if projectile == 'p' else 0
    mu = A / (1 + A)
    E_cm = E_lab * A / (1 + A)
    k = np.sqrt(2 * mu * AMU * E_cm) / HBARC
    eta = Z_proj * Z * E2 * mu * AMU / (HBARC**2 * k) if Z_proj > 0 else 0.0

    V_real, V_imag = kd02_potential_9params(r_mesh, A, Z_proj, Z, params)

    l_values = list(range(l_max + 1))
    features = prepare_features_batch(r_mesh, V_real, V_imag, E_cm, k, eta, l_values, A, l_max, r_mesh[-1])

    with torch.no_grad():
        feat_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        psi_real, psi_imag = model(feat_tensor)
        psi_real = psi_real.cpu().numpy()
        psi_imag = psi_imag.cpu().numpy()

    S_list = []
    for i, l in enumerate(l_values):
        S = compute_smatrix_from_psi(psi_real[i], psi_imag[i], r_mesh, k, eta, l, V_real, V_imag)
        S_list.append(S)

    theta_rad = np.deg2rad(theta_deg)
    sigma = compute_cross_section_from_smatrix(S_list, k, theta_rad)

    return sigma


def compute_fisher_with_eigenvectors_nn(model, projectile, A, Z, E_lab, params, theta_deg, l_max, device, eps_rel=0.01, use_log_derivatives=True):
    """
    Compute Fisher Information Matrix with eigenvectors using neural network.

    If use_log_derivatives=True, use relative sensitivities (d log sigma / d log p)
    which makes parameters with different physical units comparable.
    """
    r_mesh = np.linspace(0.1, 15.0, 100)
    n_params = len(params)
    n_angles = len(theta_deg)

    sigma_0 = compute_cross_section_nn(model, params, projectile, A, Z, E_lab, r_mesh, theta_deg, l_max, device)

    gradients = np.zeros((n_params, n_angles))

    for i in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()

        delta = eps_rel * abs(params[i])
        if delta < 1e-8:
            delta = 1e-8

        params_plus[i] += delta
        params_minus[i] -= delta

        sigma_plus = compute_cross_section_nn(model, params_plus, projectile, A, Z, E_lab, r_mesh, theta_deg, l_max, device)
        sigma_minus = compute_cross_section_nn(model, params_minus, projectile, A, Z, E_lab, r_mesh, theta_deg, l_max, device)

        if use_log_derivatives:
            # Relative sensitivity: d log(sigma) / d log(p) = p/sigma * dsigma/dp
            # This makes parameters with different units comparable
            gradients[i] = params[i] * (sigma_plus - sigma_minus) / (2 * delta) / (sigma_0 + 1e-20)
        else:
            gradients[i] = (sigma_plus - sigma_minus) / (2 * delta)

    # Fisher Information Matrix
    if use_log_derivatives:
        # For log-derivatives, use unit weights (relative error model)
        weights = np.ones(n_angles)
    else:
        weights = 1.0 / (sigma_0 ** 2 + 1e-20)

    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            F[i, j] = np.sum(weights * gradients[i] * gradients[j])

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(F)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return F, eigenvalues, eigenvectors, gradients, sigma_0


def plot_fig3(save_path, model, device):
    """
    PRL Figure 3: Eigenvector composition
    Shows what physics each constrained direction corresponds to.
    Uses CfC neural network for consistent results.
    """
    setup_style()

    # Compute for 40Ca at 50 MeV (neutron) - consistent with paper
    # Using log-derivatives for dimensionless comparison across parameters
    A, Z = 40, 20
    E = 50
    proj = 'n'
    l_max = 20

    print(f"Computing FIM for n + 40Ca @ {E} MeV using CfC network...")

    params = get_kd02_params(proj, A, Z, E)
    param_names = ['$V$', '$r_v$', '$a_v$', '$W$', '$r_w$', '$a_w$', '$W_d$', '$r_d$', '$a_d$']
    theta_deg = np.linspace(10, 170, 17)

    F, eigenvalues, eigenvectors, gradients, sigma_0 = compute_fisher_with_eigenvectors_nn(
        model, proj, A, Z, E, params, theta_deg, l_max, device
    )

    # Compute D_eff
    sum_lambda = np.sum(eigenvalues[eigenvalues > 0])
    sum_lambda2 = np.sum(eigenvalues[eigenvalues > 0]**2)
    D_eff = sum_lambda**2 / sum_lambda2

    # Eigenvalue fractions
    total_info = np.sum(eigenvalues[eigenvalues > 0])
    fractions = eigenvalues / total_info * 100

    print(f"D_eff = {D_eff:.2f}")
    print(f"Eigenvalue fractions: {fractions[:3]}")

    # Print eigenvector details for debugging
    param_names_raw = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd']
    e1 = eigenvectors[:, 0]
    e2 = eigenvectors[:, 1]

    print("\nEigenvector e1 (largest eigenvalue):")
    for i, name in enumerate(param_names_raw):
        contrib = e1[i]**2 / np.sum(e1**2) * 100
        print(f"  {name}: {e1[i]:.4f} (contrib: {contrib:.1f}%)")

    print("\nEigenvector e2 (second largest):")
    for i, name in enumerate(param_names_raw):
        contrib = e2[i]**2 / np.sum(e2**2) * 100
        print(f"  {name}: {e2[i]:.4f} (contrib: {contrib:.1f}%)")

    # Create figure - single panel showing eigenvector composition
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Use absolute values squared (contribution to variance)
    e1_contrib = e1**2
    e2_contrib = e2**2

    # Normalize
    e1_contrib = e1_contrib / np.sum(e1_contrib) * 100
    e2_contrib = e2_contrib / np.sum(e2_contrib) * 100

    x = np.arange(len(param_names))
    width = 0.35

    # Add background shading to separate Real Volume (V, rv) from Other Parameters
    ax.axvspan(-0.5, 1.5, alpha=0.15, color=COLORS['mint'], zorder=0)
    ax.axvspan(1.5, 8.5, alpha=0.08, color=COLORS['lavender'], zorder=0)
    ax.axvline(x=1.5, color=COLORS['gray'], linestyle='--', linewidth=1.5, alpha=0.7)

    # Add region labels at top
    ax.text(0.5, 95, 'Real Volume', ha='center', fontsize=16, fontweight='bold', color=COLORS['dark_green'])
    ax.text(5.0, 95, 'Other Parameters', ha='center', fontsize=16, fontweight='bold', color=COLORS['dark_purple'])

    # Determine which parameters dominate each eigenvector
    e1_dominant_idx = np.argmax(e1_contrib)
    e2_dominant_idx = np.argmax(e2_contrib)

    # Label based on actual dominant content
    # Identify which parameters dominate each eigenvector
    e1_dominant = param_names_raw[np.argmax(e1_contrib)]
    e2_dominant = param_names_raw[np.argmax(e2_contrib)]

    # Determine physical interpretation
    e1_interp = "Potential Scale" if e1_dominant == 'V' else "Volume"
    e2_interp = "Surface" if 'vd' in e2_dominant or 'd' in e2_dominant else "Volume Radius"

    e1_label = f'$\\mathbf{{e}}_1$ ({fractions[0]:.0f}% info): {e1_interp}'
    e2_label = f'$\\mathbf{{e}}_2$ ({fractions[1]:.0f}% info): {e2_interp}'

    bars1 = ax.bar(x - width/2, e1_contrib, width,
                   label=e1_label,
                   color=COLORS['green'], alpha=0.9, edgecolor=COLORS['dark_green'], linewidth=1.5)
    bars2 = ax.bar(x + width/2, e2_contrib, width,
                   label=e2_label,
                   color=COLORS['pink'], alpha=0.9, edgecolor=COLORS['dark_pink'], linewidth=1.5, hatch='//')

    ax.set_xlabel('Parameter')
    ax.set_ylabel('Contribution to Eigenvector (%)')
    ax.set_title(f'$n+^{{40}}$Ca, {E} MeV ($D_{{\\mathrm{{eff}}}}$ = {D_eff:.2f})', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.legend(loc='center right', fontsize=18)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotations for dominant components
    for i, (c1, c2) in enumerate(zip(e1_contrib, e2_contrib)):
        if c1 > 15:
            ax.annotate(f'{c1:.0f}%', (i - width/2, c1 + 2), ha='center', fontsize=18, fontweight='bold')
        if c2 > 15:
            ax.annotate(f'{c2:.0f}%', (i + width/2, c2 + 2), ha='center', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("PRL Figure 3: Eigenvector Analysis (CfC Network)")
    print("="*60)

    base_dir = os.path.dirname(__file__)

    # Load CfC model (from PRC_Neural_Solver directory)
    model_path = '/Users/jinlei/Desktop/code/PINN_CFC/PRC_Neural_Solver/stage2_bicfc_moresamples.pt'
    print(f"\nLoading CfC model: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    print(f"Model error: {checkpoint['rel_error']:.2f}%")

    model = BidirectionalCfC(
        input_size=9,
        hidden_size=config['hidden_size'],
        n_units=config['n_units']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Generate figure
    save_path = os.path.join(base_dir, 'fig3_eigenvectors.png')
    plot_fig3(save_path, model, device)

    print("\nDone!")


if __name__ == '__main__':
    main()
