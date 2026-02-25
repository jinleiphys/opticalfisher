#!/usr/bin/env python3
"""
Compute D_eff using TRUE AUTOMATIC DIFFERENTIATION.

This script uses PyTorch autograd to compute exact gradients d(log sigma)/d(log p),
instead of finite differences. This is more accurate and faster.

Key difference from recompute_deff_log.py:
- Old: finite differences (2*n_params forward passes, numerical error)
- New: autograd (1 forward + 1 backward pass, exact gradients)

Author: Jin Lei
Date: January 2025
"""

import numpy as np
import json
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# Physical constants
HBARC = 197.327  # MeV·fm
AMU = 931.494    # MeV/c²
E2 = 1.44        # e²/(4πε₀) in MeV·fm


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


# ============================================================
# PyTorch-based differentiable functions
# ============================================================

def woods_saxon_torch(r, R, a):
    """Woods-Saxon form factor (PyTorch)."""
    return 1.0 / (1.0 + torch.exp((r - R) / a))


def derivative_ws_torch(r, R, a):
    """Derivative Woods-Saxon (PyTorch)."""
    f = woods_saxon_torch(r, R, a)
    return 4.0 * f * (1.0 - f)


def coulomb_potential_torch(r, Z1, Z2, Rc):
    """Coulomb potential (PyTorch)."""
    if Z1 * Z2 < 1e-6:
        return torch.zeros_like(r)
    e2 = 1.44
    V_C = torch.zeros_like(r)
    inside = r < Rc
    V_C = torch.where(
        inside,
        Z1 * Z2 * e2 / (2 * Rc) * (3 - r**2 / Rc**2),
        Z1 * Z2 * e2 / (r + 1e-10)
    )
    return V_C


def kd02_potential_torch(r, A, Z_proj, Z, params):
    """
    KD02 potential with 9 parameters (PyTorch differentiable).
    params: tensor of shape (9,) with [V, rv, av, W, rw, aw, Wd, rvd, avd]
    """
    V, rv, av, W, rw, aw, Wd, rvd, avd = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
    A_third = A ** (1./3.)

    V_real = -V * woods_saxon_torch(r, rv * A_third, av)

    if Z_proj > 0:
        rc = 1.198 + 0.697 * A**(-2./3.) + 12.994 * A**(-5./3.)
        Rc = rc * A_third
        V_real = V_real + coulomb_potential_torch(r, Z_proj, Z, Rc)

    V_imag = -W * woods_saxon_torch(r, rw * A_third, aw)
    V_imag = V_imag - Wd * derivative_ws_torch(r, rvd * A_third, avd)

    return V_real, V_imag


def prepare_features_torch(r_mesh, V_real, V_imag, E_cm, k, eta, l_values, A, l_max, r_max, config):
    """
    Prepare features for NN (PyTorch differentiable).
    Returns tensor of shape (n_l, n_points, 9).

    NOTE: Must match prepare_features_batch in recompute_deff_log.py exactly!
    """
    n_points = len(r_mesh)
    n_l = len(l_values)
    device = r_mesh.device

    rho = k * r_mesh
    # Use k * r_max to match numpy version (NOT config['rho_max'])
    rho_max = k * r_max
    rho_norm = rho / rho_max

    V_real_rel = V_real / (E_cm + 1e-6)
    V_imag_rel = V_imag / (E_cm + 1e-6)
    V_real_norm = torch.clamp(V_real_rel / 3.0, -1, 1)
    V_imag_norm = torch.clamp(V_imag_rel / 0.5, -1, 1)

    eta_norm = eta / 20.0
    A_norm = (A ** (1./3.)) / 6.0

    dr = r_mesh[1] - r_mesh[0]
    k_local = torch.sqrt(torch.clamp(E_cm - V_real, min=0.01))
    phase = torch.cumsum(k_local * dr, dim=0)
    sin_phase = torch.sin(phase)
    cos_phase = torch.cos(phase)

    decay = torch.cumsum(-V_imag / (2 * k_local + 1e-6) * dr, dim=0)
    # Use dynamic decay_max to match numpy version (NOT config['decay_max'])
    decay_max = torch.abs(decay).max() + 1e-8
    decay_norm = decay / decay_max

    # Build features for all l values
    features = torch.zeros((n_l, n_points, 9), dtype=torch.float32, device=device)
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


def compute_smatrix_torch(psi_real, psi_imag, r, k, eta, l, V_real, V_imag):
    """
    Extract S-matrix using amplitude integral (PyTorch differentiable).
    """
    dr = r[1] - r[0]
    psi = psi_real + 1j * psi_imag
    V = V_real + 1j * V_imag

    kr = k * r
    phase = kr - l * np.pi / 2
    if eta > 0.01:
        phase = phase - eta * torch.log(2 * kr + 1e-10)
    F_l = torch.sin(phase)

    integrand = V * psi * F_l
    integral = torch.sum(integrand) * dr
    S = 1.0 + 2j * k * integral

    return S


def compute_cross_section_torch(S_list, k, theta):
    """
    Compute differential cross section from S-matrices (PyTorch differentiable).
    Returns |f(theta)|^2.
    """
    cos_theta = torch.cos(theta)
    l_max = len(S_list) - 1
    n_theta = len(theta)
    device = theta.device

    # Legendre polynomials
    P = torch.zeros((l_max + 1, n_theta), device=device, dtype=torch.float32)
    P[0] = 1.0
    if l_max >= 1:
        P[1] = cos_theta
    for l in range(2, l_max + 1):
        P[l] = ((2*l - 1) * cos_theta * P[l-1] - (l - 1) * P[l-2]) / l

    # Scattering amplitude
    f = torch.zeros(n_theta, dtype=torch.complex64, device=device)
    for l in range(l_max + 1):
        S_l = S_list[l]
        f = f + (2*l + 1) * (S_l - 1) / (2j * k) * P[l]

    # Cross section = |f|^2
    sigma = torch.abs(f) ** 2
    return sigma


def compute_cross_section_differentiable(model, params, projectile, A, Z, E_lab, r_mesh, theta, l_max, device, config):
    """
    Compute cross section with full differentiability through the NN.

    params: torch tensor with requires_grad=True
    Returns: cross section tensor (differentiable w.r.t. params)
    """
    Z_proj = 1 if projectile == 'p' else 0
    mu = A / (1 + A)
    E_cm = E_lab * A / (1 + A)
    k = np.sqrt(2 * mu * AMU * E_cm) / HBARC
    eta = Z_proj * Z * E2 * mu * AMU / (HBARC**2 * k) if Z_proj > 0 else 0.0

    # Compute potential (differentiable w.r.t. params)
    V_real, V_imag = kd02_potential_torch(r_mesh, A, Z_proj, Z, params)

    # Prepare features (differentiable)
    l_values = list(range(l_max + 1))
    features = prepare_features_torch(r_mesh, V_real, V_imag, E_cm, k, eta, l_values, A, l_max, r_mesh[-1].item(), config)

    # NN forward pass (differentiable)
    psi_real, psi_imag = model(features)

    # Compute S-matrix for each l (differentiable)
    S_list = []
    for i, l in enumerate(l_values):
        S = compute_smatrix_torch(psi_real[i], psi_imag[i], r_mesh, k, eta, l, V_real, V_imag)
        S_list.append(S)

    # Compute cross section (differentiable)
    sigma = compute_cross_section_torch(S_list, k, theta)

    return sigma


def compute_deff_autograd(model, projectile, A, Z, E_lab, params_np, theta_deg, l_max, device, config):
    """
    Compute D_eff using TRUE AUTOGRAD for log-derivatives.

    Uses torch.autograd.functional.jacobian for efficient computation.
    """
    from torch.autograd.functional import jacobian

    n_params = len(params_np)
    n_angles = len(theta_deg)

    # Convert to torch tensors (use float32 for MPS compatibility)
    r_mesh = torch.linspace(0.1, 15.0, 100, dtype=torch.float32, device=device)
    theta = torch.tensor(np.deg2rad(theta_deg), dtype=torch.float32, device=device)

    # Define function for jacobian computation
    def sigma_func(p):
        """Compute cross section as function of parameters only."""
        sigma = compute_cross_section_differentiable(model, p, projectile, A, Z, E_lab, r_mesh, theta, l_max, device, config)
        return sigma.real.float()  # Return real part as float32

    # Parameters tensor
    params = torch.tensor(params_np, dtype=torch.float32, device=device)

    # Compute Jacobian: J[i,j] = d(sigma[i]) / d(params[j])
    # Shape: (n_angles, n_params)
    J = jacobian(sigma_func, params)

    # Get sigma_0 for log-derivative normalization
    sigma_0 = sigma_func(params).detach().cpu().numpy()

    # Convert Jacobian to numpy
    J_np = J.detach().cpu().numpy()  # Shape: (n_angles, n_params)

    # Compute log-derivatives: d log(sigma) / d log(p) = p/sigma * dsigma/dp
    # gradients[i, j] = d log(sigma[j]) / d log(p[i])
    gradients = np.zeros((n_params, n_angles))
    for i in range(n_params):
        for j in range(n_angles):
            gradients[i, j] = params_np[i] * J_np[j, i] / (sigma_0[j] + 1e-20)

    # Fisher Information Matrix with unit weights
    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            F[i, j] = np.sum(gradients[i] * gradients[j])

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(F)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # D_eff (participation ratio)
    pos_eig = eigenvalues[eigenvalues > 1e-20]
    if len(pos_eig) == 0:
        return 0.0, [], 0.0, params_np.tolist()

    sum_lambda = np.sum(pos_eig)
    sum_lambda2 = np.sum(pos_eig**2)
    D_eff = sum_lambda**2 / sum_lambda2

    # V-rv correlation
    corr_matrix = np.corrcoef(gradients)
    V_rv_corr = corr_matrix[0, 1] if corr_matrix.shape[0] > 1 else 0.0

    return D_eff, eigenvalues.tolist(), V_rv_corr, params_np.tolist()


def get_kd02_params(projectile, A, Z, E_lab):
    """Get KD02 parameters."""
    from potentials import KD02Potential
    pot = KD02Potential(projectile, A, Z, E_lab)
    return np.array([
        pot.V, pot.rv, pot.av,
        pot.W, pot.rw, pot.aw,
        pot.Wd, pot.rvd, pot.avd
    ])


def main():
    print("="*70)
    print("D_eff Computation using TRUE AUTOMATIC DIFFERENTIATION")
    print("="*70)
    print("\nAdvantages over finite differences:")
    print("  - Exact gradients (no numerical truncation error)")
    print("  - Faster (1 forward + n_angles backward vs 2*n_params forward)")
    print()

    # Load model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'stage2_bicfc_moresamples.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "This is a legacy NN script. Use deff_scan_extended.py (Numerov) instead."
        )
    print(f"Loading model: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Add normalization constants if not present
    if 'rho_max' not in config:
        config['rho_max'] = 45.9475
    if 'decay_max' not in config:
        config['decay_max'] = 734.1216

    model = BidirectionalCfC(
        input_size=9,
        hidden_size=config['hidden_size'],
        n_units=config['n_units']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Keep in eval mode but don't use torch.no_grad()

    # Configuration
    nuclei = [
        ('12C', 12, 6),
        ('16O', 16, 8),
        ('27Al', 27, 13),
        ('28Si', 28, 14),
        ('40Ca', 40, 20),
        ('48Ti', 48, 22),
        ('56Fe', 56, 26),
        ('58Ni', 58, 28),
        ('90Zr', 90, 40),
        ('120Sn', 120, 50),
        ('197Au', 197, 79),
        ('208Pb', 208, 82),
    ]
    energies = [10, 20, 30, 50, 100, 150, 200]
    theta_deg = np.linspace(10, 170, 17)
    l_max = 20

    # Results storage
    results = {
        "method": "neural_network_autograd",
        "description": "True automatic differentiation (not finite differences)",
        "model": "BiCfC (PRC trained)",
        "param_names": ["V", "rv", "av", "W", "rw", "aw", "Wd", "rvd", "avd"],
        "nuclei": [n[0] for n in nuclei],
        "energies": energies,
        "data": []
    }

    total = len(nuclei) * len(energies) * 2
    count = 0
    start_time = time.time()

    print("\n" + "="*70)
    print("Computing D_eff with autograd...")
    print("="*70)

    for name, A, Z in nuclei:
        for E in energies:
            for proj in ['n', 'p']:
                count += 1
                try:
                    params = get_kd02_params(proj, A, Z, E)

                    t0 = time.time()
                    D_eff, eigenvalues, V_rv_corr, params_list = compute_deff_autograd(
                        model, proj, A, Z, E, params, theta_deg, l_max, device, config
                    )
                    dt = time.time() - t0

                    # Condition number
                    pos_eig = [e for e in eigenvalues if e > 1e-20]
                    cond = pos_eig[0] / pos_eig[-1] if len(pos_eig) > 1 and pos_eig[-1] > 0 else 1e10

                    results["data"].append({
                        "nucleus": name,
                        "A": A,
                        "Z": Z,
                        "E": E,
                        "projectile": proj,
                        "D_eff": D_eff,
                        "eigenvalues": eigenvalues,
                        "condition_number": cond,
                        "V_rv_correlation": V_rv_corr,
                        "params": params_list
                    })

                    print(f"[{count}/{total}] {proj}+{name} @ {E} MeV: D_eff = {D_eff:.3f}, V-rv = {V_rv_corr:.3f} ({dt:.2f}s)")

                except Exception as e:
                    print(f"[{count}/{total}] {proj}+{name} @ {E} MeV: ERROR - {e}")
                    import traceback
                    traceback.print_exc()

    elapsed = time.time() - start_time

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'deff_nn_9params_autograd.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Summary
    valid_data = [d for d in results["data"] if "D_eff" in d]
    all_deff_n = [d["D_eff"] for d in valid_data if d["projectile"] == "n"]
    all_deff_p = [d["D_eff"] for d in valid_data if d["projectile"] == "p"]

    print("\n" + "="*70)
    print("SUMMARY (AUTOGRAD)")
    print("="*70)
    if all_deff_n:
        print(f"Neutrons: D_eff = {np.mean(all_deff_n):.2f} ± {np.std(all_deff_n):.2f} (n={len(all_deff_n)})")
    if all_deff_p:
        print(f"Protons:  D_eff = {np.mean(all_deff_p):.2f} ± {np.std(all_deff_p):.2f} (n={len(all_deff_p)})")
    if all_deff_n and all_deff_p:
        all_deff = all_deff_n + all_deff_p
        print(f"Combined: D_eff = {np.mean(all_deff):.2f} ± {np.std(all_deff):.2f} (n={len(all_deff)})")
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/count:.2f}s per config)")


if __name__ == '__main__':
    main()
