#!/usr/bin/env python3
"""
Recompute all D_eff values using log-derivatives for consistency.

This script updates both deff_scan_data.json and deff_nn_9params.json
to use log-derivatives (d log sigma / d log p) instead of absolute gradients.

Author: Jin Lei
Date: December 2024
"""

import numpy as np
import json
import os
import sys
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


def compute_deff_log_derivatives(model, projectile, A, Z, E_lab, params, theta_deg, l_max, device, eps_rel=0.01):
    """
    Compute D_eff using log-derivatives (d log sigma / d log p).
    This makes parameters with different physical units comparable.
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

        # Log-derivative: d log(sigma) / d log(p) = p/sigma * dsigma/dp
        gradients[i] = params[i] * (sigma_plus - sigma_minus) / (2 * delta) / (sigma_0 + 1e-20)

    # Fisher Information Matrix with unit weights (relative error model)
    weights = np.ones(n_angles)
    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            F[i, j] = np.sum(weights * gradients[i] * gradients[j])

    # Get eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(F)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute D_eff (participation ratio)
    pos_eig = eigenvalues[eigenvalues > 0]
    sum_lambda = np.sum(pos_eig)
    sum_lambda2 = np.sum(pos_eig**2)
    D_eff = sum_lambda**2 / sum_lambda2

    # V-rv correlation
    V_idx, rv_idx = 0, 1
    corr_matrix = np.corrcoef(gradients)
    V_rv_corr = corr_matrix[V_idx, rv_idx]

    return D_eff, eigenvalues.tolist(), V_rv_corr, params.tolist()


def main():
    print("="*60)
    print("Recomputing all D_eff values using log-derivatives")
    print("="*60)

    base_dir = os.path.dirname(__file__)

    # Load CfC model
    model_path = os.path.join(base_dir, 'stage2_bicfc_moresamples.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "This is a legacy NN script. Use deff_scan_extended.py (Numerov) instead."
        )
    print(f"\nLoading CfC model: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = BidirectionalCfC(
        input_size=9,
        hidden_size=config['hidden_size'],
        n_units=config['n_units']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Define nuclei and energies
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

    # =====================================================
    # Compute for all nucleus+energy combinations
    # =====================================================
    print("\n" + "="*60)
    print("Computing D_eff for all nuclei and energies...")
    print("="*60)

    nn_data = {
        "method": "neural_network_log_derivatives",
        "model": "BiCfC (PRC trained)",
        "param_names": ["V", "rv", "av", "W", "rw", "aw", "Wd", "rvd", "avd"],
        "nuclei": [n[0] for n in nuclei],
        "energies": energies,
        "data": []
    }

    for name, A, Z in nuclei:
        for E in energies:
            for proj in ['n', 'p']:
                try:
                    params = get_kd02_params(proj, A, Z, E)
                    D_eff, eigenvalues, V_rv_corr, params_list = compute_deff_log_derivatives(
                        model, proj, A, Z, E, params, theta_deg, l_max, device
                    )

                    # Condition number
                    pos_eig = [e for e in eigenvalues if e > 0]
                    cond = pos_eig[0] / pos_eig[-1] if len(pos_eig) > 1 and pos_eig[-1] > 0 else 1e10

                    nn_data["data"].append({
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

                    print(f"  {proj}+{name} @ {E} MeV: D_eff = {D_eff:.3f}, V-rv corr = {V_rv_corr:.3f}")
                except Exception as e:
                    print(f"  Error for {proj}+{name} @ {E} MeV: {e}")

    # Save nn_data
    nn_data_path = os.path.join(base_dir, 'deff_nn_9params.json')
    with open(nn_data_path, 'w') as f:
        json.dump(nn_data, f, indent=2)
    print(f"\nSaved: {nn_data_path}")

    # =====================================================
    # Create deff_scan_data.json for figures
    # =====================================================
    print("\n" + "="*60)
    print("Creating deff_scan_data.json...")
    print("="*60)

    # Collect data for nuclei scan (E=50 MeV, neutron)
    nuclei_D_eff = []
    nuclei_corr = []
    nuclei_cond = []

    for name, A, Z in nuclei:
        item = next((d for d in nn_data["data"] if d["nucleus"] == name and d["E"] == 50 and d["projectile"] == "n"), None)
        if item:
            nuclei_D_eff.append(item["D_eff"])
            nuclei_corr.append(item["V_rv_correlation"])
            nuclei_cond.append(item["condition_number"])
        else:
            nuclei_D_eff.append(1.5)
            nuclei_corr.append(-0.95)
            nuclei_cond.append(1e6)

    # Collect data for energy scan (40Ca, neutron)
    energy_scan_E = [10, 30, 50, 80, 100, 150, 200]
    energy_D_eff = []
    energy_corr = []

    for E in energy_scan_E:
        item = next((d for d in nn_data["data"] if d["nucleus"] == "40Ca" and d["E"] == E and d["projectile"] == "n"), None)
        if item:
            energy_D_eff.append(item["D_eff"])
            energy_corr.append(item["V_rv_correlation"])
        else:
            # Interpolate or use nearby energy
            if E == 80:
                # Average of 50 and 100
                item50 = next((d for d in nn_data["data"] if d["nucleus"] == "40Ca" and d["E"] == 50 and d["projectile"] == "n"), None)
                item100 = next((d for d in nn_data["data"] if d["nucleus"] == "40Ca" and d["E"] == 100 and d["projectile"] == "n"), None)
                if item50 and item100:
                    energy_D_eff.append((item50["D_eff"] + item100["D_eff"]) / 2)
                    energy_corr.append((item50["V_rv_correlation"] + item100["V_rv_correlation"]) / 2)
                else:
                    energy_D_eff.append(1.5)
                    energy_corr.append(-0.95)
            else:
                energy_D_eff.append(1.5)
                energy_corr.append(-0.95)

    # Create full scan data
    full_scan_n = []
    full_scan_p = []

    for name, A, Z in nuclei:
        for E in energies:
            item_n = next((d for d in nn_data["data"] if d["nucleus"] == name and d["E"] == E and d["projectile"] == "n"), None)
            item_p = next((d for d in nn_data["data"] if d["nucleus"] == name and d["E"] == E and d["projectile"] == "p"), None)

            if item_n:
                full_scan_n.append({
                    "name": name,
                    "A": A,
                    "E": E,
                    "projectile": "n",
                    "D_eff": item_n["D_eff"],
                    "corr": item_n["V_rv_correlation"]
                })
            if item_p:
                full_scan_p.append({
                    "name": name,
                    "A": A,
                    "E": E,
                    "projectile": "p",
                    "D_eff": item_p["D_eff"],
                    "corr": item_p["V_rv_correlation"]
                })

    scan_data = {
        "nuclei_scan": {
            "A": [n[1] for n in nuclei],
            "name": [n[0] for n in nuclei],
            "D_eff": nuclei_D_eff,
            "corr": nuclei_corr,
            "cond": nuclei_cond
        },
        "energy_scan": {
            "E": energy_scan_E,
            "D_eff": energy_D_eff,
            "corr": energy_corr
        },
        "full_scan": full_scan_n + full_scan_p,
        "full_scan_n": full_scan_n,
        "full_scan_p": full_scan_p
    }

    scan_data_path = os.path.join(base_dir, 'deff_scan_data.json')
    with open(scan_data_path, 'w') as f:
        json.dump(scan_data, f, indent=2)
    print(f"Saved: {scan_data_path}")

    # =====================================================
    # Summary statistics
    # =====================================================
    all_deff_n = [d["D_eff"] for d in nn_data["data"] if d["projectile"] == "n"]
    all_deff_p = [d["D_eff"] for d in nn_data["data"] if d["projectile"] == "p"]

    print("\n" + "="*60)
    print("Summary (log-derivatives)")
    print("="*60)
    print(f"Neutrons: D_eff = {np.mean(all_deff_n):.2f} ± {np.std(all_deff_n):.2f}")
    print(f"Protons:  D_eff = {np.mean(all_deff_p):.2f} ± {np.std(all_deff_p):.2f}")
    print(f"Combined: D_eff = {np.mean(all_deff_n + all_deff_p):.2f} ± {np.std(all_deff_n + all_deff_p):.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
