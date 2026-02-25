#!/usr/bin/env python3
"""
D_eff Analysis using Neural Network + Autograd

Uses trained PRC model (BiCfC) for efficient gradient computation.
Computes Fisher Information Matrix for 9 KD02 parameters.

Author: Jin Lei
Date: December 2024
"""

import numpy as np
import sys
import os
import json
import time

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
    from ncps.torch import CfC
except ImportError:
    raise ImportError(
        "This is a legacy neural network script requiring torch and ncps.\n"
        "Use the Numerov-based scripts (deff_scan_extended.py) instead."
    )
from ncps.wirings import AutoNCP

# Physical constants
HBARC = 197.327  # MeV·fm
AMU = 931.494    # MeV/c²
E2 = 1.44        # e²/(4πε₀) in MeV·fm


#==============================================================================
# BiCfC Model (same architecture as PRC)
#==============================================================================

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


#==============================================================================
# Potential and Cross Section (NumPy for speed)
#==============================================================================

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
    """
    KD02 potential with 9 parameters.
    params = [V, rv, av, W, rw, aw, Wd, rvd, avd]
    """
    V, rv, av, W, rw, aw, Wd, rvd, avd = params
    A_third = A ** (1./3.)

    # Real volume
    V_real = -V * woods_saxon(r, rv * A_third, av)

    # Coulomb
    if Z_proj > 0:
        rc = 1.198 + 0.697 * A**(-2./3.) + 12.994 * A**(-5./3.)
        Rc = rc * A_third
        V_real = V_real + coulomb_potential(r, Z_proj, Z, Rc)

    # Imaginary volume + surface
    V_imag = -W * woods_saxon(r, rw * A_third, aw)
    V_imag = V_imag - Wd * derivative_ws(r, rvd * A_third, avd)

    return V_real, V_imag


def get_kd02_params(projectile, A, Z, E_lab):
    """Get 9 KD02 parameters for a given system."""
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

    # Phase-space coordinate
    rho = k * r_mesh
    rho_max = k * r_max
    rho_norm = rho / rho_max

    # Relative potentials
    V_real_rel = V_real / (E_cm + 1e-6)
    V_imag_rel = V_imag / (E_cm + 1e-6)
    V_real_norm = np.clip(V_real_rel / 3.0, -1, 1)
    V_imag_norm = np.clip(V_imag_rel / 0.5, -1, 1)

    # Normalized parameters
    eta_norm = eta / 20.0
    A_norm = (A ** (1./3.)) / 6.0

    # WKB (simplified)
    dr = r_mesh[1] - r_mesh[0]
    k_local = np.sqrt(np.maximum(E_cm - V_real, 0.01))
    phase = np.cumsum(k_local * dr)
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    decay = np.cumsum(-V_imag / (2 * k_local + 1e-6) * dr)
    decay_norm = decay / (np.abs(decay).max() + 1e-8)

    # Build features for all l values
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
    """Extract S-matrix using amplitude integral (simplified)."""
    dr = r[1] - r[0]
    psi = psi_real + 1j * psi_imag
    V = V_real + 1j * V_imag

    # Asymptotic Coulomb function (simplified)
    kr = k * r
    phase = kr - l * np.pi / 2
    if eta > 0.01:
        phase = phase - eta * np.log(2 * kr + 1e-10)
    F_l = np.sin(phase)

    # Amplitude integral
    integrand = V * psi * F_l
    integral = np.sum(integrand) * dr

    # S-matrix
    S = 1.0 + 2j * k * integral

    return S


def compute_cross_section_from_smatrix(S_list, k, theta):
    """Compute differential cross section from S-matrices."""
    cos_theta = np.cos(theta)
    l_max = len(S_list) - 1

    # Legendre polynomials
    P = np.zeros((l_max + 1, len(theta)))
    P[0] = 1.0
    if l_max >= 1:
        P[1] = cos_theta
    for l in range(2, l_max + 1):
        P[l] = ((2*l - 1) * cos_theta * P[l-1] - (l - 1) * P[l-2]) / l

    # Scattering amplitude
    f = np.zeros(len(theta), dtype=complex)
    for l in range(l_max + 1):
        f += (2*l + 1) * (S_list[l] - 1) / (2j * k) * P[l]

    return np.abs(f) ** 2


#==============================================================================
# Fisher Information with Neural Network
#==============================================================================

def compute_cross_section_nn(model, params, projectile, A, Z, E_lab, r_mesh, theta_deg, l_max, device):
    """Compute cross section using neural network."""
    Z_proj = 1 if projectile == 'p' else 0
    mu = A / (1 + A)
    E_cm = E_lab * A / (1 + A)
    k = np.sqrt(2 * mu * AMU * E_cm) / HBARC
    eta = Z_proj * Z * E2 * mu * AMU / (HBARC**2 * k) if Z_proj > 0 else 0.0

    # Potential
    V_real, V_imag = kd02_potential_9params(r_mesh, A, Z_proj, Z, params)

    # Prepare features for all l
    l_values = list(range(l_max + 1))
    features = prepare_features_batch(r_mesh, V_real, V_imag, E_cm, k, eta, l_values, A, l_max, r_mesh[-1])

    # Neural network inference (batched)
    with torch.no_grad():
        feat_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        psi_real, psi_imag = model(feat_tensor)
        psi_real = psi_real.cpu().numpy()
        psi_imag = psi_imag.cpu().numpy()

    # Extract S-matrices
    S_list = []
    for i, l in enumerate(l_values):
        S = compute_smatrix_from_psi(psi_real[i], psi_imag[i], r_mesh, k, eta, l, V_real, V_imag)
        S_list.append(S)

    # Cross section
    theta_rad = np.deg2rad(theta_deg)
    sigma = compute_cross_section_from_smatrix(S_list, k, theta_rad)

    return sigma


def compute_fisher_nn(model, projectile, A, Z, E_lab, params, theta_deg, l_max, device, eps_rel=0.01):
    """
    Compute Fisher Information Matrix using neural network + finite difference.

    This is faster than torch.autograd.functional.jacobian because:
    1. Neural network inference is batched for all l values
    2. Only 2*n_params forward passes needed (vs jacobian's overhead)
    """
    r_mesh = np.linspace(0.1, 15.0, 100)
    n_params = len(params)
    n_angles = len(theta_deg)

    # Reference cross section
    sigma_0 = compute_cross_section_nn(model, params, projectile, A, Z, E_lab, r_mesh, theta_deg, l_max, device)

    # Gradients via finite difference
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

        gradients[i] = (sigma_plus - sigma_minus) / (2 * delta)

    # Fisher Information Matrix
    weights = 1.0 / (sigma_0 ** 2 + 1e-20)
    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            F[i, j] = np.sum(weights * gradients[i] * gradients[j])

    return F, gradients, sigma_0


def compute_deff(F):
    """Compute effective dimensionality."""
    eigenvalues = np.linalg.eigvalsh(F)
    eigenvalues = np.maximum(eigenvalues, 0)

    if np.sum(eigenvalues) < 1e-20:
        return 0.0, eigenvalues

    D_eff = (np.sum(eigenvalues) ** 2) / (np.sum(eigenvalues ** 2) + 1e-20)
    return D_eff, eigenvalues


def compute_correlations(F):
    """Compute correlation matrix from Fisher matrix."""
    n = F.shape[0]
    std = np.sqrt(np.diag(F) + 1e-20)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = F[i, j] / (std[i] * std[j] + 1e-20)
    return corr


#==============================================================================
# Main Analysis
#==============================================================================

def run_deff_scan(model, device, log=print):
    """Run D_eff scan with 9 KD02 parameters."""

    nuclei = [
        (12, 6, '12C'), (16, 8, '16O'), (27, 13, '27Al'), (28, 14, '28Si'),
        (40, 20, '40Ca'), (48, 22, '48Ti'), (56, 26, '56Fe'), (58, 28, '58Ni'),
        (90, 40, '90Zr'), (120, 50, '120Sn'), (197, 79, '197Au'), (208, 82, '208Pb'),
    ]
    energies = [10, 20, 30, 50, 100, 150, 200]
    projectiles = ['n', 'p']

    theta_deg = np.linspace(10, 170, 17)
    l_max = 20
    param_names = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd']

    results = []
    total = len(nuclei) * len(energies) * len(projectiles)
    count = 0

    log(f"Running Neural Network D_eff scan (9 parameters)...")
    log(f"  Nuclei: {len(nuclei)}")
    log(f"  Energies: {energies}")
    log(f"  Projectiles: {projectiles}")
    log(f"  Parameters: {param_names}")
    log(f"  Total: {total} configurations")

    start_time = time.time()

    for A, Z, name in nuclei:
        for E in energies:
            for proj in projectiles:
                count += 1
                try:
                    # Get KD02 parameters
                    params = get_kd02_params(proj, A, Z, E)

                    # Skip if imaginary parts too small
                    if params[3] < 0.1 and params[6] < 0.1:  # W and Wd
                        log(f"  [{count}/{total}] {proj}+{name} @ {E} MeV: skipped (W too small)")
                        continue

                    # Compute Fisher matrix
                    F, gradients, sigma_0 = compute_fisher_nn(
                        model, proj, A, Z, E, params, theta_deg, l_max, device
                    )

                    # D_eff and correlations
                    D_eff, eigenvalues = compute_deff(F)
                    corr = compute_correlations(F)

                    # V-rv correlation (indices 0 and 1)
                    V_rv_corr = corr[0, 1] if len(corr) > 1 else 0

                    result = {
                        'nucleus': name,
                        'A': A,
                        'Z': Z,
                        'E': E,
                        'projectile': proj,
                        'D_eff': float(D_eff),
                        'eigenvalues': eigenvalues.tolist(),
                        'condition_number': float(eigenvalues.max() / (eigenvalues[eigenvalues > 1e-20].min() if np.any(eigenvalues > 1e-20) else 1e-20)),
                        'V_rv_correlation': float(V_rv_corr),
                        'params': params.tolist(),
                    }
                    results.append(result)

                    elapsed = time.time() - start_time
                    eta_min = elapsed / count * (total - count) / 60

                    log(f"  [{count}/{total}] {proj}+{name} @ {E} MeV: D_eff = {D_eff:.2f} (ETA: {eta_min:.1f} min)")

                except Exception as e:
                    log(f"  [{count}/{total}] {proj}+{name} @ {E} MeV: Error - {e}")

    return results, param_names


def run_single_case(args):
    """Run single case for multiprocessing."""
    model, proj, A, Z, name, E, theta_deg, l_max, device, param_names = args

    try:
        params = get_kd02_params(proj, A, Z, E)

        if params[3] < 0.1 and params[6] < 0.1:
            return None

        F, gradients, sigma_0 = compute_fisher_nn(
            model, proj, A, Z, E, params, theta_deg, l_max, device
        )

        D_eff, eigenvalues = compute_deff(F)
        corr = compute_correlations(F)
        V_rv_corr = corr[0, 1] if len(corr) > 1 else 0

        return {
            'nucleus': name,
            'A': A,
            'Z': Z,
            'E': E,
            'projectile': proj,
            'D_eff': float(D_eff),
            'eigenvalues': eigenvalues.tolist(),
            'condition_number': float(eigenvalues.max() / (eigenvalues[eigenvalues > 1e-20].min() if np.any(eigenvalues > 1e-20) else 1e-20)),
            'V_rv_correlation': float(V_rv_corr),
            'params': params.tolist(),
        }
    except Exception as e:
        return {'error': str(e), 'nucleus': name, 'E': E, 'projectile': proj}


def run_on_gpu(gpu_id, cases, config, param_names, result_queue, model_path):
    """Worker function for each GPU."""
    import torch
    device = torch.device(f'cuda:{gpu_id}')

    # Load model on this GPU
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = BidirectionalCfC(
        input_size=9,
        hidden_size=config['hidden_size'],
        n_units=config['n_units']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    theta_deg = np.linspace(10, 170, 17)
    l_max = 20

    results = []
    for case in cases:
        proj, A, Z, name, E = case
        try:
            params = get_kd02_params(proj, A, Z, E)

            if params[3] < 0.1 and params[6] < 0.1:
                continue

            F, gradients, sigma_0 = compute_fisher_nn(
                model, proj, A, Z, E, params, theta_deg, l_max, device
            )

            D_eff, eigenvalues = compute_deff(F)
            corr = compute_correlations(F)
            V_rv_corr = corr[0, 1] if len(corr) > 1 else 0

            result = {
                'nucleus': name,
                'A': A,
                'Z': Z,
                'E': E,
                'projectile': proj,
                'D_eff': float(D_eff),
                'eigenvalues': eigenvalues.tolist(),
                'condition_number': float(eigenvalues.max() / (eigenvalues[eigenvalues > 1e-20].min() if np.any(eigenvalues > 1e-20) else 1e-20)),
                'V_rv_correlation': float(V_rv_corr),
                'params': params.tolist(),
            }
            results.append(result)
            print(f"  GPU{gpu_id}: {proj}+{name} @ {E} MeV: D_eff = {D_eff:.2f}", flush=True)

        except Exception as e:
            print(f"  GPU{gpu_id}: {proj}+{name} @ {E} MeV: Error - {e}", flush=True)

    result_queue.put(results)


def main():
    import multiprocessing as mp

    # Must use spawn for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)

    def log(msg):
        print(msg, flush=True)

    log("="*70)
    log("D_eff Analysis using Neural Network (9 KD02 Parameters)")
    log("="*70)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log(f"\nAvailable GPUs: {n_gpus}")

    if n_gpus == 0:
        device = torch.device('cpu')
        log("Using CPU")
    else:
        device = torch.device('cuda:0')
        for i in range(n_gpus):
            log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'stage2_bicfc_moresamples.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "This is a legacy NN script. Use deff_scan_extended.py (Numerov) instead."
        )
    log(f"Loading model: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    log(f"  Model error: {checkpoint['rel_error']:.2f}%")

    # Generate all cases
    nuclei = [
        (12, 6, '12C'), (16, 8, '16O'), (27, 13, '27Al'), (28, 14, '28Si'),
        (40, 20, '40Ca'), (48, 22, '48Ti'), (56, 26, '56Fe'), (58, 28, '58Ni'),
        (90, 40, '90Zr'), (120, 50, '120Sn'), (197, 79, '197Au'), (208, 82, '208Pb'),
    ]
    energies = [10, 20, 30, 50, 100, 150, 200]
    projectiles = ['n', 'p']
    param_names = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd']

    cases = []
    for A, Z, name in nuclei:
        for E in energies:
            for proj in projectiles:
                cases.append((proj, A, Z, name, E))

    log(f"\n  Total cases: {len(cases)}")

    # Run scan
    log("\n" + "-"*70)
    start = time.time()

    if n_gpus > 1:
        # Multi-GPU parallel
        log(f"Running parallel on {n_gpus} GPUs...")

        # Split cases across GPUs
        cases_per_gpu = [cases[i::n_gpus] for i in range(n_gpus)]

        result_queue = mp.Queue()
        processes = []

        for gpu_id in range(n_gpus):
            p = mp.Process(target=run_on_gpu,
                          args=(gpu_id, cases_per_gpu[gpu_id], config, param_names, result_queue, model_path))
            p.start()
            processes.append(p)

        # Collect results
        results = []
        for _ in range(n_gpus):
            results.extend(result_queue.get())

        for p in processes:
            p.join()

    else:
        # Single GPU or CPU
        model = BidirectionalCfC(
            input_size=9,
            hidden_size=config['hidden_size'],
            n_units=config['n_units']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        results, param_names = run_deff_scan(model, device, log)

    elapsed = time.time() - start

    # Save results
    output = {
        'method': 'neural_network',
        'model': 'BiCfC (PRC trained)',
        'param_names': param_names,
        'nuclei': ['12C', '16O', '27Al', '28Si', '40Ca', '48Ti', '56Fe', '58Ni', '90Zr', '120Sn', '197Au', '208Pb'],
        'energies': [10, 20, 30, 50, 100, 150, 200],
        'data': results,
    }

    output_path = os.path.join(script_dir, 'deff_nn_9params.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Summary
    log("\n" + "="*70)
    log("SUMMARY")
    log("="*70)

    deff_n = [r['D_eff'] for r in results if r.get('projectile') == 'n' and r.get('D_eff', 0) > 0]
    deff_p = [r['D_eff'] for r in results if r.get('projectile') == 'p' and r.get('D_eff', 0) > 0]
    all_deff = deff_n + deff_p

    log(f"\nNeural Network Results (9 parameters):")
    if deff_n:
        log(f"  Neutron: D_eff = {np.mean(deff_n):.2f} ± {np.std(deff_n):.2f} (n={len(deff_n)})")
    if deff_p:
        log(f"  Proton:  D_eff = {np.mean(deff_p):.2f} ± {np.std(deff_p):.2f} (n={len(deff_p)})")
    if all_deff:
        log(f"  Combined: D_eff = {np.mean(all_deff):.2f} ± {np.std(all_deff):.2f} (n={len(all_deff)})")

    log(f"\nTime: {elapsed/60:.1f} min ({elapsed/len(cases):.1f} sec/case)")
    log(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
