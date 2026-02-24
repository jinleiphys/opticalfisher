#!/usr/bin/env python3
"""
Complete D_eff scan for 9-parameter KD02 potential.

This is the main analysis script for the PRL paper on information limits.
Scans 168 configurations: 12 nuclei × 7 energies × 2 projectiles.

Author: Jin Lei
Date: December 2024
"""

import numpy as np
import json
import sys
sys.path.insert(0, '/Users/jinlei/Desktop/code/PINN_CFC')

from potentials import KD02Potential, coulomb_potential
from scattering_fortran import ScatteringSolverFortran, HBARC, AMU, E2


def kd02_potential_9params(r, A, Z_proj, Z, params):
    """
    KD02-style potential with 9 free parameters (spin-0).
    params = [V, rv, av, W, rw, aw, Wd, rvd, avd]
    """
    V, rv, av, W, rw, aw, Wd, rvd, avd = params
    A_third = A ** (1./3.)

    # Real volume term
    R_v = rv * A_third
    f_v = 1.0 / (1.0 + np.exp((r - R_v) / av))

    # Imaginary volume term
    R_w = rw * A_third
    f_w = 1.0 / (1.0 + np.exp((r - R_w) / aw))

    # Imaginary surface term (derivative of Woods-Saxon)
    R_d = rvd * A_third
    exp_d = np.exp((r - R_d) / avd)
    f_d = exp_d / (1.0 + exp_d)**2 / avd

    # Total potential
    V_opt = -V * f_v - 1j * W * f_w - 1j * 4 * Wd * f_d

    # Add Coulomb for proton
    if Z_proj > 0:
        Rc = 1.25 * A_third
        V_coul = coulomb_potential(r, Z_proj, Z, Rc)
        V_opt = V_opt + V_coul

    return V_opt


def compute_cross_section(projectile, A, Z, E_lab, theta_deg, params, l_max=30):
    """Compute elastic cross section for 9-parameter potential."""
    A_proj = 1
    mu = A_proj * A / (A_proj + A)
    E_cm = E_lab * A / (A_proj + A)
    k = np.sqrt(2 * mu * AMU * E_cm) / HBARC

    if projectile == 'p':
        eta = Z * E2 * mu * AMU / (HBARC**2 * k)
        Z_proj = 1
    else:
        eta = 0.0
        Z_proj = 0

    pot_func = lambda r: kd02_potential_9params(r, A, Z_proj, Z, params)

    solver = ScatteringSolverFortran(r_max=40.0, hcm=0.02)
    l_values = list(range(l_max + 1))
    results = solver.solve(pot_func, E_cm, mu, Z1=Z_proj, Z2=Z, l_values=l_values)
    S_matrix = results['S_matrix']

    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)

    f_theta = np.zeros(len(theta_deg), dtype=complex)

    for l in l_values:
        S_l = S_matrix[l]
        P_l = np.polynomial.legendre.legval(cos_theta, [0]*l + [1])
        f_theta += (2*l + 1) * (S_l - 1) / (2j * k) * P_l

    if projectile == 'p' and eta > 0:
        from scattering_fortran import coulomb_phase_shift
        sigma_l = coulomb_phase_shift(eta, l_max)

        for l in l_values:
            P_l = np.polynomial.legendre.legval(cos_theta, [0]*l + [1])
            phase = np.exp(2j * sigma_l[l])
            f_theta += (2*l + 1) * (phase - 1) / (2j * k) * P_l

        f_coul = -eta / (2 * k * np.sin(theta_rad/2)**2) * \
                 np.exp(-1j * eta * np.log(np.sin(theta_rad/2)**2) + 2j * sigma_l[0])
        f_theta += f_coul

    dsigma = np.abs(f_theta)**2 * 10
    return dsigma


def compute_fisher_matrix(projectile, A, Z, E_lab, theta_deg, params, eps_rel=0.01):
    """Compute Fisher Information Matrix using finite differences."""
    n_params = len(params)
    n_angles = len(theta_deg)

    sigma_0 = compute_cross_section(projectile, A, Z, E_lab, theta_deg, params)

    gradients = np.zeros((n_params, n_angles))

    for i in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()

        delta = eps_rel * abs(params[i])
        if delta < 1e-6:
            delta = 1e-6

        params_plus[i] += delta
        params_minus[i] -= delta

        sigma_plus = compute_cross_section(projectile, A, Z, E_lab, theta_deg, params_plus)
        sigma_minus = compute_cross_section(projectile, A, Z, E_lab, theta_deg, params_minus)

        gradients[i] = (sigma_plus - sigma_minus) / (2 * delta)

    # Fisher matrix (scale-invariant D_eff)
    epsilon = 0.05
    sigma_exp = epsilon * sigma_0
    sigma_exp[sigma_exp < 1e-10] = 1e-10

    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            F[i, j] = np.sum(gradients[i] * gradients[j] / sigma_exp**2)

    return F, gradients, sigma_0


def compute_deff(F):
    """Compute effective dimensionality from Fisher matrix."""
    eigenvalues = np.linalg.eigvalsh(F)
    eigenvalues = eigenvalues[eigenvalues > 1e-20]

    if len(eigenvalues) == 0:
        return 0.0, np.array([]), np.inf

    sum_lambda = np.sum(eigenvalues)
    sum_lambda2 = np.sum(eigenvalues**2)
    D_eff = sum_lambda**2 / sum_lambda2 if sum_lambda2 > 0 else 0

    cond = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else np.inf

    return D_eff, np.sort(eigenvalues)[::-1], cond


def get_kd02_params(projectile, A, Z, E_lab):
    """Get KD02 parameters: [V, rv, av, W, rw, aw, Wd, rvd, avd]"""
    pot = KD02Potential(projectile, A, Z, E_lab)
    return np.array([
        pot.V, pot.rv, pot.av,
        pot.W, pot.rw, pot.aw,
        pot.Wd, pot.rvd, pot.avd
    ])


def compute_correlations(F, param_names):
    """Compute correlation matrix from Fisher matrix."""
    # Covariance ~ F^-1, but F is singular, so use pseudoinverse
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(F)
        # Keep only significant eigenvalues
        threshold = 1e-10 * np.max(eigenvalues)
        significant = eigenvalues > threshold

        if np.sum(significant) < 2:
            return np.eye(len(param_names))

        # Correlation from Fisher matrix directly (not covariance)
        std = np.sqrt(np.diag(F))
        std[std < 1e-20] = 1e-20
        corr = F / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        return corr
    except:
        return np.eye(len(param_names))


def main():
    """Main D_eff scan."""

    print("="*70)
    print("D_eff Scan for 9-parameter KD02 Potential")
    print("="*70)

    # Configuration
    nuclei = [
        (12, 6, '12C'),
        (16, 8, '16O'),
        (27, 13, '27Al'),
        (28, 14, '28Si'),
        (40, 20, '40Ca'),
        (48, 22, '48Ti'),
        (56, 26, '56Fe'),
        (58, 28, '58Ni'),
        (90, 40, '90Zr'),
        (120, 50, '120Sn'),
        (197, 79, '197Au'),
        (208, 82, '208Pb'),
    ]

    energies = [10, 30, 50, 65, 100, 150, 200]
    projectiles = ['n', 'p']

    param_names = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd']
    theta_deg = np.linspace(10, 170, 17)

    results = {
        'nuclei': [n[2] for n in nuclei],
        'A_values': [n[0] for n in nuclei],
        'energies': energies,
        'projectiles': projectiles,
        'param_names': param_names,
        'theta_deg': theta_deg.tolist(),
        'data': []
    }

    total = len(nuclei) * len(energies) * len(projectiles)
    count = 0

    for proj in projectiles:
        print(f"\n{'='*60}")
        print(f"Projectile: {proj}")
        print('='*60)

        for A, Z, name in nuclei:
            for E in energies:
                count += 1
                print(f"\n[{count}/{total}] {proj} + {name} @ {E} MeV")

                try:
                    params = get_kd02_params(proj, A, Z, E)
                    F, gradients, sigma_0 = compute_fisher_matrix(proj, A, Z, E, theta_deg, params)
                    D_eff, eigenvalues, cond = compute_deff(F)
                    corr = compute_correlations(F, param_names)

                    # V-rv correlation (indices 0 and 1)
                    V_rv_corr = corr[0, 1] if corr.shape[0] > 1 else 0.0

                    result = {
                        'projectile': proj,
                        'nucleus': name,
                        'A': A,
                        'Z': Z,
                        'E': E,
                        'D_eff': float(D_eff),
                        'condition_number': float(cond),
                        'V_rv_correlation': float(V_rv_corr),
                        'eigenvalues': eigenvalues.tolist(),
                        'params': params.tolist(),
                    }
                    results['data'].append(result)

                    print(f"  D_eff = {D_eff:.2f}, cond = {cond:.2e}, V-rv corr = {V_rv_corr:.3f}")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    results['data'].append({
                        'projectile': proj,
                        'nucleus': name,
                        'A': A,
                        'Z': Z,
                        'E': E,
                        'D_eff': None,
                        'error': str(e)
                    })

    # Summary statistics
    valid_data = [d for d in results['data'] if d.get('D_eff') is not None]

    neutron_data = [d for d in valid_data if d['projectile'] == 'n']
    proton_data = [d for d in valid_data if d['projectile'] == 'p']

    n_deff = [d['D_eff'] for d in neutron_data]
    p_deff = [d['D_eff'] for d in proton_data]
    all_deff = [d['D_eff'] for d in valid_data]

    results['summary'] = {
        'neutron': {
            'mean': float(np.mean(n_deff)),
            'std': float(np.std(n_deff)),
            'min': float(np.min(n_deff)),
            'max': float(np.max(n_deff)),
            'count': len(n_deff)
        },
        'proton': {
            'mean': float(np.mean(p_deff)),
            'std': float(np.std(p_deff)),
            'min': float(np.min(p_deff)),
            'max': float(np.max(p_deff)),
            'count': len(p_deff)
        },
        'combined': {
            'mean': float(np.mean(all_deff)),
            'std': float(np.std(all_deff)),
            'min': float(np.min(all_deff)),
            'max': float(np.max(all_deff)),
            'count': len(all_deff)
        }
    }

    # Save results
    outfile = '/Users/jinlei/Desktop/code/PRL_Information_Limit/deff_scan_kd02_9params.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nNeutron: D_eff = {results['summary']['neutron']['mean']:.2f} ± {results['summary']['neutron']['std']:.2f}")
    print(f"Proton:  D_eff = {results['summary']['proton']['mean']:.2f} ± {results['summary']['proton']['std']:.2f}")
    print(f"Combined: D_eff = {results['summary']['combined']['mean']:.2f} ± {results['summary']['combined']['std']:.2f}")
    print(f"\nTotal configurations: {len(valid_data)}")


if __name__ == '__main__':
    main()
