#!/usr/bin/env python3
"""
Angle-resolved sensitivity analysis.

Computes S_i(theta) = d log sigma / d log p_i on a dense angle grid
for representative nuclear systems. Addresses referee request for
angle-resolved parameter sensitivity plots.

Focuses on:
- Which parameters have distinct angular signatures
- Whether diffuseness parameters gain information at backward angles
- Cumulative Fisher information as function of theta_max

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from potentials import KD02Potential
from deff_scan_extended import (kd02_potential_11params, compute_observables_vector,
                                 get_kd02_params_11)


def compute_angle_sensitivity(proj, A, Z, E_lab, theta_dense, params,
                               rvso, avso, eps_rel=0.01, l_max=30):
    """
    Compute S_i(theta) = d log sigma / d log p_i for each parameter.

    Also computes Ay sensitivity: dAy/dp * p / delta_Ay.

    Returns:
        S_dcs: (n_params, n_angles) log-derivative of dsigma/dOmega
        S_Ay:  (n_params, n_angles) scaled derivative of Ay
        obs_0: baseline observables
    """
    n_params = len(params)
    n_angles = len(theta_dense)
    delta_Ay = 0.03

    obs_0 = compute_observables_vector(proj, A, Z, E_lab, theta_dense,
                                        params, rvso, avso, l_max)

    S_dcs = np.zeros((n_params, n_angles))
    S_Ay = np.zeros((n_params, n_angles))

    for i in range(n_params):
        params_plus = list(params)
        params_minus = list(params)

        delta = eps_rel * abs(params[i])
        if delta < 1e-8:
            delta = 1e-8

        params_plus[i] += delta
        params_minus[i] -= delta

        obs_p = compute_observables_vector(proj, A, Z, E_lab, theta_dense,
                                            params_plus, rvso, avso, l_max)
        obs_m = compute_observables_vector(proj, A, Z, E_lab, theta_dense,
                                            params_minus, rvso, avso, l_max)

        # dlog(sigma)/dlog(p)
        dcs_0 = obs_0['elastic_dcs']
        S_dcs[i] = params[i] * (obs_p['elastic_dcs'] - obs_m['elastic_dcs']) / (
            2.0 * delta) / (dcs_0 + 1e-30)

        # Ay sensitivity
        S_Ay[i] = params[i] * (obs_p['Ay'] - obs_m['Ay']) / (
            2.0 * delta) / delta_Ay

    return S_dcs, S_Ay, obs_0


def cumulative_fisher_info(S_dcs, theta_dense):
    """
    Compute cumulative Fisher information as function of theta_max.

    C_i(theta_max) = sum_{theta <= theta_max} S_i(theta)^2
    D_eff(theta_max) from the Fisher matrix built from angles up to theta_max.

    Returns:
        C_i: (n_params, n_angles) cumulative per-parameter info
        D_eff_cumul: (n_angles,) cumulative D_eff
    """
    n_params, n_angles = S_dcs.shape
    C_i = np.cumsum(S_dcs**2, axis=1)
    D_eff_cumul = np.zeros(n_angles)

    for j in range(n_angles):
        G = S_dcs[:, :j+1]  # (n_params, j+1)
        F = G @ G.T
        eigenvalues = np.linalg.eigvalsh(F)
        eigenvalues = eigenvalues[eigenvalues > 1e-20]
        if len(eigenvalues) > 0:
            s = np.sum(eigenvalues)
            s2 = np.sum(eigenvalues**2)
            D_eff_cumul[j] = s**2 / s2 if s2 > 0 else 0
    return C_i, D_eff_cumul


def main():
    print("=" * 70)
    print("Angle-Resolved Sensitivity Analysis")
    print("=" * 70)

    param_names = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd',
                   'Vso', 'Wso']

    theta_dense = np.linspace(5, 175, 35)

    cases = [
        ('n', 40, 20, '40Ca', 50),
        ('n', 208, 82, '208Pb', 30),
        ('p', 40, 20, '40Ca', 50),
        ('n', 120, 50, '120Sn', 100),
    ]

    results = {
        'param_names': param_names,
        'theta_deg': theta_dense.tolist(),
        'cases': [],
    }

    for proj, A, Z, name, E in cases:
        print(f"\n--- {proj}+{name} @ {E} MeV ---")

        params, rvso, avso = get_kd02_params_11(proj, A, Z, E)

        S_dcs, S_Ay, obs_0 = compute_angle_sensitivity(
            proj, A, Z, E, theta_dense, params, rvso, avso)

        C_i, D_eff_cumul = cumulative_fisher_info(S_dcs, theta_dense)

        case_result = {
            'projectile': proj,
            'nucleus': name,
            'A': A, 'Z': Z, 'E': E,
            'params': params,
            'S_dcs': S_dcs.tolist(),
            'S_Ay': S_Ay.tolist(),
            'cumulative_info': C_i.tolist(),
            'D_eff_cumulative': D_eff_cumul.tolist(),
            'elastic_dcs': obs_0['elastic_dcs'].tolist(),
            'Ay': obs_0['Ay'].tolist(),
            'sigma_R': float(obs_0['sigma_R']),
        }
        results['cases'].append(case_result)

        # Print key sensitivity metrics
        for i, pname in enumerate(param_names):
            rms = np.sqrt(np.mean(S_dcs[i]**2))
            max_angle = theta_dense[np.argmax(np.abs(S_dcs[i]))]
            print(f"  {pname:4s}: RMS={rms:.3f}  max at {max_angle:.0f} deg")

        print(f"  D_eff(all angles) = {D_eff_cumul[-1]:.2f}")
        # Find angle where D_eff reaches 90% of final value
        target = 0.9 * D_eff_cumul[-1]
        idx90 = np.searchsorted(D_eff_cumul, target)
        if idx90 < len(theta_dense):
            print(f"  90% of D_eff reached by theta = {theta_dense[idx90]:.0f} deg")

    # Save
    outdir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'angle_sensitivity.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
