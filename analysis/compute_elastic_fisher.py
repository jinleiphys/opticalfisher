#!/usr/bin/env python3
"""
Compute elastic-only Fisher matrices for all 168 configurations.

Reads system configs from deff_scan_extended.json, recomputes
elastic-only gradients and Fisher matrices, saves to a separate file.
This is needed for comparing all-observable vs elastic-only in the
global analysis.

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from potentials import KD02Potential
from deff_scan_extended import (compute_observables_vector, get_kd02_params_11,
                                 kd02_potential_11params)

PARAM_NAMES = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd', 'Vso', 'Wso']


def compute_elastic_fisher(proj, A, Z, E_lab, theta_deg, params, rvso, avso,
                            eps_rel=0.01, l_max=30):
    """
    Compute elastic-only 11x11 Fisher matrix using log-derivatives.

    Returns:
        F_elastic: (11, 11) Fisher matrix for elastic dσ/dΩ only
    """
    n_params = len(params)
    n_angles = len(theta_deg)

    obs_0 = compute_observables_vector(proj, A, Z, E_lab, theta_deg, params,
                                        rvso, avso, l_max)
    dcs_0 = obs_0['elastic_dcs']

    G_elastic = np.zeros((n_params, n_angles))

    for i in range(n_params):
        params_plus = list(params)
        params_minus = list(params)
        delta = eps_rel * abs(params[i])
        if delta < 1e-8:
            delta = 1e-8
        params_plus[i] += delta
        params_minus[i] -= delta

        obs_p = compute_observables_vector(proj, A, Z, E_lab, theta_deg,
                                            params_plus, rvso, avso, l_max)
        obs_m = compute_observables_vector(proj, A, Z, E_lab, theta_deg,
                                            params_minus, rvso, avso, l_max)

        # d log σ / d log p_i
        G_elastic[i] = params[i] * (obs_p['elastic_dcs'] - obs_m['elastic_dcs']) / (
            2.0 * delta) / (dcs_0 + 1e-30)

    F_elastic = G_elastic @ G_elastic.T
    return F_elastic


def main():
    print("=" * 70)
    print("Computing elastic-only Fisher matrices for all systems")
    print("=" * 70)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'data', 'deff_scan_extended.json')

    with open(data_path) as f:
        scan_data = json.load(f)

    theta_deg = np.array(scan_data['theta_deg'])
    entries = scan_data['data']
    print(f"Processing {len(entries)} configurations...")

    results = {
        'theta_deg': theta_deg.tolist(),
        'param_names': PARAM_NAMES,
        'data': [],
    }

    for idx, entry in enumerate(entries):
        proj = entry['projectile']
        A = entry['A']
        Z = entry['Z']
        E = entry['E']
        nuc = entry['nucleus']
        params = entry['params']
        rvso = entry['rvso']
        avso = entry['avso']

        F_el = compute_elastic_fisher(proj, A, Z, E, theta_deg, params,
                                       rvso, avso)

        # D_eff
        ev = np.linalg.eigvalsh(F_el)
        ev = ev[ev > 1e-20]
        D_eff = float(np.sum(ev)**2 / np.sum(ev**2)) if len(ev) > 0 else 0.0

        results['data'].append({
            'projectile': proj,
            'nucleus': nuc,
            'A': A, 'Z': Z, 'E': E,
            'fisher_matrix_elastic': F_el.tolist(),
            'D_eff_elastic': D_eff,
        })

        if (idx + 1) % 12 == 0 or idx == 0:
            print(f"  [{idx+1:3d}/{len(entries)}] {proj}+{nuc} E={E}MeV: "
                  f"D_eff(elastic) = {D_eff:.2f}")

    outfile = os.path.join(base_dir, '..', 'data', 'elastic_fisher_matrices.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
