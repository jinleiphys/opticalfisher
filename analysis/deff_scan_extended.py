#!/usr/bin/env python3
"""
Extended D_eff scan: 13-parameter KD02 with spin-orbit + multiple observables.

Computes Fisher Information and D_eff for elastic dsigma/dOmega, analyzing
power Ay, reaction cross section sigma_R, and total cross section sigma_T.

Parameters: [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso, rvso, avso]
   (all 13 KD02 parameters including SO geometry)

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import sys
import os
from multiprocessing import Pool, cpu_count

# Local imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from potentials import (KD02Potential, woods_saxon_form_factor,
                        woods_saxon_derivative, coulomb_potential)
from scattering_fortran import ScatteringSolverFortran, HBARC, AMU, E2
from observables import compute_all_observables

# Pion Compton wavelength squared (same as in potentials.py)
M_PION = 139.5706
LAMBDA_PI_SQ = (HBARC / M_PION)**2


def kd02_potential_13params(r, l, j, A, Z_proj, Z, params):
    """
    KD02-style potential with 13 free parameters, including spin-orbit geometry.

    params = [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso, rvso, avso]

    Returns complex potential array (MeV).
    """
    V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso, rvso, avso = params
    r = np.atleast_1d(np.asarray(r, dtype=float))
    A_third = A ** (1.0 / 3.0)

    # --- Central potential ---
    R_v = rv * A_third
    f_v = woods_saxon_form_factor(r, R_v, av)

    R_w = rw * A_third
    f_w = woods_saxon_form_factor(r, R_w, aw)

    R_d = rvd * A_third
    f_d = woods_saxon_form_factor(r, R_d, avd)
    g_d = 4.0 * f_d * (1.0 - f_d)  # surface form factor

    U = -V * f_v - 1j * W * f_w - 1j * Wd * g_d

    # Coulomb for protons (use KD02 radius formula)
    if Z_proj > 0:
        rc_kd02 = 1.198 + 0.697 * A**(-2./3.) + 12.994 * A**(-5./3.)
        Rc = rc_kd02 * A_third
        U = U + coulomb_potential(r, Z_proj, Z, Rc)

    # --- Spin-orbit ---
    if l > 0 and (abs(Vso) > 1e-10 or abs(Wso) > 1e-10):
        R_so = rvso * A_third
        dfdr = woods_saxon_derivative(r, R_so, avso)
        r_safe = np.where(r > 1e-6, r, 1e-6)
        # Thomas form with factor of 2 (KD02 convention: l.sigma = 2 l.s)
        U_so_radial = 2.0 * LAMBDA_PI_SQ * (Vso + 1j * Wso) * dfdr / r_safe
        # <l.s> = [j(j+1) - l(l+1) - 3/4] / 2
        ls = 0.5 * (j * (j + 1) - l * (l + 1) - 0.75)
        U = U + U_so_radial * ls

    return U


def adaptive_lmax(A, E_lab, min_lmax=30):
    """Determine l_max from grazing angular momentum: l_max = k*R + 15."""
    A_proj = 1
    mu = A_proj * A / (A_proj + A)
    E_cm = E_lab * A / (A_proj + A)
    k = np.sqrt(2 * mu * AMU * E_cm) / HBARC
    R = 1.25 * A**(1./3.)
    l_g = k * R
    return max(min_lmax, int(l_g + 15))


def compute_observables_vector(proj, A, Z, E_lab, theta_deg, params,
                                l_max=None):
    """
    Compute all observables for given parameters.

    Returns:
        dict with 'elastic_dcs', 'Ay', 'sigma_R', and possibly 'sigma_T'
    """
    A_proj = 1
    mu = A_proj * A / (A_proj + A)
    E_cm = E_lab * A / (A_proj + A)
    Z_proj = 1 if proj == 'p' else 0

    if l_max is None:
        l_max = adaptive_lmax(A, E_lab)

    def pot_lj(r, l, j):
        return kd02_potential_13params(r, l, j, A, Z_proj, Z, params)

    solver = ScatteringSolverFortran(r_max=40.0, hcm=0.02, l_max=l_max)
    results = solver.solve_spin_half(pot_lj, E_cm, mu,
                                      Z1=Z_proj, Z2=Z, l_max=l_max)

    k = results['k']
    eta = results['eta']
    S_lj = results['S_matrix_lj']
    sigma_coul = results['sigma']

    obs = compute_all_observables(k, eta, S_lj, theta_deg, l_max,
                                  is_neutron=(proj == 'n'),
                                  sigma_l=sigma_coul)
    obs['k'] = k
    obs['eta'] = eta
    return obs


def build_gradient_vector(obs_0, obs_plus, obs_minus, delta, param_val,
                          proj, delta_Ay=0.03):
    """
    Build gradient vector for one parameter perturbation.

    Uses log-derivatives for dsigma/dOmega, sigma_R, sigma_T (positive-definite).
    Uses scaled linear derivatives for Ay (can be negative).

    Returns:
        gradient: 1D array of gradient components for all observables
    """
    grads = []

    # dsigma/dOmega: d log sigma / d log p = (p/sigma) * (sigma+ - sigma-)/(2 delta)
    dcs_0 = obs_0['elastic_dcs']
    dcs_p = obs_plus['elastic_dcs']
    dcs_m = obs_minus['elastic_dcs']
    g_dcs = param_val * (dcs_p - dcs_m) / (2.0 * delta) / (dcs_0 + 1e-30)
    grads.append(g_dcs)

    # Ay: (p / delta_Ay) * dAy/dp
    Ay_p = obs_plus['Ay']
    Ay_m = obs_minus['Ay']
    g_Ay = param_val * (Ay_p - Ay_m) / (2.0 * delta) / delta_Ay
    grads.append(g_Ay)

    # sigma_R: d log sigma_R / d log p
    sr_0 = obs_0['sigma_R']
    sr_p = obs_plus['sigma_R']
    sr_m = obs_minus['sigma_R']
    g_sr = param_val * (sr_p - sr_m) / (2.0 * delta) / (sr_0 + 1e-30)
    grads.append(np.array([g_sr]))

    # sigma_T (neutrons only): d log sigma_T / d log p
    if proj == 'n':
        st_0 = obs_0['sigma_T']
        st_p = obs_plus['sigma_T']
        st_m = obs_minus['sigma_T']
        g_st = param_val * (st_p - st_m) / (2.0 * delta) / (st_0 + 1e-30)
        grads.append(np.array([g_st]))

    return np.concatenate(grads)


def compute_fisher_extended(proj, A, Z, E_lab, theta_deg, params,
                             eps_rel=0.01, l_max=None,
                             delta_Ay=0.03):
    """
    Compute Fisher matrix for 13 parameters with all observables.

    Returns:
        F: Fisher matrix (13 x 13)
        gradients: gradient matrix (13 x n_data)
        obs_0: baseline observables dict
        n_data: dict with number of data points per observable type
    """
    n_params = len(params)

    # Baseline observables
    obs_0 = compute_observables_vector(proj, A, Z, E_lab, theta_deg, params,
                                        l_max)

    n_angles = len(theta_deg)
    n_data_dcs = n_angles
    n_data_Ay = n_angles
    n_data_sr = 1
    n_data_st = 1 if proj == 'n' else 0
    n_total = n_data_dcs + n_data_Ay + n_data_sr + n_data_st

    gradients = np.zeros((n_params, n_total))

    for i in range(n_params):
        params_plus = list(params)
        params_minus = list(params)

        delta = eps_rel * abs(params[i])
        if delta < 1e-8:
            delta = 1e-8

        params_plus[i] += delta
        params_minus[i] -= delta

        obs_p = compute_observables_vector(proj, A, Z, E_lab, theta_deg,
                                            params_plus, l_max)
        obs_m = compute_observables_vector(proj, A, Z, E_lab, theta_deg,
                                            params_minus, l_max)

        gradients[i] = build_gradient_vector(obs_0, obs_p, obs_m, delta,
                                              params[i], proj, delta_Ay)

    # Fisher matrix: F = G G^T
    F = gradients @ gradients.T

    n_data = {
        'dcs': n_data_dcs,
        'Ay': n_data_Ay,
        'sigma_R': n_data_sr,
        'sigma_T': n_data_st,
        'total': n_total,
    }

    return F, gradients, obs_0, n_data


def compute_deff(F):
    """Compute effective dimensionality (participation ratio of eigenvalues)."""
    eigenvalues = np.linalg.eigvalsh(F)
    eigenvalues = eigenvalues[eigenvalues > 1e-20]

    if len(eigenvalues) == 0:
        return 0.0, np.array([]), np.inf

    sum_lambda = np.sum(eigenvalues)
    sum_lambda2 = np.sum(eigenvalues**2)
    D_eff = sum_lambda**2 / sum_lambda2 if sum_lambda2 > 0 else 0

    cond = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else np.inf

    return D_eff, np.sort(eigenvalues)[::-1], cond


def compute_deff_subsets(F_full, gradients, n_data, proj):
    """
    Compute D_eff for different observable subsets and parameter subsets.

    Returns dict with D_eff for each combination.
    """
    n_dcs = n_data['dcs']
    n_Ay = n_data['Ay']

    results = {}

    # --- Observable subsets (all 13 params) ---

    # Elastic only
    g_dcs = gradients[:, :n_dcs]
    F_dcs = g_dcs @ g_dcs.T
    D, ev, cn = compute_deff(F_dcs)
    results['elastic_13p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    # Elastic + Ay
    g_dcs_Ay = gradients[:, :n_dcs + n_Ay]
    F_dcs_Ay = g_dcs_Ay @ g_dcs_Ay.T
    D, ev, cn = compute_deff(F_dcs_Ay)
    results['elastic_Ay_13p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    # Elastic + sigma_R
    cols = list(range(n_dcs)) + [n_dcs + n_Ay]
    g_sub = gradients[:, cols]
    F_sub = g_sub @ g_sub.T
    D, ev, cn = compute_deff(F_sub)
    results['elastic_sigR_13p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    # Elastic + Ay + sigma_R
    cols = list(range(n_dcs + n_Ay)) + [n_dcs + n_Ay]
    g_sub = gradients[:, cols]
    F_sub = g_sub @ g_sub.T
    D, ev, cn = compute_deff(F_sub)
    results['elastic_Ay_sigR_13p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    # All observables (13 params)
    D, ev, cn = compute_deff(F_full)
    results['all_13p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    # --- 11-param subsets (no SO geometry: rvso, avso) ---
    g11 = gradients[:11, :]

    # Elastic only, 11 params
    g11_dcs = g11[:, :n_dcs]
    F11 = g11_dcs @ g11_dcs.T
    D, ev, cn = compute_deff(F11)
    results['elastic_11p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    # All observables, 11 params
    F11_all = g11 @ g11.T
    D, ev, cn = compute_deff(F11_all)
    results['all_11p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    # --- 9-param subsets (central only, no Vso/Wso/rvso/avso) ---
    g9 = gradients[:9, :]

    # Elastic only, 9 params
    g9_dcs = g9[:, :n_dcs]
    F9 = g9_dcs @ g9_dcs.T
    D, ev, cn = compute_deff(F9)
    results['elastic_9p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    # All observables, 9 params
    F9_all = g9 @ g9.T
    D, ev, cn = compute_deff(F9_all)
    results['all_9p'] = {'D_eff': D, 'eigenvalues': ev.tolist(), 'cond': cn}

    return results


def get_kd02_params_13(projectile, A, Z, E_lab):
    """
    Get 13 KD02 parameters: [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso, rvso, avso]
    """
    pot = KD02Potential(projectile, A, Z, E_lab)
    params = [
        pot.V, pot.rv, pot.av,
        pot.W, pot.rw, pot.aw,
        pot.Wd, pot.rvd, pot.avd,
        pot.Vso, pot.Wso,
        pot.rvso, pot.avso,
    ]
    return params


def _compute_one_config(args):
    """Worker function for parallel computation of one (proj, A, Z, E) config."""
    proj, A, Z, name, E, theta_deg = args
    try:
        params = get_kd02_params_13(proj, A, Z, E)
        F, gradients, obs_0, n_data = compute_fisher_extended(
            proj, A, Z, E, theta_deg, params
        )
        subsets = compute_deff_subsets(F, gradients, n_data, proj)

        entry = {
            'projectile': proj, 'nucleus': name,
            'A': A, 'Z': Z, 'E': E,
            'params': params,
            'sigma_R': float(obs_0['sigma_R']),
            'fisher_matrix': F.tolist(),
            'deff_results': {},
        }
        if proj == 'n':
            entry['sigma_T'] = float(obs_0['sigma_T'])
        for key, val in subsets.items():
            entry['deff_results'][key] = {
                'D_eff': float(val['D_eff']),
                'condition_number': float(val['cond']),
                'eigenvalues': val['eigenvalues'],
            }
        d9 = subsets['elastic_9p']['D_eff']
        d13 = subsets['elastic_13p']['D_eff']
        print(f"  {proj}+{name}@{E}MeV: elastic(9p)={d9:.2f} elastic(13p)={d13:.2f}"
              f"  sigma_R={obs_0['sigma_R']:.1f} mb")
        return entry
    except Exception as e:
        print(f"  ERROR {proj}+{name}@{E}MeV: {e}")
        return {'projectile': proj, 'nucleus': name, 'A': A, 'Z': Z, 'E': E,
                'error': str(e)}


def main():
    """Extended D_eff scan with 13 parameters and multiple observables."""

    print("=" * 70)
    print("Extended D_eff Scan: 13-param KD02 + spin-orbit observables")
    print("=" * 70)

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

    energies = [10, 20, 30, 50, 100, 150, 200]
    projectiles = ['n', 'p']

    param_names_13 = ['V', 'rv', 'av', 'W', 'rw', 'aw',
                      'Wd', 'rvd', 'avd', 'Vso', 'Wso', 'rvso', 'avso']
    theta_deg = np.linspace(5, 175, 35)

    # Build task list
    tasks = []
    for proj in projectiles:
        for A, Z, name in nuclei:
            for E in energies:
                tasks.append((proj, A, Z, name, E, theta_deg))

    total = len(tasks)
    print(f"Total configurations: {total}")

    # Parallel computation
    n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} parallel workers")
    with Pool(n_workers) as pool:
        data_list = pool.map(_compute_one_config, tasks)

    results = {
        'method': 'numerov_spin_half_extended',
        'nuclei': [n[2] for n in nuclei],
        'A_values': [n[0] for n in nuclei],
        'energies': energies,
        'projectiles': projectiles,
        'param_names': param_names_13,
        'theta_deg': theta_deg.tolist(),
        'observable_combinations': [
            'elastic_9p', 'elastic_11p', 'elastic_13p', 'elastic_Ay_13p',
            'elastic_sigR_13p', 'elastic_Ay_sigR_13p', 'all_13p',
            'all_11p', 'all_9p',
        ],
        'data': data_list,
    }

    # Save main results
    outdir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'deff_scan_extended.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")

    # Save gradient matrices for representative cases (for angle-resolved plots)
    representative_cases = [
        ('n', 40, 20, '40Ca', 50),
        ('n', 208, 82, '208Pb', 30),
        ('p', 40, 20, '40Ca', 50),
        ('n', 120, 50, '120Sn', 100),
    ]
    grad_data = {'theta_deg': theta_deg.tolist(), 'param_names': param_names_13,
                 'cases': []}
    for proj_r, A_r, Z_r, name_r, E_r in representative_cases:
        print(f"\nComputing gradients for {proj_r}+{name_r}@{E_r} MeV...")
        try:
            params_r = get_kd02_params_13(proj_r, A_r, Z_r, E_r)
            F_r, grads_r, obs_r, nd_r = compute_fisher_extended(
                proj_r, A_r, Z_r, E_r, theta_deg, params_r)
            grad_data['cases'].append({
                'projectile': proj_r, 'nucleus': name_r,
                'A': A_r, 'Z': Z_r, 'E': E_r,
                'params': params_r,
                'gradients': grads_r.tolist(),
                'fisher_matrix': F_r.tolist(),
                'n_data': nd_r,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
    grad_file = os.path.join(outdir, 'deff_gradients_representative.json')
    with open(grad_file, 'w') as f:
        json.dump(grad_data, f, indent=2)
    print(f"Saved: {grad_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = [d for d in results['data'] if 'deff_results' in d]

    for combo in ['elastic_9p', 'elastic_11p', 'elastic_13p',
                   'elastic_Ay_13p', 'elastic_Ay_sigR_13p', 'all_13p']:
        deff_vals = [d['deff_results'][combo]['D_eff'] for d in valid
                     if combo in d['deff_results']]
        if deff_vals:
            print(f"  {combo:25s}: D_eff = {np.mean(deff_vals):.2f} "
                  f"+/- {np.std(deff_vals):.2f} "
                  f"[{np.min(deff_vals):.2f}, {np.max(deff_vals):.2f}]")


if __name__ == '__main__':
    main()
