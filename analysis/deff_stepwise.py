#!/usr/bin/env python3
"""
Step-by-step constraint analysis: how each observable adds information.

Computes Fisher information decomposed by:
  Step 1: Elastic dσ/dΩ only
  Step 2: + Reaction cross section σ_R
  Step 3: + Analyzing power Ay
  Step 4: Multi-energy combined (Jacobian projection through 48-dim KD02 universal space)

For each step, shows:
  - Eigenvalue spectrum (which directions get constrained)
  - Information per parameter subgroup (real vol, imaginary, spin-orbit)
  - D_eff for full parameter set

Also computes the KD02 global parameter Jacobian for transforming
local (E-dependent) Fisher to global parameter Fisher.

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from deff_global_kd02 import get_kd02_universal_params, compute_log_jacobian

from deff_scan_extended import (compute_observables_vector, get_kd02_params_13,
                                 kd02_potential_13params)


# Parameter group indices
IDX_REAL = [0, 1, 2]          # V, rv, av
IDX_IMAG_VOL = [3, 4, 5]     # W, rw, aw
IDX_IMAG_SURF = [6, 7, 8]    # Wd, rvd, avd
IDX_IMAG = [3, 4, 5, 6, 7, 8]  # all imaginary
IDX_SO = [9, 10, 11, 12]     # Vso, Wso, rvso, avso

PARAM_NAMES = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd',
               'Vso', 'Wso', 'rvso', 'avso']
GROUP_NAMES = ['Real Volume', 'Imaginary', 'Spin-Orbit']
GROUP_INDICES = [IDX_REAL, IDX_IMAG, IDX_SO]


def compute_deff(F):
    """Effective dimensionality from Fisher matrix."""
    ev = np.linalg.eigvalsh(F)
    ev = ev[ev > 1e-20]
    if len(ev) == 0:
        return 0.0, np.array([])
    s = np.sum(ev)
    return s**2 / np.sum(ev**2), np.sort(ev)[::-1]


def compute_gradients_all(proj, A, Z, E_lab, theta_deg, params,
                           eps_rel=0.01, l_max=30, delta_Ay=0.03):
    """
    Compute gradient matrices for each observable separately.

    Returns:
        G_elastic: (13, n_angles) — d log σ / d log p_i
        G_Ay:      (13, n_angles) — p * dAy/dp / delta_Ay
        G_sigR:    (13, 1)        — d log σ_R / d log p_i
        G_sigT:    (13, 1) or None — d log σ_T / d log p_i (neutrons only)
        obs_0: baseline observables
    """
    n_params = len(params)
    n_angles = len(theta_deg)

    obs_0 = compute_observables_vector(proj, A, Z, E_lab, theta_deg, params,
                                        l_max)

    G_elastic = np.zeros((n_params, n_angles))
    G_Ay = np.zeros((n_params, n_angles))
    G_sigR = np.zeros((n_params, 1))
    G_sigT = np.zeros((n_params, 1)) if proj == 'n' else None

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

        p_i = params[i]

        # Elastic: d log σ / d log p
        dcs_0 = obs_0['elastic_dcs']
        G_elastic[i] = p_i * (obs_p['elastic_dcs'] - obs_m['elastic_dcs']) / (
            2.0 * delta) / (dcs_0 + 1e-30)

        # Ay: scaled derivative
        G_Ay[i] = p_i * (obs_p['Ay'] - obs_m['Ay']) / (2.0 * delta) / delta_Ay

        # σ_R: d log σ_R / d log p
        sr_0 = obs_0['sigma_R']
        G_sigR[i, 0] = p_i * (obs_p['sigma_R'] - obs_m['sigma_R']) / (
            2.0 * delta) / (sr_0 + 1e-30)

        # σ_T (neutrons only)
        if proj == 'n' and G_sigT is not None:
            st_0 = obs_0['sigma_T']
            G_sigT[i, 0] = p_i * (obs_p['sigma_T'] - obs_m['sigma_T']) / (
                2.0 * delta) / (st_0 + 1e-30)

    return G_elastic, G_Ay, G_sigR, G_sigT, obs_0


def subgroup_info(F, indices):
    """Compute trace and D_eff for a parameter subgroup (diagonal sub-block)."""
    F_sub = F[np.ix_(indices, indices)]
    ev = np.linalg.eigvalsh(F_sub)
    ev = ev[ev > 1e-20]
    tr = np.sum(ev) if len(ev) > 0 else 0
    deff = np.sum(ev)**2 / np.sum(ev**2) if len(ev) > 0 and np.sum(ev**2) > 0 else 0
    return {'trace': float(tr), 'D_eff': float(deff), 'eigenvalues': ev.tolist()}


def stepwise_analysis(proj, A, Z, name, energies, theta_deg, l_max=30):
    """
    Full step-by-step analysis for one nucleus.

    Steps:
        1. Single-energy elastic
        2. + σ_R
        3. + Ay (+ σ_T for neutrons)
        4. Multi-energy combined
        5. KD02 global parameter Fisher
    """
    n_params = 13
    result = {
        'projectile': proj, 'nucleus': name, 'A': A, 'Z': Z,
        'energies': energies, 'param_names': PARAM_NAMES,
        'steps': {},
    }

    # ---- Collect gradient matrices at all energies ----
    all_G_elastic = {}
    all_G_Ay = {}
    all_G_sigR = {}
    all_G_sigT = {}
    all_params = {}

    for E in energies:
        params = get_kd02_params_13(proj, A, Z, E)
        all_params[E] = params

        G_el, G_Ay, G_sr, G_st, obs_0 = compute_gradients_all(
            proj, A, Z, E, theta_deg, params, l_max=l_max)

        all_G_elastic[E] = G_el
        all_G_Ay[E] = G_Ay
        all_G_sigR[E] = G_sr
        all_G_sigT[E] = G_st

        print(f"  {proj}+{name}@{E}MeV: σ_R={obs_0['sigma_R']:.1f}mb "
              f"|G_el|={np.linalg.norm(G_el):.1f} "
              f"|G_sigR|={np.linalg.norm(G_sr):.2f}")

    # ---- Step 1: Single-energy elastic (use median energy) ----
    E_ref = energies[len(energies) // 2]  # 50 MeV typically
    G_el = all_G_elastic[E_ref]
    F1 = G_el @ G_el.T
    D1, ev1 = compute_deff(F1)
    result['steps']['1_elastic'] = {
        'label': f'Elastic dσ/dΩ ({E_ref} MeV)',
        'D_eff': float(D1), 'eigenvalues': ev1.tolist(),
        'subgroups': {g: subgroup_info(F1, idx)
                      for g, idx in zip(GROUP_NAMES, GROUP_INDICES)},
    }
    print(f"  Step 1 (elastic, {E_ref}MeV): D_eff = {D1:.2f}")

    # ---- Step 2: + σ_R ----
    G_sr = all_G_sigR[E_ref]
    G2 = np.hstack([G_el, G_sr])
    F2 = G2 @ G2.T
    D2, ev2 = compute_deff(F2)
    result['steps']['2_elastic_sigR'] = {
        'label': f'+ σ_R ({E_ref} MeV)',
        'D_eff': float(D2), 'eigenvalues': ev2.tolist(),
        'subgroups': {g: subgroup_info(F2, idx)
                      for g, idx in zip(GROUP_NAMES, GROUP_INDICES)},
    }
    print(f"  Step 2 (+σ_R): D_eff = {D2:.2f}")

    # ---- Step 3: + Ay (and σ_T for neutrons) ----
    G_Ay = all_G_Ay[E_ref]
    parts = [G_el, G_sr, G_Ay]
    if proj == 'n' and all_G_sigT[E_ref] is not None:
        parts.append(all_G_sigT[E_ref])
    G3 = np.hstack(parts)
    F3 = G3 @ G3.T
    D3, ev3 = compute_deff(F3)
    label3 = f'+ Ay' + (' + σ_T' if proj == 'n' else '')
    result['steps']['3_all_single_E'] = {
        'label': label3 + f' ({E_ref} MeV)',
        'D_eff': float(D3), 'eigenvalues': ev3.tolist(),
        'subgroups': {g: subgroup_info(F3, idx)
                      for g, idx in zip(GROUP_NAMES, GROUP_INDICES)},
    }
    print(f"  Step 3 ({label3}): D_eff = {D3:.2f}")

    # ---- Step 4: Multi-energy via Jacobian projection ----
    # Proper combination: project each F(E) to 48-dim universal KD02 space,
    # sum there, then project back to 13-dim local space at E_ref.
    univ_names, univ_values, univ_groups = get_kd02_universal_params()
    N_univ = len(univ_values)
    J_ref = compute_log_jacobian(univ_values, A, Z, E_ref, proj)

    F4_univ = np.zeros((N_univ, N_univ))
    for E in energies:
        parts_E = [all_G_elastic[E], all_G_sigR[E], all_G_Ay[E]]
        if proj == 'n' and all_G_sigT[E] is not None:
            parts_E.append(all_G_sigT[E])
        G_E = np.hstack(parts_E)
        F_local_E = G_E @ G_E.T
        J_E = compute_log_jacobian(univ_values, A, Z, E, proj)
        F4_univ += J_E.T @ F_local_E @ J_E

    F4 = J_ref @ F4_univ @ J_ref.T
    D4, ev4 = compute_deff(F4)
    result['steps']['4_multi_energy'] = {
        'label': f'Multi-energy ({len(energies)}E, all obs, Jacobian)',
        'D_eff': float(D4), 'eigenvalues': ev4.tolist(),
        'subgroups': {g: subgroup_info(F4, idx)
                      for g, idx in zip(GROUP_NAMES, GROUP_INDICES)},
    }
    print(f"  Step 4 (multi-E, all obs, Jacobian): D_eff = {D4:.2f}")

    # ---- Also: multi-energy elastic only (Jacobian) ----
    F4_el_univ = np.zeros((N_univ, N_univ))
    for E in energies:
        G_E = all_G_elastic[E]
        F_local_E = G_E @ G_E.T
        J_E = compute_log_jacobian(univ_values, A, Z, E, proj)
        F4_el_univ += J_E.T @ F_local_E @ J_E

    F4_el = J_ref @ F4_el_univ @ J_ref.T
    D4_el, ev4_el = compute_deff(F4_el)
    result['steps']['4a_multi_energy_elastic'] = {
        'label': f'Multi-energy ({len(energies)}E, elastic only, Jacobian)',
        'D_eff': float(D4_el), 'eigenvalues': ev4_el.tolist(),
        'subgroups': {g: subgroup_info(F4_el, idx)
                      for g, idx in zip(GROUP_NAMES, GROUP_INDICES)},
    }
    print(f"  Step 4a (multi-E, elastic only, Jacobian): D_eff = {D4_el:.2f}")

    return result


def main():
    print("=" * 70)
    print("Step-by-Step Constraint Analysis")
    print("=" * 70)

    theta_deg = np.linspace(5, 175, 35)
    energies = [10, 20, 30, 50, 100, 150, 200]

    cases = [
        ('n', 40, 20, '40Ca'),
        ('n', 208, 82, '208Pb'),
        ('p', 40, 20, '40Ca'),
        ('n', 120, 50, '120Sn'),
    ]

    all_results = {
        'theta_deg': theta_deg.tolist(),
        'energies': energies,
        'cases': [],
    }

    for proj, A, Z, name in cases:
        print(f"\n{'='*60}")
        print(f"  {proj} + {name}")
        print(f"{'='*60}")

        result = stepwise_analysis(proj, A, Z, name, energies, theta_deg)
        all_results['cases'].append(result)

        # Print summary table
        print(f"\n  {'Step':40s} {'D_eff':>6s}  {'Real':>8s}  {'Imag':>8s}  {'SO':>8s}")
        print(f"  {'-'*75}")
        for key in ['1_elastic', '2_elastic_sigR', '3_all_single_E',
                     '4a_multi_energy_elastic', '4_multi_energy']:
            step = result['steps'].get(key)
            if step is None:
                continue
            subs = step['subgroups']
            # Get subgroup D_eff or trace depending on what's available
            real_d = subs.get('Real Volume', subs.get('Geometry', {})).get('D_eff', 0)
            imag_d = subs.get('Imaginary', subs.get('Imag Depth', {})).get('D_eff', 0)
            so_d = subs.get('Spin-Orbit', subs.get('SO Depth', {})).get('D_eff', 0)
            print(f"  {step['label']:40s} {step['D_eff']:6.2f}"
                  f"  {real_d:8.2f}  {imag_d:8.2f}  {so_d:8.2f}")

    # Save
    outdir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'deff_stepwise.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
