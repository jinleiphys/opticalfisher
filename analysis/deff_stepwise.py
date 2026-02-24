#!/usr/bin/env python3
"""
Step-by-step constraint analysis: how each observable adds information.

Computes Fisher information decomposed by:
  Step 1: Elastic dσ/dΩ only
  Step 2: + Reaction cross section σ_R
  Step 3: + Analyzing power Ay
  Step 4: Multi-energy combined Fisher
  Step 5: KD02 global parameter systematics

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
from potentials import KD02Potential

from deff_scan_extended import (compute_observables_vector, get_kd02_params_11,
                                 kd02_potential_11params)


# Parameter group indices
IDX_REAL = [0, 1, 2]          # V, rv, av
IDX_IMAG_VOL = [3, 4, 5]     # W, rw, aw
IDX_IMAG_SURF = [6, 7, 8]    # Wd, rvd, avd
IDX_IMAG = [3, 4, 5, 6, 7, 8]  # all imaginary
IDX_SO = [9, 10]              # Vso, Wso

PARAM_NAMES = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd', 'Vso', 'Wso']
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


def compute_gradients_all(proj, A, Z, E_lab, theta_deg, params, rvso, avso,
                           eps_rel=0.01, l_max=30, delta_Ay=0.03):
    """
    Compute gradient matrices for each observable separately.

    Returns:
        G_elastic: (11, n_angles) — d log σ / d log p_i
        G_Ay:      (11, n_angles) — p * dAy/dp / delta_Ay
        G_sigR:    (11, 1)        — d log σ_R / d log p_i
        G_sigT:    (11, 1) or None — d log σ_T / d log p_i (neutrons only)
        obs_0: baseline observables
    """
    n_params = len(params)
    n_angles = len(theta_deg)

    obs_0 = compute_observables_vector(proj, A, Z, E_lab, theta_deg, params,
                                        rvso, avso, l_max)

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
                                            params_plus, rvso, avso, l_max)
        obs_m = compute_observables_vector(proj, A, Z, E_lab, theta_deg,
                                            params_minus, rvso, avso, l_max)

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


def kd02_global_jacobian(proj, A, Z, energies, eps_rel=0.005):
    """
    Compute Jacobian d(local_params) / d(global_params) for KD02.

    At each energy E, the 11 local params [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso]
    are determined by KD02 formulas from A, Z, E and a set of global coefficients.

    For a FIXED nucleus (A, Z), the energy-independent geometry params are:
        rv(A), av(A), rw=rv, aw=av, rvd(A), avd(A), rvso(A), avso=0.59
    The energy-dependent depths are:
        V(E), W(E), Wd(E), Vso(E), Wso(E)

    Global parameters (for given A, Z):
        [rv, av, rvd, avd,   ← geometry (4, energy-independent)
         v1, w1, d1, vso1, wso1,  ← depth amplitudes (5)
         v2, v3, w2, d2, d3, vso2, wso2,  ← depth shape (7)
         ef]  ← Fermi energy (1)
    Total: 17 global params → 11*N_E local params

    Returns:
        J: Jacobian matrix (11*N_E, N_global)
        global_param_names: list of global parameter names
        global_param_values: nominal values
    """
    N_E = len(energies)
    pot0 = KD02Potential(proj, A, Z, energies[0])
    k0 = pot0.k0

    # Extract KD02 coefficients (they depend on A, Z, proj but NOT on E)
    N_nuc = A - Z
    A_third = A ** (1.0 / 3.0)

    # Nominal global parameters
    v4 = 7.0e-9
    w2 = 73.55 + 0.0795 * A
    d2_val = 0.0180 + 3.802e-3 / (1. + np.exp((A - 156.) / 8.))
    d3 = 11.5
    vso1 = 5.922 + 0.0030 * A
    vso2 = 0.0040
    wso1 = -3.1
    wso2 = 160.

    if k0 == 1:  # neutron
        ef = -11.2814 + 0.02646 * A
        v1 = 59.30 - 21.0 * (N_nuc - Z) / A - 0.024 * A
        v2 = 7.228e-3 - 1.48e-6 * A
        v3 = 1.994e-5 - 2.0e-8 * A
        w1 = 12.195 + 0.0167 * A
        d1 = 16.0 - 16.0 * (N_nuc - Z) / A
        avd = 0.5446 - 1.656e-4 * A
    else:  # proton
        ef = -8.4075 + 0.01378 * A
        v1 = 59.30 + 21.0 * (N_nuc - Z) / A - 0.024 * A
        v2 = 7.067e-3 + 4.23e-6 * A
        v3 = 1.729e-5 + 1.136e-8 * A
        w1 = 14.667 + 0.009629 * A
        d1 = 16.0 + 16.0 * (N_nuc - Z) / A
        avd = 0.5187 + 5.205e-4 * A

    rv = 1.3039 - 0.4054 * A**(-1./3.)
    av = 0.6778 - 1.487e-4 * A
    rvd = 1.3424 - 0.01585 * A**(1./3.)

    global_names = ['rv', 'av', 'rvd', 'avd',
                    'v1', 'w1', 'd1', 'vso1', 'wso1',
                    'v2', 'v3', 'w2', 'd2', 'd3', 'vso2', 'wso2', 'ef']
    global_vals = [rv, av, rvd, avd,
                   v1, w1, d1, vso1, wso1,
                   v2, v3, w2, d2_val, d3, vso2, wso2, ef]
    N_global = len(global_vals)

    # Compute Jacobian by finite differences
    J = np.zeros((11 * N_E, N_global))

    def local_params_from_globals(gvals):
        """Compute all local params at all energies from global params."""
        rv_g, av_g, rvd_g, avd_g = gvals[0:4]
        v1_g, w1_g, d1_g, vso1_g, wso1_g = gvals[4:9]
        v2_g, v3_g, w2_g, d2_g, d3_g, vso2_g, wso2_g, ef_g = gvals[9:17]

        all_local = []
        for E in energies:
            f = E - ef_g
            V_loc = v1_g * (1. - v2_g*f + v3_g*f**2 - v4*f**3)
            W_loc = w1_g * f**2 / (f**2 + w2_g**2)
            Wd_loc = d1_g * f**2 * np.exp(-d2_g*f) / (f**2 + d3_g**2)
            Vso_loc = vso1_g * np.exp(-vso2_g * f)
            Wso_loc = wso1_g * f**2 / (f**2 + wso2_g**2)

            # local: [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso]
            # Note: rw=rv, aw=av in KD02
            local = [V_loc, rv_g, av_g, W_loc, rv_g, av_g,
                     Wd_loc, rvd_g, avd_g, Vso_loc, Wso_loc]
            all_local.extend(local)
        return np.array(all_local)

    p0 = local_params_from_globals(global_vals)

    for ig in range(N_global):
        gvals_plus = list(global_vals)
        gvals_minus = list(global_vals)
        dg = eps_rel * abs(global_vals[ig])
        if dg < 1e-10:
            dg = 1e-10
        gvals_plus[ig] += dg
        gvals_minus[ig] -= dg

        p_plus = local_params_from_globals(gvals_plus)
        p_minus = local_params_from_globals(gvals_minus)

        J[:, ig] = (p_plus - p_minus) / (2 * dg)

    return J, global_names, global_vals


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
    n_params = 11
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
        params, rvso, avso = get_kd02_params_11(proj, A, Z, E)
        all_params[E] = params

        G_el, G_Ay, G_sr, G_st, obs_0 = compute_gradients_all(
            proj, A, Z, E, theta_deg, params, rvso, avso, l_max=l_max)

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

    # ---- Step 4: Multi-energy (all obs at each E) ----
    # Block-diagonal Fisher: F_multi = Σ_E F(E)
    F4 = np.zeros((n_params, n_params))
    for E in energies:
        parts_E = [all_G_elastic[E], all_G_sigR[E], all_G_Ay[E]]
        if proj == 'n' and all_G_sigT[E] is not None:
            parts_E.append(all_G_sigT[E])
        G_E = np.hstack(parts_E)
        F4 += G_E @ G_E.T
    D4, ev4 = compute_deff(F4)
    result['steps']['4_multi_energy'] = {
        'label': f'Multi-energy ({len(energies)}E, all obs)',
        'D_eff': float(D4), 'eigenvalues': ev4.tolist(),
        'subgroups': {g: subgroup_info(F4, idx)
                      for g, idx in zip(GROUP_NAMES, GROUP_INDICES)},
    }
    print(f"  Step 4 (multi-E, all obs): D_eff = {D4:.2f}")

    # ---- Also: multi-energy elastic only ----
    F4_el = np.zeros((n_params, n_params))
    for E in energies:
        G_E = all_G_elastic[E]
        F4_el += G_E @ G_E.T
    D4_el, ev4_el = compute_deff(F4_el)
    result['steps']['4a_multi_energy_elastic'] = {
        'label': f'Multi-energy ({len(energies)}E, elastic only)',
        'D_eff': float(D4_el), 'eigenvalues': ev4_el.tolist(),
        'subgroups': {g: subgroup_info(F4_el, idx)
                      for g, idx in zip(GROUP_NAMES, GROUP_INDICES)},
    }
    print(f"  Step 4a (multi-E, elastic only): D_eff = {D4_el:.2f}")

    # ---- Step 5: KD02 global parameter systematics ----
    # Transform F_multi (11x11) to F_global (17x17)
    # F_global = J^T * F_block_diag * J
    # where F_block_diag is the block-diagonal Fisher with 11 params per energy
    # and J is the Jacobian from global to local

    J, global_names, global_vals = kd02_global_jacobian(
        proj, A, Z, energies)

    # Build full block-diagonal local Fisher (11*N_E × 11*N_E)
    N_E = len(energies)
    F_block = np.zeros((11 * N_E, 11 * N_E))
    for ie, E in enumerate(energies):
        parts_E = [all_G_elastic[E], all_G_sigR[E], all_G_Ay[E]]
        if proj == 'n' and all_G_sigT[E] is not None:
            parts_E.append(all_G_sigT[E])
        G_E = np.hstack(parts_E)
        F_E = G_E @ G_E.T
        F_block[ie*11:(ie+1)*11, ie*11:(ie+1)*11] = F_E

    # Transform to global parameters
    F_global = J.T @ F_block @ J
    D5, ev5 = compute_deff(F_global)

    # Subgroup mapping for global params:
    # [rv, av, rvd, avd, v1, w1, d1, vso1, wso1, v2, v3, w2, d2, d3, vso2, wso2, ef]
    #  0   1   2    3    4   5   6   7     8     9  10  11  12  13  14    15    16
    IDX_GLOBAL_GEOM = [0, 1, 2, 3]  # geometry
    IDX_GLOBAL_REAL_DEPTH = [4, 9, 10, 16]  # v1, v2, v3, ef (real depth params)
    IDX_GLOBAL_IMAG_DEPTH = [5, 6, 11, 12, 13]  # w1, d1, w2, d2, d3
    IDX_GLOBAL_SO = [7, 8, 14, 15]  # vso1, wso1, vso2, wso2

    global_group_names = ['Geometry', 'Real Depth', 'Imag Depth', 'SO Depth']
    global_group_indices = [IDX_GLOBAL_GEOM, IDX_GLOBAL_REAL_DEPTH,
                            IDX_GLOBAL_IMAG_DEPTH, IDX_GLOBAL_SO]

    result['steps']['5_kd02_global'] = {
        'label': f'KD02 systematics ({len(energies)}E, {len(global_names)}p)',
        'D_eff': float(D5), 'eigenvalues': ev5.tolist(),
        'N_global_params': len(global_names),
        'global_param_names': global_names,
        'subgroups': {g: subgroup_info(F_global, idx)
                      for g, idx in zip(global_group_names, global_group_indices)},
    }
    print(f"  Step 5 (KD02 global, {len(global_names)}p): D_eff = {D5:.2f}")

    return result


def main():
    print("=" * 70)
    print("Step-by-Step Constraint Analysis")
    print("=" * 70)

    theta_deg = np.linspace(10, 170, 17)
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
                     '4a_multi_energy_elastic', '4_multi_energy', '5_kd02_global']:
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
