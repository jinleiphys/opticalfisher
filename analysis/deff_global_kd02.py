#!/usr/bin/env python3
"""
Global KD02 Fisher information analysis in universal coefficient space.

Combines Fisher matrices from all (nucleus, energy, projectile) configurations
into a single Fisher matrix in the universal KD02 coefficient space (~45 params).

The KD02 global optical potential parametrizes local potential parameters
(V, rv, av, W, ...) as functions of A, Z, E using ~45 universal coefficients.
This script computes the Jacobian from universal → local for each system,
transforms each system's 11×11 Fisher matrix to the 45×45 universal space,
and sums them to get the global Fisher matrix.

Key result: D_eff(universal) shows how many KD02 global coefficients are
actually constrained by the combined dataset of all nuclei and energies.

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from potentials import KD02Potential


# =============================================================================
# Universal KD02 parameter definitions
# =============================================================================

def get_kd02_universal_params():
    """
    Return the 45 universal KD02 coefficients (names and nominal values).

    These are the coefficients in the KD02 parametrization formulas that
    determine all local potential parameters for any (A, Z, E, proj).

    Groups:
        Shared geometry (6): rv, av, rvd geometry coefficients
        Shared depth (16): v10, v1_iso, ..., wso2
        Neutron-specific (10): ef_n, v2_n, v3_n, w1_n, avd_n
        Proton-specific (13): ef_p, v2_p, v3_p, w1_p, avd_p, rc

    Note: rvso and avso are excluded because the 11-param local Fisher
    does not vary spin-orbit geometry.
    """
    names = [
        # Shared geometry (6)
        'rv0', 'rv1',       # rv = rv0 - rv1 * A^{-1/3}
        'av0', 'av1',       # av = av0 - av1 * A
        'rvd0', 'rvd1',     # rvd = rvd0 - rvd1 * A^{1/3}

        # Shared depth coefficients (16)
        'v10', 'v1_iso', 'v1_A',  # v1 = v10 ∓ v1_iso*(N-Z)/A - v1_A*A
        'v4',                       # cubic energy term
        'd10', 'd1_iso',           # d1 = d10 ∓ d1_iso*(N-Z)/A
        'w20', 'w21',              # w2 = w20 + w21*A
        'd20', 'd2_amp',           # d2 = d20 + d2_amp/(1+exp((A-156)/8))
        'd3',                       # Fermi energy denominator
        'vso10', 'vso11',          # vso1 = vso10 + vso11*A
        'vso2',                     # V_so energy decay rate
        'wso1', 'wso2',            # W_so depth and energy scale

        # Neutron-specific (10)
        'ef_n0', 'ef_n1',         # ef_n = ef_n0 + ef_n1*A
        'v2_n0', 'v2_n1',         # v2_n = v2_n0 - v2_n1*A
        'v3_n0', 'v3_n1',         # v3_n = v3_n0 - v3_n1*A
        'w1_n0', 'w1_n1',         # w1_n = w1_n0 + w1_n1*A
        'avd_n0', 'avd_n1',       # avd_n = avd_n0 - avd_n1*A

        # Proton-specific (13)
        'ef_p0', 'ef_p1',         # ef_p = ef_p0 + ef_p1*A
        'v2_p0', 'v2_p1',         # v2_p = v2_p0 + v2_p1*A
        'v3_p0', 'v3_p1',         # v3_p = v3_p0 + v3_p1*A
        'w1_p0', 'w1_p1',         # w1_p = w1_p0 + w1_p1*A
        'avd_p0', 'avd_p1',       # avd_p = avd_p0 + avd_p1*A
        'rc0', 'rc1', 'rc2',      # rc = rc0 + rc1*A^{-2/3} + rc2*A^{-5/3}
    ]

    values = [
        # Shared geometry
        1.3039, 0.4054,
        0.6778, 1.487e-4,
        1.3424, 0.01585,

        # Shared depth
        59.30, 21.0, 0.024,
        7.0e-9,
        16.0, 16.0,
        73.55, 0.0795,
        0.0180, 3.802e-3,
        11.5,
        5.922, 0.003,
        0.004,
        -3.1, 160.0,

        # Neutron-specific
        -11.2814, 0.02646,
        7.228e-3, 1.48e-6,
        1.994e-5, 2.0e-8,
        12.195, 0.0167,
        0.5446, 1.656e-4,

        # Proton-specific
        -8.4075, 0.01378,
        7.067e-3, 4.23e-6,
        1.729e-5, 1.136e-8,
        14.667, 0.009629,
        0.5187, 5.205e-4,
        1.198, 0.697, 12.994,
    ]

    # Group indices for subgroup analysis
    groups = {
        'Shared Geometry': list(range(0, 6)),
        'Shared Depth': list(range(6, 22)),
        'Neutron-specific': list(range(22, 32)),
        'Proton-specific': list(range(32, 45)),
    }

    return names, np.array(values, dtype=float), groups


def universal_to_local(univ, A, Z, E, proj):
    """
    Compute 11 local KD02 parameters from universal coefficients.

    Args:
        univ: array of 45 universal KD02 coefficients
        A, Z: target nucleus
        E: lab energy (MeV)
        proj: 'n' or 'p'

    Returns:
        local: array [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso]
    """
    N = A - Z
    A13 = A ** (1.0 / 3.0)

    # --- Unpack shared geometry ---
    rv0, rv1 = univ[0], univ[1]
    av0, av1 = univ[2], univ[3]
    rvd0, rvd1 = univ[4], univ[5]

    rv = rv0 - rv1 * A**(-1.0/3.0)
    av = av0 - av1 * A
    rvd = rvd0 - rvd1 * A13

    # rw = rv, aw = av in KD02
    rw, aw = rv, av

    # --- Unpack shared depth ---
    v10, v1_iso, v1_A = univ[6], univ[7], univ[8]
    v4 = univ[9]
    d10, d1_iso = univ[10], univ[11]
    w20, w21 = univ[12], univ[13]
    d20, d2_amp = univ[14], univ[15]
    d3 = univ[16]
    vso10, vso11 = univ[17], univ[18]
    vso2 = univ[19]
    wso1 = univ[20]
    wso2 = univ[21]

    w2 = w20 + w21 * A
    d2 = d20 + d2_amp / (1.0 + np.exp((A - 156.0) / 8.0))
    vso1 = vso10 + vso11 * A

    # --- Unpack nucleon-specific ---
    if proj == 'n':
        ef0, ef1 = univ[22], univ[23]
        v20, v21 = univ[24], univ[25]
        v30, v31 = univ[26], univ[27]
        w10, w11 = univ[28], univ[29]
        avd0, avd1 = univ[30], univ[31]

        ef = ef0 + ef1 * A
        v1 = v10 - v1_iso * (N - Z) / A - v1_A * A
        v2 = v20 - v21 * A
        v3 = v30 - v31 * A
        w1 = w10 + w11 * A
        d1 = d10 - d1_iso * (N - Z) / A
        avd = avd0 - avd1 * A
        vcoul = 0.0
    else:  # proton
        ef0, ef1 = univ[32], univ[33]
        v20, v21 = univ[34], univ[35]
        v30, v31 = univ[36], univ[37]
        w10, w11 = univ[38], univ[39]
        avd0, avd1 = univ[40], univ[41]
        rc0, rc1, rc2 = univ[42], univ[43], univ[44]

        ef = ef0 + ef1 * A
        v1 = v10 + v1_iso * (N - Z) / A - v1_A * A
        v2 = v20 + v21 * A
        v3 = v30 + v31 * A
        w1 = w10 + w11 * A
        d1 = d10 + d1_iso * (N - Z) / A
        avd = avd0 + avd1 * A

        # Coulomb correction for protons
        rc = rc0 + rc1 * A**(-2.0/3.0) + rc2 * A**(-5.0/3.0)
        Vc = 1.73 / rc * Z / A13
        f_c = E - ef
        vcoul = Vc * v1 * (v2 - 2.0*v3*f_c + 3.0*v4*f_c**2)

    # --- Energy-dependent depths ---
    f = E - ef

    V = v1 * (1.0 - v2*f + v3*f**2 - v4*f**3) + vcoul
    W = w1 * f**2 / (f**2 + w2**2)
    Wd = d1 * f**2 * np.exp(-d2 * f) / (f**2 + d3**2)
    Vso = vso1 * np.exp(-vso2 * f)
    Wso = wso1 * f**2 / (f**2 + wso2**2)

    return np.array([V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso])


def compute_log_jacobian(univ, A, Z, E, proj, eps_rel=1e-4):
    """
    Compute log-space Jacobian: J_ik = (u_k / p_i) * dp_i/du_k.

    This transforms the local log-derivative Fisher matrix to the
    universal log-derivative Fisher matrix:
        F_global = J^T @ F_local @ J

    Args:
        univ: array of universal KD02 coefficients
        A, Z, E, proj: system specification
        eps_rel: relative perturbation for finite differences

    Returns:
        J: (11, N_univ) log-space Jacobian matrix
    """
    N_univ = len(univ)
    p0 = universal_to_local(univ, A, Z, E, proj)

    J = np.zeros((11, N_univ))

    for k in range(N_univ):
        u_plus = univ.copy()
        u_minus = univ.copy()
        delta = eps_rel * max(abs(univ[k]), 1e-15)
        u_plus[k] += delta
        u_minus[k] -= delta

        p_plus = universal_to_local(u_plus, A, Z, E, proj)
        p_minus = universal_to_local(u_minus, A, Z, E, proj)

        dp_du = (p_plus - p_minus) / (2.0 * delta)

        # Log-Jacobian: J_ik = u_k/p_i * dp_i/du_k
        for i in range(11):
            if abs(p0[i]) > 1e-30:
                J[i, k] = univ[k] / p0[i] * dp_du[i]

    return J


def compute_deff(F):
    """Effective dimensionality (participation ratio of eigenvalues)."""
    ev = np.linalg.eigvalsh(F)
    ev = ev[ev > 1e-20]
    if len(ev) == 0:
        return 0.0, np.array([])
    s = np.sum(ev)
    return s**2 / np.sum(ev**2), np.sort(ev)[::-1]


def validate_jacobian(univ, A, Z, E, proj):
    """
    Validate universal_to_local against KD02Potential class.

    Returns max absolute difference between our implementation and the
    reference KD02Potential class.
    """
    p_ours = universal_to_local(univ, A, Z, E, proj)
    pot = KD02Potential(proj, A, Z, E)
    p_ref = np.array([pot.V, pot.rv, pot.av, pot.W, pot.rw, pot.aw,
                       pot.Wd, pot.rvd, pot.avd, pot.Vso, pot.Wso])
    diff = np.abs(p_ours - p_ref)
    return diff, p_ours, p_ref


# =============================================================================
# Main analysis
# =============================================================================

def main():
    print("=" * 70)
    print("Global KD02 Fisher Information in Universal Coefficient Space")
    print("=" * 70)

    # --- Load scan data ---
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'data', 'deff_scan_extended.json')

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run deff_scan_extended.py first.")
        sys.exit(1)

    with open(data_path) as f:
        scan_data = json.load(f)

    entries = [d for d in scan_data['data'] if 'fisher_matrix' in d]
    print(f"Loaded {len(entries)} configurations with Fisher matrices.")

    # --- Define universal parameters ---
    names, univ, groups = get_kd02_universal_params()
    N_univ = len(univ)
    print(f"Universal KD02 parameters: {N_univ}")

    # --- Validate against KD02Potential ---
    print("\nValidation (universal_to_local vs KD02Potential):")
    test_cases = [
        ('n', 40, 20, 50),
        ('p', 208, 82, 30),
        ('n', 12, 6, 100),
        ('p', 40, 20, 10),
    ]
    max_err = 0.0
    for proj, A, Z, E in test_cases:
        diff, p_ours, p_ref = validate_jacobian(univ, A, Z, E, proj)
        rel_diff = diff / (np.abs(p_ref) + 1e-30)
        err = np.max(rel_diff)
        max_err = max(max_err, err)
        status = "OK" if err < 1e-6 else "MISMATCH"
        print(f"  {proj}+A={A},Z={Z},E={E}: max rel err = {err:.2e} [{status}]")
    print(f"  Overall max relative error: {max_err:.2e}")

    if max_err > 1e-4:
        print("WARNING: Large discrepancy detected! Check universal_to_local.")

    # --- Compute global Fisher matrix ---
    print("\nComputing global Fisher matrix...")
    F_global = np.zeros((N_univ, N_univ))

    # Also track contributions from different subsets
    F_neutron = np.zeros_like(F_global)
    F_proton = np.zeros_like(F_global)

    # For tracking D_eff growth
    deff_growth = []  # (n_systems, D_eff, label)

    # Sort entries for reproducible ordering: by projectile, then A, then E
    entries_sorted = sorted(entries, key=lambda e: (e['projectile'], e['A'], e['E']))

    F_accum = np.zeros_like(F_global)
    n_systems = 0

    for entry in entries_sorted:
        A = entry['A']
        Z = entry['Z']
        E = entry['E']
        proj = entry['projectile']
        F_local = np.array(entry['fisher_matrix'])

        # Compute log-Jacobian
        J = compute_log_jacobian(univ, A, Z, E, proj)

        # Transform to universal space
        F_contrib = J.T @ F_local @ J
        F_global += F_contrib
        F_accum += F_contrib
        n_systems += 1

        if proj == 'n':
            F_neutron += F_contrib
        else:
            F_proton += F_contrib

        # Track D_eff growth at key milestones
        if n_systems in [1, 2, 5, 10, 20, 50, 84, 100, 168] or n_systems == len(entries_sorted):
            D, ev = compute_deff(F_accum)
            label = f"{proj}+A={A},E={E}"
            deff_growth.append({
                'n_systems': n_systems,
                'D_eff': float(D),
                'n_nonzero_ev': int(np.sum(ev > 1e-20)),
                'label': label,
            })
            print(f"  N={n_systems:3d}: D_eff = {D:.2f} "
                  f"(n_ev>0: {np.sum(ev>1e-20)}, max={ev[0]:.2e})")

    # --- Final results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    D_full, ev_full = compute_deff(F_global)
    print(f"\nFull global Fisher ({N_univ} universal params, {len(entries)} systems):")
    print(f"  D_eff = {D_full:.2f} / {N_univ}")
    print(f"  Non-zero eigenvalues: {np.sum(ev_full > 1e-20)}")
    print(f"  Condition number: {ev_full[0]/ev_full[ev_full>1e-20][-1]:.2e}")

    # Eigenvalue spectrum
    print(f"\n  Top eigenvalues:")
    for i, ev in enumerate(ev_full[:15]):
        frac = ev / np.sum(ev_full) * 100
        print(f"    λ_{i+1:2d} = {ev:12.4e}  ({frac:5.1f}% cumulative)")

    # Subgroup analysis
    print(f"\n  Subgroup D_eff:")
    for gname, gidx in groups.items():
        F_sub = F_global[np.ix_(gidx, gidx)]
        D_sub, ev_sub = compute_deff(F_sub)
        print(f"    {gname:25s}: D_eff = {D_sub:.2f} / {len(gidx)}")

    # Neutron vs Proton
    print(f"\n  Neutron data only:")
    D_n, ev_n = compute_deff(F_neutron)
    print(f"    D_eff = {D_n:.2f}")
    print(f"\n  Proton data only:")
    D_p, ev_p = compute_deff(F_proton)
    print(f"    D_eff = {D_p:.2f}")

    # --- Per-nucleus analysis ---
    # Group by (proj, nucleus) and show D_eff for each
    print(f"\n  Per-nucleus D_eff (all 7 energies combined):")
    print(f"  {'System':15s} {'D_eff':>6s} {'n_E':>4s}")
    print(f"  {'-'*30}")

    # Group entries
    sys_dict = {}
    for entry in entries_sorted:
        key = (entry['projectile'], entry['nucleus'])
        if key not in sys_dict:
            sys_dict[key] = []
        sys_dict[key].append(entry)

    per_nucleus_results = []
    for (proj, nuc), sys_entries in sorted(sys_dict.items()):
        F_sys = np.zeros((N_univ, N_univ))
        for entry in sys_entries:
            J = compute_log_jacobian(univ, entry['A'], entry['Z'],
                                      entry['E'], proj)
            F_local = np.array(entry['fisher_matrix'])
            F_sys += J.T @ F_local @ J

        D_sys, ev_sys = compute_deff(F_sys)
        per_nucleus_results.append({
            'projectile': proj, 'nucleus': nuc,
            'A': sys_entries[0]['A'], 'Z': sys_entries[0]['Z'],
            'D_eff': float(D_sys),
            'n_energies': len(sys_entries),
            'eigenvalues': ev_sys[:10].tolist(),
        })
        print(f"  {proj}+{nuc:6s}        {D_sys:6.2f}   {len(sys_entries):3d}")

    # --- D_eff growth by adding nuclei ---
    print(f"\n  D_eff growth by adding nuclei (all energies each):")
    nuclei_order = sorted(sys_dict.keys())
    F_cumul = np.zeros((N_univ, N_univ))
    nuclei_growth = []
    for i_sys, (proj, nuc) in enumerate(nuclei_order):
        for entry in sys_dict[(proj, nuc)]:
            J = compute_log_jacobian(univ, entry['A'], entry['Z'],
                                      entry['E'], proj)
            F_local = np.array(entry['fisher_matrix'])
            F_cumul += J.T @ F_local @ J

        D_cum, ev_cum = compute_deff(F_cumul)
        nuclei_growth.append({
            'n_nuclei': i_sys + 1,
            'last_added': f"{proj}+{nuc}",
            'D_eff': float(D_cum),
            'eigenvalues': ev_cum[:10].tolist(),
        })
        print(f"  +{proj}+{nuc:6s} (N={i_sys+1:2d}): D_eff = {D_cum:.2f}")

    # --- Comparison: D_eff/N_params for local vs global ---
    print(f"\n  Summary comparison:")
    # Average local D_eff (single system, 11 params)
    local_deffs = [e['deff_results']['all_11p']['D_eff'] for e in entries]
    print(f"    Local (11 params): mean D_eff = {np.mean(local_deffs):.2f} / 11"
          f" = {np.mean(local_deffs)/11*100:.0f}%")
    print(f"    Global ({N_univ} params): D_eff = {D_full:.2f} / {N_univ}"
          f" = {D_full/N_univ*100:.0f}%")

    # --- Save results ---
    results = {
        'N_universal_params': N_univ,
        'universal_param_names': names,
        'universal_param_values': univ.tolist(),
        'groups': {k: v for k, v in groups.items()},
        'N_systems': len(entries),

        'D_eff_global': float(D_full),
        'eigenvalues_global': ev_full.tolist(),
        'fisher_matrix_global': F_global.tolist(),

        'D_eff_neutron': float(D_n),
        'D_eff_proton': float(D_p),

        'deff_growth_by_system': deff_growth,
        'per_nucleus_results': per_nucleus_results,
        'nuclei_growth': nuclei_growth,

        'local_deff_mean': float(np.mean(local_deffs)),
        'local_deff_std': float(np.std(local_deffs)),
    }

    outdir = os.path.join(base_dir, '..', 'data')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'deff_global_kd02.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
