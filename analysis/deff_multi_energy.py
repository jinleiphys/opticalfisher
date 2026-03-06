#!/usr/bin/env python3
"""
Multi-energy combined Fisher information analysis.

Post-processing script: reads deff_scan_extended.json and combines
Fisher matrices across energies for each (nucleus, projectile) using
the proper Jacobian projection through the KD02 universal coefficient space:

    F_univ = sum_E J(E)^T F_local(E) J(E)     (48x48, universal space)
    F_ref  = J(E_ref) F_univ J(E_ref)^T        (13x13, local space at E_ref)

This correctly accounts for the KD02 energy dependence that ties local
parameters across energies. Also computes the naive direct sum for comparison.

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Jacobian machinery from the global analysis script
from deff_global_kd02 import (
    get_kd02_universal_params,
    compute_log_jacobian,
)


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


def main():
    print("=" * 70)
    print("Multi-Energy Combined Fisher Information Analysis")
    print("(Jacobian projection through KD02 universal coefficient space)")
    print("=" * 70)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'data', 'deff_scan_extended.json')

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run deff_scan_extended.py first.")
        sys.exit(1)

    with open(data_path, 'r') as f:
        scan_data = json.load(f)

    entries = [d for d in scan_data['data'] if 'fisher_matrix' in d]
    print(f"Loaded {len(entries)} valid entries with Fisher matrices.")

    energies = sorted(scan_data['energies'])
    param_names = scan_data['param_names']
    N_local = len(param_names)

    # Load universal KD02 parameters
    univ_names, univ_values, groups = get_kd02_universal_params()
    N_univ = len(univ_values)
    print(f"Universal KD02 parameters: {N_univ}")

    # Build index: (projectile, nucleus) -> {E: entry}
    index = {}
    for entry in entries:
        key = (entry['projectile'], entry['nucleus'])
        if key not in index:
            index[key] = {}
        index[key][entry['E']] = entry

    # Reference energy for back-projection
    E_ref = 50.0

    results = {
        'param_names': param_names,
        'energies': energies,
        'E_ref': E_ref,
        'N_universal': N_univ,
        'method': 'Jacobian projection: F_univ = sum_E J(E)^T F(E) J(E), '
                  'F_ref = J(E_ref) F_univ J(E_ref)^T',
        'combinations': [],
    }

    print(f"\nReference energy for back-projection: {E_ref} MeV")
    print(f"\n{'System':20s} {'single-E avg':>12s} {'Jacobian':>10s} "
          f"{'direct sum':>10s}  D_eff(N=1..7) [Jacobian]")
    print("-" * 100)

    for (proj, nuc), e_dict in sorted(index.items()):
        avail_E = sorted([E for E in e_dict.keys() if E in energies])
        if len(avail_E) < 2:
            continue

        A = e_dict[avail_E[0]]['A']
        Z = e_dict[avail_E[0]]['Z']

        # Single-energy D_eff values
        single_deffs = []
        for E in avail_E:
            F_single = np.array(e_dict[E]['fisher_matrix'])
            D_single, _, _ = compute_deff(F_single)
            single_deffs.append(D_single)

        # --- Jacobian-based multi-energy combination ---
        # Compute J(E_ref) for back-projection
        J_ref = compute_log_jacobian(univ_values, A, Z, E_ref, proj)

        # Accumulate F_univ = sum_E J(E)^T F(E) J(E)
        deff_vs_N = []
        F_univ_accum = np.zeros((N_univ, N_univ))

        for i, E in enumerate(avail_E):
            F_local_E = np.array(e_dict[E]['fisher_matrix'])
            J_E = compute_log_jacobian(univ_values, A, Z, E, proj)

            # Project to universal space
            F_univ_accum += J_E.T @ F_local_E @ J_E

            # Project back to local space at E_ref
            F_ref = J_ref @ F_univ_accum @ J_ref.T
            D_N, ev_N, cond_N = compute_deff(F_ref)

            deff_vs_N.append({
                'N_energies': i + 1,
                'energies_included': avail_E[:i+1],
                'D_eff': float(D_N),
                'condition_number': float(cond_N),
                'eigenvalues': ev_N.tolist(),
            })

        # --- Naive direct sum for comparison ---
        F_direct = np.zeros((N_local, N_local))
        deff_direct_vs_N = []
        for i, E in enumerate(avail_E):
            F_direct += np.array(e_dict[E]['fisher_matrix'])
            D_dir, _, _ = compute_deff(F_direct)
            deff_direct_vs_N.append({
                'N_energies': i + 1,
                'D_eff_direct': float(D_dir),
            })

        # Full combination results
        D_jacobian = deff_vs_N[-1]['D_eff']
        D_direct = deff_direct_vs_N[-1]['D_eff_direct']
        D_single_avg = np.mean(single_deffs)

        # Back-projected Fisher matrix at E_ref (full 7-energy)
        F_ref_full = J_ref @ F_univ_accum @ J_ref.T

        combo = {
            'projectile': proj,
            'nucleus': nuc,
            'A': A,
            'Z': Z,
            'single_energy_deffs': {str(E): float(d)
                                     for E, d in zip(avail_E, single_deffs)},
            'single_energy_mean': float(D_single_avg),
            'multi_energy_jacobian': deff_vs_N,
            'multi_energy_direct': deff_direct_vs_N,
            'D_eff_jacobian': float(D_jacobian),
            'D_eff_direct': float(D_direct),
            'full_combined_fisher': F_ref_full.tolist(),
        }
        results['combinations'].append(combo)

        deff_str = " ".join([f"{d['D_eff']:.2f}" for d in deff_vs_N])
        print(f"{proj}+{nuc:6s}          {D_single_avg:8.2f}   {D_jacobian:8.2f}"
              f"   {D_direct:8.2f}   [{deff_str}]")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_single = []
    all_jacobian = []
    all_direct = []
    for combo in results['combinations']:
        all_single.append(combo['single_energy_mean'])
        all_jacobian.append(combo['D_eff_jacobian'])
        all_direct.append(combo['D_eff_direct'])

    if all_single:
        print(f"  Single-energy D_eff (avg):     {np.mean(all_single):.2f} "
              f"+/- {np.std(all_single):.2f}")
        print(f"  7-energy Jacobian D_eff (avg): {np.mean(all_jacobian):.2f} "
              f"+/- {np.std(all_jacobian):.2f}")
        print(f"  7-energy direct sum (avg):     {np.mean(all_direct):.2f} "
              f"+/- {np.std(all_direct):.2f}")
        print(f"\n  Jacobian improvement factor:   "
              f"{np.mean(all_jacobian)/np.mean(all_single):.2f}x")
        print(f"  Direct sum improvement factor: "
              f"{np.mean(all_direct)/np.mean(all_single):.2f}x")

    print(f"\n  Method: Jacobian projection through {N_univ}-dim KD02 "
          f"universal coefficient space,")
    print(f"  back-projected to 13-dim local space at E_ref = {E_ref} MeV.")
    print(f"  Direct sum shown for comparison (assumes energy-independent "
          f"local parameters).")

    # Save
    outdir = os.path.join(base_dir, '..', 'data')
    outfile = os.path.join(outdir, 'deff_multi_energy.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
