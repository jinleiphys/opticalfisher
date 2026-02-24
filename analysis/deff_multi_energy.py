#!/usr/bin/env python3
"""
Multi-energy combined Fisher information analysis.

Post-processing script: reads deff_scan_extended.json and combines
Fisher matrices across energies for each (nucleus, projectile):
    F_combined = sum_E F(E)

Computes D_eff(single-E) vs D_eff(multi-E combined) and shows how
D_eff grows with number of energies.

Author: Jin Lei
Date: February 2026
"""

import numpy as np
import json
import os
import sys


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


def combine_fisher_matrices(data_entries):
    """Sum Fisher matrices from a list of scan entries."""
    F_combined = None
    for entry in data_entries:
        if 'fisher_matrix' not in entry:
            continue
        F = np.array(entry['fisher_matrix'])
        if F_combined is None:
            F_combined = np.zeros_like(F)
        F_combined += F
    return F_combined


def main():
    print("=" * 70)
    print("Multi-Energy Combined Fisher Information Analysis")
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

    # Build index: (projectile, nucleus) -> {E: entry}
    index = {}
    for entry in entries:
        key = (entry['projectile'], entry['nucleus'])
        if key not in index:
            index[key] = {}
        index[key][entry['E']] = entry

    results = {
        'param_names': param_names,
        'energies': energies,
        'combinations': [],
    }

    print(f"\n{'System':20s} {'single-E avg':>12s} {'7-E combined':>12s} "
          f"{'ratio':>8s}  D_eff(N=1..7)")
    print("-" * 90)

    for (proj, nuc), e_dict in sorted(index.items()):
        avail_E = sorted([E for E in e_dict.keys() if E in energies])
        if len(avail_E) < 2:
            continue

        # Single-energy D_eff values (from full 11x11 Fisher)
        single_deffs = []
        for E in avail_E:
            F_single = np.array(e_dict[E]['fisher_matrix'])
            D_single, _, _ = compute_deff(F_single)
            single_deffs.append(D_single)

        # Multi-energy combined D_eff: accumulate F matrices
        deff_vs_N = []
        F_accum = np.zeros((len(param_names), len(param_names)))
        for i, E in enumerate(avail_E):
            F_accum += np.array(e_dict[E]['fisher_matrix'])
            D_N, ev_N, cond_N = compute_deff(F_accum)
            deff_vs_N.append({
                'N_energies': i + 1,
                'energies_included': avail_E[:i+1],
                'D_eff': float(D_N),
                'condition_number': float(cond_N),
                'eigenvalues': ev_N.tolist(),
            })

        # Full combination
        D_full = deff_vs_N[-1]['D_eff']
        D_single_avg = np.mean(single_deffs)

        combo = {
            'projectile': proj,
            'nucleus': nuc,
            'A': e_dict[avail_E[0]]['A'],
            'Z': e_dict[avail_E[0]]['Z'],
            'single_energy_deffs': {str(E): float(d)
                                     for E, d in zip(avail_E, single_deffs)},
            'single_energy_mean': float(D_single_avg),
            'multi_energy_deff': deff_vs_N,
            'full_combined_fisher': F_accum.tolist(),
        }
        results['combinations'].append(combo)

        deff_str = " ".join([f"{d['D_eff']:.2f}" for d in deff_vs_N])
        print(f"{proj}+{nuc:6s}          {D_single_avg:8.2f}     {D_full:8.2f}"
              f"    {D_full/D_single_avg:5.2f}x  [{deff_str}]")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_single = []
    all_combined = []
    for combo in results['combinations']:
        all_single.append(combo['single_energy_mean'])
        all_combined.append(combo['multi_energy_deff'][-1]['D_eff'])

    if all_single:
        print(f"  Single-energy D_eff (avg):  {np.mean(all_single):.2f} "
              f"+/- {np.std(all_single):.2f}")
        print(f"  7-energy combined D_eff:    {np.mean(all_combined):.2f} "
              f"+/- {np.std(all_combined):.2f}")
        print(f"  Improvement factor:         {np.mean(all_combined)/np.mean(all_single):.2f}x")

    # Also compute observable-specific multi-energy combinations
    # For this we need individual observable subsets, which requires re-computation
    # Instead, just note the all_11p D_eff from the full Fisher
    print("\n  Note: The full Fisher matrix includes all observables (dcs+Ay+sigma_R+sigma_T).")
    print("  For elastic-only multi-energy, re-run with subset gradients.")

    # Save
    outdir = os.path.join(base_dir, '..', 'data')
    outfile = os.path.join(outdir, 'deff_multi_energy.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
