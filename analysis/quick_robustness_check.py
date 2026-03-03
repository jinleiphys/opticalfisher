#!/usr/bin/env python3
"""
Quick robustness checks for D_eff:
1. FIM locality: perturb KD02 reference parameters by ±10%, check D_eff stability
2. rw=rv constraint: enforce rw=rv, aw=av (11 independent params) vs free (13 params)
3. Ay weighting: scan epsilon from 0.01 to 1.0, check Ay gain vs weighting

Author: Jin Lei
Date: March 2026
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from deff_scan_extended import (
    get_kd02_params_13, compute_fisher_extended, compute_deff,
    compute_observables_vector, build_gradient_vector, kd02_potential_13params
)


def test_fim_locality(systems, theta_deg, perturbation_fracs=[0.05, 0.10, 0.15]):
    """
    Test 1: Perturb ALL parameters uniformly by ±frac, check D_eff stability.
    """
    print("=" * 70)
    print("TEST 1: FIM Locality — D_eff at perturbed parameter points")
    print("=" * 70)

    for proj, A, Z, name, E in systems:
        params_ref = get_kd02_params_13(proj, A, Z, E)
        F_ref, grad_ref, _, n_data = compute_fisher_extended(
            proj, A, Z, E, theta_deg, params_ref
        )

        # Elastic-only D_eff at reference
        n_dcs = n_data['dcs']
        g_dcs = grad_ref[:, :n_dcs]
        F_dcs = g_dcs @ g_dcs.T
        D_ref, _, _ = compute_deff(F_dcs)

        # Elastic+Ay D_eff at reference
        n_Ay = n_data['Ay']
        g_dcs_Ay = grad_ref[:, :n_dcs + n_Ay]
        F_dcs_Ay = g_dcs_Ay @ g_dcs_Ay.T
        D_ref_Ay, _, _ = compute_deff(F_dcs_Ay)

        print(f"\n{proj}+{name} @ {E} MeV:")
        print(f"  Reference: D_eff(elastic) = {D_ref:.4f}, D_eff(elastic+Ay) = {D_ref_Ay:.4f}")

        for frac in perturbation_fracs:
            D_elastic_vals = []
            D_Ay_vals = []
            # Random perturbations: 10 trials
            np.random.seed(42)
            for trial in range(10):
                # Perturb each param by uniform random in [-frac, +frac]
                factors = 1.0 + frac * (2.0 * np.random.rand(len(params_ref)) - 1.0)
                params_pert = [p * f for p, f in zip(params_ref, factors)]

                F_p, grad_p, _, nd_p = compute_fisher_extended(
                    proj, A, Z, E, theta_deg, params_pert
                )
                g_dcs_p = grad_p[:, :nd_p['dcs']]
                F_dcs_p = g_dcs_p @ g_dcs_p.T
                D_p, _, _ = compute_deff(F_dcs_p)
                D_elastic_vals.append(D_p)

                g_dcsAy_p = grad_p[:, :nd_p['dcs'] + nd_p['Ay']]
                F_dcsAy_p = g_dcsAy_p @ g_dcsAy_p.T
                D_pAy, _, _ = compute_deff(F_dcsAy_p)
                D_Ay_vals.append(D_pAy)

            D_el = np.array(D_elastic_vals)
            D_ay = np.array(D_Ay_vals)
            print(f"  ±{frac*100:.0f}% perturbation (10 trials):")
            print(f"    elastic:    {D_el.mean():.4f} ± {D_el.std():.4f}  "
                  f"(range {D_el.min():.4f}–{D_el.max():.4f}, "
                  f"max |Δ| = {np.max(np.abs(D_el - D_ref)):.4f})")
            print(f"    elastic+Ay: {D_ay.mean():.4f} ± {D_ay.std():.4f}  "
                  f"(range {D_ay.min():.4f}–{D_ay.max():.4f}, "
                  f"max |Δ| = {np.max(np.abs(D_ay - D_ref_Ay)):.4f})")


def test_rw_rv_constraint(systems, theta_deg):
    """
    Test 2: Compare D_eff with rw,aw free (13 params) vs enforced rw=rv, aw=av (11 params).
    """
    print("\n" + "=" * 70)
    print("TEST 2: rw=rv, aw=av constraint effect on D_eff")
    print("=" * 70)

    for proj, A, Z, name, E in systems:
        params_ref = get_kd02_params_13(proj, A, Z, E)
        # params = [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso, rvso, avso]

        # 13-param: full
        F_13, grad_13, _, n_data = compute_fisher_extended(
            proj, A, Z, E, theta_deg, params_ref
        )
        n_dcs = n_data['dcs']
        n_Ay = n_data['Ay']

        g_dcs_13 = grad_13[:, :n_dcs]
        F_dcs_13 = g_dcs_13 @ g_dcs_13.T
        D_13, _, _ = compute_deff(F_dcs_13)

        g_all_13 = grad_13[:, :n_dcs + n_Ay]
        F_all_13 = g_all_13 @ g_all_13.T
        D_13_Ay, _, _ = compute_deff(F_all_13)

        # 11-param: drop rw (index 4) and aw (index 5) — chain rule
        # If rw = rv, then d/d(rv) includes the contribution from rw channel
        # Jacobian: J[i,j] = d(param_13[i]) / d(param_11[j])
        # param_11 = [V, rv, av, W, Wd, rvd, avd, Vso, Wso, rvso, avso]
        # param_13 = [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso, rvso, avso]
        # With rw=rv: row 4 (rw) maps to column 1 (rv)
        # With aw=av: row 5 (aw) maps to column 2 (av)

        # Mapping: 11-param indices → 13-param indices
        # 0:V→0, 1:rv→1, 2:av→2, 3:W→3, 4:Wd→6, 5:rvd→7, 6:avd→8,
        # 7:Vso→9, 8:Wso→10, 9:rvso→11, 10:avso→12
        # Plus: rw(4)→rv(1), aw(5)→av(2)

        J = np.zeros((13, 11))
        # Direct mappings
        map_11_to_13 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 7, 6: 8,
                        7: 9, 8: 10, 9: 11, 10: 12}
        for j11, i13 in map_11_to_13.items():
            J[i13, j11] = 1.0
        # Constraints: rw=rv → d(rw)/d(rv) = 1
        J[4, 1] = 1.0  # rw depends on rv
        # aw=av → d(aw)/d(av) = 1
        J[5, 2] = 1.0  # aw depends on av

        # Project: F_11 = J^T F_13 J
        F_dcs_11 = J.T @ F_dcs_13 @ J
        D_11, _, _ = compute_deff(F_dcs_11)

        F_all_11 = J.T @ F_all_13 @ J
        D_11_Ay, _, _ = compute_deff(F_all_11)

        print(f"\n{proj}+{name} @ {E} MeV:")
        print(f"  13-param (rw,aw free):     D_eff(elastic) = {D_13:.4f},  "
              f"D_eff(elastic+Ay) = {D_13_Ay:.4f}")
        print(f"  11-param (rw=rv, aw=av):   D_eff(elastic) = {D_11:.4f},  "
              f"D_eff(elastic+Ay) = {D_11_Ay:.4f}")
        print(f"  Difference:                ΔD_eff(elastic) = {D_11 - D_13:+.4f},  "
              f"ΔD_eff(elastic+Ay) = {D_11_Ay - D_13_Ay:+.4f}")


def test_epsilon_scan(systems, theta_deg):
    """
    Test 3: Scan epsilon (elastic relative error) and show D_eff(elastic+Ay) vs epsilon.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Ay gain vs elastic precision (epsilon scan)")
    print("=" * 70)
    print("  epsilon = elastic relative error; delta_Ay = 0.03 fixed")
    print("  D_eff(combined) = f(epsilon/delta_Ay)")

    delta_Ay = 0.03
    epsilons = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 1.0]

    for proj, A, Z, name, E in systems:
        params_ref = get_kd02_params_13(proj, A, Z, E)
        F_full, grad_full, _, n_data = compute_fisher_extended(
            proj, A, Z, E, theta_deg, params_ref, delta_Ay=delta_Ay
        )
        n_dcs = n_data['dcs']
        n_Ay = n_data['Ay']

        # Elastic-only gradients (sensitivity-normalized, i.e. eps=1)
        g_dcs = grad_full[:, :n_dcs]  # S_i(theta), no eps factor
        # Ay gradients (already scaled by 1/delta_Ay)
        g_Ay = grad_full[:, n_dcs:n_dcs + n_Ay]

        # Elastic-only D_eff (scale-invariant)
        F_dcs = g_dcs @ g_dcs.T
        D_elastic, _, _ = compute_deff(F_dcs)

        print(f"\n{proj}+{name} @ {E} MeV:  D_eff(elastic only) = {D_elastic:.4f}")
        print(f"  {'epsilon':>8s}  {'eps/dAy':>8s}  {'D_eff(el+Ay)':>12s}  {'Ay gain':>10s}")
        print(f"  {'—'*8}  {'—'*8}  {'—'*12}  {'—'*10}")

        for eps in epsilons:
            # F_combined = (1/eps^2) * F_elastic + F_Ay
            # Since g_dcs is S_i (eps=1), F_elastic_code = g_dcs @ g_dcs.T
            # F_elastic(eps) = (1/eps^2) * F_elastic_code
            # F_Ay stays the same
            F_combined = (1.0 / eps**2) * (g_dcs @ g_dcs.T) + g_Ay @ g_Ay.T
            D_comb, _, _ = compute_deff(F_combined)
            gain = (D_comb - D_elastic) / D_elastic * 100
            print(f"  {eps:8.3f}  {eps/delta_Ay:8.2f}  {D_comb:12.4f}  {gain:+9.2f}%")


if __name__ == '__main__':
    theta_deg = np.linspace(5, 175, 35)

    # Representative systems
    systems = [
        ('n', 40, 20, '40Ca', 50),
        ('n', 120, 50, '120Sn', 50),
        ('p', 40, 20, '40Ca', 50),
        ('n', 208, 82, '208Pb', 50),
    ]

    test_fim_locality(systems, theta_deg)
    test_rw_rv_constraint(systems, theta_deg)
    test_epsilon_scan(systems, theta_deg)

    print("\n" + "=" * 70)
    print("All robustness checks complete.")
    print("=" * 70)
