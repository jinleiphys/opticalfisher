#!/usr/bin/env python3
"""
Compute D_eff for full KD02 potential with 9 parameters.

This directly addresses the question: does D_eff ≈ 1.7 hold for the full
9-parameter KD02 model, or is it specific to 4-parameter Woods-Saxon?

Key insight: D_eff is about DATA information content, not model complexity.
Result: D_eff = 1.65 ± 0.43 for 9 params, confirming the information limit.

Parameters for KD02 (spin-0, no spin-orbit):
  1. V   - real volume depth (MeV)
  2. rv  - real volume radius (fm)
  3. av  - real volume diffuseness (fm)
  4. W   - imaginary volume depth (MeV)
  5. rw  - imaginary volume radius (fm)
  6. aw  - imaginary volume diffuseness (fm)
  7. Wd  - imaginary surface depth (MeV)
  8. rd  - surface radius (fm)
  9. ad  - surface diffuseness (fm)

Author: Jin Lei
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/jinlei/Desktop/code/PINN_CFC')

from potentials import woods_saxon_form_factor, coulomb_potential
from scattering_fortran import ScatteringSolverFortran, HBARC, AMU, E2

# Morandi colors
COLORS = {
    'main': '#7B8B9A',
    'accent': '#C17767',
    'secondary': '#8B9A7B',
}


def kd02_potential_9params(r, A, Z_proj, Z, params):
    """
    KD02-style potential with 9 free parameters (spin-0).

    params = [V, rv, av, W, rw, aw, Wd, rd, ad]
    """
    V, rv, av, W, rw, aw, Wd, rd, ad = params

    A_third = A ** (1./3.)

    # Real volume term
    R_v = rv * A_third
    f_v = 1.0 / (1.0 + np.exp((r - R_v) / av))

    # Imaginary volume term
    R_w = rw * A_third
    f_w = 1.0 / (1.0 + np.exp((r - R_w) / aw))

    # Imaginary surface term (derivative of Woods-Saxon)
    R_d = rd * A_third
    exp_d = np.exp((r - R_d) / ad)
    f_d = exp_d / (1.0 + exp_d)**2 / ad  # df/dr * (-ad)

    # Total potential
    V_opt = -V * f_v - 1j * W * f_w - 1j * 4 * Wd * f_d

    # Add Coulomb for proton
    if Z_proj > 0:
        Rc = 1.25 * A_third
        V_coul = coulomb_potential(r, Z_proj, Z, Rc)
        V_opt = V_opt + V_coul

    return V_opt


def compute_cross_section_9params(projectile, A, Z, E_lab, theta_deg, params, l_max=30):
    """
    Compute elastic cross section for 9-parameter potential.
    """
    # Kinematics
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

    # Potential function
    pot_func = lambda r: kd02_potential_9params(r, A, Z_proj, Z, params)

    # Solve scattering
    solver = ScatteringSolverFortran(r_max=40.0, hcm=0.02)
    l_values = list(range(l_max + 1))
    results = solver.solve(pot_func, E_cm, mu, Z1=Z_proj, Z2=Z, l_values=l_values)
    S_matrix = results['S_matrix']

    # Compute cross section
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)

    f_theta = np.zeros(len(theta_deg), dtype=complex)

    for l in l_values:
        S_l = S_matrix[l]
        P_l = np.polynomial.legendre.legval(cos_theta, [0]*l + [1])
        f_theta += (2*l + 1) * (S_l - 1) / (2j * k) * P_l

    # Add Coulomb for proton
    if projectile == 'p' and eta > 0:
        from scattering_fortran import coulomb_phase_shift
        sigma_l = coulomb_phase_shift(eta, l_max)

        for l in l_values:
            P_l = np.polynomial.legendre.legval(cos_theta, [0]*l + [1])
            phase = np.exp(2j * sigma_l[l])
            f_theta += (2*l + 1) * (phase - 1) / (2j * k) * P_l

        # Point Coulomb
        f_coul = -eta / (2 * k * np.sin(theta_rad/2)**2) * \
                 np.exp(-1j * eta * np.log(np.sin(theta_rad/2)**2) + 2j * sigma_l[0])
        f_theta += f_coul

    dsigma = np.abs(f_theta)**2 * 10  # fm^2 to mb
    return dsigma


def compute_fisher_matrix_9params(projectile, A, Z, E_lab, theta_deg, params,
                                   param_names, eps_rel=0.01):
    """
    Compute Fisher Information Matrix for 9-parameter KD02 using finite differences.

    F_ij = sum_theta (1/sigma_exp^2) * (dsigma/dp_i) * (dsigma/dp_j)
    """
    n_params = len(params)
    n_angles = len(theta_deg)

    # Baseline cross section
    sigma_0 = compute_cross_section_9params(projectile, A, Z, E_lab, theta_deg, params)

    # Compute gradients via finite difference
    gradients = np.zeros((n_params, n_angles))

    for i in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()

        delta = eps_rel * params[i]
        if delta < 1e-6:
            delta = 1e-6

        params_plus[i] += delta
        params_minus[i] -= delta

        sigma_plus = compute_cross_section_9params(projectile, A, Z, E_lab, theta_deg, params_plus)
        sigma_minus = compute_cross_section_9params(projectile, A, Z, E_lab, theta_deg, params_minus)

        gradients[i] = (sigma_plus - sigma_minus) / (2 * delta)

        print(f"  Gradient {param_names[i]}: max |dσ/dp| = {np.max(np.abs(gradients[i])):.2e}")

    # Fisher matrix (assuming relative error epsilon, but D_eff is scale-invariant)
    epsilon = 0.05  # 5% relative error (doesn't affect D_eff)
    sigma_exp = epsilon * sigma_0
    sigma_exp[sigma_exp < 1e-10] = 1e-10  # Avoid division by zero

    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            F[i, j] = np.sum(gradients[i] * gradients[j] / sigma_exp**2)

    return F, gradients, sigma_0


def compute_deff(F):
    """Compute effective dimensionality from Fisher matrix."""
    eigenvalues = np.linalg.eigvalsh(F)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Keep positive

    if len(eigenvalues) == 0:
        return 0.0, np.array([]), np.inf

    # Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda2 = np.sum(eigenvalues**2)
    D_eff = sum_lambda**2 / sum_lambda2 if sum_lambda2 > 0 else 0

    # Condition number
    cond = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else np.inf

    return D_eff, eigenvalues, cond


def get_kd02_params(projectile, A, Z, E_lab):
    """
    Get KD02 parameters for a given system.
    Returns: [V, rv, av, W, rw, aw, Wd, rvd, avd]

    Note: KD02 uses rvd/avd for surface term geometry.
    """
    from potentials import KD02Potential
    pot = KD02Potential(projectile, A, Z, E_lab)

    # Extract parameters
    params = np.array([
        pot.V,    # Real volume depth
        pot.rv,   # Real volume radius
        pot.av,   # Real volume diffuseness
        pot.W,    # Imaginary volume depth
        pot.rw,   # Imaginary volume radius (= rv in KD02)
        pot.aw,   # Imaginary volume diffuseness (= av in KD02)
        pot.Wd,   # Imaginary surface depth
        pot.rvd,  # Surface radius
        pot.avd,  # Surface diffuseness
    ])

    return params


def main():
    """Main analysis."""

    print("="*70)
    print("D_eff Analysis for Full KD02 (9 parameters)")
    print("="*70)

    param_names = ['V', 'rv', 'av', 'W', 'rw', 'aw', 'Wd', 'rvd', 'avd']

    # Test cases
    cases = [
        ('n', 40, 20, '40Ca', 50),
        ('n', 208, 82, '208Pb', 50),
        ('p', 40, 20, '40Ca', 50),
        ('n', 40, 20, '40Ca', 100),
    ]

    theta_deg = np.linspace(10, 170, 17)

    results = []

    for proj, A, Z, name, E in cases:
        print(f"\n{'='*60}")
        print(f"{proj} + {name} @ {E} MeV")
        print('='*60)

        # Get KD02 parameters
        params = get_kd02_params(proj, A, Z, E)

        print(f"\nKD02 parameters (9):")
        for i, (pname, pval) in enumerate(zip(param_names, params)):
            print(f"  {pname:3s} = {pval:.4f}")

        # Compute Fisher matrix
        print(f"\nComputing Fisher matrix (finite differences)...")
        F, gradients, sigma_0 = compute_fisher_matrix_9params(
            proj, A, Z, E, theta_deg, params, param_names
        )

        # Compute D_eff
        D_eff, eigenvalues, cond = compute_deff(F)

        print(f"\nResults:")
        print(f"  D_eff = {D_eff:.2f}")
        print(f"  Condition number = {cond:.2e}")
        print(f"  Eigenvalue spectrum:")
        for i, ev in enumerate(sorted(eigenvalues, reverse=True)):
            frac = ev / np.sum(eigenvalues) * 100
            print(f"    λ_{i+1} = {ev:.2e} ({frac:.1f}%)")

        # Correlation matrix
        std = np.sqrt(np.diag(F))
        std[std < 1e-20] = 1e-20
        corr = F / np.outer(std, std)

        results.append({
            'case': f"{proj}+{name}@{E}MeV",
            'D_eff': D_eff,
            'cond': cond,
            'eigenvalues': eigenvalues,
            'corr': corr,
            'params': params,
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: D_eff for 9-parameter KD02")
    print("="*70)

    deff_values = [r['D_eff'] for r in results]
    print(f"\n{'Case':<25} {'D_eff':>8} {'Condition':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['case']:<25} {r['D_eff']:>8.2f} {r['cond']:>12.2e}")

    print(f"\n  Mean D_eff = {np.mean(deff_values):.2f} ± {np.std(deff_values):.2f}")

    # Compare with 4-param result (historical reference)
    print(f"\n  Comparison:")
    print(f"    4-param WS (old):  D_eff ~ 1.3 (simplified model)")
    print(f"    9-param KD02:      D_eff = {np.mean(deff_values):.2f} ± {np.std(deff_values):.2f}")

    if np.mean(deff_values) < 2.0:
        print(f"\n  CONCLUSION: D_eff ≈ 1-2 holds for BOTH 4-param and 9-param models!")
        print(f"  This confirms the information limit is about DATA, not MODEL.")
        print(f"  Adding parameters does NOT increase the information content.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): D_eff comparison
    ax = axes[0]
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, [1.3]*len(results), width, label='4-param WS (ref)', color=COLORS['main'], alpha=0.7)
    ax.bar(x + width/2, deff_values, width, label='9-param KD02', color=COLORS['accent'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([r['case'].replace('@', '\n@') for r in results], fontsize=9)
    ax.set_ylabel(r'$D_\mathrm{eff}$', fontsize=12)
    ax.set_title('Effective Dimensionality: 4-param vs 9-param', fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 3)

    # Panel (b): Eigenvalue spectrum for first case
    ax = axes[1]
    ev = sorted(results[0]['eigenvalues'], reverse=True)
    ev_frac = ev / np.sum(ev) * 100
    ax.bar(range(1, len(ev)+1), ev_frac, color=COLORS['secondary'], alpha=0.7)
    ax.set_xlabel('Eigenvalue index', fontsize=11)
    ax.set_ylabel('Fraction of total (%)', fontsize=11)
    ax.set_title(f"Eigenvalue spectrum ({results[0]['case']})", fontsize=11)
    ax.set_xticks(range(1, len(ev)+1))

    # Annotate
    for i, frac in enumerate(ev_frac[:3]):
        ax.annotate(f'{frac:.1f}%', (i+1, frac), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    outfile = '/Users/jinlei/Desktop/code/PINN_CFC/experiments/deff_kd02_9params.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {outfile}")

    plt.show()


if __name__ == '__main__':
    main()
