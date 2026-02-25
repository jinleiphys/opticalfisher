#!/usr/bin/env python3
"""
Quantum Scattering Solver - Fortran-compatible implementation

This module implements the scattering solver exactly as in the Fortran code scatt.f:
- Numerov method with Tx = -h²*kl/12 formulation
- S-matrix extraction using same formula as scatt.f
- Uses compiled coul90.f90 for Coulomb functions

Author: Jin Lei
Date: December 2024
"""

import numpy as np
import sys
import os

# Add coul90_wrapper directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'coul90_wrapper'))

try:
    from coul90_wrapper import coul90_mod
except ImportError:
    raise ImportError(
        "coul90_wrapper not found. Please compile it first:\n"
        "  cd coul90_wrapper && f2py -c -m coul90_wrapper coul90.f90\n"
        "Make sure to use the correct Python environment (e.g., pinn)."
    )


#==============================================================================
# Constants (nuclear physics units) - same as Fortran
#==============================================================================

HBARC = 197.3269804  # MeV·fm
AMU = 931.494        # MeV/c^2 (atomic mass unit)
E2 = 1.44            # MeV·fm (e^2, Coulomb constant)


#==============================================================================
# Coulomb Functions (using coul90)
#==============================================================================

def coul90_functions(rho, eta, lmax):
    """
    Calculate Coulomb functions using Fortran coul90.

    Parameters:
        rho: k*r (dimensionless)
        eta: Sommerfeld parameter
        lmax: maximum angular momentum

    Returns:
        F, G, Fp, Gp: arrays of Coulomb functions and derivatives (d/d(rho))
    """
    fc, gc, fcp, gcp, ifail = coul90_mod.coul90(rho, eta, 0.0, lmax, 0)
    if ifail != 0:
        print(f"Warning: coul90 ifail = {ifail}")
    return fc[:lmax+1], gc[:lmax+1], fcp[:lmax+1], gcp[:lmax+1]


def coulomb_phase_shift(eta, lmax):
    """
    Calculate Coulomb phase shifts sigma_l(eta) using coulph from coul90.

    Parameters:
        eta: Sommerfeld parameter
        lmax: maximum angular momentum

    Returns:
        sigma: array of Coulomb phase shifts
    """
    cph = coul90_mod.coulph(eta, lmax)
    return cph[:lmax+1]


#==============================================================================
# Numerov Method - exactly as in scatt.f subroutine sch
#==============================================================================

def numerov_sch(hcm, irmatch, l, mu, ecm, Vpot):
    """
    Solve radial Schrödinger equation using Numerov method.
    Exactly follows the Fortran subroutine sch in scatt.f.

    d²u/dr² + k_l(r) * u = 0

    where k_l = 2μE/ℏ² - l(l+1)/r² - 2μV(r)/ℏ²

    Parameters:
        hcm: step size (fm)
        irmatch: matching point index
        l: angular momentum
        mu: reduced mass (MeV/c²)
        ecm: center of mass energy (MeV)
        Vpot: potential array (complex, MeV) from index 0 to irmatch

    Returns:
        rwfl: wave function array (complex) from index 0 to irmatch
    """
    # Allocate arrays
    rwfl = np.zeros(irmatch + 1, dtype=complex)
    kl = np.zeros(irmatch + 1, dtype=complex)
    Tx = np.zeros(irmatch + 1, dtype=complex)
    Wx = np.zeros(irmatch + 1, dtype=complex)

    # Starting point: r0 = 2*l (to avoid centrifugal singularity)
    r0 = 2 * l

    # Boundary check: r0+2 must fit within the grid
    if r0 + 2 >= irmatch:
        # l is too large for this grid — wavefunction is negligible, return zeros
        return rwfl

    # Boundary condition: u(r0) = 0
    rwfl[r0] = 0.0

    # Initial value at r0+1
    ir = r0 + 1
    r = ir * hcm
    rwfl[ir] = hcm**(l + 1)  # arbitrary small value ~ r^(l+1)

    # Calculate kl and Tx at ir = r0+1
    kl[ir] = 2.0 * mu * ecm / HBARC**2 - l*(l+1) / r**2 - 2.0 * mu * Vpot[ir] / HBARC**2
    Tx[ir] = -hcm**2 / 12.0 * kl[ir]  # Note: negative sign as in Fortran
    Wx[ir] = (1.0 - Tx[ir]) * rwfl[ir]

    # First step at r0+2 using simple formula
    rwfl[r0 + 2] = 2.0 * rwfl[r0 + 1] - hcm**2 * kl[r0 + 1] * rwfl[r0 + 1]
    ir = r0 + 2
    r = ir * hcm
    kl[ir] = 2.0 * mu * ecm / HBARC**2 - l*(l+1) / r**2 - 2.0 * mu * Vpot[ir] / HBARC**2
    Tx[ir] = -hcm**2 / 12.0 * kl[ir]
    Wx[ir] = (1.0 - Tx[ir]) * rwfl[ir]

    # Numerov iteration from r0+2 to irmatch-1
    # Wx(ir+1) = (2 + 12*Tx + 12*Tx²) * Wx(ir) - Wx(ir-1)
    # rwfl(ir+1) = Wx(ir+1) / (1 - Tx(ir+1))
    for ir in range(r0 + 2, irmatch):
        r_next = (ir + 1) * hcm
        kl[ir + 1] = 2.0 * mu * ecm / HBARC**2 - l*(l+1) / r_next**2 - 2.0 * mu * Vpot[ir + 1] / HBARC**2
        Tx[ir + 1] = -hcm**2 / 12.0 * kl[ir + 1]

        # Numerov formula (as in Fortran)
        Wx[ir + 1] = (2.0 + 12.0 * Tx[ir] + 12.0 * Tx[ir]**2) * Wx[ir] - Wx[ir - 1]
        rwfl[ir + 1] = Wx[ir + 1] / (1.0 - Tx[ir + 1])

    return rwfl


#==============================================================================
# S-matrix extraction - exactly as in scatt.f subroutine matching
#==============================================================================

def matching(l, k, wf, hcm, nfc, ngc, nfcp, ngcp):
    """
    Extract S-matrix from wave function - exactly as in Fortran matching subroutine.

    WARNING: This method is ONLY for validating Numerov algorithm against FRESCO.
    It is NOT used for neural network training or cross section calculations.
    For production use, see smatrix_amplitude_integral() in smatrix_methods.py.

    nl*wf = 0.5*i*(H^(-) - sl*H^(+))

    where H^(+) = G + iF, H^(-) = G - iF

    Parameters:
        l: angular momentum
        k: wave number (fm^-1)
        wf: wave function array at matching region (5 points)
        hcm: step size (fm)
        nfc, ngc: F_l and G_l at matching point
        nfcp, ngcp: dF_l/d(rho) and dG_l/d(rho) at matching point

    Returns:
        sl: S-matrix element
        nl: normalization factor

    Note:
        This function matches the Fortran scatt.f implementation exactly.
        Only use this for Numerov vs FRESCO comparison tests.
    """
    # H^(+) = G + iF, H^(-) = G - iF
    hc = complex(ngc, nfc)     # H^(+)
    hc1 = complex(ngc, -nfc)   # H^(-)

    # Derivatives (w.r.t. rho)
    hcp = complex(ngcp, nfcp)    # dH^(+)/d(rho)
    hcp1 = complex(ngcp, -nfcp)  # dH^(-)/d(rho)

    # 5-point derivative formula: wfp = du/dr at wf[2] (center point)
    # wf[0] = wf(irmatch-4), wf[4] = wf(irmatch)
    # wf[2] = wf(irmatch-2) is the matching point
    wfp = (-wf[4] + 8.0*wf[3] - 8.0*wf[1] + wf[0]) / (12.0 * hcm)

    # S-matrix formula from Fortran:
    # sl = (hc1*wfp - hcp1*wf(3)*k) / (hc*wfp - hcp*wf(3)*k)
    # Note: wf(3) in Fortran 1-indexed = wf[2] in Python 0-indexed
    wf_match = wf[2]

    sl = (hc1 * wfp - hcp1 * wf_match * k) / (hc * wfp - hcp * wf_match * k)

    # Normalization factor from Fortran:
    # nl = (hc1*hcp*iu*k - hc*hcp1*iu*k) / (2*(hcp*wf(3)*k - hc*wfp))
    iu = 1j
    nl = (hc1 * hcp * iu * k - hc * hcp1 * iu * k) / (2.0 * (hcp * wf_match * k - hc * wfp))

    return sl, nl


#==============================================================================
# Complete Scattering Calculation
#==============================================================================

class ScatteringSolverFortran:
    """
    Quantum scattering solver matching Fortran scatt.f implementation.
    """

    def __init__(self, r_max=25.0, hcm=0.05, l_max=30):
        """
        Initialize solver.

        Parameters:
            r_max: matching radius (fm)
            hcm: step size (fm)
            l_max: maximum angular momentum
        """
        self.r_max = r_max
        self.hcm = hcm
        self.l_max = l_max
        self.irmatch = int(r_max / hcm)

        # Create radial mesh (ir = 0, 1, ..., irmatch corresponds to r = 0, hcm, ..., irmatch*hcm)
        self.r_mesh = np.arange(0, self.irmatch + 1) * hcm

    def solve(self, potential, E, mu, Z1=0, Z2=0, l_values=None):
        """
        Solve scattering problem for given potential and energy.

        Parameters:
            potential: function V(r) returning potential in MeV (can be complex)
            E: scattering energy (MeV, center of mass)
            mu: reduced mass (amu)
            Z1, Z2: charges of projectile and target (for Coulomb)
            l_values: list of l values to solve (default: 0 to l_max)

        Returns:
            results: dict with S-matrix, phase shifts, wave functions
        """
        # Convert mass to MeV/c²
        mu_mev = mu * AMU

        # Wave number
        k = np.sqrt(2.0 * mu_mev * E) / HBARC

        # Sommerfeld parameter
        if Z1 != 0 and Z2 != 0:
            eta = Z1 * Z2 * E2 * mu_mev / (HBARC**2 * k)
        else:
            eta = 0.0

        # Calculate potential on grid
        Vpot = np.zeros(self.irmatch + 1, dtype=complex)
        for ir in range(self.irmatch + 1):
            r = ir * self.hcm
            if r < 1e-6:
                r = 1e-6  # avoid r=0
            V_val = potential(r)
            if hasattr(V_val, '__len__'):
                Vpot[ir] = V_val[0] if len(V_val) == 1 else V_val
            else:
                Vpot[ir] = V_val

        # Get Coulomb functions at matching point
        # Matching at irmatch-2 (same as Fortran)
        rho_match = (self.irmatch - 2) * self.hcm * k

        if l_values is None:
            l_values = range(self.l_max + 1)

        max_l = max(l_values)

        # Get Coulomb functions
        nfc, ngc, nfcp, ngcp = coul90_functions(rho_match, eta, max_l)

        # Coulomb phase shifts
        sigma = coulomb_phase_shift(eta, max_l) if abs(eta) > 1e-10 else np.zeros(max_l + 1)

        # Storage
        S_matrix = {}
        phase_shifts = {}
        wave_functions = {}

        for l in l_values:
            # Solve Schrödinger equation
            rwfl = numerov_sch(self.hcm, self.irmatch, l, mu_mev, E, Vpot)

            # Boundary guard: need irmatch >= 5 for 5-point matching stencil
            if self.irmatch < 5:
                S_matrix[l] = 1.0 + 0j
                phase_shifts[l] = 0.0
                wave_functions[l] = rwfl
                continue

            # Extract S-matrix using matching at irmatch-2
            # wfmatch = rwfl[irmatch-4:irmatch+1] (5 points centered at irmatch-2)
            wfmatch = rwfl[self.irmatch-4:self.irmatch+1]

            sl, nl = matching(l, k, wfmatch, self.hcm, nfc[l], ngc[l], nfcp[l], ngcp[l])

            S_matrix[l] = sl
            phase_shifts[l] = np.angle(sl) / 2.0  # delta = arg(S)/2

            # Normalize wave function
            wave_functions[l] = rwfl * nl

        return {
            'k': k,
            'eta': eta,
            'sigma': sigma,
            'S_matrix': S_matrix,
            'phase_shifts': phase_shifts,
            'wave_functions': wave_functions,
            'r_mesh': self.r_mesh,
        }

    def solve_spin_half(self, potential_lj, E, mu, Z1=0, Z2=0, l_max=None):
        """
        Solve scattering for spin-1/2 projectile on spin-0 target.

        For each partial wave l, solves two channels j = l +/- 1/2
        (only j = 1/2 for l = 0). Spin-orbit coupling makes S_{l+} != S_{l-}.

        Parameters:
            potential_lj: function V(r_array, l, j) returning complex potential
                          array on the solver's r_mesh. Must include central,
                          Coulomb, and spin-orbit contributions for given (l, j).
            E: center-of-mass energy (MeV)
            mu: reduced mass (amu)
            Z1, Z2: charges of projectile and target
            l_max: maximum angular momentum (default: self.l_max)

        Returns:
            dict with keys:
                'k': wave number (fm^-1)
                'eta': Sommerfeld parameter
                'sigma': Coulomb phase shifts array
                'S_matrix_lj': dict {(l, j): S_lj} of S-matrix elements
        """
        if l_max is None:
            l_max = self.l_max

        mu_mev = mu * AMU
        k = np.sqrt(2.0 * mu_mev * E) / HBARC

        if Z1 != 0 and Z2 != 0:
            eta = Z1 * Z2 * E2 * mu_mev / (HBARC**2 * k)
        else:
            eta = 0.0

        # Coulomb functions at matching point (depend on l only, not j)
        rho_match = (self.irmatch - 2) * self.hcm * k
        nfc, ngc, nfcp, ngcp = coul90_functions(rho_match, eta, l_max)
        sigma = coulomb_phase_shift(eta, l_max) if abs(eta) > 1e-10 else np.zeros(l_max + 1)

        # Build r grid with r=0 replaced to avoid singularity
        r_grid = self.r_mesh.copy()
        r_grid[0] = 1e-6

        S_matrix_lj = {}

        for l in range(l_max + 1):
            j_values = [l + 0.5] if l == 0 else [l + 0.5, l - 0.5]

            for j in j_values:
                # Compute potential array for this (l, j)
                Vpot = np.asarray(potential_lj(r_grid, l, j), dtype=complex).ravel()
                if len(Vpot) != self.irmatch + 1:
                    raise ValueError(
                        f"potential_lj returned array of length {len(Vpot)}, "
                        f"expected {self.irmatch + 1} (l={l}, j={j})"
                    )

                # Solve radial Schrodinger equation via Numerov
                rwfl = numerov_sch(self.hcm, self.irmatch, l, mu_mev, E, Vpot)

                # Boundary guard: need irmatch >= 5 for 5-point matching stencil
                if self.irmatch < 5:
                    S_matrix_lj[(l, j)] = 1.0 + 0j
                    continue

                # Extract S-matrix via matching
                wfmatch = rwfl[self.irmatch - 4 : self.irmatch + 1]
                sl, nl = matching(
                    l, k, wfmatch, self.hcm,
                    nfc[l], ngc[l], nfcp[l], ngcp[l]
                )

                S_matrix_lj[(l, j)] = sl

        return {
            'k': k,
            'eta': eta,
            'sigma': sigma,
            'S_matrix_lj': S_matrix_lj,
        }


#==============================================================================
# Test
#==============================================================================

def test_with_fresco():
    """Test against FRESCO results for n + 208Pb at 30 MeV."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from potentials import KD02Potential

    print("="*70)
    print("Testing Scattering Solver (Fortran-compatible)")
    print("n + 208Pb at 30 MeV")
    print("="*70)

    # System parameters
    A_target = 208
    Z_target = 82
    E_lab = 30.0  # MeV

    # Center of mass energy
    A_proj = 1
    E_cm = E_lab * A_target / (A_proj + A_target)
    mu = (A_proj * A_target) / (A_proj + A_target)  # amu

    print(f"E_lab = {E_lab} MeV")
    print(f"E_cm = {E_cm:.4f} MeV")
    print(f"mu = {mu:.4f} amu")

    # Create KD02 potential
    pot = KD02Potential('n', A_target, Z_target, E_lab)
    print(pot)

    # Create solver (matching FRESCO parameters)
    solver = ScatteringSolverFortran(r_max=25.0, hcm=0.05, l_max=30)

    print(f"\nGrid: dr = {solver.hcm} fm, r_max = {solver.r_max} fm")
    print(f"irmatch = {solver.irmatch}")

    # Solve
    results = solver.solve(pot.potential, E_cm, mu, Z1=0, Z2=Z_target, l_values=range(15))

    print(f"\nk = {results['k']:.6f} fm^-1")
    print(f"eta = {results['eta']:.6f}")

    # FRESCO results for comparison (from running FRESCO with correct input)
    fresco_phases = {
        0: 17.0,  # approximate from previous runs
    }

    print("\n" + "="*70)
    print("Phase Shifts (degrees)")
    print("="*70)
    print(f"{'l':>3}  {'delta':>12}  {'|S_l|':>10}  {'Re(S)':>12}  {'Im(S)':>12}")
    print("-" * 55)

    for l in sorted(results['S_matrix'].keys()):
        S_l = results['S_matrix'][l]
        delta_deg = np.degrees(results['phase_shifts'][l])
        abs_S = np.abs(S_l)

        print(f"{l:3d}  {delta_deg:12.4f}  {abs_S:10.6f}  {S_l.real:12.6f}  {S_l.imag:12.6f}")

    # Reaction cross section
    sigma_R = 0.0
    k = results['k']
    for l in results['S_matrix']:
        S_l = results['S_matrix'][l]
        sigma_R += (2*l + 1) * (1.0 - np.abs(S_l)**2)
    sigma_R *= np.pi / k**2

    print(f"\nReaction cross section: sigma_R = {sigma_R:.2f} fm^2 = {sigma_R*10:.2f} mb")


def test_free_particle():
    """Test with V=0 (free particle). Should give S=1, delta=0."""
    print("\n" + "="*70)
    print("Test: Free Particle (V=0)")
    print("="*70)

    def V_zero(r):
        return 0.0 + 0.0j

    mu = 0.5  # amu
    E = 10.0  # MeV

    solver = ScatteringSolverFortran(r_max=20.0, hcm=0.05, l_max=10)
    results = solver.solve(V_zero, E, mu, Z1=0, Z2=0, l_values=range(6))

    print(f"k = {results['k']:.6f} fm^-1")
    print(f"\n{'l':>3}  {'delta (deg)':>12}  {'|S_l|':>10}")
    print("-" * 30)

    all_pass = True
    for l in sorted(results['S_matrix'].keys()):
        S_l = results['S_matrix'][l]
        delta_deg = np.degrees(results['phase_shifts'][l])
        abs_S = np.abs(S_l)

        # Check |S| = 1 and delta = 0
        if abs(abs_S - 1.0) > 0.001 or abs(delta_deg) > 0.1:
            all_pass = False
            status = "FAIL"
        else:
            status = "OK"

        print(f"{l:3d}  {delta_deg:12.4f}  {abs_S:10.6f}  {status}")

    if all_pass:
        print("\nFree particle test: PASSED")
    else:
        print("\nFree particle test: FAILED")


def test_square_well():
    """Test with square well potential (has analytical solution)."""
    print("\n" + "="*70)
    print("Test: Square Well Potential")
    print("="*70)

    V0 = 50.0  # MeV (depth)
    R = 3.0    # fm (radius)

    def V_well(r):
        if r < R:
            return -V0 + 0.0j
        return 0.0 + 0.0j

    mu = 0.5  # amu
    E = 10.0  # MeV

    solver = ScatteringSolverFortran(r_max=20.0, hcm=0.02, l_max=6)
    results = solver.solve(V_well, E, mu, Z1=0, Z2=0, l_values=range(6))

    k = results['k']

    print(f"V0 = {V0} MeV, R = {R} fm")
    print(f"E = {E} MeV, k = {k:.6f} fm^-1")
    print(f"kR = {k*R:.4f}")

    # Calculate analytical l=0 phase shift for comparison
    mu_mev = mu * AMU
    k_in = np.sqrt(2 * mu_mev * (E + V0)) / HBARC
    delta_0_ana = np.arctan(k * np.tan(k_in * R) / k_in) - k * R
    delta_0_ana = np.degrees(delta_0_ana)

    print(f"\nAnalytical l=0 phase shift: {delta_0_ana:.4f} degrees")

    print(f"\n{'l':>3}  {'delta (deg)':>12}  {'|S_l|':>10}")
    print("-" * 30)

    for l in sorted(results['S_matrix'].keys()):
        S_l = results['S_matrix'][l]
        delta_deg = np.degrees(results['phase_shifts'][l])
        abs_S = np.abs(S_l)

        print(f"{l:3d}  {delta_deg:12.4f}  {abs_S:10.6f}")

    delta_0_num = np.degrees(results['phase_shifts'][0])
    diff = abs(delta_0_num - delta_0_ana)

    if diff < 1.0:
        print(f"\nSquare well l=0 test: PASSED (diff = {diff:.4f} degrees)")
    else:
        print(f"\nSquare well l=0 test: FAILED (diff = {diff:.4f} degrees)")


if __name__ == '__main__':
    test_free_particle()
    test_square_well()
    print("\n")
    test_with_fresco()
