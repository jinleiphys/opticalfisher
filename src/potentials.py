#!/usr/bin/env python3
"""
Optical Potentials for Nuclear Scattering

This module implements optical potentials commonly used in nuclear physics:
- Cook potential for 6Li scattering
- KD02 (Koning-Delaroche) potential for nucleon (n/p) scattering

References:
    Cook: J. Cook, Nucl. Phys. A 388 (1982) 153
    KD02: A.J. Koning & J.P. Delaroche, Nucl. Phys. A 713 (2003) 231

Author: Jin Lei
Date: December 2024
"""

import numpy as np


#==============================================================================
# Physical Constants
#==============================================================================

HBARC = 197.3269804   # MeV·fm
M_PION = 139.5706     # MeV/c^2 (charged pion mass)
LAMBDA_PI_SQ = (HBARC / M_PION)**2  # ≈ 2.0 fm^2 (pion Compton wavelength squared)


#==============================================================================
# Woods-Saxon Form Factor
#==============================================================================

def woods_saxon_form_factor(r, R, a):
    """
    Woods-Saxon form factor.

    f(r) = 1 / [1 + exp((r - R) / a)]

    Parameters:
        r: radial coordinate (fm), can be array
        R: radius parameter (fm)
        a: diffuseness parameter (fm)

    Returns:
        f(r): form factor values
    """
    return 1.0 / (1.0 + np.exp((r - R) / a))


def woods_saxon_derivative(r, R, a):
    """
    Derivative of Woods-Saxon form factor with respect to r.

    df/dr = -(1/a) * f * (1 - f)

    Parameters:
        r: radial coordinate (fm), can be array
        R: radius parameter (fm)
        a: diffuseness parameter (fm)

    Returns:
        df/dr: derivative values (fm^-1)
    """
    f = woods_saxon_form_factor(r, R, a)
    return -(1.0 / a) * f * (1.0 - f)


def surface_form_factor(r, R, a):
    """
    Derivative (surface) Woods-Saxon form factor.

    g(r) = -4a * df/dr = 4 * f * (1 - f)

    Peaks at r = R with maximum value 1.

    Parameters:
        r: radial coordinate (fm), can be array
        R: radius parameter (fm)
        a: diffuseness parameter (fm)

    Returns:
        g(r): surface form factor values (dimensionless)
    """
    f = woods_saxon_form_factor(r, R, a)
    return 4.0 * f * (1.0 - f)


#==============================================================================
# Coulomb Potential
#==============================================================================

def coulomb_potential(r, Z1, Z2, Rc):
    """
    Coulomb potential for uniformly charged sphere.

    V_C(r) = Z1*Z2*e^2 / r                      for r >= Rc
           = Z1*Z2*e^2 / (2*Rc) * (3 - r^2/Rc^2) for r < Rc

    Parameters:
        r: radial coordinate (fm), can be array
        Z1: charge of projectile
        Z2: charge of target
        Rc: Coulomb radius (fm), typically Rc = rc * A^(1/3)

    Returns:
        V_C(r): Coulomb potential in MeV
    """
    # e^2 = 1.44 MeV·fm (Coulomb constant)
    e2 = 1.44

    r = np.atleast_1d(r)
    V_C = np.zeros_like(r)

    # Outside the charge distribution
    outside = r >= Rc
    V_C[outside] = Z1 * Z2 * e2 / r[outside]

    # Inside the charge distribution (uniform sphere)
    inside = r < Rc
    V_C[inside] = Z1 * Z2 * e2 / (2 * Rc) * (3 - r[inside]**2 / Rc**2)

    return V_C


#==============================================================================
# Cook Potential for 6Li Scattering
#==============================================================================

class CookPotential:
    """
    Cook optical potential for 6Li scattering.

    Reference: J. Cook, Nucl. Phys. A 388 (1982) 153

    The optical potential has the form:
        U(r) = -V0 * f_R(r) - i*W0 * f_I(r) + V_C(r)

    where f_R and f_I are Woods-Saxon form factors.

    Parameters (for 6Li):
        V0 = 109.5 MeV (real depth, constant)
        r_R = 1.326 fm, a_R = 0.811 fm (real geometry)

        W0 = 58.16 - 0.328*A + 0.00075*A^2 MeV (imaginary depth, A-dependent)
        r_I = 1.534 fm, a_I = 0.884 fm (imaginary geometry)

    Attributes:
        A_target: mass number of target nucleus
        Z_target: charge number of target nucleus
        Z_proj: charge number of projectile (3 for 6Li)
        A_proj: mass number of projectile (6 for 6Li)
    """

    # Fixed parameters for 6Li (from Cook 1982)
    V0 = 109.5      # MeV - real potential depth
    r_R = 1.326     # fm - real radius parameter
    a_R = 0.811     # fm - real diffuseness

    r_I = 1.534     # fm - imaginary radius parameter
    a_I = 0.884     # fm - imaginary diffuseness

    # Coulomb radius parameter
    r_C = 1.3       # fm - Coulomb radius parameter

    # 6Li projectile properties
    Z_proj = 3      # charge of 6Li
    A_proj = 6      # mass of 6Li

    def __init__(self, A_target, Z_target):
        """
        Initialize Cook potential for a specific target.

        Parameters:
            A_target: mass number of target nucleus
            Z_target: charge number of target nucleus
        """
        self.A_target = A_target
        self.Z_target = Z_target

        # Calculate A-dependent imaginary depth
        # W0 = 58.16 - 0.328*A + 0.00075*A^2
        A = A_target
        self.W0 = 58.16 - 0.328 * A + 0.00075 * A**2

        # Calculate radii (R = r * A^(1/3))
        A_third = A_target ** (1.0/3.0)
        self.R_R = self.r_R * A_third  # real radius
        self.R_I = self.r_I * A_third  # imaginary radius
        self.R_C = self.r_C * A_third  # Coulomb radius

        # Reduced mass (in units where hbar^2/2m = 1)
        # mu = m_proj * m_target / (m_proj + m_target)
        # Using atomic mass units, 1 amu ≈ 931.5 MeV/c^2
        self.mu = (self.A_proj * self.A_target) / (self.A_proj + self.A_target)

    def real_potential(self, r):
        """
        Real part of the optical potential (nuclear + Coulomb).

        V_R(r) = -V0 * f_R(r) + V_C(r)

        Parameters:
            r: radial coordinate (fm)

        Returns:
            V_R(r): real potential in MeV
        """
        V_nucl = -self.V0 * woods_saxon_form_factor(r, self.R_R, self.a_R)
        V_coul = coulomb_potential(r, self.Z_proj, self.Z_target, self.R_C)
        return V_nucl + V_coul

    def imaginary_potential(self, r):
        """
        Imaginary part of the optical potential.

        W(r) = -W0 * f_I(r)

        Parameters:
            r: radial coordinate (fm)

        Returns:
            W(r): imaginary potential in MeV (negative for absorption)
        """
        return -self.W0 * woods_saxon_form_factor(r, self.R_I, self.a_I)

    def potential(self, r):
        """
        Full complex optical potential.

        U(r) = V_R(r) + i*W(r) = -V0*f_R(r) - i*W0*f_I(r) + V_C(r)

        Parameters:
            r: radial coordinate (fm)

        Returns:
            U(r): complex potential in MeV
        """
        return self.real_potential(r) + 1j * self.imaginary_potential(r)

    def __repr__(self):
        return (f"CookPotential(6Li + {self.A_target})\n"
                f"  V0 = {self.V0:.1f} MeV, R_R = {self.R_R:.3f} fm, a_R = {self.a_R:.3f} fm\n"
                f"  W0 = {self.W0:.2f} MeV, R_I = {self.R_I:.3f} fm, a_I = {self.a_I:.3f} fm\n"
                f"  R_C = {self.R_C:.3f} fm")


#==============================================================================
# KD02 (Koning-Delaroche) Potential for Nucleon Scattering
#==============================================================================

class KD02Potential:
    """
    Koning-Delaroche global optical potential for nucleon scattering.

    Reference: A.J. Koning & J.P. Delaroche, Nucl. Phys. A 713 (2003) 231

    The optical potential has the form:
        U(r) = -V*f(r,rv,av) - iW*f(r,rw,aw)
               -4*a_wd * (iWd) * d/dr[f(r,rwd,awd)]
               + (Vso + iWso) * (hbar/m_pi*c)^2 * 1/r * d/dr[f(r,rvso,avso)] * (l·s)
               + Vc(r)  [for protons]

    where f(r,R,a) is the Woods-Saxon form factor.

    All parameters are energy and mass dependent (global parametrization).

    Attributes:
        projectile: 'neutron' or 'proton'
        A_target: mass number of target
        Z_target: charge number of target
        E: incident energy in MeV (lab frame)
    """

    def __init__(self, projectile, A_target, Z_target, E):
        """
        Initialize KD02 potential.

        Parameters:
            projectile: 'neutron' or 'proton' (or 'n' or 'p')
            A_target: mass number of target nucleus
            Z_target: charge number of target nucleus
            E: incident energy in MeV (lab frame)
        """
        self.A = A_target
        self.Z = Z_target
        self.N = A_target - Z_target
        self.E = E

        # Determine projectile type
        if projectile in ['neutron', 'n', 1]:
            self.k0 = 1
            self.projectile = 'neutron'
        elif projectile in ['proton', 'p', 2]:
            self.k0 = 2
            self.projectile = 'proton'
        else:
            raise ValueError(f"Unknown projectile: {projectile}")

        # Calculate all parameters
        self._calculate_parameters()

    def _calculate_parameters(self):
        """Calculate all KD02 parameters for given A, Z, E."""
        A = self.A
        Z = self.Z
        N = self.N
        E = self.E
        k0 = self.k0

        # ===== Geometry parameters (common for n and p) =====
        self.rv = 1.3039 - 0.4054 * A**(-1./3.)
        self.av = 0.6778 - 1.487e-4 * A
        self.rw = self.rv
        self.aw = self.av

        self.rvd = 1.3424 - 0.01585 * A**(1./3.)
        self.rwd = self.rvd

        self.rvso = 1.1854 - 0.647 * A**(-1./3.)
        self.rwso = self.rvso
        self.avso = 0.59
        self.awso = 0.59

        # ===== Depth parameters (common) =====
        v4 = 7.0e-9
        w2 = 73.55 + 0.0795 * A
        d2 = 0.0180 + 3.802e-3 / (1. + np.exp((A - 156.) / 8.))
        d3 = 11.5
        vso1 = 5.922 + 0.0030 * A
        vso2 = 0.0040
        wso1 = -3.1
        wso2 = 160.

        # ===== Neutron-specific parameters =====
        if k0 == 1:
            ef = -11.2814 + 0.02646 * A
            v1 = 59.30 - 21.0 * (N - Z) / A - 0.024 * A
            v2 = 7.228e-3 - 1.48e-6 * A
            v3 = 1.994e-5 - 2.0e-8 * A
            w1 = 12.195 + 0.0167 * A
            d1 = 16.0 - 16.0 * (N - Z) / A
            avd = 0.5446 - 1.656e-4 * A
            awd = avd
            rc = 0.0

        # ===== Proton-specific parameters =====
        elif k0 == 2:
            ef = -8.4075 + 0.01378 * A
            v1 = 59.30 + 21.0 * (N - Z) / A - 0.024 * A
            v2 = 7.067e-3 + 4.23e-6 * A
            v3 = 1.729e-5 + 1.136e-8 * A
            w1 = 14.667 + 0.009629 * A
            d1 = 16.0 + 16.0 * (N - Z) / A
            avd = 0.5187 + 5.205e-4 * A
            awd = avd
            rc = 1.198 + 0.697 * A**(-2./3.) + 12.994 * A**(-5./3.)

        # Store diffuseness
        self.avd = avd
        self.awd = awd
        self.rc = rc

        # ===== Energy-dependent depths =====
        f = E - ef

        # Coulomb correction for protons
        if k0 == 1:
            vcoul = 0.0
        else:
            Vc = 1.73 / rc * Z / A**(1./3.)
            vcoul = Vc * v1 * (v2 - 2.*v3*f + 3.*v4*f*f)

        # Real volume
        self.V = v1 * (1. - v2*f + v3*f**2 - v4*f**3) + vcoul

        # Imaginary volume
        self.W = w1 * f**2 / (f**2 + w2**2)

        # Real surface (always zero in KD02)
        self.Vd = 0.0

        # Imaginary surface
        self.Wd = d1 * f**2 * np.exp(-d2*f) / (f**2 + d3**2)

        # Spin-orbit real
        self.Vso = vso1 * np.exp(-vso2 * f)

        # Spin-orbit imaginary
        self.Wso = wso1 * f**2 / (f**2 + wso2**2)

        # Calculate actual radii (R = r * A^(1/3))
        A_third = A**(1./3.)
        self.Rv = self.rv * A_third
        self.Rw = self.rw * A_third
        self.Rvd = self.rvd * A_third
        self.Rwd = self.rwd * A_third
        self.Rvso = self.rvso * A_third
        self.Rwso = self.rwso * A_third
        self.Rc = rc * A_third if rc > 0 else 0.0

    def volume_real(self, r):
        """Real volume potential: -V * f(r)"""
        return -self.V * woods_saxon_form_factor(r, self.Rv, self.av)

    def volume_imag(self, r):
        """Imaginary volume potential: -W * f(r)"""
        return -self.W * woods_saxon_form_factor(r, self.Rw, self.aw)

    def surface_real(self, r):
        """Real surface potential (derivative form): always 0 in KD02"""
        return np.zeros_like(np.atleast_1d(r))

    def surface_imag(self, r):
        """
        Imaginary surface potential (derivative Woods-Saxon form).

        W_surf(r) = -Wd * g(r, Rwd, awd)

        where g = 4*f*(1-f) is the surface form factor.
        """
        r = np.atleast_1d(r)
        return -self.Wd * surface_form_factor(r, self.Rwd, self.awd)

    def real_potential(self, r):
        """
        Total real central potential (no spin-orbit).

        V_R(r) = V_volume(r) + V_surface(r) + V_Coulomb(r)
        """
        V = self.volume_real(r) + self.surface_real(r)
        if self.k0 == 2 and self.Rc > 0:  # proton
            V = V + coulomb_potential(r, 1, self.Z, self.Rc)
        return V

    def imaginary_potential(self, r):
        """
        Total imaginary central potential (no spin-orbit).

        W(r) = W_volume(r) + W_surface(r)
        """
        return self.volume_imag(r) + self.surface_imag(r)

    def spin_orbit_real(self, r):
        """
        Real spin-orbit radial form (without l·s factor).

        V_so(r) = 2 * λ²π * Vso * (1/r) * df/dr(r, Rvso, avso)

        The factor of 2 comes from the KD02/FRESCO convention: the potential
        is defined with l·σ operator, but we apply <l·s> externally.
        Since l·σ = 2*l·s for spin-1/2, the factor of 2 appears here.

        Returns V_so(r) in MeV. Multiply by <l·s> to get full SO potential.
        """
        r = np.atleast_1d(r)
        dfdr = woods_saxon_derivative(r, self.Rvso, self.avso)
        r_safe = np.where(r > 1e-6, r, 1e-6)
        return 2.0 * LAMBDA_PI_SQ * self.Vso * dfdr / r_safe

    def spin_orbit_imag(self, r):
        """
        Imaginary spin-orbit radial form (without l·s factor).

        W_so(r) = 2 * λ²π * Wso * (1/r) * df/dr(r, Rwso, awso)

        Returns W_so(r) in MeV. Multiply by <l·s> to get full SO potential.
        """
        r = np.atleast_1d(r)
        dfdr = woods_saxon_derivative(r, self.Rwso, self.awso)
        r_safe = np.where(r > 1e-6, r, 1e-6)
        return 2.0 * LAMBDA_PI_SQ * self.Wso * dfdr / r_safe

    def spin_orbit(self, r):
        """
        Complex spin-orbit radial form (without l·s factor).

        U_so(r) = V_so(r) + i*W_so(r)
        """
        return self.spin_orbit_real(r) + 1j * self.spin_orbit_imag(r)

    @staticmethod
    def ls_factor(l, j):
        """
        Expectation value of l·s for given (l, j).

        For spin-1/2 particles:
            <l·s> = [j(j+1) - l(l+1) - 3/4] / 2
                  = l/2        if j = l + 1/2
                  = -(l+1)/2   if j = l - 1/2

        Parameters:
            l: orbital angular momentum (integer)
            j: total angular momentum (half-integer, as float)

        Returns:
            <l·s>: expectation value
        """
        return 0.5 * (j * (j + 1) - l * (l + 1) - 0.75)

    def potential(self, r):
        """
        Full complex central potential (no spin-orbit).

        U(r) = V_R(r) + i*W(r)
        """
        return self.real_potential(r) + 1j * self.imaginary_potential(r)

    def full_potential(self, r, l, j):
        """
        Complete optical potential for a specific partial wave (l, j).

        U(r, l, j) = V_central(r) + i*W_central(r) + [V_so(r) + i*W_so(r)] * <l·s>

        Parameters:
            r: radial coordinate (fm)
            l: orbital angular momentum
            j: total angular momentum (half-integer, e.g. 0.5, 1.5, ...)

        Returns:
            U(r): complex potential in MeV including spin-orbit
        """
        U_central = self.potential(r)
        if l == 0:
            return U_central
        ls = self.ls_factor(l, j)
        U_so = self.spin_orbit(r) * ls
        return U_central + U_so

    def get_parameters(self):
        """Return all potential parameters as a dictionary."""
        return {
            'V': self.V, 'rv': self.rv, 'av': self.av,
            'W': self.W, 'rw': self.rw, 'aw': self.aw,
            'Vd': self.Vd, 'rvd': self.rvd, 'avd': self.avd,
            'Wd': self.Wd, 'rwd': self.rwd, 'awd': self.awd,
            'Vso': self.Vso, 'rvso': self.rvso, 'avso': self.avso,
            'Wso': self.Wso, 'rwso': self.rwso, 'awso': self.awso,
            'rc': self.rc,
        }

    def __repr__(self):
        p = self.get_parameters()
        return (f"KD02Potential({self.projectile} + A={self.A}, Z={self.Z}, E={self.E} MeV)\n"
                f"  V  = {p['V']:.3f} MeV, rv = {p['rv']:.4f} fm, av = {p['av']:.4f} fm\n"
                f"  W  = {p['W']:.3f} MeV, rw = {p['rw']:.4f} fm, aw = {p['aw']:.4f} fm\n"
                f"  Wd = {p['Wd']:.3f} MeV, rwd = {p['rwd']:.4f} fm, awd = {p['awd']:.4f} fm\n"
                f"  Vso = {p['Vso']:.3f} MeV, rvso = {p['rvso']:.4f} fm, avso = {p['avso']:.4f} fm\n"
                f"  Wso = {p['Wso']:.3f} MeV, rwso = {p['rwso']:.4f} fm, awso = {p['awso']:.4f} fm\n"
                f"  rc = {p['rc']:.4f} fm")


#==============================================================================
# Test and Visualization
#==============================================================================

def test_cook_potential():
    """Test Cook potential implementation."""
    import matplotlib.pyplot as plt

    # Test for different targets
    targets = [
        (28, 14, '28Si'),   # Silicon-28
        (40, 20, '40Ca'),   # Calcium-40
        (58, 28, '58Ni'),   # Nickel-58
        (208, 82, '208Pb'), # Lead-208
    ]

    r = np.linspace(0.1, 20, 200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Real potential for different targets
    ax = axes[0, 0]
    for A, Z, label in targets:
        pot = CookPotential(A, Z)
        V_R = pot.real_potential(r)
        ax.plot(r, V_R, label=f'6Li + {label}')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('V (MeV)')
    ax.set_title('Real Potential (Nuclear + Coulomb)')
    ax.legend()
    ax.set_xlim(0, 20)
    ax.grid(True)

    # Plot 2: Imaginary potential for different targets
    ax = axes[0, 1]
    for A, Z, label in targets:
        pot = CookPotential(A, Z)
        W = pot.imaginary_potential(r)
        ax.plot(r, W, label=f'6Li + {label}')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('W (MeV)')
    ax.set_title('Imaginary Potential (Absorption)')
    ax.legend()
    ax.set_xlim(0, 20)
    ax.grid(True)

    # Plot 3: W0 as function of A
    ax = axes[1, 0]
    A_range = np.linspace(10, 250, 100)
    W0_values = 58.16 - 0.328 * A_range + 0.00075 * A_range**2
    ax.plot(A_range, W0_values, 'b-', linewidth=2)
    ax.set_xlabel('Target Mass Number A')
    ax.set_ylabel('W0 (MeV)')
    ax.set_title('Imaginary Depth vs Target Mass')
    ax.grid(True)

    # Plot 4: Detailed view for 28Si
    ax = axes[1, 1]
    pot = CookPotential(28, 14)
    print(pot)

    V_R = pot.real_potential(r)
    W = pot.imaginary_potential(r)
    V_nucl = -pot.V0 * woods_saxon_form_factor(r, pot.R_R, pot.a_R)
    V_coul = coulomb_potential(r, pot.Z_proj, pot.Z_target, pot.R_C)

    ax.plot(r, V_R, 'b-', linewidth=2, label='V_R (total real)')
    ax.plot(r, V_nucl, 'b--', linewidth=1.5, label='V_nucl')
    ax.plot(r, V_coul, 'r--', linewidth=1.5, label='V_Coulomb')
    ax.plot(r, W, 'g-', linewidth=2, label='W (imaginary)')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('V (MeV)')
    ax.set_title('6Li + 28Si: Potential Components')
    ax.legend()
    ax.set_xlim(0, 15)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('cook_potential.png', dpi=150)
    print("\nSaved: cook_potential.png")
    plt.show()


def test_kd02_potential():
    """Test KD02 potential implementation."""
    import matplotlib.pyplot as plt

    r = np.linspace(0.1, 15, 200)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: n + 208Pb at different energies
    ax = axes[0, 0]
    energies = [10, 20, 30, 50, 100]
    for E in energies:
        pot = KD02Potential('n', 208, 82, E)
        V_R = pot.real_potential(r)
        ax.plot(r, V_R, label=f'E={E} MeV')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('V (MeV)')
    ax.set_title('n + 208Pb: Real Potential')
    ax.legend()
    ax.grid(True)

    # Plot 2: Imaginary potential
    ax = axes[0, 1]
    for E in energies:
        pot = KD02Potential('n', 208, 82, E)
        W = pot.imaginary_potential(r)
        ax.plot(r, W, label=f'E={E} MeV')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('W (MeV)')
    ax.set_title('n + 208Pb: Imaginary Potential')
    ax.legend()
    ax.grid(True)

    # Plot 3: Potential components for n + 208Pb at 30 MeV
    ax = axes[0, 2]
    pot = KD02Potential('n', 208, 82, 30)
    print("\n" + "="*60)
    print("n + 208Pb at 30 MeV:")
    print(pot)

    ax.plot(r, pot.volume_real(r), 'b-', label='V_vol (real)', linewidth=2)
    ax.plot(r, pot.volume_imag(r), 'b--', label='W_vol (imag)', linewidth=2)
    ax.plot(r, pot.surface_imag(r), 'r--', label='W_surf (imag)', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('V (MeV)')
    ax.set_title('n + 208Pb (E=30 MeV): Components')
    ax.legend()
    ax.grid(True)

    # Plot 4: p + 40Ca - with Coulomb
    ax = axes[1, 0]
    pot = KD02Potential('p', 40, 20, 30)
    print("\n" + "="*60)
    print("p + 40Ca at 30 MeV:")
    print(pot)

    V_total = pot.real_potential(r)
    V_nucl = pot.volume_real(r)
    V_coul = coulomb_potential(r, 1, 20, pot.Rc)

    ax.plot(r, V_total, 'b-', label='V_total', linewidth=2)
    ax.plot(r, V_nucl, 'b--', label='V_nuclear', linewidth=1.5)
    ax.plot(r, V_coul, 'r--', label='V_Coulomb', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('V (MeV)')
    ax.set_title('p + 40Ca (E=30 MeV): Real Potential')
    ax.legend()
    ax.grid(True)

    # Plot 5: Compare n vs p on same target
    ax = axes[1, 1]
    pot_n = KD02Potential('n', 208, 82, 30)
    pot_p = KD02Potential('p', 208, 82, 30)

    ax.plot(r, pot_n.real_potential(r), 'b-', label='n: V_real', linewidth=2)
    ax.plot(r, pot_n.imaginary_potential(r), 'b--', label='n: W_imag', linewidth=2)
    ax.plot(r, pot_p.real_potential(r), 'r-', label='p: V_real', linewidth=2)
    ax.plot(r, pot_p.imaginary_potential(r), 'r--', label='p: W_imag', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('V (MeV)')
    ax.set_title('n vs p + 208Pb (E=30 MeV)')
    ax.legend()
    ax.grid(True)

    # Plot 6: Energy dependence of potential depths
    ax = axes[1, 2]
    E_range = np.linspace(1, 150, 100)
    V_depths = []
    W_depths = []
    Wd_depths = []
    for E in E_range:
        pot = KD02Potential('n', 208, 82, E)
        V_depths.append(pot.V)
        W_depths.append(pot.W)
        Wd_depths.append(pot.Wd)

    ax.plot(E_range, V_depths, 'b-', label='V (real vol)', linewidth=2)
    ax.plot(E_range, W_depths, 'r-', label='W (imag vol)', linewidth=2)
    ax.plot(E_range, Wd_depths, 'g-', label='Wd (imag surf)', linewidth=2)
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Depth (MeV)')
    ax.set_title('n + 208Pb: Depth vs Energy')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('kd02_potential.png', dpi=150)
    print("\nSaved: kd02_potential.png")
    plt.show()


if __name__ == '__main__':
    print("Testing Cook Potential (6Li)...")
    test_cook_potential()

    print("\n" + "="*70)
    print("Testing KD02 Potential (n/p)...")
    test_kd02_potential()
