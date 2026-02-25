#!/usr/bin/env python3
"""
Observable calculations for spin-1/2 nucleon scattering.

Computes elastic cross section, analyzing power, reaction and total cross
sections from the (l, j)-resolved S-matrix.

Physics:
    f(theta) = f_C(theta) + (1/2ik) sum_l e^{2i sigma_l}
               [(l+1)(S_{l+}-1) + l(S_{l-}-1)] P_l(cos theta)
    g(theta) = (1/2ik) sum_{l>=1} e^{2i sigma_l}
               [S_{l+} - S_{l-}] P_l^1(cos theta)
    dsigma/dOmega = |f|^2 + |g|^2
    Ay = 2 Im(f g*) / (|f|^2 + |g|^2)

Author: Jin Lei
Date: February 2026
"""

import numpy as np
from scipy.special import lpmv


def coulomb_amplitude(k, eta, theta_rad, sigma_0):
    """
    Point Coulomb scattering amplitude (Rutherford).

    f_C(theta) = -eta/(2k sin^2(theta/2))
                 * exp(-i eta ln(sin^2(theta/2)) + 2i sigma_0)

    Parameters:
        k: wave number (fm^-1)
        eta: Sommerfeld parameter
        theta_rad: scattering angles in radians (array)
        sigma_0: Coulomb phase shift for l=0

    Returns:
        f_C: complex Coulomb amplitude array (fm)
    """
    if abs(eta) < 1e-10:
        return np.zeros_like(theta_rad, dtype=complex)

    sin_half_sq = np.sin(theta_rad / 2.0) ** 2

    # Guard against forward-angle singularity (theta -> 0)
    safe = sin_half_sq > 1e-30
    f_C = np.zeros_like(theta_rad, dtype=complex)
    f_C[safe] = (-eta / (2.0 * k * sin_half_sq[safe])
                 * np.exp(-1j * eta * np.log(sin_half_sq[safe]) + 2j * sigma_0))
    # At theta=0 the Coulomb amplitude diverges; leave as 0 (physical: never measured at 0°)
    return f_C


def scattering_amplitudes(k, eta, S_matrix_lj, theta_deg, l_max,
                           sigma_l=None):
    """
    Compute direct f(theta) and spin-flip g(theta) scattering amplitudes.

    Parameters:
        k: wave number (fm^-1)
        eta: Sommerfeld parameter
        S_matrix_lj: dict {(l, j): S_lj} from solve_spin_half
        theta_deg: scattering angles in degrees (array)
        l_max: maximum angular momentum included
        sigma_l: Coulomb phase shifts (optional, computed if None)

    Returns:
        f_theta, g_theta: complex amplitude arrays (fm)
    """
    theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=float))
    cos_theta = np.cos(theta_rad)
    n_theta = len(theta_rad)

    f_theta = np.zeros(n_theta, dtype=complex)
    g_theta = np.zeros(n_theta, dtype=complex)

    # Coulomb phase shifts
    if sigma_l is None:
        if abs(eta) > 1e-10:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from scattering_fortran import coulomb_phase_shift
            sigma_l = coulomb_phase_shift(eta, l_max)
        else:
            sigma_l = np.zeros(l_max + 1)

    # Coulomb amplitude
    if abs(eta) > 1e-10:
        f_theta += coulomb_amplitude(k, eta, theta_rad, sigma_l[0])

    # Precompute Legendre polynomials
    P_l_arr = np.zeros((l_max + 1, n_theta))
    P1_l_arr = np.zeros((l_max + 1, n_theta))
    for l in range(l_max + 1):
        P_l_arr[l] = np.polynomial.legendre.legval(
            cos_theta, [0] * l + [1]
        )
        if l >= 1:
            P1_l_arr[l] = lpmv(1, l, cos_theta)

    # Nuclear partial wave sum
    for l in range(l_max + 1):
        phase = np.exp(2j * sigma_l[l])

        S_lp = S_matrix_lj.get((l, l + 0.5), 1.0)   # j = l + 1/2
        S_lm = S_matrix_lj.get((l, l - 0.5), 1.0)   # j = l - 1/2

        # Direct amplitude f(theta)
        if l == 0:
            f_theta += phase * (S_lp - 1.0) / (2j * k) * P_l_arr[l]
        else:
            f_theta += (phase / (2j * k)
                        * ((l + 1) * (S_lp - 1.0) + l * (S_lm - 1.0))
                        * P_l_arr[l])

        # Spin-flip amplitude g(theta), l >= 1
        if l >= 1:
            g_theta += (phase * (S_lp - S_lm) / (2j * k)
                        * P1_l_arr[l])

    return f_theta, g_theta


def elastic_cross_section(f_theta, g_theta):
    """
    Elastic differential cross section.

    dsigma/dOmega = |f|^2 + |g|^2  (mb/sr)

    Parameters:
        f_theta, g_theta: scattering amplitudes (fm)

    Returns:
        dcs: differential cross section array (mb/sr)
    """
    return (np.abs(f_theta)**2 + np.abs(g_theta)**2) * 10.0  # fm^2 -> mb


def analyzing_power(f_theta, g_theta):
    """
    Vector analyzing power.

    Ay(theta) = 2 Im(f g*) / (|f|^2 + |g|^2)

    Parameters:
        f_theta, g_theta: scattering amplitudes (fm)

    Returns:
        Ay: analyzing power array (dimensionless, -1 to 1)
    """
    denom = np.abs(f_theta)**2 + np.abs(g_theta)**2
    return 2.0 * np.imag(f_theta * np.conj(g_theta)) / (denom + 1e-30)


def reaction_cross_section(k, S_matrix_lj):
    """
    Reaction (absorption) cross section.

    sigma_R = (pi/k^2) sum_l [(l+1)(1-|S_{l+}|^2) + l(1-|S_{l-}|^2)]  (mb)

    Parameters:
        k: wave number (fm^-1)
        S_matrix_lj: dict {(l, j): S_lj}

    Returns:
        sigma_R: reaction cross section (mb)
    """
    sigma_R = 0.0
    for (l, j), S_lj in S_matrix_lj.items():
        weight = (l + 1) if abs(j - (l + 0.5)) < 0.1 else l
        sigma_R += weight * (1.0 - np.abs(S_lj)**2)
    return sigma_R * np.pi / k**2 * 10.0  # fm^2 -> mb


def total_cross_section(k, S_matrix_lj):
    """
    Total cross section via optical theorem (neutrons only).

    sigma_T = (2pi/k^2) sum_l [(l+1)(1-Re S_{l+}) + l(1-Re S_{l-})]  (mb)

    Not valid for charged particles (Coulomb divergence).

    Parameters:
        k: wave number (fm^-1)
        S_matrix_lj: dict {(l, j): S_lj}

    Returns:
        sigma_T: total cross section (mb)
    """
    sigma_T = 0.0
    for (l, j), S_lj in S_matrix_lj.items():
        weight = (l + 1) if abs(j - (l + 0.5)) < 0.1 else l
        sigma_T += weight * (1.0 - S_lj.real)
    return sigma_T * 2.0 * np.pi / k**2 * 10.0  # fm^2 -> mb


def compute_all_observables(k, eta, S_matrix_lj, theta_deg, l_max,
                            is_neutron=True, sigma_l=None):
    """
    Compute all scattering observables from (l, j)-resolved S-matrix.

    Parameters:
        k: wave number (fm^-1)
        eta: Sommerfeld parameter
        S_matrix_lj: dict {(l, j): S_lj}
        theta_deg: angles for differential observables (degrees)
        l_max: maximum angular momentum
        is_neutron: if True, also compute sigma_T
        sigma_l: Coulomb phase shifts (optional)

    Returns:
        dict with:
            'elastic_dcs': dsigma/dOmega(theta) in mb/sr
            'Ay': analyzing power Ay(theta)
            'sigma_R': reaction cross section (mb)
            'sigma_T': total cross section (mb, neutrons only)
    """
    f, g = scattering_amplitudes(k, eta, S_matrix_lj, theta_deg, l_max,
                                  sigma_l)

    result = {
        'elastic_dcs': elastic_cross_section(f, g),
        'Ay': analyzing_power(f, g),
        'sigma_R': reaction_cross_section(k, S_matrix_lj),
    }

    if is_neutron:
        result['sigma_T'] = total_cross_section(k, S_matrix_lj)

    return result
