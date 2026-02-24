# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nuclear physics research computing the **effective dimensionality (D_eff)** of optical potentials in nuclear scattering. Core result: elastic scattering can only constrain D_eff ≈ 1.5 parameter combinations out of 9 (or 11 with spin-orbit) in the KD02 optical potential, explaining the Igo ambiguity. Target: PRL publication.

**Strategy: Numerov method exclusively.** All computation uses the Numerov solver with finite-difference gradients, validated against FRESCO. Neural network scripts are historical artifacts only.

The code supports both **spin-0** (9-param, central only) and **spin-1/2** (11-param, with spin-orbit coupling) scattering, including analyzing power Ay, reaction cross section σ_R, and total cross section σ_T.

## Setup & Compilation

The Fortran Coulomb function wrapper must be compiled before any analysis can run:
```bash
cd src/coul90_wrapper
f2py -c -m coul90_mod coul90.f90
```

No requirements.txt exists. Key dependencies: numpy, matplotlib, json. Some legacy scripts also use torch/ncps (neural network — not needed for Numerov path).

## Running Analysis

```bash
# Single-case D_eff computation (quick test)
python analysis/deff_kd02_9params.py

# Full 168-configuration scan (12 nuclei × 7 energies × 2 projectiles)
python analysis/deff_scan_kd02_9params_log.py    # spin-0, 9 params (log-derivatives)
python analysis/deff_scan_kd02_9params.py         # spin-0, absolute derivatives version

# Extended scan: spin-1/2, 11 params, multiple observables (dσ/dΩ, Ay, σ_R, σ_T)
python analysis/deff_scan_extended.py             # ← RECOMMENDED for referee response

# Multi-energy combined Fisher analysis (post-processing)
python analysis/deff_multi_energy.py

# Angle-resolved sensitivity analysis
python analysis/angle_resolved_sensitivity.py

# Generate publication figures (all Numerov-based, no NN dependencies)
python paper/plot_fig1_final.py           # D_eff heatmap
python paper/plot_fig2_final.py           # D_eff vs mass/energy/condition
python paper/plot_fig3_final.py           # Eigenvector analysis (Numerov)
python paper/plot_fig4_final.py           # Information geometry (real gradients)
python paper/plot_fig_complementarity.py  # Observable complementarity + multi-E
python paper/plot_fig_sensitivity.py      # Angle-resolved sensitivity

# Sync figures/LaTeX to Overleaf
./sync_overleaf.sh
```

## Architecture

### Computational Pipeline

```
KD02Potential (src/potentials.py)
  → 9-param central potential U(r) = V(r) + iW(r)            [spin-0]
  → 11-param potential U(r,l,j) = central + spin-orbit        [spin-1/2]
    → ScatteringSolverFortran (src/scattering_fortran.py)
      → solve():          Numerov per l, returns S_l           [spin-0]
      → solve_spin_half(): Numerov per (l,j), returns S_{l,j}  [spin-1/2]
        → Observables (src/observables.py)
          → f(θ), g(θ) scattering amplitudes
          → dσ/dΩ = |f|² + |g|², Ay = 2Im(fg*)/(|f|²+|g|²)
          → σ_R, σ_T from S-matrix
            → Fisher Information Matrix via finite-difference gradients
              → D_eff = (Σλ)²/Σλ² (participation ratio of eigenvalues)
```

### Import Path Issue

Analysis scripts currently hardcode `sys.path.insert(0, '/Users/jinlei/Desktop/code/PINN_CFC')` to find `potentials` and `scattering_fortran`. These should point to `src/` instead:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

### Key Modules

- **`src/potentials.py`** — `KD02Potential` class: energy/mass-dependent global optical potential (Koning-Delaroche 2003). Includes `full_potential(r, l, j)` for spin-orbit. Also `CookPotential` for 6Li.
- **`src/scattering_fortran.py`** — `ScatteringSolverFortran` class: Numerov solver matching Fortran `scatt.f` exactly. `solve()` for spin-0, `solve_spin_half()` for spin-1/2 with (l,j)-resolved S-matrix.
- **`src/observables.py`** — Spin-1/2 scattering observables: amplitudes f(θ)/g(θ), dσ/dΩ, Ay, σ_R, σ_T. Validated against FRESCO (S-matrix to |ΔS|<0.002, Ay matches iT11 via −√2 Madison convention factor).
- **`src/coul90_wrapper/`** — Compiled Fortran module for Coulomb wave functions F_l, G_l and phase shifts σ_l.

### 9-Parameter Model (KD02, spin-0)

```python
params = [V, rv, av, W, rw, aw, Wd, rvd, avd]
#  indices: 0   1   2   3   4   5   6    7    8
```
V=real volume depth (MeV), rv/av=radius/diffuseness (fm), W=imaginary volume, Wd=imaginary surface.

### 11-Parameter Model (KD02, spin-1/2)

```python
params = [V, rv, av, W, rw, aw, Wd, rvd, avd, Vso, Wso]
#  indices: 0   1   2   3   4   5   6    7    8    9    10
```
Extends 9-param with Vso (real spin-orbit depth) and Wso (imaginary spin-orbit depth). Spin-orbit geometry (rvso, avso) fixed at KD02 values. Thomas form with factor of 2 (l·σ = 2 l·s convention).

### Physical Constants (consistent across all files)

```python
HBARC = 197.3269804  # MeV·fm
AMU = 931.494        # MeV/c²
E2 = 1.44            # MeV·fm (Coulomb constant)
```

## Data Files

- `data/deff_scan_kd02_9params_log.json` — Spin-0, 9-param results (Numerov + log-derivatives)
- `data/deff_scan_extended.json` — **Spin-1/2, 11-param results with multiple observables + full Fisher matrices**
- `data/deff_gradients_representative.json` — Full gradient matrices for representative cases
- `data/deff_multi_energy.json` — Multi-energy combined Fisher analysis results
- `data/angle_sensitivity.json` — Angle-resolved sensitivity S_i(theta) on dense grid
- `data/deff_scan_data.json` — Plotting-ready format used by `paper/plot_fig*.py`
- `data/deff_nn_9params.json` — Neural network results (historical reference only)

## Overleaf Sync

```bash
./sync_overleaf.sh   # Copies paper/*.{tex,bib} + paper/figures/*.pdf → Overleaf
```
Overleaf URL: `https://git@git.overleaf.com/6951e717227815c0789a012a`
Local clone: `/tmp/overleaf_opticalfisher`

## Conventions

- `_log.py` suffix = uses log-derivatives d log σ/d log p (scale-invariant, preferred)
- Plotting color scheme: green/mint for neutrons, pink/rose for protons (pastel Morandi palette)
- Cross sections in mb (millibarns = 10 fm²), energies in MeV, distances in fm
- Solver grid: r_max=25-40 fm, dr=0.02-0.05 fm, l_max=30
