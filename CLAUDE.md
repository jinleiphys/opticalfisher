# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nuclear physics research computing the **effective dimensionality (D_eff)** of optical potentials in nuclear scattering. Core result: elastic scattering can only constrain D_eff ≈ 1.5 parameter combinations out of 9 in the KD02 optical potential, explaining the Igo ambiguity. Target: PRL publication.

**Strategy: Numerov method only.** Neural network scripts exist as reference but are not the active development path.

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
python analysis/deff_scan_kd02_9params_log.py    # ← RECOMMENDED (log-derivatives)
python analysis/deff_scan_kd02_9params.py         # absolute derivatives version

# Generate publication figures
python paper/plot_fig1_final.py   # D_eff heatmap
python paper/plot_fig2_final.py   # D_eff vs mass/energy/condition
python paper/plot_fig3_final.py   # Eigenvector analysis (currently uses NN — needs Numerov rewrite)
python paper/plot_fig4_final.py   # Information geometry

# Sync figures/LaTeX to Overleaf
./sync_overleaf.sh
```

## Architecture

### Computational Pipeline

```
KD02Potential (src/potentials.py)
  → 9-param optical potential U(r) = V(r) + iW(r)
    → ScatteringSolverFortran (src/scattering_fortran.py)
      → Numerov integration of radial Schrödinger equation per l
      → S-matrix extraction via matching to Coulomb functions (coul90)
        → Cross section dσ/dΩ(θ) from partial wave sum
          → Fisher Information Matrix via finite-difference gradients
            → D_eff = (Σλ)²/Σλ² (participation ratio of eigenvalues)
```

### Import Path Issue

Analysis scripts currently hardcode `sys.path.insert(0, '/Users/jinlei/Desktop/code/PINN_CFC')` to find `potentials` and `scattering_fortran`. These should point to `src/` instead:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

### Key Modules

- **`src/potentials.py`** — `KD02Potential` class: energy/mass-dependent global optical potential (Koning-Delaroche 2003). Also `CookPotential` for 6Li.
- **`src/scattering_fortran.py`** — `ScatteringSolverFortran` class: Numerov solver matching Fortran `scatt.f` exactly. Uses `coul90_wrapper` for Coulomb functions.
- **`src/coul90_wrapper/`** — Compiled Fortran module for Coulomb wave functions F_l, G_l and phase shifts σ_l.

### 9-Parameter Model (KD02, spin-0)

```python
params = [V, rv, av, W, rw, aw, Wd, rvd, avd]
#  indices: 0   1   2   3   4   5   6    7    8
```
V=real volume depth (MeV), rv/av=radius/diffuseness (fm), W=imaginary volume, Wd=imaginary surface.

### Physical Constants (consistent across all files)

```python
HBARC = 197.3269804  # MeV·fm
AMU = 931.494        # MeV/c²
E2 = 1.44            # MeV·fm (Coulomb constant)
```

## Data Files

- `data/deff_scan_kd02_9params_log.json` — **Primary result for PRL** (Numerov + log-derivatives)
- `data/deff_scan_data.json` — Plotting-ready format used by `paper/plot_fig*.py`
- `data/deff_nn_9params.json` — Neural network results (reference only, known 36% discrepancy)

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
