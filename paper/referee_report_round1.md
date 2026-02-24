# Referee Report — Round 1

## Summary

The referee finds the work presents an interesting approach but does not meet PRL standards. Core criticism: the analysis only considers single-energy elastic scattering angular distributions, whereas the community already uses multiple data types.

## Major Criticisms

### (1) Limited observable scope

The author focuses exclusively on elastic scattering angular distributions, whereas KD02 was constrained using:

- (i) elastic-scattering angular distributions
- (ii) analyzing powers
- (iii) total cross sections
- (iv) average resonance parameters

**Referee's key point**: If a comprehensive analysis of ALL data types still showed reduced effective parameter space, THAT would be a significant new insight.

### (2) Fisher analysis methodology concerns

The single-energy Fisher analysis does not capture:

- (i) whether a parameter is required for globally acceptable fits
- (ii) whether a parameter is needed to reproduce energy trends

Diffuseness is primarily constrained through energy systematics, integrated observables, and correlations across energies — none captured in single-energy Fisher analysis.

**Specific suggestion**: Plot angle-resolved sensitivity to diffuseness S_a(θ) — backward angle effects may be washed out when summing over all angles.

### (3) Path to PRL-level impact

A comprehensive Fisher analysis should include at least:

- (i) reaction cross sections
- (ii) energy systematics

## Minor Points

### (A) Abstract framing

First sentence ("why the optical model potential remains ambiguous despite decades of precise scattering measurements") is misleading — the actual scope is narrower: how strongly does elastic scattering at a given energy constrain the OMP parameters.

### (B) PCA undefined

"PCA" (Principal Component Analysis) is used without definition in the third paragraph of the Introduction.

### (C) Conclusion on nuclear forces

The link between "sloppy" parameters and chiral nuclear forces needs refinement. Microscopic nuclear forces fitted to NN scattering can produce unphysical OMP shapes — elastic nucleon-nucleus scattering DOES constrain these. The conclusion should acknowledge this.

## Response Strategy

### Must-do extensions (code work needed)

1. Add reaction cross section σ_R = π/k² Σ_l (2l+1)(1-|S_l|²) to Fisher matrix
2. Add total cross section σ_T = 2π/k² Σ_l (2l+1)(1-Re[S_l]) to Fisher matrix
3. Multi-energy combined Fisher analysis: F_combined = Σ_E F(E)
4. Plot angle-resolved sensitivity S_a(θ) for each parameter (especially diffuseness)

### Writing fixes

5. Revise abstract opening — narrower, more precise framing
6. Define PCA in Introduction
7. Revise conclusion on nuclear forces connection

### Key scientific question

After adding σ_R + σ_T + multi-energy:
- If D_eff still ≈ 2-3 → fundamental information limit (strong PRL case)
- If D_eff → 5-7 → quantifies complementarity of different observables (also publishable)
