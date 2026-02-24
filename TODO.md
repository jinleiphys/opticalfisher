# TODO — Referee Response

## Phase 1: Code Extensions (Priority)

- [ ] Add reaction cross section σ_R to Fisher matrix computation
- [ ] Add total cross section σ_T (optical theorem) to Fisher matrix
- [ ] Implement multi-energy combined Fisher analysis: F_total = Σ_E F(E)
- [ ] Plot angle-resolved sensitivity S_a(θ) for each parameter, especially diffuseness
- [ ] Re-run full 168-configuration scan with extended observables
- [ ] Fix sys.path imports to use local src/ instead of PINN_CFC

## Phase 2: Analysis & Figures

- [ ] Compare D_eff: elastic-only vs elastic+σ_R vs elastic+σ_R+σ_T vs multi-energy
- [ ] New figure: D_eff progression as observables are added
- [ ] New figure: angle-resolved sensitivity S_a(θ) showing diffuseness at backward angles
- [ ] Update heatmap (Fig 1) with comprehensive D_eff values

## Phase 3: Paper Revision

- [ ] Revise abstract — narrower framing, remove overly ambitious first sentence
- [ ] Define PCA (Principal Component Analysis) in Introduction
- [ ] Revise conclusion — acknowledge elastic scattering CAN constrain microscopic forces
- [ ] Add discussion of observable complementarity (which data types break degeneracies)
- [ ] Cite KD02 fitting procedure explicitly (elastic + analyzing powers + σ_T + resonance)
- [ ] Decide target journal: revised PRL or PRC
