# Response to Referee Report

**Manuscript:** Scale-Invariant Information Limit in Nuclear Optical Potential Extraction

Jin Lei

---

We thank the referee for the careful reading of the manuscript and the constructive suggestions. The referee's core criticism---that the original analysis was limited to single-energy elastic angular distributions---was entirely well-taken. We have substantially extended the analysis to include (i) analyzing power $A_y$, (ii) reaction cross section $\sigma_R$, (iii) total cross section $\sigma_T$, (iv) multi-energy combined Fisher analysis, (v) the KD02 global parameterization at three distinct levels (13 local, 19 per-nucleus, and 48 universal parameters), and (vi) angle-resolved sensitivity. We have also added a detailed comparison with the Bayesian uncertainty quantification of King *et al.* [PRL **122**, 232502 (2019)], establishing the Fisher information matrix as the rigorous theoretical foundation for the empirical findings of that landmark study.

The revised manuscript presents these extensions as a **step-by-step constraint analysis**, progressively adding observables and energies to decompose the constraining power of each data type. The End Matter now includes a complete inventory of all 48 KD02 universal coefficients (Table I), with a clear distinction between the three levels of parameter counting. This addresses all of the referee's major and minor criticisms. Below we respond to each point in detail.

---

## Major Criticisms

### (1) Limited observable scope

> *The author focuses exclusively on elastic scattering angular distributions, whereas KD02 was constrained using (i) elastic-scattering angular distributions, (ii) analyzing powers, (iii) total cross sections, and (iv) average resonance parameters. If a comprehensive analysis of ALL data types still showed reduced effective parameter space, THAT would be a significant new insight.*

We have now extended the analysis to include three of the four scattering-region observable types the referee lists: elastic angular distributions, analyzing powers $A_y$, and total cross sections $\sigma_T$ (plus the reaction cross section $\sigma_R$). The fourth type---average resonance parameters---constrains the potential at low energies ($E \lesssim 1$ MeV) in the resonance region, which lies outside the 10--200 MeV continuum regime of the present analysis. Below $\sim$5 MeV, individual compound-nucleus resonances dominate and smooth optical potentials such as KD02 are not designed to apply, as their Woods-Saxon forms cannot capture the rapid energy dependence near individual resonances. The revised manuscript now explicitly states this scope limitation in the Introduction. Nevertheless, the Fisher framework generalizes straightforwardly to include resonance data: one simply adds their contribution to the Fisher matrix, $F_\mathrm{total} = F_\mathrm{scattering} + F_\mathrm{resonance}$. The revised manuscript presents a step-by-step decomposition (new Fig. 4):

- **Step 1: Elastic $d\sigma/d\Omega$ alone**: $D_\mathrm{eff} \approx 1.2$ (50 MeV), with the 168-configuration average $D_\mathrm{eff} = 1.7 \pm 0.4$.
- **Step 2: $+ \sigma_R$**: $D_\mathrm{eff}$ increases by $<0.001$. A single integrated datum is overwhelmed by 17 elastic angular bins.
- **Step 3: $+ A_y + \sigma_T$**: $D_\mathrm{eff}$ rises from 1.19 to 1.62 for $n+{}^{40}$Ca, from 1.17 to 1.84 for $n+{}^{120}$Sn. The analyzing power provides genuinely new information through the spin-flip amplitude $g(\theta)$, which probes spin-orbit interference. The spin-orbit subgroup trace increases by two orders of magnitude.
- **Step 4: Multi-energy (7 energies)**: $D_\mathrm{eff}$ increases to ~2.4--2.8. This is the single most effective route.
- **Step 5: KD02 per-nucleus global parameterization (19 parameters)**: $D_\mathrm{eff} \approx 1.0$ of 19 per-nucleus global parameters. (The 13 local parameters per energy collapse to 19 per-nucleus parameters because geometry is energy-independent while depths follow polynomial energy dependence; see End Matter, Table I for the complete parameter hierarchy.) Importantly, this does *not* mean that information is lost relative to Step 4---the data still determine the same local-parameter combinations at each energy. The lower $D_\mathrm{eff}$ arises because KD02 expresses each local parameter as a polynomial in energy (e.g., $V(E)$ via 4 coefficients), but the data can only pin down the leading 1--2 coefficients; the higher-order terms ($v_3$, $v_4$, $d_3$, etc.) are redundant. This is analogous to fitting a straight line with a cubic: the slope is well-determined, but the curvature coefficients are arbitrary.

The key new insight, as the referee anticipated, is precisely this: even after including all scattering-region observables available within 10--200 MeV ($d\sigma/d\Omega$, $A_y$, $\sigma_T$, and $\sigma_R$), $D_\mathrm{eff}$ remains ~2--3, far below the 13 local parameters. The information limit is intrinsic to the scattering physics. The step-by-step decomposition reveals *which* observables contribute *what*: $A_y$ lifts the spin-orbit degeneracy, multi-energy data lift the imaginary-parameter degeneracy, but single-energy $\sigma_R$ provides negligible additional constraint.

**Physical picture of the Igo ambiguity**: The eigenvector analysis (revised Fig. 3) provides a precise characterization of the classical Igo ambiguity. The dominant eigenvector $\mathbf{e}_1$ has $V$ and $r_v$ entering with the *same sign*: both increasing together makes the volume integral $J_V \propto V r_v^3$ larger, producing a measurably different cross section. This is the "stiff" (well-constrained) direction. The Igo ambiguity is the *orthogonal* "sloppy" direction, where $V$ increases while $r_v$ decreases (or vice versa) so that $J_V$ stays approximately constant---the data cannot distinguish a deep, narrow potential from a shallow, wide one. In short, there are two directions in the $V$-$r_v$ plane: the data pin down the volume integral (stiff), but cannot resolve the depth-radius tradeoff (sloppy). This eigenvalue ratio exceeds $10^5$, making the tradeoff direction effectively unmeasurable.

The computational infrastructure now includes:
- Full spin-1/2 scattering solver (`solve_spin_half()`) computing $(l,j)$-resolved $S$-matrix, validated against FRESCO ($|\Delta S| < 0.002$)
- Observables module computing $f(\theta)$, $g(\theta)$, $d\sigma/d\Omega$, $A_y = 2\mathrm{Im}(fg^*)/(|f|^2+|g|^2)$, $\sigma_R$, and $\sigma_T$
- Separate gradient matrices for each observable type, enabling the additive Fisher decomposition

### (2) Fisher analysis methodology concerns

> *The single-energy Fisher analysis does not capture (i) whether a parameter is required for globally acceptable fits, or (ii) whether a parameter is needed to reproduce energy trends. Diffuseness is primarily constrained through energy systematics, integrated observables, and correlations across energies.*

We now directly address both points:

**(i) Multi-energy combined Fisher analysis**: The combined Fisher matrix $F_\mathrm{combined} = \sum_E F(E)$ captures the information from measurements across 7 energies (10--200 MeV). Combining elastic data alone across energies raises $D_\mathrm{eff}$ from ~1.2 to ~2.1--2.6 (new Fig. 4, Step 4). The energy lever arm is the key: at low energies the real potential dominates, while at high energies absorption becomes stronger, probing orthogonal parameter combinations. The imaginary subgroup $D_\mathrm{eff}$ increases from ~1.1 (single energy) to ~2.3 (7 energies).

**(ii) KD02 per-nucleus global systematics**: We compute the Jacobian $J = \partial\boldsymbol{\theta}_\mathrm{local}/\partial\boldsymbol{\theta}_\mathrm{global}$ relating the 13 local parameters per energy to the 19 per-nucleus KD02 parameters, and project the multi-energy Fisher matrix onto the global space: $F_\mathrm{global} = J^T F_\mathrm{combined} J$. The result ($D_\mathrm{eff} \approx 1.0$ of 19 parameters) reveals that the polynomial functional forms in KD02 introduce additional degeneracies beyond those inherent in the scattering physics. At this level, each quantity like $E_f^n$ or $v_1$ is a single number for a fixed nucleus; the 19 parameters comprise 6 energy-independent geometry ($r_v, a_v, r_d, a_d, r_{so}, a_{so}$) and 13 energy-dependent depth amplitudes.

**(iii) Full multi-system global analysis**: To address the concern that a systematic optical potential should be evaluated across *all* systems rather than a single nucleus, we have extended the analysis to the **full universal KD02 coefficient space**. The KD02 global potential parametrizes all local potential parameters as functions of mass $A$, charge $Z$, and energy $E$ through 48 universal coefficients (counted from Tables 10--11 and Eqs. (25)--(37) of Ref. [KD02]; see End Matter, Table I for a complete inventory). At this level, each per-nucleus parameter is decomposed into its $A$-dependent coefficients: e.g., $E_f^n = -11.2814 + 0.02646\,A$ contributes *two* coefficients, and $r_v = 1.3039 - 0.4054\,A^{-1/3}$ contributes two more. The 48 coefficients comprise 9 shared geometry (including $a_{so} = 0.59$), 16 shared depth, 10 neutron-specific, and 13 proton-specific. We compute the Jacobian from these 48 universal parameters to the 13 local parameters for each of the 168 (nucleus, energy, projectile) configurations, and sum the transformed Fisher matrices:

$$F_\mathrm{universal} = \sum_{i=1}^{168} J_i^T \, F_i^\mathrm{local} \, J_i.$$

The result is striking: $D_\mathrm{eff} = 2.9$ out of 48 universal parameters. The dominant eigenvector (46% of total information) is the real potential radius coefficient $r_{v0}$, followed by $r_{v1}$ (36%) and the surface imaginary radius $r_{d0}$ (7%). Adding nuclei saturates $D_\mathrm{eff}$ after ~5 systems; the improvement from going from one nucleus to 24 (nucleus, projectile) pairs is only a factor of ~1.1. This confirms that the information limit is not a single-system artifact but an intrinsic feature of how nucleon-nucleus scattering encodes potential information.

We also performed the global analysis using only elastic scattering data (no $A_y$, $\sigma_R$, or $\sigma_T$), yielding slightly *higher* $D_\mathrm{eff}$ than the all-observable result. This counterintuitive result arises because $A_y$ data strongly reinforce the already-constrained "stiff" directions (individual eigenvalues increase by factors of 300--500), making the eigenvalue spectrum more peaked and thus lowering the participation ratio. The total Fisher information increases by orders of magnitude, but concentrates along already-stiff directions rather than opening new ones. In both cases, only ~3 effective parameter combinations are constrained.

The per-nucleus analysis in the universal space shows $D_\mathrm{eff} = 1.9$--$3.8$ (all 7 energies combined), with lighter nuclei like $^{12}$C having higher values. These results are presented in the revised End Matter with a new figure (Fig. 5) showing the $D_\mathrm{eff}$ saturation, eigenvalue spectrum, and eigenvector composition in the 48-dimensional universal coefficient space.

**Diffuseness**: The referee's specific concern is validated by our analysis. At a single energy, diffuseness parameters ($a_v$, $a_w$, $a_d$) contribute $<1\%$ to the dominant eigenvectors (Fig. 3). With multi-energy data, the imaginary subgroup (which includes $a_w$ and $a_d$) becomes better constrained ($D_\mathrm{eff}^\mathrm{imag}: 1.1 \to 2.3$). This quantitatively confirms the referee's statement that diffuseness is primarily constrained through energy systematics.

---

> *Specific suggestion: Plot angle-resolved sensitivity to diffuseness $S_a(\theta)$---backward angle effects may be washed out when summing over all angles.*

Done. The new Fig. 7 shows $|S_i(\theta)|$ for all 13 parameters. The referee's intuition is confirmed: diffuseness parameters $a_v$, $a_w$, and $a_d$ become most distinguishable at backward angles ($\theta > 130°$), where their sensitivity curves diverge from the depth/radius parameters. The cumulative $D_\mathrm{eff}(\theta_\mathrm{max})$ shows that ~90% of the total constraining power is captured by $\theta \approx 140°$. Backward-angle measurements are disproportionately valuable for breaking the parameter degeneracy---a practical recommendation for experimental design. The Fisher analysis does not deny the physical importance of diffuseness parameters, but quantifies that their information content comes primarily from backward angles combined with multi-energy data---precisely the regime where the referee's insight applies.

### (3) Path to PRL-level impact

> *A comprehensive Fisher analysis should include at least (i) reaction cross sections and (ii) energy systematics.*

Both are now included. Additionally, we provide a new element not anticipated in the original submission: a detailed comparison with the Bayesian analysis of King *et al.* [PRL **122**, 232502 (2019)]. This comparison (End Matter of the revised manuscript) demonstrates that:

1. The Fisher information matrix provides a **rigorous theoretical foundation** for the empirical findings of King *et al.*: in the Gaussian approximation, the posterior covariance equals $F^{-1}$, so our $D_\mathrm{eff} \approx 1.7$ directly explains why only $V$ and $r_v$ showed strong correlation in their MCMC scatter plots. Specifically, $V$ and $r_v$ both enter the dominant eigenvector $\mathbf{e}_1$ with the same sign (the volume integral direction), producing the elongated correlation structure observed in the MCMC posterior. The orthogonal direction ($V$ and $r_v$ trading off at constant $J_V$) is the "sloppy" Igo direction with an eigenvalue $10^5$ times smaller.
2. The FIM eigenvalue hierarchy (condition number $>10^7$) predicts the elongated "cigar-shaped" Bayesian posterior and explains why the frequentist $\chi^2$ approach produced spurious correlations.
3. The Fisher approach is ~3700x more efficient ($2N_p + 1 = 27$ evaluations for $N_p = 13$ vs. ~100,000 MCMC samples), enabling the systematic 168-configuration scan that would be prohibitively expensive with MCMC.
4. The Fisher analysis yields provably scale-invariant quantities ($D_\mathrm{eff}$), a fundamental property difficult to establish empirically from finite MCMC chains.

The two approaches are complementary: Fisher efficiently maps global constraint structure and proves fundamental properties, while Bayesian MCMC captures non-Gaussian posterior features. This positions the present work as the theoretical counterpart to the Bayesian UQ program of the MSU group [6--11], establishing a unified framework for understanding parameter constraints in nuclear reactions.

---

## Minor Points

### (A) Abstract framing

> *First sentence is misleading---the actual scope is narrower.*

Revised. The abstract now opens with "How many independent parameter combinations can nucleon-nucleus scattering data actually constrain in the optical model potential?" and explicitly states the baseline constraint from elastic scattering ($D_\mathrm{eff} \approx 1.7$), the step-by-step decomposition, and the priority ordering for augmenting elastic data: multi-energy systematics $\gg$ $A_y$ $\gg$ $\sigma_R$.

### (B) PCA undefined

> *"PCA" used without definition.*

Fixed: "principal component analysis (PCA) studies" (Introduction, paragraph 3).

### (C) Conclusion on nuclear forces

> *The link between "sloppy" parameters and chiral nuclear forces needs refinement.*

Revised. The Conclusion now states: "The 'sloppy' parameter directions do not imply that these aspects of the potential are unconstrained by all data---rather, they are insensitive to single-energy scattering measurements." The multi-energy analysis (Fig. 4) demonstrates that combining data across energies probes orthogonal directions in parameter space, providing a constructive path forward.

---

## Summary of Changes

1. Extended from 9-parameter central potential to **13-parameter model** including spin-orbit coupling ($V_{so}$, $W_{so}$, $r_{so}$, $a_{so}$)
2. Added spin-1/2 scattering solver with $(l,j)$-resolved $S$-matrix, validated against FRESCO
3. Added observables: $A_y$, $\sigma_R$, $\sigma_T$ (in addition to $d\sigma/d\Omega$)
4. New step-by-step constraint analysis decomposing the information contribution of each observable and multi-energy data (new Fig. 4)
5. New angle-resolved sensitivity analysis showing $|S_i(\theta)|$ for all 13 parameters (new Fig. 7)
6. **KD02 parameter hierarchy analysis at three levels** (End Matter, Table I):
   - Level 1: 13 local parameters per (A, Z, E) configuration
   - Level 2: 19 per-nucleus global parameters (fixed A, Z)
   - Level 3: 48 universal coefficients (full global model, counted from Tables 10--11 and Eqs. (25)--(37) of KD02)
7. Detailed Fisher--Bayesian comparison in End Matter, connecting to King *et al.* PRL (2019)
8. Revised abstract, introduction, discussion, and conclusion
9. All calculations use Numerov solver with finite-difference gradients (no neural network dependencies)
