# Response to Referee Report

**Manuscript:** Intrinsic Information Limit in Nuclear Optical Potential Extraction

Jin Lei

---

I thank the referee for the careful reading of the manuscript and the constructive suggestions. The referee's core criticism, that the original analysis was limited to single-energy elastic angular distributions, was entirely well-taken. I have substantially extended the analysis to include (i) analyzing power $A_y$, (ii) reaction cross section $\sigma_R$, (iii) total cross section $\sigma_T$, (iv) multi-energy combined Fisher analysis, (v) the KD02 global parameterization at two distinct levels (13 local parameters and 48 universal coefficients), and (vi) angle-resolved sensitivity. I have also added a detailed comparison with the Bayesian uncertainty quantification of King *et al.* [PRL **122**, 232502 (2019)], establishing the Fisher information matrix as the rigorous theoretical foundation for the empirical findings of that landmark study.

The revised manuscript presents these extensions as a **step-by-step constraint analysis**, progressively adding observables and energies to decompose the constraining power of each data type. The End Matter now includes a complete inventory of all 48 KD02 universal coefficients (Table I), with a clear distinction between local and universal parameter levels. This addresses all of the referee's major and minor criticisms. Below I respond to each point in detail.

---

## Major Criticisms

### (1) Limited observable scope

> *The author focuses exclusively on elastic scattering angular distributions, whereas KD02 was constrained using (i) elastic-scattering angular distributions, (ii) analyzing powers, (iii) total cross sections, and (iv) average resonance parameters. If a comprehensive analysis of ALL data types still showed reduced effective parameter space, THAT would be a significant new insight.*

I have now extended the analysis to include three of the four scattering-region observable types the referee lists: elastic angular distributions, analyzing powers $A_y$, and total cross sections $\sigma_T$ (plus the reaction cross section $\sigma_R$). The fourth type, average resonance parameters, constrains the potential at low energies ($E \lesssim 1$ MeV) in the resonance region, which lies outside the 10--200 MeV continuum regime of the present analysis. Below $\sim$5 MeV, individual compound-nucleus resonances dominate and smooth optical potentials such as KD02 are not designed to apply, as their Woods-Saxon forms cannot capture the rapid energy dependence near individual resonances. The revised manuscript now explicitly states this scope limitation in the Introduction. Nevertheless, the Fisher framework generalizes straightforwardly to include resonance data: one simply adds their contribution to the Fisher matrix, $F_\mathrm{total} = F_\mathrm{scattering} + F_\mathrm{resonance}$. The revised manuscript presents a step-by-step decomposition (new Fig. 4):

- **Step 1: Elastic $d\sigma/d\Omega$ alone**: $D_\mathrm{eff} = 1.22$ (50 MeV, $n+{}^{40}$Ca), with the 168-configuration average $D_\mathrm{eff} \approx 1.8 \pm 0.5$.
- **Step 2: $+ \sigma_R$**: $D_\mathrm{eff}$ increases by $<0.001$. A single integrated datum is overwhelmed by 35 elastic angular bins.
- **Step 3: $+ A_y$**: Adding $A_y$ raises $D_\mathrm{eff}$ from 1.22 to 1.53 for $n+{}^{40}$Ca, and from 1.15 to 1.72 for $n+{}^{120}$Sn (at sensitivity normalization $\epsilon = 1$, $\delta_{A_y} = 0.03$). The improvement is entirely due to $A_y$, which provides genuinely new information through the spin-flip amplitude $g(\theta)$ probing spin-orbit interference. The total cross section $\sigma_T$ (obtained from $\mathrm{Im}\,f(0)$ via the optical theorem) and $\sigma_R$ provide no measurable additional gain, as their integral constraints are already captured by the absolutely-normalized angular distribution. The quantitative gain from $A_y$ depends on the relative precision ratio $\epsilon/\delta_{A_y}$: at typical experimental elastic precision ($\epsilon = 5\%$--$10\%$), the gain ranges from ~5% to ~35% across systems; at $\epsilon = 1$ (equal statistical weight), the gain is ~25%--50%. The qualitative hierarchy multi-energy $\gg$ $A_y$ $\gg$ $\sigma_R$ holds at all precision ratios tested ($\epsilon = 1\%$--$100\%$). See End Matter, "Robustness checks."
- **Step 4: Multi-energy (7 energies)**: $D_\mathrm{eff}$ increases to ~2.4--2.6. This is the single most effective route.
- **Step 5: KD02 global systematics (48 universal coefficients)**: Projecting all 168 single-system Fisher matrices onto the 48-dimensional universal KD02 coefficient space yields $D_\mathrm{eff} = 3.0$ out of 48 parameters. With neutron data alone, $D_\mathrm{eff}$ peaks at ~3.2 around 7 systems; adding proton data *decreases* it to 3.0 because protons reinforce the same dominant directions rather than opening new ones. The dominant eigenvector (44% of information) is the real potential radius coefficient $r_{v0}$. See End Matter, Fig. 5 and Table I for details.

The key new insight, as the referee anticipated, is precisely this: even after including all scattering-region observables available within 10--200 MeV ($d\sigma/d\Omega$, $A_y$, $\sigma_T$, and $\sigma_R$), $D_\mathrm{eff}$ reaches only ~2.4--2.6 in the 13-parameter local space (and $D_\mathrm{eff} \approx 3.0$ out of 48 in the universal KD02 coefficient space), far below the number of model parameters. Within the KD02 framework, the information limit is intrinsic to the scattering physics. The step-by-step decomposition reveals *which* observables contribute *what*: $A_y$ lifts the spin-orbit degeneracy, multi-energy data lift the imaginary-parameter degeneracy, but single-energy $\sigma_R$ provides negligible additional constraint.

**Physical picture of the Igo ambiguity**: The eigenvector analysis (revised Fig. 3) provides a precise characterization of the classical Igo ambiguity. The dominant eigenvector $\mathbf{e}_1$ has $V$ and $r_v$ entering with the *same sign*: both increasing together makes the volume integral $J_V \propto V r_v^3$ larger, producing a measurably different cross section. This is the "stiff" (well-constrained) direction. The Igo ambiguity is the *orthogonal* "sloppy" direction, where $V$ increases while $r_v$ decreases (or vice versa) so that $J_V$ stays approximately constant, the data cannot distinguish a deep, narrow potential from a shallow, wide one. In short, there are two directions in the $V$-$r_v$ plane: the data pin down the volume integral (stiff) but cannot resolve the depth-radius tradeoff (sloppy). This eigenvalue ratio exceeds $10^5$, making the tradeoff direction effectively unmeasurable.

The computational extensions include a full spin-1/2 Numerov solver computing the $(l,j)$-resolved $S$-matrix (validated against FRESCO to $|\Delta S| < 0.002$), scattering amplitudes $f(\theta)$ and $g(\theta)$ yielding $d\sigma/d\Omega$, $A_y$, $\sigma_R$, and $\sigma_T$, and separate gradient matrices for each observable type enabling the additive Fisher decomposition.

### (2) Fisher analysis methodology concerns

> *The single-energy Fisher analysis does not capture (i) whether a parameter is required for globally acceptable fits, or (ii) whether a parameter is needed to reproduce energy trends. Diffuseness is primarily constrained through energy systematics, integrated observables, and correlations across energies.*

I now directly address both points:

**(i) Multi-energy combined Fisher analysis**: The combined Fisher matrix $F_\mathrm{combined} = \sum_E F(E)$ captures the information from measurements across 7 energies (10--200 MeV). Combining elastic data alone across energies raises $D_\mathrm{eff}$ from ~1.2 to ~2.4--2.6 in the 13-parameter local space (new Fig. 4, Step 4). The energy lever arm is the key: at low energies the real potential dominates, while at high energies absorption becomes stronger, probing orthogonal parameter combinations. This gain relies jointly on the additional data *and* the KD02 analytical energy dependence that ties parameters across energies.

**(ii) Full multi-system global analysis**: To address the concern that a systematic optical potential should be evaluated across *all* systems rather than a single nucleus, I have extended the analysis to the **full universal KD02 coefficient space**. The KD02 global potential parametrizes all local potential parameters as functions of mass $A$, charge $Z$, and energy $E$ through 48 universal coefficients (counted from Tables 10--11 and Eqs. (25)--(37) of Ref. [KD02]; see End Matter, Table I for a complete inventory). Each per-nucleus parameter is decomposed into its $A$-dependent coefficients: e.g., $E_f^n = -11.2814 + 0.02646\,A$ contributes *two* coefficients, and $r_v = 1.3039 - 0.4054\,A^{-1/3}$ contributes two more. The 48 coefficients comprise 9 shared geometry (including $a_{so} = 0.59$), 16 shared depth, 10 neutron-specific, and 13 proton-specific. I compute the Jacobian from these 48 universal parameters to the 13 local parameters for each of the 168 (nucleus, energy, projectile) configurations, and sum the transformed Fisher matrices:

$$F_\mathrm{universal} = \sum_{i=1}^{168} J_i^T \, F_i^\mathrm{local} \, J_i.$$

The result is striking: $D_\mathrm{eff} = 3.0$ out of 48 universal parameters. The dominant eigenvector (44% of total information) mixes $r_{v0}$ (74%) and $r_{so,0}$ (20%), the second eigenvector (37%) reverses these weights ($r_{so,0}$ 73%, $r_{v0}$ 20%), and the third (8%) is 92% $r_{d0}$ (the surface imaginary radius). With neutron data alone, $D_\mathrm{eff}$ peaks at ~3.2 around 7 systems; adding proton data *decreases* it to the final value of 3.0, because protons reinforce the same dominant directions rather than opening new ones. This confirms that the information limit is not a single-system artifact but an intrinsic feature of how nucleon-nucleus scattering encodes potential information.

I also performed the global analysis using only elastic scattering data (no $A_y$, $\sigma_R$, or $\sigma_T$), yielding slightly *higher* $D_\mathrm{eff}$ than the all-observable result. This counterintuitive result arises because $A_y$ data strongly reinforce the already-constrained "stiff" directions (individual eigenvalues increase by factors of 300--500), making the eigenvalue spectrum more peaked and thus lowering the participation ratio. The total Fisher information increases by orders of magnitude, but concentrates along already-stiff directions rather than opening new ones. In both cases, information is overwhelmingly concentrated in ~3 stiff directions out of 48 (as measured by the participation ratio $D_\mathrm{eff}$).

These results are presented in the revised End Matter with a new figure (Fig. 5) showing the $D_\mathrm{eff}$ saturation, eigenvalue spectrum, and eigenvector composition in the 48-dimensional universal coefficient space.

**Robustness of the Fisher analysis**: The revised End Matter includes three dedicated robustness checks: (1) *Evaluation point*: perturbing all 13 KD02 parameters by ±10% yields $D_\mathrm{eff}$ in the range 1.1--2.5 with means within 0.2 of the reference values, confirming that the conclusions do not depend on the specific evaluation point. (2) *Geometry constraints*: enforcing the KD02 ties $r_w = r_v$ and $a_w = a_v$ (reducing to 11 parameters) increases $D_\mathrm{eff}$ by 0.1--0.3, so the 13-parameter analysis is conservative. (3) *Observable weighting*: an $\epsilon$-scan from 1% to 100% elastic relative error shows that the Ay-induced $D_\mathrm{eff}$ gain varies from ~1% (high elastic precision) to ~50% (low elastic precision), but the qualitative hierarchy is invariant. These checks address the natural concern that the Fisher analysis might be sensitive to specific numerical choices.

**Diffuseness**: The referee's specific concern is validated by the analysis. At a single energy, diffuseness parameters ($a_v$, $a_w$, $a_d$) contribute $<1\%$ to the dominant eigenvectors (Fig. 3). With multi-energy data (exploiting the KD02 energy dependence to tie parameters across energies), the total $D_\mathrm{eff}$ more than doubles (from 1.22 to 2.59 for $n+{}^{40}$Ca), as different energies probe different radial regions of the potential, opening new constraining directions that include diffuseness. This confirms the referee's statement that diffuseness is primarily constrained through energy systematics.

---

> *Specific suggestion: Plot angle-resolved sensitivity to diffuseness $S_a(\theta)$---backward angle effects may be washed out when summing over all angles.*

Done. The new Fig. 6 plots normalized sensitivity directions $S_i(\theta)/\|\mathbf{S}_i\|$ for $V$, $r_v$, and the diffuseness $a_v$, together with the cumulative $D_\mathrm{eff}(\theta_\mathrm{max})$. Normalizing to unit length removes the overall magnitude and isolates the angular shape: overlapping curves indicate collinear (degenerate) parameters, while separated curves indicate independent information. Since $D_\mathrm{eff}$ is scale-invariant, this representation directly visualizes what determines $D_\mathrm{eff}$.

The result partially confirms the referee's intuition, but with an important nuance. The diffuseness $a_v$ does diverge from $V$ and $r_v$ at backward angles ($\theta > 100°$), showing that an independent angular structure develops there. However, the backward-angle cross sections are orders of magnitude smaller than the forward-angle values, so this angular independence contributes negligible Fisher information. The cumulative $D_\mathrm{eff}(\theta_\mathrm{max})$ [Fig. 6(b)] confirms this: it peaks at $\sim$2.0 at $\theta_\mathrm{max} \approx 25°$ in the Coulomb-nuclear interference region (where the diffraction minimum briefly separates the sensitivity directions), then decreases as information accumulates along the dominant $V$-$r_v$ direction, settling to $D_\mathrm{eff} = 1.22$ at full coverage. The referee's intuition that backward angles contain information about diffuseness is correct in terms of angular structure, but this information is suppressed by the small backward-angle cross sections. Diffuseness is constrained primarily through multi-energy systematics (Step 4), not through angular coverage at a single energy.

### (3) Path to PRL-level impact

> *A comprehensive Fisher analysis should include at least (i) reaction cross sections and (ii) energy systematics.*

Both are now included. Additionally, I provide a new element not anticipated in the original submission: a detailed comparison with the Bayesian analysis of King *et al.* [PRL **122**, 232502 (2019)]. This comparison (End Matter of the revised manuscript) demonstrates that:

1. The Fisher information matrix provides a **rigorous theoretical foundation** for the empirical findings of King *et al.*: in the Gaussian approximation, the posterior covariance equals $F^{-1}$, so the $D_\mathrm{eff} \approx 1.8$ found here directly explains why only $V$ and $r_v$ showed strong correlation in their MCMC scatter plots. Specifically, $V$ and $r_v$ both enter the dominant eigenvector $\mathbf{e}_1$ with the same sign (the volume integral direction), producing the elongated correlation structure observed in the MCMC posterior. The orthogonal direction ($V$ and $r_v$ trading off at constant $J_V$) is the "sloppy" Igo direction with an eigenvalue $10^5$ times smaller.
2. The FIM eigenvalue hierarchy (condition number $>10^7$) predicts the elongated "cigar-shaped" Bayesian posterior and explains why the frequentist $\chi^2$ approach produced spurious correlations.
3. The Fisher approach is ~3700x more efficient ($2N_p + 1 = 27$ evaluations for $N_p = 13$ vs. ~100,000 MCMC samples), enabling the systematic 168-configuration scan that would be prohibitively expensive with MCMC.
4. The Fisher analysis yields provably scale-invariant quantities ($D_\mathrm{eff}$), a fundamental property difficult to establish empirically from finite MCMC chains.

The two approaches are complementary: Fisher efficiently maps global constraint structure and proves fundamental properties, while Bayesian MCMC captures non-Gaussian posterior features. This positions the present work as the theoretical counterpart to the Bayesian UQ program of the MSU group [6--11], establishing a unified framework for understanding parameter constraints in nuclear reactions.

**Why this work meets PRL criteria.** I respectfully submit that the revised manuscript meets PRL's standards of broad significance and substantial advance, for the following reasons:

1. *It answers a decades-old open question quantitatively.* The Igo ambiguity has been known since 1958, but has never been characterized beyond qualitative statements ("the potential is only sensitive near the surface"). This work provides the first number: $D_\mathrm{eff} \approx 1.8$ out of 13, verified across 168 configurations within the KD02 framework. This transforms a qualitative observation into a precise and falsifiable statement.

2. *It establishes a fundamental no-go result.* The scale-invariance of $D_\mathrm{eff}$ proves that uniform error reduction cannot change the geometric anisotropy of the constraint structure: the ratio of uncertainties along stiff and sloppy directions is fixed regardless of overall precision. Breaking the bottleneck requires qualitatively different measurements, not simply better statistics.

3. *It provides the theoretical foundation for a PRL result.* King *et al.* [PRL **122**, 232502 (2019)] empirically observed that only $V$ and $r_v$ are correlated in Bayesian posteriors. The present work explains *why*: the FIM eigenvalue hierarchy predicts exactly this correlation structure from first principles, and the $\sim$3700-fold computational speedup enables the systematic survey that MCMC cannot.

4. *It delivers actionable experimental guidance.* The step-by-step decomposition (Fig. 4) and global analysis (Fig. 5) provide a clear priority ordering for rare-isotope beam facilities: multi-energy systematics $\gg$ $A_y$ $\gg$ $\sigma_R$. This is immediately relevant to experimental programs at FRIB, RIKEN, and other facilities where beam time allocation requires quantitative justification.

5. *It connects nuclear physics to a broader universality class.* Identifying the optical potential as a "sloppy model" links nuclear reaction theory to a well-established framework in systems biology, condensed matter, and particle physics [Waterfall *et al.*, PRL (2006); Machta *et al.*, Science (2013)]. This cross-disciplinary connection broadens the audience beyond nuclear physics.

---

## Minor Points

### (A) Abstract framing

> *First sentence is misleading---the actual scope is narrower.*

Revised. The abstract now opens with "How many independent parameter combinations can nucleon-nucleus scattering data actually constrain in the optical model potential?" and explicitly states the baseline constraint from elastic scattering ($D_\mathrm{eff} \approx 1.8 \pm 0.5$), the step-by-step decomposition with local and global parameter spaces clearly distinguished ($D_\mathrm{eff} \to 2.4$--$2.6$ in the 13-parameter local space; $D_\mathrm{eff} \approx 3.0$ out of 48 universal KD02 coefficients), and the priority ordering for augmenting elastic data: multi-energy systematics $\gg$ $A_y$ $\gg$ $\sigma_R$. Claims are scoped to the KD02 framework and the configurations sampled. The $A_y$ contribution is described qualitatively ("provides the next-largest gain through orthogonal spin-orbit constraints") rather than with a specific percentage, because the quantitative gain depends on the relative precision of elastic and $A_y$ data (see End Matter, Robustness checks, for the full $\epsilon$-dependence).

### (B) PCA undefined

> *"PCA" used without definition.*

Fixed: the full term "principal component analysis" is now used throughout (Introduction, paragraph 3); the abbreviation "PCA" is not used elsewhere so no parenthetical definition is needed.

### (C) Conclusion on nuclear forces

> *The link between "sloppy" parameters and chiral nuclear forces needs refinement.*

Revised. The Discussion now clarifies that the scale-invariance of $D_\mathrm{eff}$ means that "uniform error reduction cannot change the geometric anisotropy of the constraint structure," and that "only qualitatively different measurements (systematic multi-energy fitting, spin observables)" can partially lift it. The multi-energy analysis (Fig. 4) demonstrates that combining data across energies probes orthogonal directions in parameter space, providing a constructive path forward.

---

## Summary of Changes

### Title and scope
- Title changed from "Scale-Invariant Information Limit..." to "Intrinsic Information Limit in Nuclear Optical Potential Extraction"
- Scope expanded from single-energy elastic scattering (9 parameters) to a comprehensive multi-observable, multi-energy, multi-system analysis (13 local parameters and 48 universal KD02 coefficients)

### Method
- Replaced the neural network / automatic differentiation approach with a conventional Numerov solver validated against FRESCO ($|\Delta S| < 0.002$). The finite-difference gradients are verified to be fully converged: two-point and five-point stencils agree to three to four significant figures, and $D_\mathrm{eff}$ is stable over three orders of magnitude in step size (see End Matter). The neural network is no longer used or referenced.
- Extended from 9-parameter spin-0 central potential to **13-parameter spin-1/2 model** including spin-orbit coupling ($V_{so}$, $W_{so}$, $r_{so}$, $a_{so}$) with $(l,j)$-resolved $S$-matrix
- Added observables: analyzing power $A_y$ (via spin-flip amplitude $g(\theta)$), reaction cross section $\sigma_R$, and total cross section $\sigma_T$ (via optical theorem)

### New analyses
- **Step-by-step constraint decomposition** (new Fig. 4): progressively adds $\sigma_R$, $A_y$, $\sigma_T$, and multi-energy data, quantifying the information contribution of each
- **Multi-energy combined Fisher analysis** (7 energies, 10--200 MeV): $D_\mathrm{eff}$ increases from $\sim$1.2 to $\sim$2.4--2.6 in the 13-parameter local space, through the energy lever arm and KD02 energy dependence tying
- **KD02 global analysis in the 48-dimensional universal coefficient space** (new Fig. 5, Table I): $D_\mathrm{eff} = 3.0$ out of 48, with a complete inventory of all KD02 universal coefficients
- **Angle-resolved sensitivity** (new Fig. 6): normalized sensitivity directions $S_i(\theta)/\|\mathbf{S}_i\|$ for $V$, $r_v$, and $a_v$, plus cumulative $D_\mathrm{eff}(\theta_\mathrm{max})$
- **Fisher--Bayesian comparison** (End Matter): connects to King *et al.* [PRL **122**, 232502 (2019)], showing that the FIM eigenvector structure predicts their MCMC correlation patterns
- **Robustness checks** (End Matter): (i) FIM evaluation-point perturbation ($\pm 10\%$), (ii) $r_w = r_v$, $a_w = a_v$ constraint effect, (iii) $\epsilon$-scan of observable weighting

### Updated results
- $D_\mathrm{eff}$ updated from $1.7 \pm 0.4$ (9-param, NN) to $1.8 \pm 0.5$ (13-param, Numerov), consistent within uncertainties
- Eigenvector composition updated: dominant mode now $r_v$-dominated (62%) rather than $V$-dominated (87%), reflecting the 13-parameter analysis where the radius sensitivity $\partial\log J_V/\partial\log r_v = 3$ exceeds the depth sensitivity $\partial\log J_V/\partial\log V = 1$
- Experimental hierarchy established: multi-energy $\gg$ $A_y$ $\gg$ $\sigma_R$

### Writing
- Abstract rewritten with narrower, more precise framing
- Introduction expanded with additional references (Hodgson, Mahaux, Furnstahl, Phillips)
- "principal component analysis" spelled out (was undefined "PCA")
- Discussion revised: removed diffraction-limit analogy, added constructive path forward via multi-energy and spin observables
- End Matter added (numerical method, robustness, condition number interpretation, KD02 parameter hierarchy, global analysis, Fisher--Bayesian comparison)
- Sensitivity normalization explicitly specified ($\epsilon = 1$, $\delta_{A_y} = 0.03$) with quantitative $\epsilon$-dependence in End Matter
