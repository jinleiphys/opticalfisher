# Response to the Second Report of Referee A and the Report of Referee B: LM20073

*Manuscript: "Intrinsic information limit in nuclear optical potential extraction" (J. Lei, sole author).*

I thank both Referees for their careful reading. Referee A's second report raises one technical question about the treatment of forward- and backward-angle data in the relative error model; answering it uncovered a wording error in the End Matter, which I have corrected, and I am grateful for the catch. Referee B raises one conceptual concern (whether the optical potential belongs to the sloppy-model class) and one physics concern (that radii, diffuseness, and depths are constrained by established physical understanding), together with a recommendation about venue. I address each point below. Figure and section numbers refer to the revised manuscript.

Let me first restate what the manuscript does. It asks how many independent parameter combinations elastic nucleon-nucleus scattering itself determines in a global optical potential, and answers with the effective dimensionality $D_\mathrm{eff}$, the participation ratio of the Fisher information eigenvalues. Across 168 configurations ($A = 12$ to $208$, $E = 10$ to $200$ MeV, both projectiles) the answer is $D_\mathrm{eff} \approx 1.7 \pm 0.5$ out of 13 parameters, and the step-by-step decomposition identifies which additional data lift the limit and by how much: multi-energy systematics first, analyzing power second, $\sigma_R$ essentially not at all. All conclusions are scoped to the Koning-Delaroche (KD02) framework.

---

## Point 1 (Referee A): forward- and backward-angle data under the relative error model

> *Isn't the author assuming a relative error model, e.g., text around Eqs. (1) & (2)? Then the absolute magnitude of the cross section should scale out of the Fisher Information Matrix entries, right? A backward-angle point that is tiny in absolute cross section but highly shape-sensitive can matter just as much as a forward-angle point if measured to the same relative precision. This statement by the author makes me worried that the forward- and backward-angle data are not being treated on equal footing, even though in a relative error model they should be. This point needs to be addressed.*

The Referee is right, and I thank him for the catch. The sentence in question was a misstatement in the prose, not in the analysis. Every Fisher matrix in the paper is built from logarithmic derivatives $\partial \log\sigma / \partial \log p_i$, the relative error model of Eqs. (1) and (2); the absolute magnitude of the cross section cancels in every matrix element, so forward- and backward-angle points enter on exactly the same footing. The cumulative $D_\mathrm{eff}(\theta_\mathrm{max})$ of Fig. 6(b) was computed in this model from the start, so no number in the paper changes. What was wrong was the explanatory sentence, which attributed to absolute magnitude an effect whose actual origin is collinearity.

The quantitative answer, for the $n+{}^{40}$Ca 50 MeV case of Fig. 6, is as follows. First, in the relative error model the backward angles carry more information, not less: the root-mean-square logarithmic sensitivity of $r_v$ grows from 5.7 at forward angles to 16.5 at backward angles, the opposite of "negligible". Second, this large backward-angle information is almost entirely redundant: 96% of the backward-angle sensitivity of $a_v$ lies in the plane spanned by the $V$ and $r_v$ sensitivities, and the orthogonal remainder carries Fisher information of only $2\times10^{-3}$ of the leading eigenvalue. Third, and this settles the equal-footing worry directly, restricting the Fisher matrix to backward angles alone gives $D_\mathrm{eff} = 1.20$, against 1.19 for forward angles alone and 1.22 for full coverage: the degeneracy is present in every angular region separately and is not an artifact of how the regions are weighted.

Both offending sentences have been replaced by the collinearity argument with these numbers.

**Revised End Matter (angle-resolved sensitivity section):** *"Because the Fisher matrix is built from logarithmic derivatives, the absolute magnitude of the cross section cancels in every matrix element, and backward-angle points enter on exactly the same footing as forward-angle points; the relative sensitivities are in fact largest at backward angles (the root-mean-square of $S_{r_v}$ grows from 5.7 at $\theta < 100^\circ$ to 16.5 at $\theta \geq 100^\circ$). The reason the independent structure of $a_v$ contributes little to $D_\mathrm{eff}$ is collinearity, not magnitude: 96% of the backward-angle sensitivity of $a_v$ lies in the plane spanned by $\mathbf{S}_V$ and $\mathbf{S}_{r_v}$, and the orthogonal remainder carries Fisher information of only $2\times10^{-3}$ of the leading eigenvalue $\lambda_1$. Restricting the Fisher matrix to backward angles alone gives $D_\mathrm{eff} = 1.20$, essentially identical to the forward-angle value of 1.19 and the full-coverage value of 1.22: the degeneracy is present in every angular region separately, rather than being an artifact of how the regions are weighted."*

**Revised Fig. 6 caption:** *"However, this independent component is small: 96% of the backward-angle sensitivity of $a_v$ remains collinear with the $V$--$r_v$ plane (see text)."*

---

## Point 2 (Referee B): the sloppy-model classification

> *However already the abstract of such reference states "We explain why such systems so often are sloppy: the system behavior depends only on a few 'stiff' combinations of the parameters and is unchanged as other 'sloppy' parameter combinations vary by orders of magnitude." In the case of the optical model, parameters that cannot be uniquely fixed never change by orders of magnitude.*

I take the concern seriously, because the classification is central to the paper, and the abstract the Referee quotes does invite this reading. The criterion for sloppiness in the works the manuscript cites is a property of the Fisher (or Hessian) eigenvalue spectrum: eigenvalues spread roughly evenly over many decades, so that the data constrain a few stiff combinations and leave the rest undetermined. The manuscript's spectra satisfy this definition directly: the single-energy eigenvalue spectrum drops by seven orders of magnitude, the global 48-coefficient spectrum spans nine orders of magnitude below $\lambda_1$, and the condition numbers are of order $10^6$ or larger, often far above $10^{10}$.

The clause "sloppy parameter combinations vary by orders of magnitude" describes what the data permit, not what published fits display. The sloppy combinations can vary by orders of magnitude while the observables stay fixed; the manuscript states the optical-model version of this quantitatively in the End Matter:

**End Matter (condition number and parameter extraction):** *"If the data constrain the stiff combination (dominated by $r_v$) to 1%, the sloppiest combination is simultaneously constrained to only $\sqrt{\kappa} \times 1\% \approx 7000\%$, i.e., essentially unconstrained."*

This is a statement, in the local Fisher metric, about what the data determine. The Referee is right that fitted values do not wander in practice, and the reason is instructive: fitters pin the sloppy directions with starting values, priors, and systematics, all of which is information imported from outside the data being fit. The empirical record shows this mechanism at work. In the Bayesian-frequentist comparison of King *et al.* [Phys. Rev. Lett. 122, 232502 (2019), Ref. [13] of the manuscript], the unconstrained $\chi^2$ minimization drove imaginary-part parameters into unphysical territory, and the authors had to freeze them at physically plausible values. Stability of published parameters is evidence of external pinning, not of constraint by the elastic data alone.

The classification therefore follows the standard definition, applied to computed spectra.

---

## Point 3 (Referee B): radii, diffuseness, and depths as established constraints

> *Actually by trying to understand the physical meaning of the potential model parameters, beyond the technicalities of statistical models and/or quantification of uncertainties, many concepts have been clarified. Radii are now constraint by direct measurements, diffuseness has been shown to be linked to the separation energies of valence nucleons, and as a consequence depths can be constrained by systematical analysis of scattering in a large energy range.*

I agree with the physics history the Referee summarizes, and the manuscript is built on the same picture. Each of these constraints is real, and each is external to the elastic-scattering angular distribution whose information content the manuscript quantifies. That the community needs them is the manuscript's thesis: elastic scattering by itself determines about 1.7 parameter combinations, and the rest of what we know about the parameters is imported from other measurements and from theory.

On radii: electron scattering and laser spectroscopy determine the charge density and charge radius. Carrying that knowledge to the potential radius $r_v$ requires folding the density with an effective nucleon-nucleon interaction, the program founded by Greenlees, Pyle, and Tang [Phys. Rev. 171, 1115 (1968), Ref. [10] of the manuscript]. The folding step is model input; in Bayesian terms it is a prior on the direction that the elastic data leave flat. The eigenvector analysis of Fig. 3 shows why this import is necessary: the stiff combination is the volume integral $J_V \propto V r_v^3$, while $r_v$ separately is sloppy.

On diffuseness: the connection to separation energies operates through the asymptotic tail of the density or of the overlap function, or through dispersive self-energy fits that include bound-state information [Mahaux and Sartor, Ref. [6] of the manuscript]. Both are again channels of external information; neither is an independent determination of the Woods-Saxon $a_v$ from a single elastic angular distribution. Heavy-ion systems provide an independent cautionary example: fusion and elastic scattering demand mutually incompatible Woods-Saxon diffuseness values [Newton *et al.*, Phys. Rev. C 70, 024605 (2004), now cited], a different sector but the same phenomenon of observable-class dependence that the manuscript quantifies for the nucleon-nucleus case.

On depths from wide-energy-range systematics: this is computed in the manuscript. Step 4 of the analysis shows that multi-energy systematics is the largest single improvement available, raising $D_\mathrm{eff}$ from 1.7 to about 2.4 in the 13-parameter local space and to 3.0 out of 48 universal coefficients. The Referee's sentence is the qualitative version of Fig. 4.

The submitted version left this complementarity implicit; the Discussion now states it directly:

**Revised Discussion:** *"This is consistent with long-standing practice: the physical meaning attached to optical-model parameters has historically been imported from outside elastic scattering, through matter densities folded with an effective nucleon-nucleon interaction or dispersive constraints linking real and imaginary parts across energies; in Bayesian language, these are priors on the directions that the elastic data leave unconstrained. The observable-class dependence found here has a well-known heavy-ion counterpart, where fusion and elastic scattering demand incompatible Woods-Saxon diffuseness values."*

---

## Point 4 (Referee B): scope of the conclusions

> *This paper studies only the case of the Koning-Delaroche potential for nucleon-nucleus scattering and I do not agree with the "extreme" conclusions it reaches. However it is a valuable application of interesting statistical techniques. As such I deem it more appropriate to be published in a more specialized journal, provided the physical conclusions are amended along the lines I suggested above.*

The conclusions are scoped to what was computed, and the qualifier appears throughout the abstract, the Discussion, and the conclusions, for example:

**Discussion:** *"Within the KD02 optical model framework, single-energy elastic scattering constrains only $D_\mathrm{eff} \approx 1.7$ out of 13 parameter combinations; systematic multi-energy fitting raises $D_\mathrm{eff}$ to only $\sim$2.4 on average (up to $\sim$3 for the lightest systems) in the 13-parameter local space, and $D_\mathrm{eff} \approx 3.0$ out of 48 in the universal coefficient space."*

The manuscript does not claim that optical-model parameters can never be determined. It separates what elastic scattering determines by itself from what must be imported from other measurements and theory, and it ranks the imports by their information gain. That is the same physics picture the Referee describes; the disagreement is confined to the statistical terminology addressed in Point 2. The amendment the Referee asks for, an explicit acknowledgment that external physics constrains what elastic data do not, is now stated in the Discussion (Point 3). Verification with global parameterizations beyond KD02 is flagged in the manuscript as a natural extension.

---

## Summary of changes

| # | Referee point | Section(s) modified |
|---|---|---|
| 1 | Backward-angle Fisher information under the relative error model (A) | End Matter angle-resolved paragraph and Fig. 6 caption rewritten: collinearity explanation with quantitative decomposition (96% collinear, orthogonal remainder $2\times10^{-3}\,\lambda_1$, backward-only $D_\mathrm{eff}=1.20$ vs forward-only 1.19 vs full 1.22) |
| 2 | Sloppy-model classification (B) | No change; the eigenvalue-spectrum definition and its quantitative optical-model form are already in the manuscript (End Matter, condition-number passage) |
| 3 | Radii, diffuseness, depths as constraints (B) | Discussion: two sentences added framing density-folding and dispersive constraints as complementary priors; new reference Newton *et al.*, Phys. Rev. C 70, 024605 (2004) |
| 4 | Scope of conclusions (B) | No change; KD02 scoping statements present throughout |
