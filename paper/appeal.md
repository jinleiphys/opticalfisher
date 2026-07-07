# Appeal on Manuscript LM20073

*"Intrinsic information limit in nuclear optical potential extraction", J. Lei (sole author). Addressed to the Editors of Physical Review Letters.*

I ask the Editors to reconsider the decision on LM20073 through the formal appeal procedure. The grounds are specific. Referee A, after a full exchange, finds the work "comprehensive and quite convincing" and states that it "could be broadly impactful across a large subfield of the nuclear physics reaction community"; the single technical question in the second report is answered below, and it turned out to concern a wording error, not an analysis error. Referee B does not dispute any calculation and calls the work "a valuable application of interesting statistical techniques"; the recommendation to publish elsewhere rests on two objections which I address point by point: the reading of the sloppy-model criterion, where the manuscript follows the standard Fisher-spectrum definition of the works it cites, and a list of established parameter constraints that draw on information beyond a single-energy elastic angular distribution, which is precisely the separation the manuscript quantifies. Figure and section numbers below refer to the revised manuscript.

Let me restate plainly what the manuscript does. It asks how many independent parameter combinations elastic nucleon-nucleus scattering itself determines in a global optical potential, and answers with the effective dimensionality $D_\mathrm{eff}$, the participation ratio of the Fisher information eigenvalues. Across 168 configurations ($A = 12$ to $208$, $E = 10$ to $200$ MeV, both projectiles) the answer is $D_\mathrm{eff} \approx 1.7 \pm 0.5$ out of 13 parameters, and the decomposition identifies which additional data lift the limit and by how much: multi-energy systematics first, analyzing power second, $\sigma_R$ essentially not at all. Everything is scoped to the Koning-Delaroche (KD02) framework.

---

## Point 1 (Referee A): forward- and backward-angle data under the relative error model

> *Isn't the author assuming a relative error model, e.g., text around Eqs. (1) & (2)? Then the absolute magnitude of the cross section should scale out of the Fisher Information Matrix entries, right? A backward-angle point that is tiny in absolute cross section but highly shape-sensitive can matter just as much as a forward-angle point if measured to the same relative precision. This statement by the author makes me worried that the forward- and backward-angle data are not being treated on equal footing, even though in a relative error model they should be. This point needs to be addressed.*

The Referee is right, and I am grateful for the catch. The sentence in question was a misstatement in the prose, not in the analysis. Every Fisher matrix in the paper is built from logarithmic derivatives $\partial \log\sigma / \partial \log p_i$, the relative error model of Eqs. (1) and (2); the absolute magnitude of the cross section cancels in every matrix element, so forward- and backward-angle points enter on exactly the same footing. The cumulative $D_\mathrm{eff}(\theta_\mathrm{max})$ of Fig. 6(b) was computed in this model from the start, so no number in the paper changes. What was wrong was the explanatory sentence, which attributed to absolute magnitude an effect whose actual origin is collinearity.

The quantitative answer, for the $n+{}^{40}$Ca 50 MeV case of Fig. 6, is as follows. First, in the relative error model the backward angles carry more information, not less: the root-mean-square logarithmic sensitivity of $r_v$ grows from 5.7 at forward angles to 16.5 at backward angles, the opposite of "negligible". Second, this large backward-angle information is almost entirely redundant: 96% of the backward-angle sensitivity of $a_v$ lies in the plane spanned by the $V$ and $r_v$ sensitivities, and the orthogonal remainder carries Fisher information of only $2\times10^{-3}$ of the leading eigenvalue. Third, and this settles the equal-footing worry directly, restricting the Fisher matrix to backward angles alone gives $D_\mathrm{eff} = 1.20$, against 1.19 for forward angles alone and 1.22 for full coverage: the degeneracy is present in every angular region separately and is not an artifact of how the regions are weighted.

Both offending sentences have been replaced by the collinearity argument with these numbers.

**Revised End Matter (angle-resolved sensitivity section):** *"Because the Fisher matrix is built from logarithmic derivatives, the absolute magnitude of the cross section cancels in every matrix element, and backward-angle points enter on exactly the same footing as forward-angle points; the relative sensitivities are in fact largest at backward angles (the root-mean-square of $S_{r_v}$ grows from 5.7 at $\theta < 100^\circ$ to 16.5 at $\theta \geq 100^\circ$). The reason the independent structure of $a_v$ contributes little to $D_\mathrm{eff}$ is collinearity, not magnitude: 96% of the backward-angle sensitivity of $a_v$ lies in the plane spanned by $\mathbf{S}_V$ and $\mathbf{S}_{r_v}$, and the orthogonal remainder carries Fisher information of only $2\times10^{-3}$ of the leading eigenvalue $\lambda_1$. Restricting the Fisher matrix to backward angles alone gives $D_\mathrm{eff} = 1.20$, essentially identical to the forward-angle value of 1.19 and the full-coverage value of 1.22: the degeneracy is present in every angular region separately, rather than being an artifact of how the regions are weighted."*

**Revised Fig. 6 caption:** *"However, this independent component is small: 96% of the backward-angle sensitivity of $a_v$ remains collinear with the $V$--$r_v$ plane (see text)."*

---

## Point 2 (Referee B): the sloppy-model classification

> *However already the abstract of such reference states "We explain why such systems so often are sloppy: the system behavior depends only on a few 'stiff' combinations of the parameters and is unchanged as other 'sloppy' parameter combinations vary by orders of magnitude." In the case of the optical model, parameters that cannot be uniquely fixed never change by orders of magnitude.*

The criterion for sloppiness in the works the manuscript cites is a property of the Fisher (or Hessian) eigenvalue spectrum: eigenvalues spread roughly evenly over many decades, so that the data constrain a few stiff combinations and leave the rest undetermined. The manuscript's spectra satisfy this definition directly: the single-energy eigenvalue spectrum drops by seven orders of magnitude, the global 48-coefficient spectrum spans nine orders of magnitude below $\lambda_1$, and the condition numbers are of order $10^6$ or larger, reaching above $10^{10}$.

The clause the Referee quotes, "sloppy parameter combinations vary by orders of magnitude", describes the sloppy combinations: in the local Fisher metric, the data permit uncertainties orders of magnitude larger along these directions than along the stiff ones. It does not describe a drift of published best-fit values. The manuscript already states the optical-model version of this quantitatively, in the End Matter:

**End Matter (condition number and parameter extraction):** *"If the data constrain the stiff combination (dominated by $r_v$) to 1%, the sloppiest combination is simultaneously constrained to only $\sqrt{\kappa} \times 1\% \approx 7000\%$, i.e., essentially unconstrained."*

This is a statement, in the local Fisher metric, about what the data determine; it does not claim that fitted values wander in practice. They do not, and the reason is instructive: fitters pin the sloppy directions with starting values, priors, and systematics, all of which is information imported from outside the data being fit. The empirical record shows exactly this mechanism at work. In the Bayesian-frequentist comparison of King *et al.* [Phys. Rev. Lett. 122, 232502 (2019)], the prior-free $\chi^2$ minimization drove imaginary-part parameters into unphysical territory, and the authors had to freeze them at physically plausible values. In the same study, the frequentist fits for $^{90}$Zr, initialized from the Becchetti-Greenlees global values, returned proton and neutron geometry parameters so different from each other as to be physically implausible, and the authors had to initialize the proton fit from the neutron result. The stability of published parameters thus reflects the physical systematics, density information, and dispersive constraints that enter alongside the elastic data, rather than a constraint delivered by a single-energy elastic angular distribution alone.

To prevent this ambiguity in the manuscript itself, I have added an explicit sentence to the Introduction:

**Revised Introduction:** *"Here ``sloppy'' refers to the hierarchy of Fisher eigenvalues, not to a requirement that published best-fit parameters drift by orders of magnitude; external physical constraints and global systematics can pin directions that one observable class leaves flat."*

The classification therefore follows the standard definition, applied to computed spectra.

---

## Point 3 (Referee B): radii, diffuseness, and depths as established constraints

> *Actually by trying to understand the physical meaning of the potential model parameters, beyond the technicalities of statistical models and/or quantification of uncertainties, many concepts have been clarified. Radii are now constraint by direct measurements, diffuseness has been shown to be linked to the separation energies of valence nucleons, and as a consequence depths can be constrained by systematical analysis of scattering in a large energy range.*

The key distinction is not whether these constraints exist; they do. The distinction is whether they come from a single-energy elastic angular distribution alone. Radii and diffuseness constraints import density, bound-state, or dispersive information, while depth constraints from wide-energy-range systematics are precisely the multi-energy information source quantified in Step 4. That the community needs these inputs is the manuscript's thesis, not a counterexample to it. Moreover, the mapping from these external inputs to Woods-Saxon radii, diffusenesses, and depths is conditional on the assumed density, folding interaction, dispersive ansatz, or global energy dependence.

The revised Abstract and Introduction now state the practical motivation explicitly. Optical-potential UQ is more than a statistical exercise on fitted parameters. In complex reaction calculations, especially for weakly bound and rare-isotope systems, disagreement with data is ambiguous until one can decide whether the dominant error comes from the optical interaction, the reaction mechanism, the structure input, or missing channels. Recent reaction-theory UQ work and the FRIB-TA optical-potential program make this point explicitly.

This distinction is not hypothetical. In the (d,p) UQ study of King, Lovell, and Nunes [Phys. Rev. C 98, 044623 (2018)], once correlated optical-potential uncertainties were propagated, the transfer data could no longer discriminate ADWA from DWBA. That is exactly the situation in which one needs an optical-potential error budget before assigning a disagreement to the reaction mechanism.

**Revised Abstract:** *"For downstream reaction calculations, this identifies which optical-potential directions must be pinned or propagated before discrepancies are assigned to reaction or structure physics."*

**Revised Introduction:** *"The motivation goes beyond attaching error bars to optical-model fits: in transfer, breakup, charge-exchange, and other reactions involving weakly bound or rare isotopes, a disagreement with data cannot be interpreted until the error budget separates optical-potential input from reaction-mechanism approximations, structure overlaps, and missing channels~\cite{King2018,Lovell2021}. This need is explicit in the rare-isotope-beam literature, where quantifying and reducing reaction-model uncertainties, especially those associated with nuclear optical potentials, has been identified as a targeted priority~\cite{Hebborn2023}."*

On radii: electron scattering and laser spectroscopy determine the charge density and charge radius. Carrying that knowledge to the potential radius $r_v$ requires folding the density with an effective nucleon-nucleon interaction, the program founded by Greenlees, Pyle, and Tang [Phys. Rev. 171, 1115 (1968), already cited in the manuscript]. The folding step is model input; in Bayesian terms it is a prior on the direction that the elastic data leave flat. The manuscript's eigenvector analysis shows precisely why this import is necessary: the stiff combination is the volume integral $J_V \propto V r_v^3$, while $r_v$ separately is sloppy.

On diffuseness: the connection to separation energies operates through the asymptotic tail of the density or of the overlap function, or through dispersive self-energy fits that include bound-state information [Mahaux and Sartor, Adv. Nucl. Phys. 20, 1 (1991)]. Both are again channels of external information; neither is an independent determination of the Woods-Saxon $a_v$ from a single elastic angular distribution. Heavy-ion systems provide an independent cautionary example: fusion and elastic scattering demand mutually incompatible Woods-Saxon diffuseness values [Newton *et al.*, Phys. Rev. C 70, 024605 (2004)], a different sector but the same phenomenon of observable-class dependence that the manuscript quantifies for the nucleon-nucleus case.

On depths from wide-energy-range systematics: this is computed in the manuscript. Step 4 of the analysis shows that multi-energy systematics is the largest single improvement available, raising $D_\mathrm{eff}$ from 1.7 to about 2.4 in the 13-parameter local space (Fig. 4 of the manuscript); Step 5 extends the combination to all 168 systems in the 48-coefficient universal space, where $D_\mathrm{eff}$ reaches 3.0 (Fig. 5). The Referee's sentence is the qualitative version of this analysis.

To make this complementarity explicit rather than implicit, the Discussion now states it directly:

**Revised Discussion:** *"This is consistent with long-standing practice: the physical meaning attached to optical-model parameters has historically been imported from outside a single-energy elastic angular distribution, through matter densities folded with an effective nucleon-nucleon interaction or dispersive constraints linking real and imaginary parts across energies; in Bayesian language, these are priors on the directions that the isolated angular distribution leaves unconstrained. Such external constraints are valuable, but the resulting constraints on Woods-Saxon radii, diffusenesses, and depths are conditional on the assumed density, folding interaction, dispersive ansatz, or global energy dependence, rather than being model-independent information contained in one angular distribution. The observable-class dependence found here has a well-known heavy-ion counterpart, where fusion and elastic scattering demand incompatible Woods-Saxon diffuseness values."*

**Revised closing paragraph:** *"In practical terms, an elastic fit with a small $\chi^2$ does not by itself determine the optical input needed for a transfer or breakup calculation; the unconstrained directions must either be pinned by external physics or propagated as uncertainty to the reaction observable."*

---

## Point 4 (Referee B): scope and the "extreme" conclusions

> *This paper studies only the case of the Koning-Delaroche potential for nucleon-nucleus scattering and I do not agree with the "extreme" conclusions it reaches. However it is a valuable application of interesting statistical techniques. As such I deem it more appropriate to be published in a more specialized journal, provided the physical conclusions are amended along the lines I suggested above.*

The conclusions are scoped to what was computed. The abstract, the Discussion, and the conclusions all carry the qualifier explicitly, for example:

**Discussion:** *"Within the KD02 optical model framework, single-energy elastic scattering constrains only $D_\mathrm{eff} \approx 1.7$ out of 13 parameter combinations; systematic multi-energy fitting raises $D_\mathrm{eff}$ to only $\sim$2.4 on average (up to $\sim$3 for the lightest systems) in the 13-parameter local space, and $D_\mathrm{eff} \approx 3.0$ out of 48 in the universal coefficient space."*

The manuscript nowhere claims that optical-model parameters can never be determined. It separates what a single-energy angular distribution determines, what multi-energy systematics adds, and what is imported from other measurements and theory, then ranks these inputs by their information gain. That is the same physics picture the Referee describes in the report; the disagreement is confined to statistical terminology, which Point 2 addresses. The amendment the report asks for, an acknowledgment that external physics constrains what isolated elastic angular distributions do not, is in substance already in the manuscript and is now explicit (Point 3).

---

## Why this belongs in Physical Review Letters

Referee A's report makes the impact case directly: the work "effectively packages" the known degeneracies "into a compact quantitative metric $D_\mathrm{eff}$", and "given the wide use of optical potentials by both theorists and experimentalists, I believe this work could be broadly impactful across a large subfield of the nuclear physics reaction community." Optical potentials enter essentially every reaction calculation used at FRIB, FAIR, and HIAF. For these systems, fitting an elastic angular distribution is only one part of the problem; the harder task is to assign the source of a disagreement in a downstream reaction calculation. The manuscript gives the first quantitative characterization of a nearly seventy-year-old ambiguity, identifies the optical potential as a member of the sloppy-model universality class alongside systems biology and nuclear density functionals, and provides an actionable ranking of which measurements add information. Referee B does not identify a numerical inconsistency in the Fisher analysis, and the physical framing the report asks for is now stated explicitly in the revised manuscript.

I therefore ask that the manuscript be reconsidered on the basis of the point-by-point response above.

---

## Summary of changes

| # | Referee point | Change in the revised manuscript |
|---|---|---|
| 1 | Backward-angle Fisher information under the relative error model (A) | End Matter angle-resolved paragraph and Fig. 6 caption rewritten: collinearity explanation with quantitative decomposition (96% collinear, orthogonal remainder $2\times10^{-3}\,\lambda_1$, backward-only $D_\mathrm{eff}=1.20$ vs forward-only 1.19 vs full 1.22) |
| 2 | Sloppy-model classification (B) | Introduction sentence added to define "sloppy" as a Fisher-spectrum statement, not a claim that published best-fit parameters must drift by orders of magnitude; End Matter condition-number passage retained |
| 3 | Radii, diffuseness, depths as constraints (B) | Abstract and Introduction add the error-source motivation for optical-potential UQ in transfer, breakup, charge-exchange, and rare-isotope reactions; Discussion frames density-folding and dispersive constraints as complementary priors and adds the practical point that a small elastic $\chi^2$ does not determine the optical input needed for a downstream reaction calculation; new reference Newton *et al.*, Phys. Rev. C 70, 024605 (2004) |
| 4 | Scope of conclusions (B) | No additional manuscript change beyond the KD02 scoping statements in the Abstract and Discussion and the clarifications added under Points 2 and 3 |
