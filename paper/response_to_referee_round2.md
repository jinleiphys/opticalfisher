# Response to the Second Report of Referee A and the Report of Referee B: LM20073

*Manuscript: "Intrinsic information limit in nuclear optical potential extraction" (J. Lei, sole author).*

I thank both Referees for their careful reading. Referee A's second report raises one technical question about the treatment of forward- and backward-angle data in the relative error model; answering it uncovered a wording error in the End Matter, which I have corrected, and I am grateful for the catch. Referee B raises one conceptual concern (whether the optical potential belongs to the sloppy-model class) and one physics concern (that radii, diffuseness, and depths are constrained by established physical understanding), together with a recommendation about venue. I address each point below. Figure and section numbers refer to the revised manuscript.

Let me first restate what the manuscript does. It asks how many independent parameter combinations elastic nucleon-nucleus scattering itself determines in a global optical potential, and answers with the effective dimensionality $D_\mathrm{eff}$, the participation ratio of the Fisher information eigenvalues. Across 168 configurations ($A = 12$ to $208$, $E = 10$ to $200$ MeV, both projectiles) the answer is $D_\mathrm{eff} \approx 1.7 \pm 0.5$ out of 13 parameters. The step-by-step decomposition then identifies which additional data open independent parameter directions: multi-energy systematics first, analyzing power second, $\sigma_R$ essentially not at all. This is a ranking of spectral broadening, not of total Fisher information. All conclusions are scoped to the Koning-Delaroche (KD02) framework and the stated error models.

---

## Point 1 (Referee A): forward- and backward-angle data under the relative error model

> *Isn't the author assuming a relative error model, e.g., text around Eqs. (1) & (2)? Then the absolute magnitude of the cross section should scale out of the Fisher Information Matrix entries, right? A backward-angle point that is tiny in absolute cross section but highly shape-sensitive can matter just as much as a forward-angle point if measured to the same relative precision. This statement by the author makes me worried that the forward- and backward-angle data are not being treated on equal footing, even though in a relative error model they should be. This point needs to be addressed.*

The Referee is right, and I am grateful for the catch. The sentence in question was a misstatement in the prose, not in the analysis. Every Fisher matrix in the paper is built from logarithmic derivatives $\partial \log\sigma / \partial \log p_i$, the relative error model of Eqs. (1) and (2); the absolute magnitude of the cross section cancels in every matrix element, so forward- and backward-angle points enter on exactly the same footing. The cumulative $D_\mathrm{eff}(\theta_\mathrm{max})$ of Fig. 6(b) was computed in this model from the start, so no number in the paper changes. What was wrong was the explanatory sentence, which attributed to absolute magnitude an effect whose actual origin is collinearity.

The quantitative answer, for the $n+{}^{40}$Ca 50 MeV case of Fig. 6, is as follows. First, backward-angle points are not suppressed by their smaller cross sections in the relative error model: the root-mean-square logarithmic sensitivity of $r_v$ grows from 5.7 at forward angles to 16.5 at backward angles. Second, this large sensitivity is almost entirely redundant: 96% of the squared norm of the backward-angle $a_v$ sensitivity lies in the plane spanned by the $V$ and $r_v$ sensitivities, and the orthogonal remainder carries Fisher information of only $2\times10^{-3}$ of the leading eigenvalue. Third, restricting the Fisher matrix to backward angles alone gives $D_\mathrm{eff} = 1.20$, against 1.19 for forward angles alone and 1.22 for full coverage. The degeneracy is present in every angular region separately and is not an artifact of how the regions are weighted.

Both offending sentences have been replaced by the collinearity argument with these numbers.

**Revised End Matter (angle-resolved sensitivity section):** *"Because the Fisher matrix is built from logarithmic derivatives, the absolute magnitude of the cross section cancels in every matrix element, and backward-angle points are not down-weighted by their smaller cross sections; the root-mean-square logarithmic sensitivity of $r_v$ in fact grows from 5.7 at $\theta < 100^\circ$ to 16.5 at $\theta \geq 100^\circ$. The reason the independent structure of $a_v$ contributes little to $D_\mathrm{eff}$ is collinearity, not magnitude: 96% of the squared norm of the backward-angle $a_v$ sensitivity lies in the plane spanned by $\mathbf{S}_V$ and $\mathbf{S}_{r_v}$, and the orthogonal remainder carries Fisher information of only $2\times10^{-3}$ of the leading eigenvalue $\lambda_1$. Restricting the Fisher matrix to backward angles alone gives $D_\mathrm{eff} = 1.20$, essentially identical to the forward-angle value of 1.19 and the full-coverage value of 1.22: the degeneracy is present in every angular region separately, rather than being an artifact of how the regions are weighted."*

**Revised Fig. 6 caption:** *"However, this independent component is small: 96% of the squared norm of the backward-angle $a_v$ sensitivity lies in the $V$--$r_v$ plane (see text)."*

---

## Point 2 (Referee B): the sloppy-model classification

> *However already the abstract of such reference states "We explain why such systems so often are sloppy: the system behavior depends only on a few 'stiff' combinations of the parameters and is unchanged as other 'sloppy' parameter combinations vary by orders of magnitude." In the case of the optical model, parameters that cannot be uniquely fixed never change by orders of magnitude.*

I take the concern seriously, because the classification is central to the paper, and the abstract the Referee quotes does invite this reading. The calculation establishes a local spectral statement: in logarithmic KD02 coordinates, the single-energy Fisher spectrum drops by seven orders of magnitude, the global 48-coefficient spectrum spans nine orders of magnitude below $\lambda_1$, and the condition numbers are of order $10^6$ or larger, reaching above $10^{10}$.

The Referee is right that this local calculation does not establish how far a parameter combination can move on the global model manifold. I have therefore restricted the claim to the standard local Fisher-spectrum signature and revised the End Matter to state what the condition number does, and does not, imply:

**Revised End Matter (condition number and parameter extraction):** *"This ratio is a diagnostic of the local anisotropy, not a claim that the quadratic approximation remains valid over a literal 7000% displacement; establishing the global range requires a profile-likelihood or model-manifold calculation."*

Published fits nevertheless do not wander in practice. Starting values, physical constraints, shared parameter relations, and global systematics stabilize directions that one data class leaves flat. The empirical record shows this mechanism at work. In the Bayesian-frequentist comparison of King *et al.* [Phys. Rev. Lett. 122, 232502 (2019)], the prior-free $\chi^2$ minimization drove imaginary-part parameters into unphysical territory, and the authors had to freeze them at physically plausible values. In the same study, the frequentist fits for $^{90}$Zr, initialized from the Becchetti-Greenlees global values, returned proton and neutron geometry parameters so different from each other as to be physically implausible, and the authors had to initialize the proton fit from the neutron result. The stability of published parameters therefore reflects information and constraints entering alongside an isolated elastic angular distribution.

To prevent this ambiguity in the manuscript itself, I have added an explicit sentence to the Introduction:

**Revised Introduction:** *"This is a local spectral classification, not a requirement that published best-fit parameters drift by orders of magnitude; physical constraints, shared parameter relations, and global systematics can pin directions that one observable class leaves flat."*

The classification is therefore the standard local Fisher-spectrum diagnosis, applied in logarithmic KD02 coordinates and for the stated error models. The manuscript no longer uses it as evidence for a literal global parameter excursion.

---

## Point 3 (Referee B): radii, diffuseness, and depths as established constraints

> *Actually by trying to understand the physical meaning of the potential model parameters, beyond the technicalities of statistical models and/or quantification of uncertainties, many concepts have been clarified. Radii are now constraint by direct measurements, diffuseness has been shown to be linked to the separation energies of valence nucleons, and as a consequence depths can be constrained by systematical analysis of scattering in a large energy range.*

I agree with the physics history the Referee summarizes, and the manuscript is built on the same picture. The key distinction is not whether these constraints exist; they do. The distinction is which observable or model relation supplies them. Radii and diffuseness constraints can enter through density information, bound-state asymptotics, folding structure, or dispersive relations, while depth constraints from wide-energy-range systematics are precisely the multi-energy source quantified in Step 4. A single-energy angular distribution by itself determines about 1.7 parameter combinations; additional measurements, theory, and shared parameter relations constrain further combinations. The mapping to Woods-Saxon radii, diffusenesses, and depths remains conditional on the assumed density, folding interaction, dispersive ansatz, or global energy dependence.

I have also made the motivation for this separation explicit in the Abstract and Introduction. Optical-potential UQ is more than a statistical exercise on fitted parameters. In complex reaction calculations, especially for weakly bound and rare-isotope systems, disagreement with data is ambiguous until one can decide whether the dominant error comes from the optical interaction, the reaction mechanism, the structure input, or missing channels. Recent reaction-theory UQ work and the FRIB-TA optical-potential program make this point explicitly.

This distinction is not hypothetical. In the (d,p) UQ study of King, Lovell, and Nunes [Phys. Rev. C 98, 044623 (2018)], once correlated optical-potential uncertainties were propagated, the transfer data could no longer discriminate ADWA from DWBA. That is exactly the situation in which one needs an optical-potential error budget before assigning a disagreement to the reaction mechanism.

**Revised Abstract:** *"For downstream reaction calculations, this identifies which optical-potential directions must be pinned or propagated before discrepancies are assigned to reaction or structure physics."*

**Revised Introduction:** *"The motivation goes beyond attaching error bars to optical-model fits: in transfer, breakup, charge-exchange, and other reactions involving weakly bound or rare isotopes, a disagreement with data cannot be interpreted until the error budget separates optical-potential input from reaction-mechanism approximations, structure overlaps, and missing channels~\cite{King2018,Lovell2021}. This need is explicit in the rare-isotope-beam literature, where quantifying and reducing reaction-model uncertainties, especially those associated with nuclear optical potentials, has been identified as a targeted priority~\cite{Hebborn2023}."*

On radii: electron scattering and laser spectroscopy determine charge-density information and charge radii, not a Woods-Saxon potential radius directly. Greenlees, Pyle, and Tang [Phys. Rev. 171, 1115 (1968)] showed the more precise point relevant here: a folding representation can determine an rms matter-radius combination rather stably even while its underlying radius and diffuseness parameters trade off. Folding therefore supplies model structure that reorganizes the poorly determined Woods-Saxon coordinates into a better-constrained physical combination. This is consistent with the eigenvector analysis of Fig. 3, where the stiff direction follows the volume integral $J_V \propto V r_v^3$ while the orthogonal depth-radius tradeoff remains sloppy.

On diffuseness: the separation-energy argument enters through the asymptotic tail of a bound-state density or overlap, while dispersive optical models connect positive- and negative-energy information through the self-energy [Mahaux and Sartor, Adv. Nucl. Phys. 20, 1 (1991)]. These are model-dependent relations, not a one-to-one determination of the Woods-Saxon $a_v$ from a single elastic angular distribution. The manuscript cites Mahaux and Sartor for the dispersive relation, not as evidence for a universal separation-energy-to-diffuseness map. Heavy-ion systems provide an independent cautionary example: fusion and elastic scattering demand incompatible Woods-Saxon diffuseness values [Newton *et al.*, Phys. Rev. C 70, 024605 (2004), now cited], a different sector but the same phenomenon of observable-class dependence quantified here for nucleon-nucleus scattering.

The fully nonlocal dispersive analysis of $^{40}$Ca by Mahzoon *et al.* [Phys. Rev. Lett. 112, 162503 (2014), now cited] provides a direct nucleon-nucleus test. Local and nonlocal absorptive potentials reproduce equivalent elastic differential cross sections but yield different angular-momentum absorption profiles, and only the nonlocal form simultaneously recovers particle number and charge density. Elastic equivalence therefore does not imply that the structure-bearing self-energy has been identified. This supports the Referee's point that established physical information matters, while showing exactly what that information adds beyond one elastic observable class.

On depths from wide-energy-range systematics: this is computed in the manuscript. Step 4 of the analysis shows that multi-energy systematics is the largest single improvement available, raising $D_\mathrm{eff}$ from 1.7 to about 2.4 in the 13-parameter local space (Fig. 4); Step 5 extends the combination to all 168 systems in the 48-coefficient universal space, where $D_\mathrm{eff}$ reaches 3.0 (Fig. 5). The wide-energy systematics emphasized by the Referee are therefore the largest source of additional independent information found in the calculation.

The submitted version left this complementarity implicit; the Discussion now states it directly:

**Revised Discussion (citations omitted):** *"This is consistent with long-standing practice: physical interpretations of optical-model parameters are supplied by additional observables and by model structure, through matter densities folded with an effective nucleon-nucleon interaction or dispersive relations linking real and imaginary parts across energies. These inputs can constrain combinations that an isolated angular distribution leaves flat. A direct nucleon-nucleus example is the nonlocal dispersive analysis of $^{40}$Ca: local and nonlocal absorptive potentials reproduce equivalent elastic differential cross sections but give different angular-momentum absorption profiles, and only the nonlocal form also recovers particle number and charge density. The resulting constraints on Woods-Saxon radii, diffusenesses, and depths remain conditional on the assumed density, folding interaction, dispersive ansatz, or global energy dependence, rather than being model-independent information contained in one angular distribution. The observable-class dependence found here has a well-known heavy-ion counterpart, where fusion and elastic scattering demand incompatible Woods-Saxon diffuseness values."*

**Revised closing paragraph:** *"In practical terms, an elastic fit with a small $\chi^2$ does not by itself determine the optical input needed for a transfer or breakup calculation; the unconstrained directions must either be pinned by external physics or propagated as uncertainty to the reaction observable."*

---

## Point 4 (Referee B): scope of the conclusions

> *This paper studies only the case of the Koning-Delaroche potential for nucleon-nucleus scattering and I do not agree with the "extreme" conclusions it reaches. However it is a valuable application of interesting statistical techniques. As such I deem it more appropriate to be published in a more specialized journal, provided the physical conclusions are amended along the lines I suggested above.*

The conclusions are scoped to what was computed, and the qualifier appears throughout the abstract, the Discussion, and the conclusions, for example:

**Discussion:** *"Within the KD02 optical model framework, single-energy elastic scattering constrains only $D_\mathrm{eff} \approx 1.7$ out of 13 parameter combinations; systematic multi-energy fitting raises $D_\mathrm{eff}$ to only $\sim$2.4 on average (up to $\sim$3 for the lightest systems) in the 13-parameter local space, and $D_\mathrm{eff} \approx 3.0$ out of 48 in the universal coefficient space."*

The manuscript does not claim that optical-model parameters can never be determined. It separates what a single-energy angular distribution determines, what multi-energy systematics adds, and what additional observables or model relations supply. It then ranks these inputs by how effectively they broaden the constrained subspace, not by total Fisher information. That is the same physics picture the Referee describes. The amendment the Referee asks for, an explicit acknowledgment that further physics constrains what isolated elastic angular distributions do not, is now stated in the Discussion (Point 3). Verification with global parameterizations beyond KD02 is flagged as a natural extension.

---

## Why the revised manuscript remains appropriate for Physical Review Letters

Referee A's second report is directly relevant to the venue question. After the first-round expansion, Referee A describes the work as "comprehensive and quite convincing" and states that it "could be broadly impactful across a large subfield of the nuclear physics reaction community." I respectfully agree with that assessment. Optical potentials enter essentially every reaction calculation used at FRIB, FAIR, HIAF, and related rare-isotope programs. For these systems, fitting an elastic angular distribution is only one part of the problem; the harder task is to assign the source of a disagreement in a downstream reaction calculation. The manuscript maps the effective dimensionality of the Igo ambiguity across 168 configurations and provides an experimental hierarchy for opening independent parameter directions.

The revised manuscript now makes the interpretive limits explicit: ``sloppy'' is a local spectral diagnosis in logarithmic KD02 coordinates, and radii, diffusenesses, and depths acquire further constraints through additional observables and model relations. Within that scope, the result is a compact statement about the structural information bottleneck in optical-potential extraction.

---

## Summary of changes

| # | Referee point | Section(s) modified |
|---|---|---|
| 1 | Backward-angle Fisher information under the relative error model (A) | End Matter and Fig. 6 caption rewritten: relative-error weighting and collinearity stated explicitly; 96% identified as a squared-norm fraction; angular split fixed at $\theta\geq100^\circ$ |
| 2 | Sloppy-model classification (B) | Introduction scopes "sloppy" to the local Fisher spectrum in logarithmic KD02 coordinates; End Matter distinguishes local anisotropy from a global parameter excursion |
| 3 | Radii, diffuseness, depths as constraints (B) | Abstract and Introduction add the downstream error-budget motivation; Discussion distinguishes additional observables from folding/dispersive model structure; Mahzoon *et al.* supplies a direct nucleon-nucleus elastic-equivalence test; Greenlees and Newton *et al.* delimit the radius/diffuseness interpretation |
| 4 | Scope of conclusions and venue (B) | No additional manuscript change beyond the KD02 scoping statements in the Abstract and Discussion and the clarifications added under Points 2 and 3; the venue case is made in this reply |
