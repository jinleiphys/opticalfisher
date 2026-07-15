# Verification notes, 2026-07-14 (pre-appeal audit; internal, not for submission)

Round-3 defense record. Two audit rounds (Claude recompute + independent Codex
cross-validation, twice) plus a root-cause fix. Every number in main_v3.tex now
traces to the converged artifacts regenerated on this date.

## 0. THE ROOT-CAUSE FIX: partial-wave convergence

`adaptive_lmax` used l_max = k*R + 15 (R = 1.25 A^1/3). Insufficient above
E ~ 100 MeV because Fisher log-derivatives converge in l more slowly than the
cross section. Certified against l_max = 60/90/120 (mutually consistent <0.002):

| config | old (underconverged) | converged | shift |
|---|---|---|---|
| p+208Pb@200 | 2.879 | 2.164 | -0.72 |
| n+208Pb@200 | 1.862 | 1.616 | -0.25 |
| n+40Ca@200  | 1.667 | 1.520 | -0.15 |
| n+120Sn@150 | 1.614 | 1.569 | -0.05 |
| everything at E <= 50 MeV | unchanged | unchanged | ~0 |

Fix: analysis/deff_scan_extended.py adaptive_lmax margin +15 -> +50 (documented
in the docstring); analysis/deff_stepwise.py hardcoded l_max=30 -> None (adaptive);
analysis/angle_resolved_sensitivity.py hardcoded l_max=30 -> None (Codex round-3
catch; the paper-quoted n+40Ca@50 angle numbers are unchanged to 4 decimals).
ALL artifacts regenerated in one chain (scan -> multi_energy -> stepwise ->
global -> angle -> census). This also explained the earlier stepwise-vs-multi_energy
disagreement on Sn (stepwise had l_max=30, multi_energy had the old adaptive;
both underconverged differently).

FULL-CENSUS CERTIFICATION (data/lmax_certification.json, script certify_lmax.py
in the session scratchpad): all 168 configurations re-evaluated at l_max = 90
and 120 versus production (adaptive kR+50, values 51-72). Max |Delta D_eff|:
elastic 0.0101 (l90) / 0.01133 (l120), all-obs 0.0100 / 0.0108; worst config
n+58Ni@200 (1.7525 -> 1.7412). The letters state the hard bound "no more than
0.012" (strict-inequality safe; an earlier "0.011" failed Codex round-4 because
0.01133 > 0.011, and "below the precision quoted anywhere" failed because the
worst shift exceeds the 0.01 two-decimal resolution).

## 1. Paper numbers as pinned (old -> new, all declared in response row 5)

- Headline: 1.7+/-0.5 -> **1.7+/-0.4** (1.659+/-0.397); n 1.63+/-0.33 -> **1.59+/-0.29**;
  p 1.79+/-0.55 -> **1.73+/-0.47**.
- Step 4 (Ca/Sn): elastic-only 2.52/2.27 -> **2.47/2.19**; all-obs 2.18/2.20 -> **2.16/2.27**
  (Sn all-obs now ABOVE its elastic-only value; Step-4 sentence reworded accordingly).
- Multi-energy averages over 24 systems: all-obs 2.36 -> **2.34 (~2.3)**;
  elastic-only -> **2.47 (~2.5)**; abstract/Discussion now say "~2.3--2.5".
- Global 48-coefficient: 3.0 -> **2.9** (2.936); lambda1 = 4.18e8 carries 43.8% ("44%");
  lambda2 37.5% ("38%"); lambda3 7.6% ("8%"); top-4 92.8% ("93%"); top-9 99.3% ("99%");
  cond 8.74e8 ("9e8"); e1 = rv0 70% + rso0 24%; e2 = rso0 69% + rv0 25%; e3 = rd0 91%.
  Per-system growth (as plotted in Fig. 5a, nuclei_growth): neutron start 2.37,
  peak 3.09 at 3 systems, settle 2.96; protons essentially neutral, final 2.94.
  (A per-configuration accumulation instead peaks 3.36 around 20 configs; the
  figure and text use the per-system view.) lambda2 = 37.47% -> "37%"; top-4
  92.8% -> "over 90%" matching the figure annotation.
  Elastic-only global D_eff = 3.17 > all-obs 2.94 (End Matter claim holds).
  Leading eigenvalue growth factors all/elastic top-8: 437/414/377/284/446/260/379/471
  ("roughly 250--500").
- Census (converged): eps=5%: A_y gain positive **113/168** (n 68/84 med +0.163;
  p 45/84 med +0.012); eps=10%: **92/168** (n 61/84 med +0.152; p 31/84 med -0.107).
  sigma_R: 168/168 positive, median +0.00062; only >0.05: p+197Au@10 (+0.532),
  p+208Pb@10 (+0.140), sub-barrier (barriers 10.6/10.9 MeV vs Ecm 9.95).
- Geometry tie census (chain rule rv<-rv+rw, av<-av+aw): increase **115/168**,
  range **-0.35..+0.49** ("-0.4 to +0.5"); showcase 50 MeV +0.14/+0.20/+0.27/+0.31.
- Back-projection conventions (converged): paper-convention mean 2.340, pinv 2.899,
  pinv larger in **23/24**; rcond-insensitive 1e-8..1e-12.
- Multi-energy gain Discussion: stated exactly, Delta D_eff = 1.24 (Ca) and 1.03 (Sn)
  with A_y gains 0.31 and 0.57 (raw 1.2448/1.0312/0.3108/0.5651); the earlier
  "~1.2" and "1.0--1.2" phrasings are superseded.
- Unchanged at 50 MeV (verified identical): 1.222/1.533 (Ca), 1.154/1.719 (Sn),
  sigma_R +0.0004, eigenvectors 90.1%/7.1%/97.2%, kappa 5.1e7, sqrt(kappa)~7000,
  angle-resolved 5.73/16.52, 96.0%, 2.0e-3 (backward-lambda1 convention, script
  angular_split_diagnostics; dividing by full-coverage lambda1 gives 1.8e-3, both
  round to 2e-3), D_eff fwd/bwd/full 1.19/1.20/1.22, cumulative peak 2.05@25deg,
  first diffraction minimum 30.84deg ("near 31deg").

## 1b. Codex round-3 catches (final acceptance audit), all applied

- angle_resolved_sensitivity.py still had l_max=30 -> fixed, artifact regenerated
  (paper numbers unchanged).
- "certified stable from l_max = 60 to 120" was wider than the repo record ->
  replaced in both letters by the measured full-census statement (max shift 0.011
  at l_max 90/120, record in data/lmax_certification.json).
- Residual "universal / depends neither on" in Results and two captions ->
  "stable across the configurations sampled, with only weak dependence on"
  (Spearman rho = -0.21, p = 0.006 across 168; a weak but significant mass trend
  exists, so "depends neither" was not defensible); declared in row 5.
- Response row 5 ranges synced to the converged text (1.0--1.2; 250--500;
  -0.4..+0.5; proton effect "essentially neutral").

## 1c. Codex round-4 strict-identity catches, all applied

Rule: printed value must equal the artifact rounded to the displayed digits;
ranges and "no more than" bounds must bracket the raw artifact values.

- l_max certification bound 0.011 -> 0.012 (raw max 0.011334); "below the
  precision quoted anywhere" clause deleted.
- Fig. 2 caption "no A-dependence" -> "no strong A-dependence" (Spearman
  rho = -0.67, p = 0.017 for n@50, 12 points).
- Discussion "no significant dependence" -> "only weak dependence"
  (rho = -0.21, p = 0.006 across 168).
- "regardless of how many systems are measured" dropped (not certified by
  the stored artifacts).
- "seven orders of magnitude" -> "nearly eight" (lambda1/lambda13 = 5.086e7,
  10^7.706); also in the response letter narrative.
- "50 to 10^7" -> "roughly 50 to 5e7" (raw 52.68 .. 5.086e7).
- Discussion multi-energy gains now exact: Delta D_eff = 1.24 (Ca), 1.03 (Sn);
  A_y gains 0.31, 0.57 (raw 1.2448/1.0312/0.3108/0.5651).
- Representative geometry interval +0.1..+0.3 -> +0.14..+0.31
  (raw 0.1407/0.2027/0.2660/0.3068).
- lambda1 = 4.1e8 -> 4.2e8 (raw 4.1807e8, two-sig-fig rounding).
- Row 5 synced to all of the above.

## 2. Earlier corrections (Codex round-2 catches, all applied and declared)

- Igo constant-J_V direction: NOT an eigenvector; Fisher information along it is
  a factor ~52 (orthogonalized variant ~60) below lambda1, not 1e5. Text now says
  "roughly a factor of 50".
- Bayesian-section sloppy range: mode ratios 12.7/52.7/205/245/465/1813/.../5.1e7
  -> text "50 to 1e7" (was 1e3--1e7).
- "means within 0.2" -> 0.25 (worst offset +0.237, n+40Ca, seed-42 protocol).
- Geometry-constraint "increases 0.1--0.3 hence conservative" was representative-
  systems-only; rescoped to the census (see above).
- Abstract no longer claims "orthogonal spin-orbit sensitivity" as the A_y gain
  mechanism (A_y Fisher trace is mostly central-parameter: SO-block median share
  14%, only 28% outside the elastic top-2 modes). Step 3's g(theta) physics
  statement stands (A_y vanishes without spin-orbit coupling).
- "Three tests" -> "Four checks"; ~38 -> 39; "individual" -> "leading" eigenvalues;
  timing hardware-neutralized ("a few seconds per configuration").
- sigma_R sub-barrier wording: elastic absorptive-block gradients are NOT zero
  (Frobenius 1.26/0.63 for Au/Pb); text says "constrains the absorptive parameters
  only weakly" instead of "carries the only information".

## 3. Archives and package hygiene

- paper/cover_letter.txt: archived first submission (banner added). DO NOT SUBMIT.
- paper/cover_letter_v2.txt: restored to the as-sent round-2 version (title
  "Intrinsic...", 1.8+/-0.5 as then reported) + archive banner. DO NOT EDIT/REUSE.
  The current letter to the editors is appeal.md.
- main_v2.tex, response_to_referee.md: historical round-1 documents, untouched
  (they contain era-correct numbers incl. the superseded 2.33/1.81).
- data/elastic_fisher_matrices.json: legacy artifact (17 angles, 11 params),
  predates the current pipeline; not used by any current figure or number.
- Appeal package = main_v3.tex/pdf + response_to_referee_round2.md + appeal.md.
- Census assets now in-repo: analysis/census_gradients.py + data/census_gradients.npz
  (regenerate ~3-10 min parallel). Figures 1, 2, 4 (stepwise), 5 (global)
  regenerated from converged data and synced to paper/ and paper/figures/;
  Figs. 3 and 6 (both n+40Ca@50) unchanged.

## 3b. Independent from-scratch re-audit, 2026-07-15 (code <-> paper strict identity)

Full re-derivation of every load-bearing number in main_v3.tex directly from the
committed artifacts (scratchpad verify_paper.py), plus a formula-by-formula read
of src/{observables,potentials,scattering_fortran}.py and all analysis scripts.
Formulas: ALL match the manuscript (dsigma/dOmega=|f|^2+|g|^2; Ay=2Im(fg*)/(...);
sigma_R, sigma_T optical-theorem neutron-only; log-derivative S_i; F=G G^T;
Thomas factor 2; delta=0.01|p|; 35 angles; D_eff=(sum l)^2/sum l^2; Step-4
Jacobian projection; 48-coeff universal grouping 9/16/10/13; pinv back-projection).
Numbers: all reproduce (headline 1.59/1.73/1.66; steps 1.222/1.533/2.467/2.159 Ca,
1.154/1.719/2.185/2.271 Sn; multi-E gains 1.2448/1.0312; global 2.936/e1-e2-e3;
angle 5.73/16.52/96.0%/2.0e-3/1.19-1.20-1.22; pinv 2.899 vs 2.340, 23/24;
lmax cert 0.01133) EXCEPT two isolated drifts, both fixed this pass:

- **Line 71 condition-number range**: text said "$10^6$--$10^{11}$" for Fig.2(c),
  but Fig.2(c) is n+A@50 across nuclei with true max 7.56e10 (=10^10.88, worst
  = 12C); never reaches 10^11. Both the Fig.2 caption AND response letter already
  said "to above $10^{10}$" -> line 71 was the lone outlier. FIXED to
  "$10^6$ to above $10^{10}$" (matches caption line 76 + response letter).
- **Line 128 Sn A_y gain**: text said "0.57"; committed deff_stepwise.json gives
  the Sn single-energy A_y gain rounding to **0.56** under every definition
  (Step3-Step1 = 0.564738, Step3-Step2 = 0.564263, elastic_Ay_13p - elastic_13p
  = 0.564716; all -> 0.56, none -> 0.57). The round-4 note's raw "0.5651" was
  superseded by the converged artifact. FIXED to "0.56". (Ca gain 0.3102 -> 0.31
  unchanged.)
- **Line 128 rounding-optics (Codex 2026-07-15 point d)**: Step-3 quotes the Sn
  endpoints as 1.15 and 1.72 (each raw-rounded; locked as "showcase unchanged" in
  response row 5), whose naive 2-dp difference 0.57 collides with the corrected
  gain 0.56. This is rounding non-additivity (raw 1.7188886 - 1.1541509 =
  0.5647377 -> 0.56), NOT a numerical error, and the multi-E gains 1.24/1.03 are
  likewise raw differences (not 2.47-1.22=1.25). Resolved by appending "all
  computed from the unrounded single-energy D_eff values" to the Discussion gain
  clause, so endpoints stay 1.15/1.72 and gains stay raw-rounded 0.31/0.56 with no
  apparent conflict. Response letter quotes no A_y-gain number, so no letter edit.

main_v3.tex recompiles clean (11 pp). No other main-text number moved. If the
appeal package is re-declared, these two last-digit corrections join response row 5.

## 4. Changelog discipline

All post-round-2 changes declared in response Summary rows 1-6, including the
convergence fix, the wording pass, all numeric corrections, the census/convention
additions, and administrative items; full source diff offered on request.
Reference diff: `git diff 6434324 -- paper/main_v3.tex`.
