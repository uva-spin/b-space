# Finite-Y completion handoff

Status: lambda=1 Tevatron finite-Y candidate validated; LHCb fiducial W/Y remains a diagnostic candidate pending input normalization/covariance closure; frozen production unchanged.

Paths under `outputs/`, `production_frozen/`, and the original identifiability
campaign are working-archive references.  They are intentionally omitted from
the compact public checkout; the frozen public package is under
`../../production/lambda1_empirical_reference_full96x50/`, and the manifests
record how to reconnect excluded artifacts.

## Lambda-one 353-row test-fit launch (2026-08-19)

The differentiable test-fit runner was corrected after an input-provenance
check found that its historical default still pointed at the lambda=0.5
Collins candidate.  It now starts from the registered lambda=1 endpoint
`dataset_identifiability_campaign_2026/outputs/lambda1_start_expansion96_s353_cont120000`.
The 329 accepted core rows are loaded with `Y=0` by construction; the 24
Tevatron boundary rows receive the unitary finite-Y profile.  The boundary
sample extends to `qT/Q=0.2967`.

The isolated test outputs are:

```text
systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/outputs/lambda1_unitary_boundary_test_frozen_s353/
systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/outputs/lambda1_unitary_boundary_test_early_frozen_s353/
systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/outputs/lambda1_unitary_boundary_test_late_frozen_s353/
systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/outputs/lambda1_unitary_boundary_test_refit_s353/
```

The frozen-FNP fits pass their nuisance stationarity gate for all three
transition profiles.  Their total chi2/row values are 0.4566, 0.4599, and
0.4615 for the early, central, and late profiles; the corresponding boundary
chi2/row values are 1.027, 1.074, and 1.100.  The accepted-core prediction
shift from nuisance refitting is about 1.16% at maximum.

The free-FNP central refit reaches total chi2/row 0.4558 and reduces the
boundary chi2/row to 1.042, but it fails the FNP stationarity gate
(`fnp_gradient_l2=1.40`, threshold `1e-4`).  It is therefore a fit-impact
diagnostic, not a promoted extraction.

The next systematic range step is now well-defined: retain the 329-row core,
add the complete 24-row Tevatron boundary through `qT/Q~0.30`, and vary the
unitary transition profile.  No additional Tevatron rows exist beyond this
published boundary scope in the current input.  The four high-qT LHCb rows
remain excluded from the global test until their observable and covariance
closure is approved.

## Important baseline correction (2026-08-17)

The first frozen-FNP fit/replica audit in this package accidentally loaded the
older source state
`collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303`
instead of a verified lambda=1 empirical-reference endpoint. Those numerical
fit/replica results are therefore superseded and must not be interpreted as
the lambda=1 finite-Y result. The unitary algebra, NLO provenance, node
convergence, and profile-variation checks are independent of this mistake.

The correct lambda=1 production candidate is the empirical-reference
construction registered at
`dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50`;
its concrete endpoint states are under the corresponding
`outputs/lambda1_start_expansion96_s*_cont120000` directories. The finite-Y
fit and replica audit must be rerun with that state (and with boundary W values
recomputed from it) before any finite-Y fit-impact conclusion is accepted.

That correction has now been completed in the isolated reports under
`reports/lambda1_unitary_endpoint_recompute/`. Every one of the 96 registered
lambda=1 endpoints was evaluated on the accepted rows and on the 24 Tevatron
unitary boundary rows. Relative to the mistakenly used lambda=0.5 endpoint,
the boundary W values shift by as much as 6.59% (median endpoint maximum
shift 3.99%); all corrected matched predictions remain positive.

## Starting evidence

The prior experimental packages establish:

- the ordinary additive `W + (FO - ASY)` pilot fails because the resummed W
  is not close to ASY near the boundary;
- the strict ASY expansion itself closes at the tested node;
- 24 Tevatron rows have complete genuine-NLO fixed-order results and passed
  the numerical convergence and NLO scale pilots;
- Simpson node integration at `n_b=640` passes the 5% numerical gate for all
  early/central/late profiles;
- the previous frozen-FNP fit-impact test fails without theory nuisances, so
  fit impact must not be confused with mathematical validity of the Y term;
- high-qT LHCb node-level fiducial acceptance is now available for the four
  boundary rows, but the LHCb NNLO observable/covariance closure remains open.

## Candidate definition

The candidate is

`Y_unitary = p(r) [FO_NLO - W]`,

with `r=qT/Q` and a C2 smootherstep profile. Therefore

`matched = (1-p) W + p FO_NLO`.

It is explicitly called a unitary finite-Y transition. It is not presented as
the conventional asymptotic `FO-ASY` term.

## Decision criteria

The construction is valid for the isolated Tevatron scope if endpoint limits,
continuity, node convergence, external NLO provenance, profile variation,
lambda=1 endpoint fit impact, and the 50-replica propagation all pass. Those
lambda=1 checks are complete. Universal production promotion additionally
requires the LHCb observable, fixed-order, and correlated-covariance closure.

## Frozen-state rule

No file below `production_frozen/`, the active lambda=1 package, the published
source checkout, or the paper may be overwritten by this campaign.

## Current decision

The isolated Tevatron validation is complete. The candidate passes algebraic
endpoint/C2 tests, exact row reconstruction, existing `n_b=320` versus
`n_b=640` Simpson convergence for all 24 rows and all three profile windows,
positive-prediction checks, and the completed genuine-NLO input gate. The
maximum node-integration shift is 3.58%; the maximum early/late profile
envelope is 25.96% and is retained as a model-form uncertainty.

The valid result is therefore:

`Y_unitary = p(qT/Q) [FO_NLO - W]`,

with `matched=(1-p)W+p FO_NLO`, for the 24-row Tevatron scope. It is not a
validated conventional `FO-ASY` Y term. The earlier additive construction is
rejected because its W/ASY cancellation fails at the transition node.

This validates a finite-Y transition construction for the Tevatron scope, not
yet a universal production promotion. The corrected lambda=1 fit-impact audit
uses all 96 endpoints and all three profile windows. All 288 endpoint/profile
fits converge and pass the operational pull/nuisance gate. For the central
profile, total chi2 per row has q16/median/q84 = 1.068/1.070/1.072 and data
chi2 per row = 0.939/0.941/0.942. The corrected 50-replica propagation covers
96 endpoints and all three profiles (14,400 fits); all optimizers converge and
the central total chi2-per-row q16/median/q84 is 1.455/1.893/2.410. The
replica maximum-pull q95 is about 4.1--4.2, which is a Gaussian-replica
fluctuation diagnostic, not a reason to reject the construction without a
separate replica-distribution criterion.

Three staged differentiable FiLM/FNP central-refit trials were also completed:
an unanchored 50,000-epoch run, an anchor-`lambda=1000` 50,000-epoch run, and
an anchor-`lambda=1e5` 20,000-epoch control. None reached the plateau or
stationarity gates. Their total objectives were respectively `0.4830`,
`0.4961`, and `0.5328` per row, with nonzero FNP gradients. A newly refit
flexible FNP therefore cannot be promoted from these trials; the established
FNP is retained for the valid unitary-Y audit.

The fiducial audit finds 46 high-qT candidate rows across Tevatron and LHCb.
The unitary node/kernel campaign is complete for the 24 Tevatron rows and the
four LHCb boundary rows (`LHCb_7:10`--`:13`). The LHCb grid reproduces
independent full-bin DYTurbo fiducial/inclusive ratios to 0.75%, so the former
node-coverage gap is closed. Broad Tevatron+LHCb production remains
unapproved for the separate reason that the LHCb NNLO fixed-order prediction
and published covariance still do not yield an acceptable universal boundary
fit.

The machine-readable reports are `reports/decision_status.json`,
`reports/tevatron_unitary_validation.json`,
`reports/fnp_refit_promotion_audit.json`,
`reports/frozen_unitary_replica_stability.json`, and
`reports/fiducial_acceptance_audit.json`. The current lambda=1 and LHCb
follow-up records are `reports/lambda1_unitary_fit_impact.json`,
`reports/lambda1_unitary_boundary_replicas.json`,
`reports/lhcb_true_nnlo_scale_scan/summary.json`, and
`reports/lhcb_correlated_covariance_audit/summary.json`.

## LHCb high-qT closure update (2026-08-17)

The explicit DYTurbo acceptance grid and node-level fiducial W kernels are now
complete for `LHCb_7:10`, `:11`, `:12`, and `:13` at `n_b=640`.  The merged
2x6 (rows 10--12) / 4x6 (row 13) acceptance grid reproduces independent
full-bin DYTurbo fiducial/inclusive ratios to a maximum relative mismatch of
0.75%; all nodes are finite and positive where the inclusive rate is nonzero.
The kernels and endpoint W/unitary predictions are stored under
`reports/lhcb_fiducial_w_kernels_nb640/` and
`reports/lambda1_lhcb_unitary/`.

During the first LHCb fit-impact pass two provenance problems were found and
fixed in the isolated campaign.  DYTurbo text tables are in fb/bin, not
pb/bin, and the earlier cards labelled NLO had `doVJREAL=false` and
`doVJVIRT=false`; they therefore evaluated only the V+jet LO piece.  A fresh
true-NLO run with both switches enabled and Vegas V+jet integration is now in
`reports/lhcb7_external_true_nlo/dyturbo_true_nlo_summary.csv`, and the
lambda=1 recompute uses those values after fb/bin -> pb/GeV conversion.

The resulting LHCb diagnostic fit has positive matched predictions and
reasonable data pulls (central-profile median data chi2/row about 1.31 and
maximum pull about 1.85), but the fixed-normalization fit requires an NLO
scale nuisance of about 4.25 sigma.  Thus it is not a production promotion:
the four source rows are still marked `diagnostic_fiducial_acceptance_candidate`,
have `sysNorm_rel=0` and diagonal released errors rather than an approved
correlated covariance model.  The apparent factor-of-two external
DYTurbo/MCFM discrepancy has since been traced to MCFM's absolute-rapidity
cut; the remaining positive-arm data-versus-true-NLO normalization is still
unresolved.  This failure is currently an LHCb observable/provenance issue,
not evidence that the Tevatron unitary algebra or W-kernel construction is
invalid.  The isolated fit report is
`reports/lambda1_lhcb_unitary/lhcb_fit_impact_summary.json`.

## Conventional FO-ASY rejection is not a study stop (2026-08-17)

The label “Conventional (Y=FO-ASY) — rejected” applies only to that additive
matching construction at the tested transition point. It does not reject the
finite-Y objective or authorize stopping the campaign. The original additive
pilot was also run before the explicit-y W evaluator removed the backend's
inclusive Tevatron rapidity factor, so a corrected reassessment was required.

That reassessment is stored in
`reports/conventional_y_reassessment/cdf_run_2_36.json`. With the rapidity
factor removed, the central-node values are approximately:

| $n_b$ | $W_{\rm corrected}$ | ASY | $W+FO-ASY$ | FO |
|---:|---:|---:|---:|---:|
| 160 | 1.916 | −42.754 | 46.728 | 2.057 |
| 320 | 2.193 | −45.226 | 49.476 | 2.057 |
| 640 | 2.255 | −45.845 | 50.158 | 2.057 |

Thus the rapidity normalization issue was not the cause of the conventional
failure. At $q_T/Q\simeq0.20$, the fully resummed W is not close to its
strict fixed-order asymptotic expansion, so adding (FO-ASY) produces a large
overshoot. This is a domain-of-validity failure of that candidate, not a
conclusion that finite-Y work is complete. The production path remains the
separately validated unitary transition, followed by the remaining Tevatron
production promotion and LHCb input/covariance closure.

## LHCb rapidity-sign closure update (2026-08-17)

The apparent factor-of-two DYTurbo/MCFM discrepancy in the new high-qT LHCb
check has been traced to MCFM's `gencuts_user.f90`, which applies
`abs(yraptwo(3,4,pjet))` to `y34min/y34max`. MCFM's explicit-y cards therefore
include both (+y) and (-y) pp arms, while the LHCb/DYTurbo card is the
positive arm. A mirrored negative-y DYTurbo no-lepton run sums to the MCFM
absolute-y result within 1.1%; positive-arm DYTurbo agrees with half of MCFM
within about 1--3% for the available explicit bins. The code closure is
therefore understood and is recorded in
`reports/lhcb_external_closure/summary.json`.

The remaining LHCb blocker is independent: the positive-arm true-NLO DYTurbo
prediction is only about 0.52--0.58 of the candidate data in rows 10--13, and
those rows still have diagnostic-only released diagonal errors, no approved
correlated covariance, and no accepted normalization nuisance. That
data/observable provenance question must be resolved before universal LHCb
finite-Y promotion; it does not invalidate the Tevatron unitary W/Y
construction.

## Full LHCb true-NLO shape check (2026-08-17)

To determine whether the LHCb discrepancy is a single high-qT normalization
problem or an all-spectrum mistake, an isolated true-NLO DYTurbo run was
completed for all 14 candidate bins. The result is recorded in
`reports/lhcb_true_nlo_shape_all14/summary.json` and
`reports/lhcb7_external_true_nlo_all14/`.

The low-qT rows are not a valid fixed-order replacement for W: the
true-NLO/data ratio ranges from 8.30 at qT/Q=0.012 to 0.82 at qT/Q=0.088,
which is the expected fixed-order singular/resummation-domain behavior. The
transition-approach rows (0.10 <= qT/Q < 0.20) have a consistent ratio of
0.62--0.73. The four high-qT rows used for the finite-Y boundary have a
pointwise ratio of 0.521--0.583 and a bin-integrated ratio of 0.546.

This establishes that the factor-of-two code convention is already resolved,
and that the remaining issue is a smooth positive-arm data-versus-true-NLO
normalization/observable mismatch that becomes visible before and through the
finite-Y transition. It is not evidence against the unitary construction. A
production promotion still requires the LHCb input team to identify the
observable convention and supply an approved correlated covariance/normal-
ization treatment; multiplying the fixed-order term by the observed ratio
would be an unapproved data-driven rescaling and is not being done.

An optional independent MCFM process-41 NLO probe for rows `LHCb_7:10` and
`:11` was attempted under `reports/lhcb7_external_mcfm_true_nlo_probe/`.
MCFM remained in adaptive real-emission warmup and repeatedly increased the
call count without producing a converged integral, so the process was stopped
and no number was used. This is an integration-cost limitation, not a contrary
physics result; the accepted NLO evidence remains the completed DYTurbo
true-NLO all-14-bin run together with the converged MCFM LO rapidity-sign
closure.

### Independent MCFM NLO retry (2026-08-17)

Because the initial MCFM probe used generic integration settings, a second
isolated attempt was made using MCFM's own `input_Zjet.ini`-style controls:
five-million NLO-real initial calls, `warmupchisqgoal=12.5`, intermediate
snapshots, `ndmx=40`, ten warmup iterations, and the standard call boost. The
first boundary bin completed the five-million and seven-million call blocks,
but MCFM still reported that the 25% relative warmup precision goal was not
reached and requested a 9.8-million block. The retry was stopped before a
final integral rather than escalating this optional CPU-only check without
bound. The log and exact status are in
`reports/lhcb7_external_mcfm_true_nlo_retry/`.

This does not weaken or overturn the finite-Y conclusion: it is an
independent integration-cost limitation. No MCFM NLO number is used for
promotion; the converged DYTurbo true-NLO/NNLO results remain the fixed-order
evidence, and the converged MCFM LO result remains the rapidity-sign closure.

## LHCb true-NLO scale closure (2026-08-17)

The four finite-Y boundary bins were rerun at the central scale and the six
standard noncentral combinations
`(muR,muF)=(0.5,0.5),(0.5,1),(1,0.5),(1,2),(2,1),(2,2)`.
The isolated result is in `reports/lhcb_true_nlo_scale_scan/summary.json`.
The largest pointwise theory/data ratio is 0.666 and the largest ratio in the
first three transition bins is 0.628; the bin-integrated ratio over all four
boundary bins reaches only 0.598 at the most favorable scale point. Scale
variation therefore does not cover the approximately 45% positive-arm
data/theory deficit. The discrepancy cannot be absorbed into the declared
perturbative scale envelope, and no data-driven FO rescaling is authorized.

## LHCb NNLO fixed-order follow-up (2026-08-17)

The NLO deficit prompted the next required test rather than ending the finite-Y
campaign. DYTurbo supports `order=2` for the fixed-order NNLO calculation;
the LHCb paper also compares the fiducial total rate to NNLO theory. The
isolated NNLO run used the same positive-arm lepton cuts, PDF convention, and
fb/bin-to-pb/GeV conversion as the NLO closure, with 10 million V+jet calls at
the central scale. The six noncentral scale points used 5 million calls each.

The complete NNLO scale summary is
`reports/lhcb_true_nnlo_scale_scan/summary.json`. Central theory/data ratios
for the four high-qT boundary bins (`LHCb_7:10`--`:13`) are approximately
`0.849, 0.819, 0.773, 0.888`. Thus NNLO raises the prediction substantially
relative to true NLO (`0.52`--`0.58`) and confirms that the rejected additive
candidate did not exhaust the perturbative options. The standard six-point
scale envelope reaches maxima of approximately `0.849, 0.891, 0.850, 0.968`
in the same bins. It therefore nearly covers the last bin, but remains below
the data in the first three; the all-four-bin integrated ratio is at most about
`0.867`.

This is an active follow-up result, not a rejection of finite-Y work. It
changes the next candidate from an NLO fixed-order input to an NNLO fixed-order
input for the LHCb closure. Universal production promotion is still withheld
because the residual first-three-bin mismatch, released diagonal-only source
errors, and missing approved correlated normalization/covariance treatment
remain. A data-driven rescaling is explicitly not allowed. The NNLO run is
diagnostic and leaves all frozen production inputs unchanged.

## LHCb boson-rapidity observable check (2026-08-17)

The published LHCb (p_T^Z) table is inclusive in the boson rapidity while
retaining the two-muon fiducial cuts. To test whether the repeated metadata
field `2<y<4.25` was causing the NNLO residual, the positive-arm DYTurbo run
was repeated with (0<y_Z<6) and with the lepton cuts unchanged. For rows
`:11`--`:13` the result is numerically identical within the Vegas uncertainty
to the `2<y_Z<4.25` cards; row `:10` changes from about `0.852` to `0.827`
pb/GeV (a roughly 3% Monte-Carlo-level shift). Thus the positive-arm
rapidity window is not the source of the 15--23% NNLO residual in the first
three bins. A diagnostic `-10<y_Z<10` run approximately doubles the rate,
which is the expected inclusion of the negative rapidity pp arm and is not the
LHCb observable.

The corrected positive-arm NNLO observable check is therefore consistent with
the existing NNLO scale summary; no rapidity-window rescaling is authorized.

## LHCb NNLO PDF convention check (2026-08-17)

To test whether the remaining boundary discrepancy is a PDF-choice effect,
the positive-arm NNLO calculation was repeated at 2 million V+jet calls for
`NNPDF40_nnlo_as_01180`, `CT18NNLO`, and `MSHT20nnlo_as118`. The complete table
is `reports/lhcb_nnlo_pdf_scan/summary.json`. Across the three sets, theory/data
ratios range only from `0.787--0.825` (row 10), `0.779--0.819` (row 11),
`0.758--0.773` (row 12), and `0.882--0.892` (row 13). PDF choice therefore
changes the prediction by a few percent, not enough to close the first-three-
bin residual. No PDF-driven rescaling is authorized.

## LHCb published pT covariance audit (2026-08-17)

The LHCb paper supplies a (p_T^Z) correlation matrix in Table 10, excluding
the beam-energy and luminosity components. The isolated audit in
`reports/lhcb_correlated_covariance_audit/summary.json` reconstructs the full
four-bin high-qT covariance by combining that matrix with the released
per-bin statistical/systematic errors and fully correlated beam/luminosity
vectors. The covariance is positive definite after the stated components are
included.

Using the NNLO positive-arm input and the same unitary endpoint/profile
ensemble, the central transition profile has median data
χ²/row ≈ `6.58` (q16--q84 `6.12--6.86`) and median total χ²/row ≈ `10.74`.
The early and late profiles are no better. Correlations therefore do not
remove the LHCb mismatch; they make the diagnostic tension more explicit. The
source manifest still requires formal covariance/provenance promotion, so this
audit remains diagnostic and does not alter production inputs.
The numeric covariance and its decomposed error components are exported as
`reports/lhcb_correlated_covariance_audit/lhcb_pT_covariance_pb2_per_GeV2.csv`
and `lhcb_pT_covariance_components.csv` for any future isolated fit.

## LHCb NNLO electroweak convention check (2026-08-17)

The final direct-theory convention sweep is recorded in
`reports/lhcb_nnlo_convention_scan/summary.json`. For row `LHCb_7:10`, removing
γ* or changing the DYTurbo electroweak input scheme shifts the NNLO/data ratio
only from the baseline `0.825` to `0.802--0.830`; none closes the roughly 18%
residual. The baseline γ*/(G_μ) convention is retained and no convention-driven
rescaling is authorized.

## LHCb NNLO unitary endpoint audit (2026-08-17)

The four isolated positive-arm NNLO DYTurbo boundary rows were combined in
`reports/lhcb7_external_true_nnlo_positive_y_10m_combined/` and used to
recompute the complete 96-endpoint lambda=1 unitary W/Y ensemble in
`reports/lambda1_lhcb_unitary_nnlo/`. All 96 endpoints, all three transition
profiles, and all four LHCb rows produced finite positive matched predictions.
The endpoint spread is negligible in rows 11--13 because those bins are
already at the fixed-order end of the profile; row 10 retains the expected
small W-side endpoint variation.

The corresponding covariance fit is in
`reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_fit_impact_summary.json`.
It uses the published pT correlations plus fully correlated beam/luminosity
components and a symmetric largest-absolute six-point NNLO scale excursion.
All 288 endpoint/profile fits converge and remain positive. Median total
chi2 per row is approximately 4.56 (early), 6.69 (central), and 7.90 (late),
with the central-profile matching and NNLO-scale nuisance pulls about 2.56 and
2.28 sigma. This is an improvement over the NLO input, but it is not a
production-quality LHCb fit and does not justify silently rescaling the data
or fixed-order term.

The finite-Y construction is therefore algebraically and numerically complete
for the verified Tevatron scope, and the LHCb W/Y endpoint machinery is now
complete as an isolated diagnostic. Universal production promotion remains
blocked specifically by the unresolved LHCb high-qT observable/theory
normalization residual and its formal covariance/provenance gate, not by the
rejected additive FO-ASY ansatz or by endpoint instability.

## LHCb NNLO unitary replica propagation (2026-08-17)

The LHCb published covariance was sampled into 50 reproducible Gaussian
pseudo-replicas (`seed=20260817`) and propagated through all 96 lambda=1
endpoints and all three matching profiles. The complete result is in
`reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_replica_summary.json`,
with the row-level fits in
`reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_lambda1_nnlo_replica_fits.csv`.
All 14,400 endpoint/profile/replica fits converged and remained positive.

For the central profile, the median total chi2 per row is about 7.16 (q16--q84
`5.57--9.23`), with matching and NNLO-scale nuisance medians about 2.67 and
2.27 sigma. The replica spread is now explicitly quantified, but it does not
remove the systematic theory/data residual. This closes the experimental
replica-propagation part of the isolated LHCb finite-Y audit; it does not
authorize universal production promotion.

## Completion audit and scope decision (2026-08-17)

The requirement-level audit is recorded in
`reports/finite_y_completion_audit.json`. The verified lambda=1 unitary
finite-Y candidate is complete at production-level quality for the 24-row
Tevatron scope, with all endpoint, positivity, node-convergence, fit-impact,
and replica checks recorded. The LHCb fiducial node grid, W kernels, positive-
arm convention, NNLO endpoint ensemble, and 50-replica propagation are also
complete and numerically stable as isolated diagnostics.

Universal LHCb promotion is not approved because the remaining issue is an
external observable/input closure: the first three high-qT LHCb bins remain
above the converged positive-arm NNLO fixed-order prediction, even after
scale, rapidity, PDF, electroweak, covariance, and replica checks. No
data-driven rescaling is permitted. This is an explicit exhausted blocker
requiring either formal LHCb covariance/normalization clarification or an
independently validated fixed-order result that closes the spectrum. Frozen
production files remain unchanged.

## Metric comparability correction (2026-08-18)

The earlier juxtaposition of `chi2/row ≈ 7.16` with the production value
`≈ 0.43` was misleading. The exact comparison is recorded in
`reports/metric_comparability_audit.json`.

The `0.434` production number is

```text
(140.073 data chi2 + 2.775 normalization penalty) / 329 rows
```

from the low-qT W-only production-objective fit. It includes six low-qT LHCb
rows, but not the four high-qT finite-Y boundary rows. The production training
scope is qT/Q ≤ 0.20 and the historical backend has `y_mode=zero`.

The `7.16` number is the median of a 50-replica diagnostic involving only the
four LHCb boundary rows (`qT/Q=0.242--1.85`). It uses the published correlated
pT covariance and includes matching and NNLO-scale nuisance penalties. The
central-profile endpoint-only fit has median total chi2/row `6.69`; before
nuisance profiling the same central model has covariance chi2/row about
`21.12`.

Therefore `7.16` is a local high-qT LHCb closure metric, not the chi2/row of
the whole production fit. As an illustrative arithmetic average only,
adding four rows at that local value to the 329-row production total gives
about `0.515` over 333 rows; this is not a refit, but demonstrates why the
global and local numbers cannot be compared directly. The local LHCb tension
is nevertheless real and remains the finite-Y promotion blocker.

## FNAL finite-Y chi2 scope clarification and global refit (2026-08-18)

The earlier handoff incorrectly said that no global calculation existed. An
older isolated 329-accepted-row + 24-boundary-row joint refit did exist, but it
was initialized from the lambda=0.5 production state and every long run failed
the stationarity/plateau gate. It therefore was not the requested lambda=1
W+Y production refit.

The missing like-for-like calculation was run in isolated output tags. The
first completed step is a lambda=1 endpoint (`s353`) with FNP held fixed and
all normalization/matching/scale nuisance parameters profiled:

```text
total:       162.3328 / 353 = 0.459866
accepted:    131.0158 / 329 = 0.398224
boundary:     25.7732 / 24  = 1.073883
```

This is a global W+Y fit-impact check, not the final refit because the
lambda=1 FNP endpoint was held fixed. The corresponding unfreezed long run
(`lambda1_global_unitary_refit_s353_long`) used 50,000 epochs with
`0:2e-5,10000:2e-6,30000:2e-7`. It reached

```text
total:       160.7061 / 353 = 0.455258
accepted:    130.3478 / 329 = 0.396194
boundary:     24.9019 / 24  = 1.037577
```

but failed the stationarity gate: FNP gradient norm per-row objective
`1.169e-2` versus the `1e-4` threshold (nuisance gradient `4.762e-3`). The
FNP moved by relative parameter norm `1.945%`, with a maximum accepted
prediction shift of `9.63%`. Thus this is a formal global W+Y *attempt*, not a
converged production refit. A 1,000-iteration L-BFGS polish lowered the
objective only to `0.455258161` and the FNP gradient to `7.243e-3`, still far
above threshold. A start-to-start global stability study cannot be authorized
until the single-start optimization itself reaches stationarity.

The completed 24-row boundary study remains useful as a 96-endpoint and
50-replica diagnostic:

```text
endpoint-only total chi2/N = 1.0696 (q16--q84: 1.0675--1.0715), N=24
50-replica median          = 1.8930 (q16--q84: 1.4553--2.4099), N=24
```

The historical `chi2/N ≈ 0.43` remains the separate 329-row W-only
production objective and is not the corrected W+Y global value. The complete
machine-readable record is
`reports/lambda1_global_fnal_wy_refit_status.json` and the run outputs are
under `experimental_unitary_transition/outputs/lambda1_global_unitary_*`.

One subtlety is essential: all 329 accepted rows satisfy `qT/Q <= 0.20`, and
the selected unitary profile has `p=0` throughout that domain. Therefore the
corrected finite-Y term is identically zero on those rows. There cannot be a
distinct nonzero-Y 329-row result without changing the matching profile or
adding boundary rows. The first nontrivial global W+Y scope is consequently
the 329 core rows plus the 24 genuine-NLO boundary rows.

## Global W+Y comparison correction (2026-08-18)

The first global run exposed a bookkeeping mismatch: it loaded a lambda=1 FNP
endpoint but initialized dataset normalizations from the old lambda=0.5
production predictions. The accepted initial data chi2 was consequently
`138.0825`, whereas the same lambda=1 endpoint's own 329-row record is
`130.1058`. The isolated runner now accepts `--initial-norms-path`, and the
endpoint model plus endpoint `control_norm` values reproduce `130.1058`
exactly.

The corrected decomposition shows no significant change to the accepted core
from adding Y. With FNP fixed and nuisances profiled, the accepted-core
objective is `(131.0158 + 2.1292)/329 = 0.40476`, compared with the lambda=1
endpoint's `0.40344`. The full 353-row number `0.45987` includes the 24 new
boundary rows and a `3.4145` matching/scale theory penalty. The historical
`0.43419` is not a like-for-like comparator: it is the old lambda=0.5,
329-row W-only objective.

The consistent unfreezed attempt reaches `0.45530` over 353 rows, but its FNP
gradient is `4.39e-2` versus the `1e-4` stationarity threshold. Its roughly
7% accepted prediction movement is therefore an unconverged FiLM response to
the added boundary rows, not evidence that the unitary Y term changes the
329-row core. The full decomposition is in
`reports/lambda1_global_wy_decomposition.json`.

## LHCb scope clarification for the global comparison (2026-08-18)

The six `LHCb_7` rows in the 329-row accepted core are `LHCb_7:0`--`5`, with
`qT/Q=0.0122`--`0.0883`. They remain in the strict TMD domain where the
unitary profile has `p=0`; no finite-Y correction was applied. The consistent
lambda=1 endpoint/global-nuisance pulls are approximately
`[0.083, -0.350, 1.068, 0.220, 1.458, -0.847]` sigma. This is an inherited
low-qT fit result, not a new LHCb finite-Y closure.

The unresolved LHCb finite-Y issue is the separate high-qT fiducial set
`LHCb_7:10`--`13`, which has its own acceptance and NNLO diagnostic campaign.
Those rows were not included in the 24-row Tevatron global boundary fit (all
24 boundary rows are `CDF_RUN_1`, `CDF_RUN_2`, or `D0_RUN_1`). Universal LHCb
promotion remains withheld.

## Way forward for high-qT LHCb rows 10--13 (2026-08-18)

There is a technically valid path forward, but it is an LHCb observable/theory
closure path rather than another $F_{\rm NP}$ tuning path. The four rows
`LHCb_7:10`--`13` have now been evaluated with the isolated fiducial W-kernel
campaign (`reports/lhcb_fiducial_w_kernels_nb640/`) using the validated DYTurbo
node acceptance grid. The grid reproduces the full-bin fiducial acceptance to
better than 0.8% (the broad row 13 is the limiting case), and all four
`n_b=640` kernels are complete. The lambda=1 endpoint W and unitary matched
tables are in `reports/lambda1_lhcb_unitary/`; the NNLO variant is in
`reports/lambda1_lhcb_unitary_nnlo/`.

The best current fixed-order diagnostic is the positive-arm DYTurbo NNLO
input. Its central theory/data ratios for rows 10--13 are approximately
`0.85, 0.82, 0.77, 0.89`; the standard scale envelope reaches at most
approximately `0.97`, only in row 13, while the first three rows remain below
the data. The 96-endpoint covariance fit has central-profile median total
chi2 per row about `6.69`; the 50-replica propagation gives about `7.16`.
All fits are converged and positive, so this is not an endpoint or
non-uniqueness failure. It is a residual observable-level closure issue.

The allowed next sequence is:

1. Formally freeze and provenance-audit the LHCb observable definition,
   positive-rapidity-arm convention, bin normalization, and published
   covariance (including beam/luminosity correlations). The old factor of two
   is already explained as MCFM's absolute-rapidity convention; no data-driven
   factor or normalization adjustment is allowed.
2. Re-run the lambda=1 fiducial W+unitary endpoint fit using the completed
   kernels and the highest-statistics validated fixed-order input, then repeat
   the covariance/replica check. This remains an isolated diagnostic until
   input provenance is promoted.
3. If the residual persists, perform one independently validated higher-order
   calculation (for example an independently checked NNLO or NNLO+resummed
   fiducial prediction). The prior MCFM true-NLO retry was numerically
   inconclusive and should not be used to rescale the data.
4. Only if this closes the four rows should they enter universal production.
   Otherwise retain them as a documented external finite-Y closure diagnostic
   and keep the validated Tevatron production scope unchanged.

Thus there is a way forward, but there is no justified method-level fix that
can make these four rows agree by altering $F_{\rm NP}$ or the unitary
matching profile.

## NNLO acceptance consistency check (2026-08-18)

Because the W kernels use an NLO node-level fiducial grid while the best
fixed-order Y input is NNLO, an isolated bin-integrated NNLO acceptance check
was run at 10M real/virtual calls per row. The NNLO/NLO acceptance ratios are

```text
LHCb_7:10  1.01614
LHCb_7:11  0.97457
LHCb_7:12  1.02269
LHCb_7:13  1.00311
```

The bin-level integrals are positive and numerically stable. Applying these
ratios as a deliberately conservative sensitivity proxy to the W term does
not close the LHCb residual: the central-profile median total chi2/row moves
from `6.69` to `7.32`, and the late-profile value from `7.90` to `9.18` (the
early profile changes by less than `0.04`). This is not a substitute for a
full NNLO node-kernel calculation, but it shows that acceptance-order
consistency cannot plausibly supply the missing 15--23% normalization.

An attempted low-statistics NNLO node grid was explicitly rejected by a
numeric gate: individual nodes produced sign-changing near-zero Monte Carlo
integrals. It is recorded under
`reports/lhcb_node_acceptance_grid_nnlo_q2y6_row10/` with status
`isolated_lhcb_node_acceptance_grid_failed_numeric_gate`; it must not be used
as a fiducial weight grid. The reliable bin-integrated result is in
`reports/lhcb_bin_acceptance_nnlo_all4_10m/`, and the fit-impact sensitivity
is in `reports/lhcb_nnlo_acceptance_effect/`.

## LHCb observable/provenance audit (2026-08-18)

The local data conversion was independently checked against the published
LHCb 7 TeV measurement (arXiv:1505.07024, JHEP 08 (2015) 039). The released
14-bin table has contiguous pT edges, the stored per-GeV values equal the
bin-integrated values divided by the bin widths, and the bin-integrated sum is
`75.976 pb`, consistent with the published `76.0 pb` total. The candidate
manifest reproduces the released per-GeV values exactly. The fiducial cuts
recorded in the manifest are $p_T^\mu>20$ GeV, $2<\eta^\mu<4.5$, and
$60<M_{\mu\mu}<120$ GeV at 7 TeV, consistent with the publication.

The reconstructed 14-by-14 pT correlation matrix is symmetric with unit
diagonal, and its high-qT sub-covariance is positive definite after adding
the released fully correlated beam-energy and luminosity components. The
machine-readable record is `reports/lhcb_observable_provenance_audit.json`.
This closes the hidden global-normalization/conversion question. The
remaining formal input gate is provenance packaging for the correlation
matrix components, not evidence for an arbitrary rescaling of rows 10--13.

## Convex-hull closure bound (2026-08-18)

The decisive method-level test is now complete. For the unitary construction

```text
matched = (1-p) W + p FO,  with 0 <= p <= 1,
```

an arbitrary profile and endpoint choice can only produce a convex combination
of the W and fixed-order terms. Using all 96 lambda=1 endpoints and the
NNLO scale-high fixed-order value gives the following optimistic upper bounds:

```text
row 10: 0.8520 / 1.0031 = 0.8494 of data
row 11: 0.5127 / 0.5753 = 0.8913 of data
row 12: 0.1697 / 0.1996 = 0.8503 of data
row 13: 0.007086 / 0.007324 = 0.9675 of data
```

Every W endpoint and every NNLO scale-high FO value is below the measured
value. Therefore no $F_{\rm NP}$ endpoint, local start, or unitary profile
can close these rows. The machine-readable result is
`reports/lhcb_convex_hull_closure/summary.json`.

This changes the scientific decision: rows `LHCb_7:10`--`13` cannot be
promoted into universal production through a method-level modification of the
TMD model. The remaining possible resolution is an independently validated
higher-order/fiducial fixed-order prediction or a formally revised external
experimental input. Until one exists, the four rows remain a documented
external finite-Y closure diagnostic, while the validated Tevatron production
scope remains unchanged.

## 2026-08-18 freeze and conventional-matching handoff

The validated lambda=1 identifiability result is frozen as the comparison
baseline. The finite-Y artifacts in this package remain immutable diagnostics;
the unitary `p*(FO_NLO-W)` construction is not being promoted as the final
conventional Y term. The hash-locked baseline record and new candidate scope
are under
`../full_n3ll_wy_production_2026/manifests/frozen_lambda1_baseline.json` and
`../full_n3ll_wy_production_2026/HANDOFF.md`.

The next candidate will implement and validate conventional
`Y=FO_NLO-ASY_NLO` for the 24-row Tevatron boundary data, while retaining the
329-row low-qT core in the same global 353-row scope. No frozen production
file or existing finite-Y report is overwritten.
## 2026-08-18 supersession by full NNLO-side target

The earlier finite-Y diagnostics and their `FO_NLO` notation are retained as
historical lower-order validation.  The active follow-up target is now the
isolated `full_n3ll_wy_production_2026` campaign with conventional
`Y_NNLO = FO_NNLO - ASY_NNLO` for the Tevatron scope.  The unitary `p*(FO-W)`
construction remains a cross-check only; neither it nor the earlier NLO-side
candidate is promoted.  The frozen lambda=1 production package is unchanged.
