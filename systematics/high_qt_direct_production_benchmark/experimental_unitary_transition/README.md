# Experimental unitary W-to-FO transition

This separately tagged subtree tests a unitary transition

```text
sigma = (1 - p) W + p FO
```

after the additive `W + (FO - ASY)` pilot failed. It does not modify or
supersede the accepted qT/Q <= 0.20 production scheme. Promotion requires
continuity, profile variation, full row coverage, fit-impact, replica, and
b-space stability gates.

The first pilot uses a C2 smootherstep profile from r=qT/Q=0.20 to 0.30 and
two bracketing transition windows. Since the available inputs are bin averages,
the pilot computes a bin-averaged profile and treats W and FO as constant
within that bin. A production candidate must instead integrate both spectra at
node level.

## Boundary pilot: CDF_RUN_2:36

- corrected Richardson W: 1.81875 pb/GeV
- external DYTurbo/MCFM average: 2.05700 pb/GeV
- central bin-averaged profile: 0.0000516
- central unitary prediction: 1.81876 pb/GeV
- early/central/late profile envelope: 1.81875--1.83375 pb/GeV

Algebraic convexity and C2 profile tests pass. Node-level spectral integration,
full Tevatron tier-1 coverage, fit impact, and replica stability remain false.
LHCb remains out of scope until a high-qT fiducial acceptance calculation is
available.

```text
outputs/unitary_smootherstep_v0_binavg/cdf_run_2_36/status.json
```

## Full Tevatron node campaign

All 24 Tier-1 Tevatron rows were evaluated with explicit qT/y quadrature and
the backend inclusive-rapidity approximation removed. Uniform-grid trapezoid
integration failed convergence. Simpson integration at n_b=320 and 640 passes
the 5% rowwise gate for every unitary profile:

- early profile maximum shift: 1.87%
- central profile maximum shift: 2.74%
- late profile maximum shift: 3.58%
- central profile maximum difference from FO for qT/Q>=0.28: 1.62%

The early/late profile envelope reaches 26.0% on one intermediate row, so
matching-profile variation must be included in fit-impact testing. The
node-level Tevatron numerical gate passes and a separately tagged central fit
is authorized. Replicas remain unauthorized until central-fit impact passes.

```text
summaries/unitary_smootherstep_v1_simpson_node_campaign/gate_status.json
summaries/unitary_smootherstep_v1_simpson_node_campaign/nb320_vs_nb640_rows.csv
```

## Frozen-state fit-impact precheck

Before optimization, the 24 added rows were evaluated with the accepted q020
dataset normalizations. The central unitary profile contributes chi2=2186.4,
or 91.1 per added row; early and late variants give 81.9 and 99.0 per row.
These pulls are too large to justify a refit. In the high-profile subset the
prediction is fixed-order dominated, so changing F_NP cannot resolve the
discrepancy.

The current external fixed-order normalization/order must be closed against
the measured observable, or a defensible higher-order K-factor and uncertainty
must be established, before central fits or replicas. Central-refit and replica
authorization are therefore revoked for this v1 candidate.

```text
summaries/unitary_smootherstep_v1_frozen_fit_impact/gate_status.json
summaries/unitary_smootherstep_v1_frozen_fit_impact/added_row_pulls.csv
```

## Fixed-order order/normalization audit

The apparent DYTurbo/MCFM NLO agreement was an order-labeling trap. All 24
external pairs contain only the first nonzero-qT contribution:

- MCFM explicitly uses `part = lo` and writes `Z_1jet_lo` output.
- DYTurbo uses `order = 1`, which its log calls NLO in inclusive-DY counting,
  but enables only `doVJ`; `doVJREAL` and `doVJVIRT` are both false.

Thus the two-code closure validates Z+jet LO, O(alpha_s), not genuine Z+jet
NLO, O(alpha_s^2). The 66--116 GeV mass windows and per-bin to pb/GeV
conversions match on all rows. Data/LO ranges from 1.61 to 2.01 (median 1.91),
and data remain 1.34--1.70 above even the largest existing LO scale variation.
An electroweak, mass-window, or bin-normalization factor is therefore not
indicated. A genuine real-plus-virtual Z+jet NLO campaign is required; an ad
hoc K-factor, fits, and replicas remain unauthorized.

A controlled genuine-NLO pilot has now completed for the fixed-order-dominated
row CDF_RUN_2:51. Independent configurations agree within 1.65% (0.50 times
their combined Monte Carlo uncertainty). The reference result is
1.4228 +/- 0.0279 pb/GeV, compared with 0.9356 pb/GeV at Z+jet LO and
1.736 pb/GeV in data. Thus NLO gives K=1.521 and explains most of the original
deficit, but data remain 22.0% higher (3.78 sigma including data and integration
uncertainties). Representative-row coverage and NLO scale variation are needed
before considering a 24-row campaign or revisiting fit impact.

That representative pilot now covers CDF_RUN_2:36, :45, and :51. Central NLO
K-factors range from 1.490 to 1.597, while data/central-NLO ranges from 1.147 to
1.327, confirming that a universal K-factor is inadequate. A standard
seven-point scale pilot on :51 gives 1.332--1.702 pb/GeV, or -6.4%/+19.6%
relative to the converged central result. The measured 1.736 pb/GeV point lies
just above the raw envelope but is consistent with the converged low-scale
endpoint after data and Monte Carlo uncertainties. The competing lower mixed
endpoint has now been converged to 1.3281 +/- 0.0229 pb/GeV.

A uniform ten-iteration seven-point study on the midpoint row gives
1.8178--2.3182 pb/GeV, or -6.8%/+18.8% relative to its central result. Its
2.590 pb/GeV measurement is above the raw envelope but only 1.61 sigma from
the low-scale endpoint after experimental and integration uncertainties. The
midpoint thus reproduces the anchor scale pattern. The representative and full
NLO scale gates pass, authorizing a 24-row central genuine-NLO campaign. Fits
and replicas remain blocked until that campaign is completed and its fit
impact is reassessed.

The authorized 24-row central NLO campaign is now complete at 500k real and
virtual calls with ten result iterations per row. NLO/LO varies from 1.432 to
1.645, confirming the need for rowwise NLO predictions; data/central-NLO ranges
from 1.039 to 1.327 with median 1.234. Substituting these results into the
frozen unitary profiles improves the central-profile added-row chi2 per row
from 91.1 to 64.8, but does not pass the fit-impact gate. The two rows with
central profile weight at least 0.95 still give chi2/row=11.6. Central refits
and replicas therefore remain blocked pending a correlated 24-row NLO scale-
uncertainty treatment.

That correlated treatment has now been tested. MCFM's internal scale
reweighting was rejected because its production-statistics anchor response
(+9.3% at correlated low scales) does not reproduce the independently
converged +19.6% anchor and +18.8% midpoint responses. Using their mean as a
fully correlated asymmetric Gaussian nuisance gives -6.75%/+19.22% on the NLO
component. The central profile prefers a 2.59-sigma upward excursion but still
has data chi2/row=50.4; its two fixed-order-locked rows retain chi2/row=11.35.
Ordinary correlated NLO scale uncertainty therefore does not rescue the
candidate. The next investigation is the transition-region W-to-NLO mismatch
and profile choice, not a fit or replica campaign.

A profile-window scan completes that diagnosis. The unconstrained optimum is
an extremely early/narrow 0.17--0.21 transition, with profiled data chi2/row
0.94 and a 1.23-sigma upward scale nuisance. It succeeds only by making the
prediction NLO dominated at the accepted qT/Q=0.20 boundary. Requiring the
transition to start at or above 0.20 gives a best chi2/row of 24.0; even the
broader constraint start>=0.18 and width>=0.08 gives 20.1. Thus no profile that
preserves the accepted low-qT production domain passes. This unitary candidate
must not be promoted; the remaining issue is the underlying W/NLO
incompatibility at the boundary.

## Boundary uncertainty correction

A subsequent audit found that the experimental-error-only frozen-impact gate
was not consistent with the accepted q020 fit. The exact-bin W prediction is
smooth across the cutoff, and its factor-of-two deficit is already present on
the last accepted rows. Those rows carry the accepted Collins-factorization
uncertainty

```text
sigma_fact = |data| * 0.5 * turn_on(r) * (r/0.1)^2,
```

which is roughly 200% of the data by r=0.20. Tapering this uncertainty by the
W weight while turning on the correlated NLO scale nuisance makes all three
profiles trivially consistent: data chi2/row is 0.17, 0.05, and 0.07 for the
early, central, and late profiles. The median tapered W uncertainty for the
central profile is 4.96 pb/GeV, much larger than the data themselves. Thus the
earlier raw-error rejection and profile-window optimum are superseded as
promotion evidence. The candidate is underdetermined, not statistically
excluded, and remains unauthorized because the honest uncertainty model has
almost no discriminating power. Matching should not resume until the W
factorization uncertainty near r=0.20 is independently calibrated or reduced.

A calculable diagnostic replacement is substantially more informative. Taking
`sigma_match=(1-profile)*|W-NLO|`, combining it with experimental errors, and
profiling the correlated NLO scale nuisance gives chi2/row 1.72, 1.55, and
1.66 for the early, central, and late profiles. The central profile has maximum
pull 1.74 and prefers a 1.57-sigma upward NLO scale shift. This indicates that a
controlled matching treatment may still be viable once the oversized accepted
factorization uncertainty is replaced. The proxy is not yet formally
authorized: its perturbative interpretation, correlation structure, and
double-counting with scale variation must be established before any fit.

An endpoint-separated correlated implementation passes that test operationally:
one Gaussian nuisance follows `(1-profile)*(NLO-W)` and a second follows
`profile*NLO*0.192174`. Each is fully correlated over rows, while the two are
independent; the matching direction vanishes in the NLO limit and the scale
direction vanishes in the W limit. Across early, central, and late profiles the
data chi2/row is 1.05--1.13, the matching nuisance is 1.45 sigma, the scale
nuisance is 1.16--1.21 sigma, and the largest pull is 2.37. This authorizes only
a separately tagged exploratory central fit implementing both nuisances.
Production promotion, an ordinary central refit, and replicas remain blocked.

The first separately tagged exploratory fit is complete with the accepted FNP
state frozen. It jointly profiles the three Tevatron dataset normalizations and
the two correlated theory nuisances. For the central profile, data chi2/row is
0.935, the matching and scale nuisances are 1.35 and 0.94 sigma, the maximum
pull is 2.43, and fitted normalization factors are 1.004 (CDF Run 1), 1.035
(CDF Run 2), and 1.000 (D0 Run 1). Early and late profiles give similarly
stable results. This frozen-FNP profile-fit gate passes and authorizes building
differentiable added-row kernels. It does not yet authorize an FNP refit,
replicas, or production promotion.

The differentiable added-row cache is now complete for all 24 rows at n_b=640
with 2x2 qT/rapidity quadrature and Simpson b integration. Each row stores the
perturbative b-space kernels and node kinematics independently. Reconstructing
the frozen accepted FNP prediction from these kernels agrees with the converged
node campaign on every row; the maximum relative difference is 1.37e-14. The
kernel gate passes and a separately tagged differentiable central FNP refit is
authorized. Replicas and production promotion remain blocked.

## Controlled regularized FNP convergence campaign

The unconstrained differentiable FNP refit was prediction-stable across three
starts but functionally non-identifiable, with pointwise FNP ranges as large as
78%. A separately tagged refit now anchors `log(F_NP)` to the accepted FNP on
an 8-by-161 `(x,b_T)` grid. A strength-100 calibration remained too weak: one
start retained a localized 25% FNP branch near `x=0.01`, `b_T=1`. At strength
1000, the maximum three-start FNP range where `F_NP>0.05` falls to 0.86%, the
maximum accepted-prediction range is 0.024 experimental sigma, and the maximum
boundary-prediction range is 0.00061 experimental sigma.

This passes the declared 2% functional local-stability criterion without
freezing the FNP, but all three best epochs remain near the 20,000-epoch
horizon and none passes the unchanged plateau criterion. Replicas and
production promotion therefore remain unauthorized pending resolution or a
documented justification of the persistent convergence behavior.

A subsequent staged joint polish used forced Adam rates of `2e-6` and `2e-7`
followed by L-BFGS. It did not establish stationarity: two starts retained FNP
gradient norms of 0.026--0.115, while the third reopened a 10.5% FNP branch.
The predeclared fallback therefore froze each stable strength-1000 FNP and
solved only the dataset-normalization and theory nuisances from cached W
predictions using double-precision nonlinear least squares. All three
conditional solves pass the `1e-4` nuisance-gradient gate (final norms are of
order `1e-6`) while preserving the 0.86% FNP range and prediction stability.

This is conditional frozen-FNP convergence, not joint FNP convergence. It does
not authorize ordinary FNP replicas. The next scientific choice is either a
fixed-FNP nuisance-only replica study or a reduced identifiable FNP
parameterization.

## Production-only FNP uniqueness control

An isolated control subsequently repeated the local-start test using only the
accepted 329-row production objective: the same FiLM architecture and accepted
data, `sigma_used`, W/Y kernels, dataset-normalization penalties, and no FNP
anchor or transition rows. The accepted checkpoint is reconstructed to within
`7e-6` experimental sigma, but it has a large objective gradient and its
original best epoch was the final epoch 1800. After 20,000 additional epochs,
none of three starts reaches a plateau and their best epochs remain at the
horizon. Cross-start predictions remain stable (maximum range 0.063
experimental sigma), while the FNP range reaches 23% where `F_NP>0.05`.

Thus the transition campaign did not create the basic non-identifiability; it
exposed a limitation already present in the unanchored production objective.
The earlier qmax=0.20 replica ensemble was not this control: it used fluctuated
pseudodata, head-only 500-epoch training, and a log-FNP anchor of strength 3.
It measures anchored uncertainty propagation rather than uniqueness of the
unanchored central solution. The accepted production files remain untouched.

Two derivative-only tail constraints were then tested on the isolated
production objective. A normalized curvature penalty on the damping rate
`h=-d_b log(F_NP)` at strength `3e-5` makes all three starts plateau and reduces
the `b_T<=1` range to 1.3%, but permits different smooth broad slopes and gives
a 43% range farther into the tail. A minimum-length penalty on `A(x,b_T)` in
`F_NP=exp(-b_T^2 A)` at strength `1e-2` makes each tail nearly constant in A,
but different starts choose different constant amplitudes; the maximum range
is 50% and moves inward to `x=0.01`, `b_T=0.65`.

Neither constraint is selected. Minimum bending or length controls tail
geometry but cannot determine its boundary amplitude when the data do not.
The next defensible model experiment is a reduced low-dimensional `A(x)` tail
with explicit smooth matching at the data-supported onset, not a stronger
derivative penalty or an implicit value anchor.

That reduced-tail gate was tested with nine positive log-x knot amplitudes,
smooth C2 matching to the FiLM core, the fixed production late-b floor, and an
initialization fitted to the accumulated accepted `log(F_NP)` tail. A matching
window from `b_T=1` to 2 targets the unstable region but gives data chi2=208.1
after 5000 epochs, versus 147.9 for the same-horizon unconstrained control. A
later window from 2 to 3 gives a more tolerable chi2=151.5, but begins after
the observed instability and therefore cannot resolve it. Neither candidate
passes the pilot gate, so a three-start campaign is not justified.

The failure is structural: the accepted FiLM solution uses important
b-dependent accumulated damping in the same region that would have to be
replaced to remove the degeneracy. If model reduction is pursued further, it
should test a globally reduced FNP form rather than grafting a constant-A tail
onto the flexible core.

A global reduced-model ladder was then tested directly against the 329-row
production objective with a learning rate appropriate to the small parameter
sets. Positive monotone Gaussian-like models with 9 and 18 amplitudes give
data chi2 values of 1367 and 2372 after 5000 epochs. Positive bilinear
`A(log x,b_T)` splines with 54 and 153 amplitudes improve this only to chi2=487
and 511, still far above the same-horizon unconstrained value 147.9. Increasing
the b-grid resolution therefore does not rescue the compact shared-x model.

No global reduced candidate passes the fit-quality pilot, so multi-start tests
are not launched. Further knot expansion would recreate the original flexible
model without establishing identifiability. The remaining honest choices are
to report only observable-level combinations that the data identify, or to
adopt an explicit physics prior and quantify its prior dependence.

```text
summaries/unitary_smootherstep_v1_regularized_fnp_anchor100_convergence_campaign/campaign_status.json
summaries/unitary_smootherstep_v1_regularized_fnp_anchor1000_convergence_campaign/campaign_status.json
summaries/unitary_smootherstep_v1_regularized_fnp_anchor1000_staged_polish_convergence_campaign/campaign_status.json
summaries/unitary_smootherstep_v1_regularized_fnp_anchor1000_scipy_nuisance_polish_convergence_campaign/campaign_status.json
summaries/production_fnp_stability_control/campaign_status.json
summaries/production_fnp_tail_regularization_scan/scan_status.json
summaries/production_fnp_reduced_tail_pilots/pilot_status.json
summaries/production_fnp_global_reduced_pilots/pilot_status.json
```

```text
summaries/unitary_smootherstep_v1_fo_order_audit/gate_status.json
summaries/unitary_smootherstep_v1_fo_order_audit/row_order_normalization_audit.csv
summaries/unitary_smootherstep_v1_fo_order_audit/nlo_pilot_status.json
summaries/unitary_smootherstep_v1_fo_order_audit/mcfm_nlo_pilot_convergence.csv
summaries/unitary_smootherstep_v1_fo_order_audit/nlo_representative_status.json
summaries/unitary_smootherstep_v1_fo_order_audit/mcfm_nlo_representative_rows.csv
summaries/unitary_smootherstep_v1_fo_order_audit/mcfm_nlo_anchor_scale_points.csv
summaries/unitary_smootherstep_v1_fo_order_audit/mcfm_nlo_midpoint_scale_points.csv
outputs/mcfm_zjet_nlo_24row_central_500k_i10/campaign_status.json
outputs/mcfm_zjet_nlo_24row_central_500k_i10/mcfm_benchmark_summary.csv
summaries/unitary_smootherstep_v1_nlo_24row/gate_status.json
summaries/unitary_smootherstep_v1_nlo_24row/mcfm_nlo_24row_audit.csv
summaries/unitary_smootherstep_v1_nlo_24row/nlo_frozen_unitary_pulls.csv
summaries/unitary_smootherstep_v1_nlo_correlated_scale/gate_status.json
summaries/unitary_smootherstep_v1_nlo_correlated_scale/profiled_correlated_scale_pulls.csv
summaries/unitary_smootherstep_v1_nlo_correlated_scale/profile_window_scan.csv
summaries/unitary_smootherstep_v1_w_nlo_boundary_audit/gate_status.json
summaries/unitary_smootherstep_v1_w_nlo_boundary_audit/boundary_pairs.csv
summaries/unitary_smootherstep_v1_w_nlo_boundary_audit/combined_uncertainty_pulls.csv
outputs/unitary_smootherstep_v1_exploratory_profile_fit/fit_status.json
outputs/unitary_smootherstep_v1_exploratory_profile_fit/fitted_rows.csv
outputs/unitary_smootherstep_v1_differentiable_kernels_nb640_nqt2_ny2/campaign_status.json
outputs/unitary_smootherstep_v1_differentiable_kernels_nb640_nqt2_ny2/kernel_manifest.csv
outputs/unitary_smootherstep_v1_differentiable_kernels_nb640_nqt2_ny2/frozen_fnp_reconstruction.csv
```
