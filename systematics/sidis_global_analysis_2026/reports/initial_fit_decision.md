# Initial DY + SIDIS fit decision (2026-08-26)

The first isolated joint objective has now been executed. It combines the
frozen lambda=1 DY W-only FiLM model with the identified COMPASS isoscalar
collinear pi/K release in the 2026 addendum. The SIDIS input is the
provenance-closed 746-row subset from
`data/derived/compass_collinear_provisional/`; one K- row is excluded because
the fixed-order NNFF10 central member gives a negative collinear ratio there.
No positivity clamp was applied.

## Result

The run is stored in `outputs/initial_joint_dy_compass_collinear_pilot/` and
is marked `initial_joint_dy_sidis_collinear_pilot_complete_not_production`.

| component | rows | chi2 | chi2/row |
|---|---:|---:|---:|
| frozen-lambda=1 DY W-only | 329 | 129.78 | 0.394 |
| COMPASS identified collinear SIDIS | 745 | 92,692.00 | 124.42 |

The DY component remains at the expected lambda=1 quality. The SIDIS value is
a deliberate closure failure, not a promoted fit: a central-point LO-like
FF/PDF ratio with four 10%-scale nuisance directions cannot reproduce the
published dM/dz shape. The failure identifies missing physics/interface work
(bin integration and the proper coefficient functions/normalization), not
evidence that the COMPASS data should be discarded.

## What is and is not established

* The LHAPDF adapter, NNFF10 NNLO central member, target-isospin bookkeeping,
  objective wiring, DY non-regression, and output provenance are exercised by
  a real joint run.
* The addendum is collinear only; it cannot constrain a transverse TMDFF width
  or validate the b_T/k_T transform.
* The public table has stat and point-to-point systematic components but no
  full covariance; quadrature is used only for this pilot objective.
* HERMES zxpt-3D values/covariance and the 1,547-row benchmark remain gated.
* The DY input is the frozen W-only lambda=1 anchor; this run does not claim a
  finite-Y global prediction.

The next required step is an independent scalar closure implementation for
the COMPASS multiplicity definition: integrate the numerator and denominator
over each published x/y/z bin, use the declared perturbative coefficient
order, and verify the result against the public correction/reweighting
information before any SIDIS row is promoted. A fit with a flexible empirical
correction would hide this failure and is therefore not authorized.

## Corrected pilot and external-FF closure follow-up

The first run initialized all four SIDIS normalization directions at one,
which left this deliberately small shared-F_iLM learning rate far from the
scalar closure minimum. The driver now initializes each scale to the median
data/theory ratio (the 10% log-normal prior is unchanged) and gives only those
four scalar parameters a separate learning rate. The resulting pilots are
recorded in `reports/initial_fit_trials.{json,md}`.

The primary independent-FF bin-average run,
`outputs/initial_joint_dy_compass_collinear_binavg_pilot_converged/`, gives:

| component | rows | chi2 | chi2/row |
|---|---:|---:|---:|
| frozen lambda=1 DY W-only | 329 | 129.719 | 0.3943 |
| COMPASS identified collinear SIDIS | 745 | 12,759.460 | 17.127 |

The longer 3000-epoch reoptimization is retained in
`...binavg_pilot_reoptimized/` and gives SIDIS chi2/row = 21.943. This spread
is an optimizer diagnostic, not a physical uncertainty estimate. Both DY
anchors remain at the expected 0.394/row quality. One of the 746 source rows is
excluded explicitly because the central signed NNFF10 fixed-order ratio is
negative; no positivity clamp is applied.

As a theory comparison only, the HAPS NNLO central grids give SIDIS
chi2/row = 2.941 for all 746 rows in
`...haps_binavg_pilot_reoptimized/`. HAPS is not independent here: its FFs
were fitted using modern COMPASS SIDIS information. This result therefore
cannot close the independent-data gate or be promoted as the first global
fit.

The all-member NNFF10 profile is in
`outputs/nnff10_replica_profile_diagnostic/` (midpoint) and
`outputs/nnff10_replica_profile_binavg_diagnostic/` (bin-average). In the
midpoint profile, the central member has chi2/valid-row = 17.56 for 745 valid
rows. The raw lowest-objective member becomes non-positive for hundreds of
rows and is invalid; the best member retaining all rows positive remains a
poor closure candidate. FF replicas therefore do not explain the mismatch.

### Decision

The campaign has now produced a genuine initial joint DY+SIDIS software fit,
but it is an explicitly rejected closure pilot, not a production result. The
next gate is a validated NNLO SIDIS coefficient-function plus inclusive-DIS
denominator/kinematic-normalization interface. HERMES identity and covariance,
full covariance treatment, transverse TMDFF data, and finite-Y scope remain
open. Frozen DY production files are unchanged.
