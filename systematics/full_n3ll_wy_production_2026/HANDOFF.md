# Full N3LL+NNLO W+Y production handoff

Last updated: 2026-08-26

## Current status pointer (2026-08-26)

This remains an isolated W+Y research archive. The 122-row Tevatron
unprimed-N3LL+NNLO external grid is complete and numerically checked as an
observable candidate, but no coupled TMD candidate in this directory has been
promoted. The corrected 329/353-row start and replica campaigns are diagnostic
comparisons to the immutable lambda=1 `full96x50` package. Six LHCb rows remain
W-only diagnostics because finite-Y observable, acceptance, subtraction, and
covariance closure is not production-ready. Read the repository
[`CODEX_HANDOFF.md`](../../CODEX_HANDOFF.md) for the cross-workstream status;
do not treat older running snapshots in this chronology as live jobs.

## Purpose

This is a new isolated research workspace for a genuine conventional
matched prediction,

$$
\begin{aligned}
\mathrm{d}\sigma = W_{\mathrm{N^3LL}} + Y_{\mathrm{NNLO}}, \qquad Y_{\mathrm{NNLO}} = \mathrm{FO}_{\mathrm{NNLO}}-\mathrm{ASY}_{\mathrm{NNLO}}.
\end{aligned}
$$

The first executable scope is the 122 published Tevatron qT bins (CDF Run I,
CDF Run II, and D0 Run I) plus the 24 Tevatron boundary rows for which
fiducial genuine-NNLO inputs can be generated with the installed DYTurbo
engine. The 329-row fixed-target/low-qT production core and the four high-qT
LHCb rows remain outside this external-engine candidate until their observable
and covariance conventions are separately closed.

All work in this directory is isolated.  No file below the active lambda=1
package, `production_frozen/`, the published source checkout, or the paper
may be overwritten by this campaign.

Many historical `outputs/` and `production_frozen/` paths below refer to the
working archive and are not present in the compact public checkout.  The
committed `reports/`, manifests, scripts, and `production/` package are the
portable record; excluded raw grids and checkpoints must be restored from the
archive before rerunning a numerical campaign.

## Frozen comparison baseline

The current identifiability result is frozen as the comparison reference, not
as evidence that it is already a full matched N3LL+Y prediction:

- production package: `../dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50/`;
- 96 stationary starts and 50 conditional experimental replicas;
- 4,800 crossed members per flavor;
- current Fig. 2 and Fig. 6 artifacts are recorded in
  `manifests/frozen_lambda1_baseline.json`;
- the baseline uses the accepted low-$q_T$ W-dominated result and has no
  promoted conventional finite-$Y$ term.

The freeze is hash-verified and must be checked with
`scripts/verify_frozen_baseline.py` before any candidate comparison.

## Accuracy target

The initial target is **unprimed N3LL+NNLO**, with an explicit convention table
for the cusp/noncusp evolution, hard function, OPE coefficients, PDF and
$\alpha_s$ orders, scale profiles, and bin integration.  A later N3LL' + NNLO
extension is not assumed by this campaign.

For the accuracy nomenclature, the Tevatron N3LL study used as the external
standard states that N3LL+NNLO contains the cusp anomalous dimension through
$\mathcal O(\alpha_s^4)$, non-cusp evolution through
$\mathcal O(\alpha_s^3)$, hard/collinear boundary terms through
$\mathcal O(\alpha_s^2)$, and NNLO fixed-order matching (see
`Fermilab-PUB-22-374-T`, Table I).  This is why the campaign target was raised
from the earlier NLO-side pilot to a genuine NNLO-side construction.

The primary matching formula must be the conventional additive remainder
`FO_NNLO - ASY_NNLO`.  The previously validated unitary transition
`p*(FO_NNLO-W)` is retained only as a diagnostic cross-check; it is not a
substitute for the conventional $Y$ term in the primary candidate.

## Promotion gates

1. The frozen baseline manifest verifies without hash drift.
2. The N3LL coefficient and convention inventory is complete; no config label
   is accepted as proof of perturbative accuracy.
3. The small-$b_T$ expansion of the exact W implementation reproduces the
   ASY subtraction at the declared order in the same scheme and normalization.
4. Independent NNLO fixed-order predictions reproduce the 24 Tevatron boundary
   observables after cuts, bin integration, and nuisance conventions.
5. `W+Y` is continuous, positive where required, numerically stable under
   integration refinement, and approaches fixed order in the matching region.
6. A frozen-FNP central fit reaches stationarity and preserves the cross-section
   quality gate.
7. The full experimental-replica and start-variation propagation is rerun for
   the candidate before any production consideration.

Failure of a gate records a rejected candidate and does not alter the frozen
lambda=1 result.

## Current state

The workspace was initialized on 2026-08-18.  The first executable tasks were:

1. verify the immutable baseline hashes;
2. build the exact N3LL convention/coefficient inventory from the published v22
   source and the current backend;
3. implement a same-scheme NNLO ASY expansion and compare it point-by-point with
   the external Tevatron NNLO grids;
4. only after those checks, construct a tagged W+Y diagnostic cache.

The existing finite-Y reports remain evidence and inputs, not production
outputs.  In particular, the 24-row unitary validation is useful for numerical
cross-checks, while the old additive route failed because its W and ASY pieces
were not mutually consistent near the transition node.

## Initial source audit (2026-08-18)

`reports/accuracy_inventory_v1.json` is the first candidate-side audit.  It
confirms that the current source is not yet a genuine full N3LL+NNLO W+Y
implementation: the `n3llp`/`n3ll_pilot` label returns the same `A1--A3,
B1--B2` set as NNLL, the available hard/OPE modules are one-loop, and the
default finite-tail path is still `y_mode=zero` or a development
`FO_real_dev - singular_dev` construction.  The v22 strict one-loop W
expansion is useful for lower-order diagnostics, but it does not close the
conventional `FO_NNLO - ASY_NNLO` production gate by itself.  The installed
DYTurbo 1.4.2 engine is an available NNLO fixed-order source; its Tevatron
row-level card and numerical-uncertainty validation is a separate gate.  The
candidate-side Gaussian profile and its first diagonal pseudo-replica layer are
now complete, but promotion remains unauthorized.

An isolated capability probe for `CDF_RUN_2:17` passed with DYTurbo order 3,
`primed=false`, and both real/virtual V+jet pieces enabled; see
`reports/dyturbo_nnlo_probe/probe_status.json`.  The probe is deliberately
boson-level and is not yet a production fixed-order grid.  It establishes that
the external NNLO engine is available, while the remaining work is to reproduce
all Tevatron boundary bins with the exact observable conventions and construct
the same-scheme NNLO asymptotic subtraction.

The complete 24-row boundary run is now launched by
`scripts/run_tevatron_boundary_nnlo.py`; all cards, logs, tables, and the
candidate-local input snapshot are written below
`reports/dyturbo_nnlo_boundary/`.  The coefficient and runtime provenance for
this route is recorded in `reports/dyturbo_n3ll_source_map.json`.  These files
are validation inputs only and do not authorize a production refit.

The 24-row run completed successfully.  DYTurbo's NNLO central values are
finite and positive with 3.65% mean (7.50% maximum) Monte-Carlo integration
uncertainty.  Its unnormalized ratio to the data is 0.788 median (0.720--0.957
range); after a per-dataset normalization diagnostic, the statistical-only
χ²/row is 0.75 for CDF Run I, 2.03 for CDF Run II, and 0.37 for D0.  This is
not yet a fit or a final closure claim: the result still needs the production
nuisance/covariance convention and a same-scheme `ASY_NNLO` comparison.  The
unit conversion is explicit in `boundary_status.json` and the companion CSV
(`DYTurbo` text is fb/bin; the reported diagnostic is pb/GeV).

The complete external matched calculation then ran over the same 24 rows with
resummed W, NNLO counterterm, and NNLO V+jet real/virtual pieces all enabled:
`reports/dyturbo_full_n3ll_nnlo_boundary/`.  All rows are finite and positive;
the median full-W+Y/data ratio is 0.974 (range 0.852--1.179).  Three bins were
refined at 15 million Vegas calls per component, reducing the maximum
integration uncertainty to 7.29% (the unrefined 3-million-call pass peaked at
12.7%).  This establishes a numerically usable perturbative observable oracle,
not yet a fitted TMD or a promoted production result.

An all-Tevatron grid run (41 CDF Run I, 61 CDF Run II, and 20 D0 Run I qT
bins) was completed under
`reports/dyturbo_full_n3ll_nnlo_tevatron_grid/`.  The initial pass used one
million calls per Vegas component; cancellation-dominated rows were then
refined before using the grid for candidate diagnostics.

That g1=0 coverage grid is now complete and has been selectively refined:
the median full-W+Y/data ratio is 0.974, mean MC relative uncertainty 4.18%,
and maximum 7.93% after the 100-million-call check of the worst low-qT bin.
The unphysical-looking excess at qT=0 (ratios about 2) is the expected
zero-NP test, not a perturbative failure.  A candidate-local Gaussian NP pilot
(`reports/dyturbo_npff_pilot/`) shows that g1 around 1 GeV^2 brings the qT=0
CDF Run II bin to ratio 1.01 while leaving the qT≈9 GeV bin at ratio 0.96.
The independent g1=1 full grid is complete and has its own numerical checks. It
uses the same unprimed order-3 DYTurbo engine, with a candidate Gaussian
nonperturbative factor $\exp[-g_1 b_T^2]$, $g_1=1\ \mathrm{GeV}^2$. After
30-million-call refinements of the cancellation-dominated bins and a final
100-million-call check of the three worst rows, the 122-row grid has median
full-W+Y/data ratio 1.001, mean relative integration uncertainty 4.19%, and
maximum 8.99%. After the final low-qT refinements, the direct candidate fit
has stat-only chi2/row 1.58, RMS pull 1.26, and median prediction/data ratio
1.001. The candidate comparison gives stat-only chi2/row 1.58,
compared with 12.79 for the zero-NP control; these are external-grid
diagnostics, not the final correlated production fit. The g1=1 grid is stored
under `reports/dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0/`.

The conventional additive decomposition was also validated independently for
`CDF_RUN_2:17` at 30 million calls per Vegas component. DYTurbo returns
`RES=5689.608`, `CT=-5204.185`, and `VJ=5221.492` fb/bin, so

$$
\begin{aligned}
W=\mathrm{RES},\qquad ASY=-\mathrm{CT},\qquad FO=\mathrm{VJ},\qquad Y=FO-ASY=\mathrm{VJ}+\mathrm{CT},
\end{aligned}
$$

and the reconstructed `RES+CT+VJ=5706.915` fb/bin agrees exactly with the
all-terms run at the same integration settings. This establishes the
candidate-side conventional $Y=FO_{\rm NNLO}-ASY_{\rm NNLO}$ implementation
for the external engine; it does not yet constitute a fitted 353-row
production extraction.
The compact perturbative-closure record is
`reports/accuracy_closure_v2.json`.

The same $g_1=1$ candidate was rerun on all 24 Tevatron boundary rows. After
15-million-call refinements of the three cancellation-dominated rows, all
values remain finite and positive; the median ratio to data is 0.996, mean
relative integration uncertainty 3.96%, and maximum 7.16%. The result is in
`reports/dyturbo_full_n3ll_nnlo_boundary_g1_1p0/`.

The first direct candidate summary is
`reports/tevatron_n3ll_nnlo_candidate_fit_status.json`. On the 122-row
Tevatron grid, the $g_1=1$ candidate has stat-only chi2/row 1.577, RMS pull
1.26, and median prediction/data ratio 1.001. This is a candidate central
observable fit only: it has no DNN starts and no experimental-replica band,
and therefore remains explicitly non-production.
The candidate artifact hashes are recorded in
`manifests/tevatron_g1_1_candidate.json`.

The next gate is an isolated Tevatron fit using this explicit W+Y engine (with
the Gaussian NP parameters treated as candidate model parameters), followed
by stationarity, start variation, and replica propagation. No candidate is
authorized to replace the frozen lambda=1 DNN result until those gates pass.

An endpoint profile over the two directly evaluated grids gives a provisional
best $g_1=1.024\ \mathrm{GeV}^2$ in the endpoint-linear profile, with
stat-only chi2/row 1.571 versus 1.577
at the directly evaluated $g_1=1$ grid. The small difference means $g_1=1$
is a suitable direct candidate while the fit runner is built; the profile is
not being mistaken for a direct evaluation. The profile and its per-dataset
normalization diagnostics are in
`reports/tevatron_gaussian_np_candidate_profile.json`.

An independent-random-seed $g_1=1$ 122-row grid was run under
`reports/dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p0_seed1357911/`. Its cards
use unique DYTurbo table names; the earlier cache-collision attempt was not
accepted. This was a numerical-integration stability check, not an additional
model-start distribution.

The independent-seed check is complete after additional 30-million-call
refinements of the largest discrepancies and a 100-million-call check of
`CDF_RUN_2:1`. The two 122-row grids have RMS difference 0.80 combined MC
sigma and maximum 1.74 combined MC sigma; the remaining largest relative
differences are all in cancellation-sensitive bins and are covered by their
reported integration errors. This validates numerical integration stability,
not uniqueness or experimental uncertainty. Results are summarized in
`reports/tevatron_external_seed_stability.json`.

A first 50-member point-to-point pseudo-replica propagation is complete in
`reports/tevatron_gaussian_np_replica_profile/`. Profiling the common Gaussian
parameter gives $g_1=1.031^{+0.019}_{-0.032}\ \mathrm{GeV}^2$ (16--84%
quantiles). This is only the diagonal experimental-error layer; correlated
normalization covariances, PDF replicas, and model-form variations remain
open, and no final 1-sigma production band is claimed.

## Freeze and active next phase (2026-08-18)

The lambda=1 empirical reference is now an immutable comparison baseline. The
freeze audit reports `frozen_baseline_verified`, with all 16 recorded artifacts
matching their SHA-256 hashes and zero writes performed. The audit is
`scripts/verify_frozen_baseline.py`; its manifest is
`manifests/frozen_lambda1_baseline.json`. The new candidate manifest is
`manifests/tevatron_g1_1_candidate.json`; it is also explicitly non-production.

The active production candidate is the installed external DYTurbo route with
unprimed order 3 (`primed=false`), conventional
`W=RES`, `ASY=-CT`, `FO=VJ`, and `Y=VJ+CT`, with a Gaussian
`exp(-g_1 b_T^2)` nonperturbative factor. The directly evaluated Tevatron
candidate uses `g_1=1 GeV^2`; the endpoint-linear profile prefers
`1.024 GeV^2` but lies just beyond the directly evaluated `[0,1]` interval and
is not promoted without a direct rerun.

The current external candidate has passed observable-level finite/positive,
term-reconstruction, 24-row boundary, 122-row Tevatron-grid, and independent
integration-seed checks. It has **not** yet passed production promotion: the
DYTurbo grid is an observable-level Gaussian candidate rather than a refit of
the 353-row TMD/F_NP model, the diagonal 50-member profile is not a correlated
experimental/PDF/model-form uncertainty, and the fixed-target 329-row core has
not been evaluated with this external NNLO engine. The next automatic work is
therefore to (i) directly evaluate the profiled Gaussian point as a controlled
candidate variation, (ii) build the Tevatron W+Y candidate fit and its complete
correlated/start/replica propagation, (iii) generate candidate Fig. 2/Fig. 6
comparisons, and (iv) promote only if every gate in this handoff passes.
The isolated candidate-only Fig. 2/6 renderings are frozen under
`reports/tevatron_candidate_fig2_fig6_g1_1p0/`, with their artifact hashes in
`manifests/tevatron_candidate_figures_g1_1p0.json`. They use the frozen
perturbative b-space factors times the Gaussian candidate and the diagonal
50-member (g_1) profile; they are visual diagnostics only and must not be
copied over the production figures.

## Fixed-target x_F observable audit (2026-08-19)

The first fixed-target DYTurbo convention probe used a narrow central-y slice
for E288.  That is not a valid comparison for the published E288 rows, which
carry finite x_F intervals.  The isolated diagnostic
`scripts/probe_fixed_target_xf_observable.py` therefore evaluates E288_200:0
with `dsdxf=true`, `edsdp3=true`, the published x_F interval, an explicit Be
target (`Z=4`, `A=9`), and the same unprimed N3LL+NNLO W+Y terms used by the
Tevatron candidate.  Its result is stored under
`reports/fixed_target_xf_observable_probe/`.

The run is numerically complete but not a convention closure: DYTurbo returns
8363.17 in its native fb-based output for the whole-nucleus card; dividing by
the target mass number and by 1000 for pb still leaves a factor about 235
relative to the tabulated E288_200:0 `CS`.  The discrepancy is not interpreted
as a physics result because the published invariant-cross-section unit,
x_F/mass-bin averaging, and target/acceptance normalization have not yet been
reconciled.  The fixed-target scope remains fail-closed and no 353-row
production grid is authorized.  Do not reuse the earlier central-y probe as an
x_F validation.

## Direct profiled candidate check (2026-08-18)

The endpoint profile point was evaluated directly rather than accepted from
the profile interpolation. The controlled candidate uses the same external
DYTurbo unprimed order-3 cards and conventional decomposition, with

$$
\begin{aligned}
g_1=1.0241911542738864\ \mathrm{GeV}^2 .
\end{aligned}
$$

The clean 122-row Tevatron grid was run at 30 million calls per row, with the
largest cancellation-dominated rows refined to 100 million calls. All
predictions are finite and positive. The direct candidate gives a stat-only
$\chi^2/N=1.0963$, RMS statistical pull 1.047, median prediction/data
ratio 0.9944, mean relative integration uncertainty 1.92%, and maximum
10.9%. The directly evaluated $g_1=1$ reference has
$\chi^2/N=1.5771$; this improvement is an observable-level candidate
comparison, not a replacement of the frozen F_NP extraction.

The 24-row Tevatron boundary oracle was independently rerun at the profiled
point with 3 million calls and 15-million-call refinement of the three worst
cancellation rows. It gives $\chi^2/N=0.9321$, median ratio 0.9960,
mean integration uncertainty 3.96%, and maximum 7.16%. The corresponding
$g_1=1$ boundary value is $\chi^2/N=0.9317$, so the profile changes the
boundary quality negligibly.

These checks establish a stronger external observable-level candidate, but
not production promotion. The candidate still has no full 353-row refit of
the DNN/F_NP model, no correlated experimental/PDF/model-form uncertainty,
and no external-engine closure for the 329 fixed-target low-qT rows or the
four unresolved high-qT LHCb rows. The profiled candidate figures are frozen
as diagnostics under
`reports/tevatron_candidate_fig2_fig6_g1_1p024191/`, with hashes in
`manifests/tevatron_candidate_figures_g1_1p02419.json`; the complete
candidate manifest is `manifests/tevatron_g1_1p02419_candidate.json`.
Promotion remains unauthorized until the Tevatron W+Y fit, stationarity,
start distribution, and full replica propagation are complete.

The first covariance-aware observable-level fit is now recorded in
`reports/tevatron_external_candidate_correlated_fit.json`. It profiles one
common Gaussian (g_1) together with the released one-per-dataset
normalization nuisances, using the diagonal point-to-point errors and the
published normalization fractions. The interpolated minimum is
$g_1=1.017\ \mathrm{GeV}^2$ with $\chi^2/N=0.7853$ over the 122 rows.
This is a fit-quality diagnostic only: it uses direct DYTurbo grids at
$g_1=0,1,1.024191$ and piecewise-linear interpolation between them, so the
fitted point must be checked by a direct external run before any promotion.
That direct run and a seven-point $(\mu_R,\mu_F)$ envelope are active in
the isolated campaign.

## Direct controlled profile point and 150-member diagnostic (2026-08-18)

To separate a profile-interpolation artifact from a directly evaluated candidate,
the external unprimed N3LL+NNLO grid was rerun at the controlled point
`g1=1.017 GeV^2`. The 122-row grid used 30 million Vegas calls per
component, followed by 100-million and 300-million-call refinements of the
cancellation-dominated rows. The final grid is under
`reports/dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p017_30m/`; its largest
reported relative integration uncertainty is 5.29%. All values are finite and
positive. The direct diagnostics are:

* stat-only chi2/row = 0.9653, RMS statistical pull = 0.9825;
* median prediction/data ratio = 0.9975;
* mean relative integration uncertainty = 1.77%, maximum = 5.29%.

The one-normalization-nuisance covariance diagnostic at the same direct point
gives chi2/row = 0.8164. A piecewise interpolation through directly evaluated
`g1=0`, `1`, `1.017`, and `1.024191` grids has an interpolated minimum at
`g1=1.013` with chi2/row = 0.7399, but that value is not accepted as a result
until directly evaluated. The direct `g1=1.017` point is therefore the
controlled candidate used for the next propagation step.

An isolated 150-member diagonal point-to-point pseudo-replica layer was also
run at this candidate using seed `20260819`. It gives a profile
`g1=1.0278` with q16--q84 = `[0.9952, 1.0526] GeV^2`; the corresponding
chi2/row quantiles are median 2.516, q16 2.274, q84 2.756. This layer uses
only released point-to-point errors: it has no correlated covariance, PDF
replicas, or model-form/start variation and is explicitly not a formal 1sigma
production band. Candidate-only Fig. 2 and Fig. 6 renderings based on this
layer are frozen under
`reports/tevatron_candidate_fig2_fig6_g1_1p017/`, with hashes in
`manifests/tevatron_candidate_figures_g1_1p017.json`.

These checks strengthen the observable-level conventional
`Y=FO_NNLO-ASY_NNLO` candidate, but they do not yet authorize production:
the external route covers the 122 published Tevatron qT bins plus the
24-row Tevatron boundary oracle, not the fixed-target 329-row DNN/F_NP core;
the external Gaussian candidate is not a full refit of the flexible F_NP
model; and the complete correlated experimental/PDF/model-form/start
propagation is still open. The seven-point scale variation at the profiled
candidate is running under
`reports/tevatron_scale_variations_g1_1p024191/`. Frozen lambda=1 baseline
artifacts remain unchanged and are rechecked before any promotion decision.

The perturbative provenance distinction is now explicit: the external DYTurbo
order-3 unprimed engine does provide the same-scheme counterterm needed for
`ASY_NNLO` (`ASY=-CT` in the candidate convention), and the observable-level
`W+Y=RES+CT+VJ` closure is therefore genuine for the Tevatron rows evaluated
here. The coefficients have not been imported into the older fitted v22 W
backend; that older backend remains a separate lower-order/pilot path and is
not being silently called N3LL+NNLO.

The first 150-member diagnostic used an older endpoint interpolation centered
on the g1=1 grid. It is retained for audit history, but the controlled
candidate now uses the corrected centered layer
`reports/tevatron_gaussian_np_replica_profile_150_g1_1p017/`. This layer uses
all four directly evaluated grids (`g1=0, 1, 1.017, 1.024191`) and profiles a
dense piecewise-linear observable curve around the direct `g1=1.017` point.
Its profile is `g1=1.0125` with q16--q84 = `[1.0110, 1.0200] GeV^2` and
chi2/row quantiles median 1.831, q16 1.666, q84 2.023. The resulting
candidate-only figures are under
`reports/tevatron_candidate_fig2_fig6_g1_1p017_centered/`, with the frozen
hash manifest `manifests/tevatron_candidate_figures_g1_1p017_centered.json`.
This remains a diagonal point-to-point diagnostic, not a formal production
1sigma band.

## Perturbative scale-variation closure (2026-08-18)

The first seven-point `(muR,muF)` pass at three million Vegas calls per
component was completed over all 122 Tevatron bins. It is deliberately not
interpreted as a physical scale envelope: the low-qT cancellation floor left
three negative point estimates and gave a central stat-only chi2/N=83.7, with
relative MC errors above 50% in 22 row/scale cards. The raw diagnostic is under
`reports/tevatron_scale_variations_g1_1p024191/scale_variation_status.json`;
its large apparent envelope is therefore a numerical-integration artifact until
refinement. The isolated high-statistics refinement runner
`scripts/refine_tevatron_scale_variation_rows.py` is now rerunning those 22
cards at 30 million calls per component. A refined scale envelope will only be
used after all selected rows are finite, positive, and stable under the stated
MC uncertainties.

The 22-card refinement completed with all selected values finite and positive;
the record is `reports/tevatron_scale_variations_g1_1p024191/scale_variation_refinement_status.json`.
Three central `(muR,muF)=(1,1)` rows were then rerun at 100 million calls per
component (`scale_variation_central_refinement_status.json`). The updated
central diagnostic has chi2/N=3.17 and maximum remaining relative MC error
46.8%; the scale half-width maximum is about 52.5%. These numbers are still
integration-limited rather than a final perturbative uncertainty band. The
candidate W+Y central grid remains valid independently; the scale campaign is
being retained as a conservative numerical/perturbative diagnostic until the
remaining high-error rows are either refined or explicitly excluded with a
documented precision floor.

The direct interpolation check at `g1=1.013 GeV^2` is now complete under
`reports/dyturbo_full_n3ll_nnlo_tevatron_grid_g1_1p013_30m/`. All 122 values
are finite and positive; mean relative MC uncertainty is 2.14%, maximum
24.1%, and median prediction/data ratio 0.9973. Its direct stat-only
chi2/N=1.277, which is worse than the directly evaluated `g1=1.017`
candidate (chi2/N=0.9653). The earlier interpolated minimum at 1.013 was
therefore a profile-interpolation artifact and is rejected. The direct
g1=1.017 candidate remains the controlled center for subsequent propagation.

The direct profile decision is recorded in
`reports/tevatron_g1_direct_profile_decision.json`: among the directly
evaluated points, `g1=1.017` is the retained center (chi2/N=0.9653), while
`g1=1.013` is rejected (chi2/N=1.2771), despite its lower interpolated
profile value. The 24-row boundary oracle at the retained point is now running
under `reports/dyturbo_full_n3ll_nnlo_boundary_g1_1p017/`.

The `g1=1.017` boundary oracle completed with all 24 rows finite and positive.
After 15-million-call refinements of CDF Run II rows 44, 46, and 47, the
median full-W+Y/data ratio is 0.99585, the mean relative MC uncertainty is
3.96%, and the maximum is 7.16%. This matches the numerical quality of the
earlier g1=1 and profiled boundary checks and is recorded in
`boundary_full_wy_status.json`.

## Live batch state and resume instructions (2026-08-19)

The isolated genuine W+Y batch is currently running and must not be confused
with a frozen production replacement. The central high-statistics run is:

```
python scripts/run_tevatron_full_n3ll_nnlo_grid.py \
  --g1 1.017 --calls 100000000 --seed 20260818 \
  --out reports/tevatron_n3ll_nnlo_wy_production_g1_1p017
```

It executes CDF Run I first. An independent 30-million-call run with seed
`20260820` executes the same external engine sequentially over the three
datasets under
`reports/tevatron_n3ll_nnlo_wy_stationarity_g1_1p017_seed_20260820/`.
The stationarity comparison is not accepted until both complete.

An isolated supervisor watches the central CDF-I log. When its DYTurbo
summary appears, it launches the central CDF-II and D0 cards in parallel at
100 million calls, merges the three tables with
`scripts/merge_tevatron_grid_parts.py`, and runs
`scripts/finalize_tevatron_n3ll_production_batch.py`. Separate supervisors
then run the stationarity comparison, re-verify the frozen baseline, and
launch the 500-member diagnostic replica profile. If a session is
interrupted, inspect the two grid logs and resume only missing dataset parts;
never overwrite the frozen lambda=1 package.

The current external scope is 122 published Tevatron qT bins plus the
24-row transition oracle. It is not a 353-row N3LL claim: the 329-row
fixed-target DNN/F_NP core remains numerically unclosed in the external
engine. The running batch is therefore a genuine Tevatron W+Y candidate,
with promotion still blocked on stationarity, uncertainty layers, and
fixed-target closure. The last freeze audit during this batch reports
`frozen_baseline_verified` (16/16 hashes, zero writes).

## Freeze-and-launch decision (2026-08-19)

The lambda=1 empirical 96-start by 50-replica extraction is now the
immutable reference baseline for all comparisons. Its 16-artifact manifest
passes the read-only verifier with 16/16 hashes and zero writes. No files in
`dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50/`
may be edited by this W+Y study.

The new production candidate is an isolated, directly evaluated unprimed
N3LL+NNLO Tevatron calculation at the retained direct Gaussian center
`g1=1.017 GeV^2`, using

$$
\begin{aligned}
d\sigma = W_{\rm N^3LL} + Y_{\rm NNLO},\qquad Y_{\rm NNLO}=FO_{\rm NNLO}-ASY_{\rm NNLO},
\end{aligned}
$$

with DYTurbo's term identity `W=RES`, `ASY=-CT`, `FO=VJ`, so the evaluated
observable is `RES+CT+VJ`. The high-statistics batch covers the 122
published CDF Run-I, CDF Run-II, and D0 Run-I Tevatron qT bins. Its parts are
row-count guarded; a partial dataset cannot be treated as a complete grid.
An independent 122-row stationarity batch is already complete. Once the
primary reaches 122 rows, the automated guard runs precision refinement,
finalization, stationarity comparison, the 500-member diagnostic replica
propagation, frozen-baseline verification, candidate figures, and a
review-only candidate freeze manifest. The candidate remains unpromoted
until all uncertainty and closure gates pass.

The subsequent seven-point scale watcher is downstream of the candidate
freeze and uses a fresh `g1=1.017` run; it remains diagnostic until the
numerical precision floor is demonstrated. Fixed-target rows and unresolved
high-qT LHCb rows remain outside this genuine Tevatron W+Y claim.

## Frozen baseline and new genuine Tevatron production run (2026-08-18)

Before starting the new perturbative-accuracy run, the lambda=1 empirical
reference was verified again with `scripts/verify_frozen_baseline.py`:
16/16 recorded production artifacts match their expected SHA256 values and
the verifier performed zero writes. The frozen package is therefore the
comparison reference and is not an input that this campaign may modify.

The new isolated production target is the directly evaluated, unprimed
N3LL+NNLO observable

$$
\begin{aligned}
d\sigma = W_{\rm N3LL} + Y_{\rm NNLO},\qquad Y_{\rm NNLO}=FO_{\rm NNLO}-ASY_{\rm NNLO},
\end{aligned}
$$

with the DYTurbo decomposition `W=RES`, `ASY=-CT`, `FO=VJ`, and hence
`W+Y=RES+CT+VJ`. It is being run at the retained direct Gaussian
nonperturbative center `g1=1.017 GeV^2`, not at the rejected interpolation
minimum. The first production batch covers every externally achievable
published Tevatron qT bin (CDF Run I, CDF Run II, D0 Run I; 122 rows), with
the separate 24-row Tevatron transition oracle retained alongside it. The
329-row fixed-target DNN/F_NP core remains explicitly outside this external
DYTurbo batch until a same-scheme fixed-target implementation is validated;
it must not be silently labeled N3LL+NNLO.

The high-statistics batch is isolated at
`reports/tevatron_n3ll_nnlo_wy_production_g1_1p017/`, with its own cards,
logs, grid, and status manifest. It is a genuine W+Y calculation, but it is
not yet a promoted replacement: promotion still requires numerical
integration closure, a stationarity check, and the full experimental,
replica/PDF, model-form, and F_NP/start propagation. The scale-variation
campaign remains a diagnostic until its cancellation-dominated rows reach a
documented precision floor.

An isolated fixed-target capability probe was also attempted for `E288_200:0`
using the same unprimed N3LL+NNLO card with `ih1=ih2=1`. At one million calls
the cancellation produced a negative noisy estimate; at 30 million calls the
estimate was `2.76 +- 6.24 fb` for the bin. This is finite and positive only
as a central point but remains integration-limited by a factor greater than
two, so it is not a usable fixed-target production prediction. The probe is
recorded under `reports/fixed_target_n3ll_nnlo_capability_probe/`; the present
high-statistics production scope is therefore the externally stable Tevatron
set plus its 24 transition rows. Fixed-target rows require a better numerical
integration strategy or a separately validated fixed-target implementation
before they can be included in a genuine full N3LL+NNLO fit.

Once the 100M-call central grid is complete, an automated 500-member
Tevatron replica propagation is queued at
`reports/tevatron_n3ll_nnlo_wy_replica_profile_500_g1_1p017/`. It includes the
released diagonal point-to-point errors and one profiled normalization
nuisance per CDF/D0 dataset. PDF replicas, F_NP/start variation, model-form
variation, and the scale envelope remain open gates; this q16--q84 result is
therefore an uncertainty diagnostic, not a formal production 1-sigma band.

## Fixed-target integration-method probe (2026-08-19)

The first fixed-target capability probe used automatic Vegas integration for
the NNLO V+jet term and returned `2.8 +- 6.2 fb` for `E288_200:0`, so it was
not usable. A new isolated probe,
`scripts/run_fixed_target_quadrature_probe.py`, tests the documented
quadrature route for fixed-target bins, which have qT, rapidity, and mass
boundaries but no nontrivial lepton cuts. With `VJquad=true`,
`intDimVJ=3` (and independently 5), and `makecuts=false`, it returns
`6.366533 +- 0.000254 fb` in 13 seconds. The d3 and d5 no-cut results agree
bit-for-bit. The cut-aware d5 variant (`makecuts=true`) exits with code 255
and no table, so it is retained as a failed diagnostic rather than silently
used.

This result is evidence that the earlier fixed-target failure was dominated
by the high-dimensional cut-aware V+jet integration, but it is not yet a
329-row closure. The no-cut route must still be checked against the exact
fixed-target observable conventions, all RES/ASY/VJ terms, bin units, and
independent numerical settings before a full fixed-target production batch
can be launched. The probe artifacts are under
`reports/fixed_target_quadrature_probes/`; the frozen lambda=1 package was not
read or written by the test.

## Fixed-target nuclear-card check (2026-08-19)

The fixed-target quadrature route was then checked with DYTurbo's explicit
`nuclearpdf` branch rather than the earlier proton-target placeholder. For
the historical target approximations already used by the internal backend
(E288: Be, (Z=4,A=9); E605: Cu, (Z=29,A=63.546)), the card must represent
the beam proton as (Z_1=A_1=1); setting those beam counts to zero silently
returns a zero cross section. With the corrected card, E288_200:0 gives
`78.4242 +- 0.0024 fb` and E605:0 gives `42.384 +- 0.031 fb` under the
no-cut `VJquad=true`, `intDimVJ=3` route. DYTurbo's nuclear branch returns
the whole-nucleus PDF sum, so these numbers cannot yet be compared to the
fit CSV without a verified per-nucleon/normalization convention. E772's
target composition is still absent from the source CSV and must not be
silently assigned a nucleus. These are isolated card/integration checks,
not a fixed-target production closure; the external Tevatron batch remains
the only live N3LL+NNLO production scope.

## Tevatron batch restartability correction (2026-08-19)

The first 100M-call CDF Run-I subprocess reached its final bins but was
terminated by the isolated runner's one-hour subprocess timeout before its
DYTurbo table was written. This was an execution limit, not a numerical
physics result. The runner now exposes a per-dataset timeout (default 7200 s;
the restart supervisor uses 10800 s), and the batch is being rerun from the
incomplete CDF Run-I table with a fail-and-retry supervisor. Completion is
recognized only from a valid 41-row CDF Run-I status, followed by independent
61-row CDF Run-II and 20-row D0 Run-I parts and a 122-row merge. The original
stationarity run remains independent and unchanged.
### Row-count-guarded post-processing (2026-08-19)

The earlier background watchers were retired because they treated the
existence of `grid_status.json` as completion.  The runner writes that file
after each dataset, so CDF Run-I alone can temporarily appear as a valid
41-row grid.  The new `scripts/supervise_tevatron_postbatch.sh` waits for
`row_count == 122` in both the high-statistics primary and independent-seed
stationarity outputs before running the finalizer, stationarity comparison,
500-member replica diagnostic, and frozen-baseline verifier.  This guard is
candidate-side only and does not alter the frozen lambda=1 package.

The same post-processing step writes
`reports/tevatron_n3ll_nnlo_wy_campaign_summary.json` through
`scripts/summarize_tevatron_n3ll_campaign.py`.  That summary is deliberately
fail-closed: a complete external W+Y grid is not treated as a finished TMD
production result while PDF, F_NP/start/model-form, scale-floor, and
fixed-target closure gates remain open.

The fail-closed finalizer was also corrected to compute integration precision
from the runner's raw per-bin uncertainty columns; it is syntax-checked and
will be exercised only after the merged 122-row table exists.
The finalized status also records the exact card-level switches
(`fixedorder_only=false`, `order=3`, `primed=false`, `doBORN/doCT/doVJREAL/doVJVIRT=true`,
`VJquad=false`) and the term identity `RES+CT+VJ = W+(FO_NNLO-ASY_NNLO)`.
DYTurbo's human-readable banner calls this “N3LO+N3LL” because its inclusive
order counter includes the Born channel; in the unprimed convention used here,
`order=3` maps to N3LL resummation plus NNLO V+jet, as recorded in the source
map and term-decomposition audit.  It is not an NNLO+N3LL claim based only on
the banner text.

After those checks, `scripts/freeze_tevatron_candidate.py` will create
`manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json`, hashing the primary
grid, independent-seed grid, conventional-Y term decomposition, boundary
oracle, precision-refinement status, and campaign summary.  This freezes the candidate for review only;
the manifest explicitly remains unpromoted.
The postprocessor also creates the candidate-only data/W+Y comparison figure
under `reports/tevatron_n3ll_nnlo_wy_production_g1_1p017/figures/`; it is a
bin-level perturbative diagnostic, not a replacement for the frozen TMD
Fig. 2/Fig. 6.

## Live status at the latest handoff edit (2026-08-19)

The persistent high-statistics supervisor has completed the 41-row CDF Run-I
part and is currently evaluating the 61-row CDF Run-II part at 100M calls per
Vegas component with the extended 10,800-second timeout. The independent 30M-
call stationarity run has completed its 122-row grid; the post-batch
supervisor is waiting for the high-statistics primary merge, after which it
will run precision refinement, the candidate-center conventional-Y closure,
finalization, the replica diagnostic, and candidate freezing automatically.
No final candidate grid, replica profile, or candidate freeze has been
declared yet.

The completed stationarity file records all 122 rows finite/positive, median
prediction/data ratio 1.0051 (range 0.713--1.217), mean MC fraction 2.31%, and
maximum MC fraction 24.1%.  These are integration diagnostics only; the
primary high-statistics values and the refinement pass determine the retained
candidate table.

Before finalization it now invokes `scripts/refine_tevatron_primary_grid.py`.
This reevaluates only rows with MC relative uncertainty larger than 0.5 times
the released data relative error, using 300M calls and up to three isolated
workers.  The criterion is a numerical integration floor tied to the data
precision; it is not a physics prior or an uncertainty combination rule.

An additional persistent watcher,
`scripts/supervise_tevatron_scale_campaign.sh`, waits for the candidate freeze
and then launches a fresh seven-point $(\mu_R,\mu_F)$ scan at $g_1=1.017$
(3M calls/component), followed by the existing 30M-call refinement of rows
whose scale-run MC fraction exceeds 50%.  This is intentionally downstream of
the central batch and remains a diagnostic until the precision floor is
demonstrated.

## Candidate-center conventional-Y closure correction (2026-08-19)

Before the post-batch freeze, an audit found that the pre-existing isolated
RES/CT/VJ decomposition used `g1=1.0`, whereas the retained direct candidate
center is `g1=1.017`. That decomposition remains a convention check, but it
must not be the closure artifact hashed into the new candidate. The
row-count-guarded postprocessor now runs the same three-term diagnostic at
`g1=1.017` in `reports/dyturbo_term_decomposition_g1_1p017/`, and the
candidate-freeze script requires that center explicitly. The identity remains
`W=RES`, `ASY=-CT`, `FO=VJ`, and `Y=FO-ASY=VJ+CT`; this correction only makes
the closure point consistent with the candidate's retained nonperturbative
center.

The campaign summary was also corrected to read the downstream scale scan
from `reports/tevatron_scale_variations_g1_1p017/`, matching the watcher and
the retained center; the older `g1=1.024191` directory is historical and is
not used for this candidate's perturbative-variation summary.

For the same reason, the postprocessor now writes
`reports/accuracy_closure_g1_1p017.json` from the complete primary grid, the
24-row boundary oracle, and the candidate-center RES/CT/VJ decomposition.
The candidate freeze and campaign summary consume this center-matched record;
the historical `accuracy_closure_v2.json` remains preserved as provenance but
is no longer used as the candidate's closure artifact.

## 122-row Tevatron W+Y grid completed; precision refinement active (2026-08-19)

The primary directly evaluated unprimed N3LL+NNLO W+Y grid has now completed
for all 122 Tevatron bins: CDF Run-I (41), CDF Run-II (61), and D0 Run-I (20).
The retained direct nonperturbative center is `g1=1.017 GeV^2`.  The merged
grid is finite and positive in every row; before refinement its mean relative
DYTurbo integration uncertainty is 1.370%, with a 10.055% maximum in the
cancellation-dominated tail.  The median full-W+Y/data ratio is 1.002916
(range 0.778671--1.191633).  These values are recorded in
`reports/tevatron_n3ll_nnlo_wy_production_g1_1p017/grid_status.json` and are
not a promoted TMD result.

The row-count guard has advanced automatically to the 300M-call precision
refinement.  Twenty-four rows were selected by the documented criterion
`MC_relative_error > 0.5 * data_relative_error`; they are being reevaluated
in batches of three concurrent isolated DYTurbo workers.  Candidate-center RES/CT/VJ
decomposition, accuracy closure, stationarity comparison, replica diagnostic,
scale watcher, and the review-only candidate freeze remain downstream of this
refinement.  No frozen lambda=1 file or result has been modified.

The read-only frozen-baseline verifier was rerun immediately after the merged
grid appeared: all 16 recorded hashes still pass and `writes_performed` is
false.

## 122-row Tevatron grid finalized and candidate frozen (2026-08-19)

The isolated direct unprimed N3LL+NNLO W+Y grid is now complete and frozen for
review at `g1=1.017 GeV^2`. It covers 122 Tevatron bins (CDF Run-I 41, CDF
Run-II 61, D0 Run-I 20) with `W=RES`, `ASY=-CT`, `FO=VJ`, and
`Y=FO-ASY=VJ+CT`; the candidate formula is `RES+CT+VJ`. The candidate-center
30M-call RES/CT/VJ decomposition and accuracy-closure record both pass at the
same g1 value. The independent 122-row stationarity grid is also complete.

The 24 rows selected by the documented numerical-precision criterion were all
reevaluated at 300M calls per Vegas component. The initial primary maximum
relative integration estimate was 10.055%; the refined table's maximum estimate
is 16.545% because cancellation-dominated rows have noisier error estimates at
the higher-statistics evaluation. The refined 122-row table remains finite and
positive, with median W+Y/data ratio 1.00355 (range 0.77867--1.19163). These
integration estimates are diagnostics, not physics uncertainties.

The first aggregate refinement writer stopped after successfully updating the
grid because of a candidate-side column-name typo. The cached 300M-call tables
were preserved; `scripts/recover_tevatron_precision_refinement_status.py`
reconstructed all 24 records from the append-only log without rerunning any
integration. The corrected status explicitly records this recovery. This
bookkeeping incident did not touch the frozen lambda=1 package.

The 500-member diagonal-plus-normalization replica diagnostic, candidate grid
figure, campaign summary, read-only baseline audit, and review-only freeze
manifest are complete. The candidate manifest is
`manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json`; it remains explicitly
unpromoted. A downstream seven-point `(mu_R,mu_F)` scale scan at g1=1.017 is
now running automatically. Fixed-target rows and LHCb high-qT rows remain
outside this genuine Tevatron scope pending target/normalization and same-scheme
closure work; no 353-row production claim is allowed.

## Fixed-target boundary remains fail-closed (2026-08-19)

The isolated `VJquad=true`, no-cut quadrature probes demonstrate that DYTurbo
can evaluate the same unprimed N3LL+NNLO W+Y ingredients for fixed-target
kinematics. Explicit nuclear cards were tested for E288 Be (`Z=4,A=9`) and E605
Cu (`Z=29,A=63.546`), with stable numerical quadrature. DYTurbo returns a
whole-nucleus PDF sum; dividing by A gives values that still do not establish
the published per-nucleon normalization (for example, the E288_200:0 and E605:0
probes do not match their tabulated units under the obvious divisions). The
processed E772 card explicitly flags target, unit, and normalization review, so
no fixed-target production rows are being inserted by assumption. The existing
probes remain under `reports/fixed_target_quadrature_probes/` and are capability
tests only. A same-scheme fixed-target closure is required before any 353-row
claim or final all-data TMD propagation.

The explicit numeric comparison is recorded in
`reports/fixed_target_quadrature_probes/fixed_target_normalization_audit.json`:
for E288_200:0 the Be whole-nucleus ratio is about 19.84 and the simple
per-nucleon ratio about 2.20; for E605:0 the corresponding ratios are about
34.30 and 0.540. These are evidence of a convention/units problem, not a
candidate fit failure.

## Scale-scan numerical recovery (2026-08-19)

The first seven-point scale pass was correctly quarantined because the
low-qT `CDF_RUN_2:1` point at `(mu_R,mu_F)=(0.5,1)` had a negative central
Vegas estimate with an integration error larger than the estimate.  This is a
numerical-resolution failure, not an accepted negative cross section.  The
isolated scale runner now retains such rows as `unresolved_rows` and lets
`scripts/refine_tevatron_scale_variation_rows.py` rerun them at 30M calls per
component.  The companion scale status is promoted to `all_positive=true`
only after that refinement has actually replaced every unresolved value with
a finite positive result.  Until then the scale gate and the final W+Y
supervisor remain fail-closed.

## Frozen review and final Tevatron W+Y launch (2026-08-19)

The λ=1 identifiability package remains hash-locked at
`dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50/`;
the read-only verifier reports all recorded hashes passing and no writes.  The
isolated candidate review package is also frozen by
`manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json`.  That manifest is a
review freeze only: it does not promote the candidate or replace the λ=1
production result.

After the candidate freeze, an automatic supervisor was launched for a new
genuine external unprimed N3LL+NNLO W+Y Tevatron production grid.  It waits for
the seven-point scale scan and its precision refinement, rechecks the frozen
baseline, and then evaluates all 122 Tevatron bins at `g1=1.017 GeV^2`, using
100M Vegas calls per DYTurbo component and a new seed.  The output is isolated
under `reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/` and is finalized by
`scripts/finalize_tevatron_final_wy_production.py` with promotion explicitly
disabled.  The launch parameters and prerequisite gate files are recorded in
`manifests/tevatron_n3ll_nnlo_wy_final_launch.json`.
After the direct grid, the same supervisor will run the isolated 500-member
diagonal-plus-dataset-normalization replica diagnostic and write a grid-level
comparison figure.  That layer is intentionally labeled diagnostic: it does
not substitute for the DNN `F_NP`/start ensemble, PDF replicas, or a model-form
envelope.

The scale campaign is additionally guarded by the persistent user unit
`full-n3ll-wy-scale-persistent.service`; it waits for the original process and
resumes the restartable scale supervisor if the terminal process disappears.

To avoid serial idle time, `full-n3ll-wy-final-grid-now.service` now runs the
fresh 100M-call-per-component, 122-row Tevatron grid in parallel with the
seven-point scale scan.  It writes the separate final-candidate directory;
the downstream final supervisor checks the row-count guard and reuses the
completed grid after the scale/refinement gate rather than rerunning it.

The first final-grid card has exposed a cancellation-sensitive numerical edge:
DYTurbo reported a nonfinite V+J-real value for CDF Run I's
`qT=0.5--1 GeV` bin at this seed.  The pipeline is now fail-closed on
nonfinite/nonpositive rows.  `full-n3ll-wy-final-grid-recovery.service` waits
for the first attempt, quarantines any invalid table/status pair under
`reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/recovery_invalid_attempts/`, and
tries seed-rotated reruns before the finalizer can consume the grid.  No
invalid table is eligible for the final figures or replica propagation.  The
first invalid attempt was stopped after the confirmed NaN and the recovery is
currently running with seed `20260823`.
The launch manifest now records the recovery seed list (`20260823`,
`20260824`, `20260825`) and the automatic handoff writer.

This is the complete scope that is currently defensible for conventional
`Y=FO_NNLO-ASY_NNLO`: CDF Run I (41), CDF Run II (61), and D0 Run I (20).
Fixed-target rows and the four high-qT LHCb rows are not silently inserted;
their target/normalization or same-scheme observable closures remain required
before any 353-row or all-data claim.  The new Tevatron run is therefore a
genuine perturbative W+Y production grid, but not yet a final F_NP/start/
replica-propagated TMD publication result.  Its remaining gates are recorded
in its final manifest and include independent-start/model-form propagation,
correlated experimental/PDF propagation, and the fixed-target closure.

The finalizer records the `random_seed` from the actually accepted grid in
the final candidate identifier.  Thus a seed-rotated recovery cannot be
mistaken for the initial failed attempt in the provenance manifest.

The isolated scale runner and its 30M-call refinement have also been made
fail-closed for nonfinite, nonpositive, or negative-uncertainty DYTurbo rows;
such a table is deleted and cannot generate a passing perturbative-accuracy
status.

The unattended final-production supervisor now calls the idempotent
`record_final_tevatron_handoff.py` step after the final grid, scale gate,
replica diagnostic, and figures are present, so completion metrics are added
here automatically without any write to frozen production results.
It also writes a separate seven-point scale-envelope figure under the final
candidate's `figures/` directory.

After the accepted final grid is complete, the supervisor extracts an
explicit 122-row conventional-Y table from the same DYTurbo term logs
(`W=RES`, `ASY=-CT`, `FO=VJREAL+VJVIRTUAL`, `Y=VJ+CT`) and requires its
finite/positive reconstruction gate before recording completion.
Before that extraction it applies the candidate's 300M-call precision
refinement to rows whose 100M-call integration error exceeds half the data
relative error.

## Scale-scan cancellation edge (2026-08-19)

The first unattended 3M-call scale pass reached the CDF Run-I
`(mu_R,mu_F)=(2,2)` card and exposed a cancellation-sensitive low-qT table
entry: the `0.5--1 GeV` total was `-64.95 +/- 3643 fb/bin`.  This is a
diagnostic failure, not a physical negative cross section and not an accepted
scale variation.  The table/status pair is kept as an evidentiary attempt;
the persistent supervisor will quarantine it when the stale pass exits and
rerun the complete seven-point scan with the patched finite/positive validator
and a rotated seed.  No scale envelope or final production manifest may use
the first-pass table.

The persistent scale guard was tightened after this event to validate the
contents of both aggregate status files, rather than only testing file
existence.  Thus an invalid first-pass status cannot block the rotated-seed
retry or the downstream final-production supervisor.

## 353-row scope audit (2026-08-19)

The exact requested selection is now resolved and recorded without running or
promoting it in `reports/tevatron_353_scope_audit.json`. It consists of the
unique 329-row low-qT core selected by the source-production prediction table
plus the explicit 24 CDF/D0 Tevatron boundary rows. The audit handles legacy
compressed fixed-target row labels by matching published bin coordinates and
cross sections against the current fit-ready source tables; it does not use a
positional join. This proves the row count, not perturbative validity: the
fixed-target nuclear/unit convention, E772 target composition, and LHCb
fiducial acceptance remain diagnostic-only. No 353-row production claim is
authorized until those gates are closed.

## Pre-production review freeze and genuine W+Y run (2026-08-19)

Before the new final run, the validated review inputs were hash-locked in
`manifests/preproduction_review_freeze_20260819.json`. This freeze records
the immutable lambda=1 comparison package, the candidate perturbative
closure/decomposition/stationarity inputs, the fixed-target normalization
audit, and the resolved 353-row scope audit. It is an isolated review
checkpoint: it neither copies nor modifies any frozen production artifact and
does not authorize promotion.

The new genuine external production is active under
`reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/`. It uses DYTurbo 1.4.2,
unprimed order 3, `W=RES`, `ASY=-CT`, `FO=VJ`, and the conventional
`Y=FO_NNLO-ASY_NNLO=VJ+CT`, with `g1=1.017 GeV^2`, 100M Vegas calls per
component, and seed-rotated recovery for any nonfinite/cancellation-dominated
row. The Tevatron scope is 122 qT rows; the 24 boundary rows are retained as
an explicit oracle/check, not silently merged into the fit. The supervisor
will not finalize the run until the scale scan/refinement and finite/positive
row gates pass, after which it extracts the explicit conventional-Y table,
performs the documented precision refinement, and writes isolated diagnostics.
Fixed-target and LHCb rows remain fail-closed pending their observable,
target, and normalization closures.

## Feasibility boundary for fully matched production (2026-08-19)

The perturbative construction itself is feasible for the Tevatron scope: the
isolated DYTurbo path evaluates the unprimed N3LL resummed term and the NNLO
fixed-order term in the same runtime convention, with the conventional
matching identity

```text
W + Y = RES + (FO_NNLO - ASY_NNLO)
      = RES + CT + VJ
```

The high-statistics 122-row grid and seven-point scale campaign are still
running.  This does not yet constitute a final TMD extraction because the
external engine currently supplies an observable-level perturbative oracle;
the fitted DNN/F_NP model, independent-start ensemble, and experimental/PDF
replica propagation still need to be connected to the accepted candidate.
The 353-row global claim remains fail-closed until the fixed-target and LHCb
observable/normalization gates are closed.  No frozen production artifact is
modified by this campaign.

## Freeze confirmation and active Tevatron production snapshot (2026-08-19)

The requested pre-run freeze was rechecked at launch time.  The read-only
verifier `scripts/verify_frozen_baseline.py` reports
`status=frozen_baseline_verified`, all 16 recorded hashes pass, and
`writes_performed=false`.  The immutable lambda=1 package remains the
reference production result; it has not been copied over, edited, or
relabelled by this study.

The isolated candidate freeze is recorded in
`manifests/preproduction_review_freeze_20260819.json` and
`manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json`, both with promotion
disabled.  The new genuine external run is active under
`reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/`.  It evaluates the attainable
Tevatron scope (CDF Run-I 41 rows, CDF Run-II 61 rows, D0 Run-I 20 rows) with
DYTurbo 1.4.2, unprimed order 3, `g1=1.017 GeV^2`, and 100M Vegas calls per
component.  Its conventional matching is fixed before looking at the result:

```text
W = RES,  ASY = -CT,  FO = VJ,
Y = FO - ASY = VJ + CT,
W + Y = RES + CT + VJ.
```

The seven-point scale campaign and the 122-row high-statistics grid are
running as separate restartable jobs.  Neither is eligible for final figures
until its finite/positive status gate passes; cancellation-sensitive rows are
quarantined and refined rather than silently accepted.  After those gates,
the supervisor will extract the explicit conventional-Y table and run the
96-start x 50-replica F_NP/TMD propagation.  The resulting figures remain
isolated diagnostics until the external W+Y connection and all required
scope/normalization gates are validated.  Fixed-target and LHCb rows remain
excluded fail-closed, so no 353-row production claim is being made at this
stage.

## Fixed-target external W+Y capability recheck (2026-08-19)

The representative E288_200:0 capability probe was repeated at 30M Vegas
calls per component after the earlier 10k/1M noisy attempts.  The unprimed
N3LL+NNLO card returned a finite positive observable,
`2.96695 +/- 6.15026 fb`, using proton-on-proton kinematics and the same
`W=RES`, `ASY=-CT`, `FO=VJ`, `Y=VJ+CT` identity.  This removes the earlier
low-statistics negative value as evidence of a physical failure, but it does
not close the fixed-target gate: the probe still omits the nuclear target
mixture and does not yet prove the invariant/fit-ready `CS=A/PreFactor`
normalization for all 243 rows.  The result is therefore recorded as a
capability diagnostic only, with production authorization still disabled.

The downstream completion guard
`scripts/supervise_tevatron_postgrid_completion.sh` was launched separately
because the original long-running shell predates the final propagation patch.
It is restart-safe and waits for the authoritative finite/positive grid and
scale/refinement statuses before running the conventional-Y extraction,
precision/replica diagnostics, full start/replica propagation, and figures.

## Decision boundary: can the target be completed? (2026-08-19)

Yes, for the Tevatron perturbative scope.  The isolated DYTurbo calculation is
already a genuine unprimed order-3 construction: its runtime maps order 3 to
NNLO V+jet, includes the N3LL Sudakov/coefficient content recorded in the
source map, and evaluates the conventional matching identity

```text
W + Y = RES + (FO_NNLO - ASY_NNLO)
      = RES + CT + VJ.
```

The term-level closure at `CDF_RUN_2:17` is finite and positive and reconstructs
`RES+CT+VJ` consistently with the direct full result.  Thus there is no
perturbative-formalism obstruction to a fully matched Tevatron W+Y prediction.

This is not yet the final production extraction.  The 122-row high-statistics
Tevatron grid and seven-point scale campaign are still required to pass their
authoritative finite/positive/refinement gates, and the accepted lambda=1
F_NP/start/replica ensemble still has to be connected to the external W+Y
oracle.  The current external run uses the validated Gaussian `g1=1.017`
input; it must not be described as a fitted DNN-F_NP W+Y result until that
coupling is explicit and checked.  Fixed-target rows additionally require the
nuclear target and fit-ready normalization closure, while the LHCb high-qT
rows remain outside this Tevatron candidate.  Consequently the current status
is:

```text
Tevatron W+Y formalism: demonstrated, production candidate running
full 353-row W+Y production: not yet authorized (fail-closed)
frozen lambda=1 baseline: unchanged and verified
```

Three additional fixed-target capability probes were launched after this
review, each with a unique output tag so concurrent DYTurbo tables cannot
collide: E288_300:0 with the Be convention `(Z,A)=(4,9.0121831)`, E605:0
with the Cu convention `(29,63.546)`, and E772:0 with the current diagnostic
isoscalar convention `(0.5,1)`.  These are 30M-call-per-component probes
only; their completion can inform the 353-row gate but cannot authorize
production without the observable and per-nucleon normalization closure.

The corrected no-lepton-cut fixed-target pilots are now summarized in
`reports/fixed_target_n3ll_nnlo_grid_pilot_summary.json`.  They use the
published 0.2-GeV E288/E605 and 0.25-GeV E772 qT bins and compare the explicit
per-nucleon DYTurbo value with `CS=A/PreFactor`.  The first rows give ratios
E288_200=4.75, E288_300=2.51, E288_400=0.42, E605=1.13, and the diagnostic
E772 isoscalar point=1.13.  Because the finite-order pieces are strongly
cancellation-sensitive in these low-qT fixed-target bins, these 30M-call
ratios are not sufficient to distinguish a convention mismatch from Monte
Carlo under-resolution.  The 243-row fixed-target grid therefore remains
fail-closed pending the high-statistics, row-level target/observable closure
campaign; this does not block the validated Tevatron W+Y candidate.

## Fixed-target xF probe follow-up (2026-08-19)

The explicit DYTurbo observable switches were checked in isolation with
`dsdxf=true` and `edsdp3=true` for the representative E288_200:0 row. The
card uses the published xF interval and a no-lepton-cut inclusive setup, but
its raw result is not numerically comparable to the fit-ready `CS=A/PreFactor`
value: after the candidate whole-nucleus normalization the per-nucleon ratio
is still about 235 for that row. The earlier E288/E605/E772 convention probe
also mixes incompatible normalizations and is not a production result. This
shows that the xF card is not a valid replacement for the fit-ready convention.
Together with the cancellation-sensitive y-bin results, this means that both
the invariant-observable/unit conversion and numerical stability still need
to be checked row by row; it is not evidence for a missing finite-Y capability.
No fixed-target rows are promoted until a high-statistic row-level conversion
reproduces the fit-ready convention across E288, E605, and E772.

The authorized scope remains the 122-row Tevatron grid plus the separate
24-row Tevatron boundary oracle. The final supervisor is waiting for the
high-statistics grid and scale/refinement finite-positive gates before it
writes conventional-Y, replica, F_NP/start, and figure artifacts. Frozen
production files remain unchanged.

### Live execution snapshot (09:02 EDT, 2026-08-19)

The unattended workers are still alive. The final 100M-call central grid has
completed CDF Run-I and CDF Run-II and is processing D0 Run-I (currently in
the 4--5 GeV qT bin). The rotated seven-point scale worker is processing the
D0 Run-I card for the `(mu_R,mu_F)=(1,0.5)` variation after two earlier
low-qT cancellation-sensitive passes were correctly quarantined. No final
grid or scale status is accepted until the finite/positive gates pass, and the
post-grid watchdog remains running to continue precision refinement, explicit
Y extraction, replica/start propagation, and figures automatically.

## Fixed-target pilot interpretation reassessed (2026-08-19)

The initial 30M-call fixed-target ratios must not be treated as a definitive
normalization rejection. The full W+Y value is a cancellation-sensitive sum:
the representative E288 nuclear cards contain VJREAL and VJVIRTUAL terms of
order tens of thousands of fb whose difference is only order tens of fb. In
the existing isolated E288_200 comparison, the full value changed from about
`170 +/- 65 fb` at 30M calls to about `30 +/- 37 fb` at 100M calls, while the
fit-ready value is `CS=3.9537` in the corresponding convention. This is a
clear warning that apparent 30M central ratios can be dominated by unresolved
Monte-Carlo cancellation, rather than evidence for a physical global
normalization mismatch.

There is also a useful but limited historical check: the published E288_400
fixed-order VJ-only card (`makecuts=false`, y-bin convention) gives a raw value
consistent with the A-like convention. It does not validate the complete
N3LL+NNLO W+Y sum, so it cannot close the production gate by itself. The
explicit xF probe is likewise not a valid replacement: its observable switches
produce a value incompatible with the fit-ready convention and are retained
only as a diagnostic.

The fixed-target decision is therefore updated from “non-uniform pilot ratios
prove a normalization failure” to “the observable and normalization remain
unresolved and require high-statistics, row-level closure tests.” The next
isolated campaign will compare proton and nuclear/per-nucleon cards, preserve
all RES/CT/VJ components, and repeat representative E288, E605, and E772 rows
at substantially higher calls and independent seeds. It will be accepted only
if the reconstructed `RES+CT+VJ` and the fit-ready `CS=A/PreFactor` convention
are stable at the row level. Until then, fixed-target and LHCb rows remain
fail-closed; this correction does not alter the authorized Tevatron candidate
or any frozen production artifact.

## High-statistic fixed-target closure supervisor launched (2026-08-19)

The restart-safe supervisor
`scripts/supervise_fixed_target_highstat_closure.sh` is now active (current
PID 1592869; an earlier terminal-scoped PID 1591884 was superseded). It waits
for the authoritative 122-row Tevatron grid and scale
gates, then runs representative E288, E605, and E772 nuclear/per-nucleon
cards at 100M calls with three independent seeds, plus proton-target E288
comparisons. It records every card, log, raw value, quoted integration error,
and per-nucleon conversion under
`reports/fixed_target_highstat_wy_closure_20260819/`. The resulting summary is
diagnostic only and is not allowed to promote the fixed-target or LHCb scope.
The persistent detached supervisor is currently PID 1592869; its log records
the gate wait and every later probe/reuse decision.
Before any probe started, the runner was patched to accept an explicit
`--out` root; the supervisor now directs all high-statistic artifacts into its
own closure directory rather than the historical capability-probe directory.
The two Tevatron downstream supervisors were also relaunched with a shared
`downstream_completion.lock` (PIDs 1597018 and 1597019); this prevents duplicate
precision/refinement work when both observe the gates together.

### Live execution snapshot (09:17 EDT, 2026-08-19)

The central D0 Run-I high-statistics worker has advanced through the 10--12
GeV bin and is now processing 12--14 GeV; ten DYTurbo workers remain active
for that dataset. The scale campaign is in the `(mu_R,mu_F)=(1,1)` pass after
the earlier low-qT cancellation-sensitive attempt. Neither status file has
been accepted yet. The fixed-target supervisor remains in its intentional
wait state, so no extra fixed-target process is competing with the current
Tevatron production.

### Live execution snapshot (09:28 EDT, 2026-08-19)

D0 has completed the 25--30 GeV bin and is integrating the 30--35 GeV bin;
the central grid remains active with its parallel DYTurbo workers. The scale
campaign is still finishing the CDF Run-II central-scale card. No grid,
scale, conventional-Y, or propagation status has been promoted early.

The later live check clarified that the scale runner had then advanced into
the final two variation triplets rather than its final dataset card: after
completing the `(1,2)` triplet it began `(2,1)`, with six dataset cards still
to run before `(2,2)`. This is normal sequential progress, not a retry or
duplicate production run.

### Central Tevatron grid gate passed (09:34 EDT, 2026-08-19)

The new high-statistics 122-row unprimed N3LL+NNLO Tevatron grid is complete
and passed its finite/positive gate. It used 100M calls per Vegas component
with `g1=1.017`, has mean MC relative uncertainty 1.3426%, maximum 11.7016%,
and full-W+Y/data ratio median 1.00420 (range 0.77540--1.16826). The central
artifact is `reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/tevatron_full_wy_grid.csv`;
the scale gate is still pending, so no downstream result is promoted yet.

At the latest check-in, the `(1,2)` scale triplet is complete and the runner
has begun D0 Run-I for `(2,1)`; the remaining `(2,1)` D0 card and the three
`(2,2)` cards still precede scale consolidation/refinement. This ordering is
recorded here to make the handoff restart-safe and to avoid mistaking a
partial scale table for a completed uncertainty envelope.

The seven-point scan subsequently completed all 21 cards and wrote
`scale_variation_status.json`. It is finite but still fail-closed because one
low-qT point (`CDF_RUN_2:1` at `(mu_R,mu_F)=(1,0.5)`) has a negative 3M-call
estimate with an integration error larger than the estimate. The dedicated
30M-call scale-row refinement is active; downstream completion remains gated
until its all-finite/all-positive status is published.

### Scale closure passed and downstream precision refinement active (10:42 EDT, 2026-08-19)

The isolated seven-point scale campaign is now numerically closed. All 21
scale cards are finite and positive after 17 targeted 30M-call refinements;
the previously negative `CDF_RUN_2:1` `(mu_R,mu_F)=(1,0.5)` estimate was
resolved to a positive value. The authoritative scale artifacts are
`reports/tevatron_scale_variations_g1_1p017/scale_variation_status.json` and
`scale_variation_refinement_status.json`. The refined scale-envelope
half-width has median 14.29% and maximum 32.48%; this is a perturbative
scale diagnostic, not a statistical confidence band or a production
promotion.

The primary Tevatron downstream supervisor now owns the shared lock and is
running the 300M-call precision refinement on the nine central-grid rows whose
Monte-Carlo error exceeded half of the data relative error. Three rows have
completed in each of the first two batches; the final three are in the active
three-worker DYTurbo batch. It will publish
`precision_refinement/primary_refinement_status.json`
before explicit conventional-Y extraction, 500-replica perturbative
diagnostics, F_NP/start propagation, and figures can proceed.

An isolated supervisor deadlock was found when the post-grid watchdog held
the lock while waiting for the primary supervisor. The watchdog was patched
to rely solely on the shared lock (no cross-wait), the stale inherited sleep
was removed, and the primary precision run started successfully. This
control-flow correction does not touch frozen production artifacts.

The fixed-target high-statistic closure probes also completed: seven
representative nuclear/proton groups, each at three seeds and 100M calls per
Vegas component, are finite and recorded in
`reports/fixed_target_highstat_wy_closure_20260819/closure_summary.json`.
They remain diagnostic and fail-closed because the fit-ready fixed-target
conversion is not yet validated for full production scope. The supervisor
summary-writing typo was repaired and the completed summary was regenerated.
The authorized production scope therefore remains the 122-row Tevatron
candidate (plus the separate boundary oracle), not the full 353-row scope.

## Final isolated Tevatron W+Y handoff record

The unattended genuine external run has passed its isolated Tevatron row and perturbative-scale gates.  This is a review artifact, not a replacement of the frozen lambda=1 package.
- Candidate: `tevatron_n3ll_nnlo_wy_final_g1_1p017_seed20260823`; formula `W_N3LL + (FO_NNLO - ASY_NNLO)`.
- Scope: 122 rows ({'CDF_RUN_1': 41, 'CDF_RUN_2': 61, 'D0_RUN_1': 20}); fixed-target and LHCb high-qT rows remain excluded pending closure.
- Grid MC relative uncertainty: mean 1.1773%, max 16.5446%.
- Central-grid precision refinement: 19 rows at 300,000,000 calls/component; reported maximum after refinement 16.5446%.
- W+Y/data ratio: median 1.002973, range [0.775403, 1.168256].
- Seven-point scale envelope (3M-call pass): median half-width 14.2900%, max 95.7953%; 30M-call refinement selected 17 rows.
- Explicit conventional-Y grid: 122 rows; median term-reconstruction residual 2.947e-04 pb/GeV, maximum 9.070e-03 pb/GeV.
- These are perturbative/integration diagnostics.  The final TMD publication band still requires correlated experimental/PDF and F_NP/start/model-form propagation.
- Final manifest: `reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/FINAL_PRODUCTION_MANIFEST.json`.

## Current status correction and 353-row continuation (2026-08-19)

The preceding live-execution snapshots are historical.  The central Tevatron
candidate and all of its currently authorized downstream diagnostics have now
finished; no active worker from this campaign is running.  The authoritative
post-refinement values are:

- 122-row Tevatron grid: complete, finite, positive, and still isolated/not
  promoted.  Nineteen rows received the 300M-call/component refinement.
- Mean grid Monte-Carlo relative uncertainty: 1.1773%; maximum after
  refinement: 16.5446%.  The maximum increased because the refined estimate
  resolved a previously underestimated integration error; this is not a claim
  of improved precision.
- Conventional ($Y=\mathrm{FO}_{\mathrm{NNLO}}-\mathrm{ASY}_{\mathrm{NNLO}}$):
  complete for all 122 rows, with 19 precision-term overrides.  The maximum
  reconstructed-term residual is 0.00907 pb/GeV and the median is
  0.000295 pb/GeV.
- Full isolated propagation: 96 optimizer starts crossed with 50 experimental
  replicas.  The resulting b-space and k-space bands are diagnostic envelopes,
  not assigned confidence intervals; no individual-start curves are used in
  the publication-style figures.

The generated diagnostic artifacts are
`reports/tevatron_fnp_start_replica_propagation/fig2_bspace_fnp_start_replica.png`
and
`reports/tevatron_fnp_start_replica_propagation/fig6_kspace_ud_fnp_start_replica.png`,
with corresponding CSV/PDF files.  The 353-row scope audit remains
`353_row_scope_resolved_not_yet_run`: it consists of the 329-row core selection
plus 24 Tevatron boundary rows, whereas the completed external grid is the
separate 122-row Tevatron candidate plus its boundary oracle.

Seven fixed-target nuclear/proton high-statistic probe groups also completed
(three seeds, 100M calls per Vegas component), all finite and seed-stable in
the numerical probe.  They are not promoted because the fit-ready
fixed-target ($CS=A/\mathrm{PreFactor}$) convention, target normalization, and
the E772 isoscalar treatment still require an explicit same-scheme closure.
The six LHCb core rows likewise remain outside the authorized production
scope pending their fiducial/high-(q_T) closure.  Frozen lambda=1 files and
all existing production outputs remain unchanged.

## Fixed-target convention audit update (2026-08-19)

The full 353-row W+Y run remains intentionally fail-closed.  The numerical
DYTurbo probes are finite, but the earlier fixed-target runner compared a raw
per-nucleon DYTurbo bin to the fit-ready ``CS=A/PreFactor`` table without first
reconstructing the published observable.  That comparison is not authorized
for production.

The source-observable audit has now resolved the direction of the correction:

- E288 is tabulated at fixed rapidity.  Across its rows,
  ``PreFactor * 2*pi*qT`` is approximately one, consistent with the invariant
  cross-section conversion after the fixed-y bin is integrated.
- E605 is a fixed-``x_F`` measurement.  Its effective prefactor tracks the
  ``dx_F/dy`` Jacobian divided by ``2*pi*qT``; a generic y-bin prediction is
  therefore not the correct observable.
- E772 is an invariant ``E d^3sigma/d^3q`` result per nucleon in finite
  ``x_F`` intervals (the original table uses 0.1--0.3 for the first mass
  bin).  Its prefactor is not a universal y-bin factor.
- DYTurbo supports the needed diagnostic switches ``dsdxf=true``,
  ``edsdp3=true`` and the ``xfmin/xfmax`` bounds.  These are capability
  inputs only until representative rows reproduce the fit-ready convention
  with the correct finite qT, Q and xF/y bins.

Consequently the next gate is a three-observable representative closure:
E288 fixed-y, E605 fixed-xF/invariant, and E772 finite-xF/invariant, with the
published target treatment and per-nucleon normalization.  Only after those
ratios and units close will the isolated 353-row grid be launched.  No frozen
lambda=1 file or production cache has been modified.

## Fixed-target closure result and fail-closed decision (2026-08-19)

The representative closure was rerun with the explicit DYTurbo
``dsdqt2=true``, ``dsdxf=true``, ``edsdp3=true`` configuration for E605 and
E772, while retaining fixed-y E288.  The run was finite and numerically
stable, but the conversion did not close:

- E288_200:0: converted CS = 3.52587 versus fit-ready CS = 3.95371,
  ratio 0.89179.
- E605:0: converted CS = 1824.65 versus 1.23567, ratio 1476.65.
- E772:0: converted CS = 8975.82 versus 29.0088, ratio 309.42.

The exact inputs and cards are recorded in
``reports/fixed_target_cs_convention_closure/closure_status.json`` and
``closure_rows.csv``.  These numbers are a diagnostic failure, not a
phenomenological prediction.  In particular, the direct replacement of a
rapidity-bin width by a finite xF width is not justified for the
non-integrable DYTurbo observable switches; it creates the large E605/E772
factors above.

An independent prefactor audit now gives the corrected provenance finding:

- E288 fixed-y rows follow ``1/(2*pi*qT)``;
- E605 rows follow ``[1/(2*pi*qT)]*[2*sqrt(Q^2+qT^2)*cosh(y)/sqrt(s)]``;
- E772 follows the historical ``1/(2*pi*qT*Q)`` A-convention to high
  accuracy, apart from its separately encoded Q=9.46, qT=0.25 block.

This audit is stored in
``reports/fixed_target_prefactor_kinematic_audit/audit_status.json`` and
``prefactor_kinematic_rows.csv``.  It confirms that the fit-ready table has
dataset-specific observable conventions; it does not authorize changing the
table or assuming that the DYTurbo ``dsdxf`` output is already the published
CS observable.

The full 353-row W+Y production run therefore remains blocked by a technical
convention gate, not by GPU capacity or numerical instability.  The next
isolated test will remove the invalid xF-width conversion and compare
DYTurbo's integrated, fixed-y and fixed-xF outputs against the published
E288/E605/E772 definitions one observable at a time.  Until that closes, no
fixed-target rows are promoted, no 353-row propagation is launched, and all
frozen lambda=1 production files remain unchanged.

## Integrated-dsdxf follow-up (2026-08-19)

The next isolated test used DYTurbo's finite-xF integration directly: the
``dxF/dy`` Jacobian was enabled through ``dsdxf=true``, ``edsdp3=true`` was
enabled, and all ``ptbinwidth``/``ybinwidth`` flags were disabled.  The prior
invalid division by a nominal xF width was removed.  The two 30M-call rows
were finite, but the recorded candidates are still convention-dependent:

- E605: raw integrated 2947.52 fb per nucleus (46.3840 fb per nucleon);
  candidate ``pi*sigma/(1000*A*Delta-qT*Delta-Q)`` = 0.72860, or 0.58964 of
  the fit-ready CS.
- E772: raw integrated 353.986 fb per nucleon; the same candidate conversion
  = 4.44832, or 0.15334 of the fit-ready CS.

The full rows are in
``reports/fixed_target_dsdxf_integrated_probe/probe_status.json`` and
``probe_rows.csv``.  These results do not establish a conversion: ``edsdp3``
changes the differential observable before integration, while the published
E288/E605 quantity is related to
``(1/pi) * integral(dQ^2) d sigma/(dQ^2 d(qT^2) d eta)``.  The next audit must
therefore reconstruct that equation directly, including the qT-squared and
Q-squared bin measures and the fixed-y versus fixed-xF reporting variable,
rather than infer a universal conversion from the output label.

No data table, fit cache, frozen package, or production result was modified.

## Eq. (3.3) measure audit (2026-08-19)

The next test used the published fixed-target relation directly, with
``dsdqt2=true``, fixed-y bins, no bin-width flags, and the candidate
conversion

``CS = sigma_fb / (1000*A*pi*Delta(qT^2)*Delta-y)``.

At 30M calls/component it gave finite results, substantially better behaved
than the invalid xF-width conversion but not yet closed:

- E288_200:0: Eq.(3.3) candidate 7.91712 versus data 3.95371, ratio 2.00245.
- E605:0: Eq.(3.3) candidate 1.64353 versus data 1.23567, ratio 1.33007.

Artifacts are in ``reports/fixed_target_eq33_probe/``.  The factor-of-two
E288 result is especially useful: it indicates a remaining azimuth/target or
qT-bin-measure convention mismatch, rather than an unstable numerical
integral.  E605 is within the scale of a normalization/target convention
effect but is not a closure.  This is still a diagnostic-only result; the
353-row run remains fail-closed.

## Fixed-target point-limit qT audit (2026-08-19)

To test whether the preceding discrepancy was caused by comparing a finite
qT-bin average with the tabulated qT point, the representative E288_200:0
and E605:0 calculations were repeated in shrinking windows centered at the
published qT=0.1 GeV point.  The Eq. (3.3) conversion was unchanged and the
windows were 0.20, 0.05, 0.02, and 0.01 GeV wide.  At 10M calls per Vegas
component all runs were finite and numerically stable:

- E288_200:0 converged from a ratio 2.002 (0.20-GeV window) to 2.053
  (0.01-GeV window) relative to the fit-ready CS value.
- E605:0 converged from 1.330 to 1.258 over the same sequence.

Thus the factor-of-two E288 discrepancy is not a finite-qT-bin averaging
artifact.  The E605 result also retains a roughly 26% normalization/observable
discrepancy in the point limit.  Full rows and cards are recorded in
``reports/fixed_target_eq33_point_probe/probe_status.json`` and
``probe_rows.csv``.  The probe remains diagnostic-only; no conversion factor
has been promoted and the 353-row W+Y run is still unauthorized.

The literature provenance audit also confirms that the published low-energy
data use a 40% proton/60% neutron treatment for the E288/E605 target ensemble,
whereas the current DYTurbo diagnostic cards use Be and Cu isotope
approximations.  This composition difference is small compared with the
factor-of-two E288 mismatch, but it must be corrected or explicitly bounded
before a fixed-target production claim.

## Native normalized-observable correction (2026-08-19)

The preceding Eq. (3.3) point-limit numbers were not all compared in the same
observable.  The fit-ready files contain both an invariant quantity ``A`` and a
reported ``CS`` with ``A = PreFactor * CS``.  The earlier diagnostic called the
Eq. (3.3) candidate ``CS`` even when it was the invariant ``A``.  Therefore the
old E288 ``factor-of-two`` statement must not be used as a production-quality
closure claim.

The corrected native-observable probe is recorded in
``reports/fixed_target_normalized_observable_probe/probe_status.json``.  For
E288_200:0, the fixed-y Eq. (3.3) result is ``A_pred/A_data = 1.290`` at
1M calls/component (the corresponding fit-ready CS ratio is identical).  This
is a residual normalization/physics difference, not a factor-of-two measure
error.  For E605:0 and E772:0, the direct ``dsdxf=true`` path remains unusable:
its inferred ratios are 180 and 46 at a narrow xF window, and changing the
xF window gives non-physical scaling.  Thus the DYTurbo xF observable or its
conversion is not yet validated for these rows.

The corrected conclusion is fail-closed: qT averaging is not the dominant
issue, but no fixed-target CS/A conversion has been promoted for E605/E772 and
the 353-row W+Y production run remains unauthorized.  The lambda=1 baseline,
the 122-row Tevatron W+Y candidate, all frozen manifests, and all production
outputs remain unchanged.

## E605 fixed-y versus native-xF identity probe (2026-08-19)

An isolated identity test used the same E605:0 mass and qT bins and the same
Cu target.  A fixed-y rectangle corresponding approximately to the narrow
``xF=[-0.105,-0.095]`` window gave 20.8163 fb per nucleus.  The native
``dsdxf=true`` card with that same xF window gave 1415.662 fb per nucleus,
while a control with ``dsdxf=false`` and unrestricted ``y=[-10,10]`` gave
1223.326 fb per nucleus.  The native xF result is therefore close to an
unrestricted-y integral rather than the fixed-y xF-equivalent result.

This is an identity failure in the DYTurbo xF path (restriction/Jacobian or its
interaction with the full W+Y integrand), not a statistical fluctuation.  The
full report is ``reports/fixed_target_xf_identity_probe/identity_status.json``;
the cards and logs are in the adjacent ``identity_probe/`` directory.  Native
``dsdxf`` output remains prohibited for E605/E772 production conversion.  The
next valid route is an explicit fixed-y quadrature along the xF(Q) mapping, or
an independently audited correction to the external-program path.

## E288 high-Q inverse-transform stabilization audit (2026-08-19)

The explicit fixed-y route was then tested on the first incomplete E288 row,
E288_200:30 (Q=6--7 GeV, y=0.3727--0.4315, qT=0.02--0.2 GeV). This row is a
representative stress test for the full N3LL+NNLO W+Y calculation. The pilot
used 100k calls/component and varied only isolated DYTurbo card settings; no
DYTurbo source or frozen production file was modified.

The complete trial inventory is in
``reports/dyturbo_stabilization_pilot_e288_200_30/all_trials_summary.json``.
The 18 variants covered bmax=0.5/1/2, inverse-transform tolerances,
``sumlogs``, all-b* switches, the local b* prescription, fixed versus dynamic
qT subtraction cuts, qT lower edges 0.05--0.15 GeV, and modified logarithms.
None satisfies the acceptance gate of finite positive output, zero abnormal
inverse-Bessel warnings, and a physically finite normalization. The apparent
zero-warning local-b* result at bmax=2 is approximately 3.3e11 times the
fit-ready E288_200:30 A value; the bmax=0.5 and bmax=1 results are negative or
similarly divergent. Relaxing quadrature accuracy, changing subtraction cuts,
raising the qT lower edge, or changing modified logs either timed out or
produced divergent output.

Therefore the remaining E288 high-Q rows are a numerical/kinematic boundary of
the current external W+Y implementation, not an ordinary Monte Carlo
precision issue. The fixed-target full grid remains diagnostic-only and the
353-row promotion stays fail-closed until either (i) an independently validated
fixed-target W+Y implementation handles this region, or (ii) the scope is
explicitly restricted to rows with a demonstrated finite observable conversion.

## Fixed-xF mapped-Q diagnostic (2026-08-19)

As a replacement for the invalid native ``dsdxf`` route, an isolated probe now
partitions each fixed-target Q bin and evaluates fixed-y DYTurbo rectangles
whose y edges are mapped from the published xF edges using
``xF=2*sqrt(Q^2+qT^2)*sinh(y)/sqrt(s)``.  The integrated result is converted
with the xF Jacobian to the fit-ready A/CS convention.  This is implemented in
``scripts/probe_fixed_xf_mapped_q2.py`` and does not touch frozen production.

At 100k calls per Q slice and four slices for row 0:

- E605:0 is finite with ``A_pred/A_data=1.91`` using the explicit diagnostic
  xF window ``[-0.105,-0.095]`` (the table gives only xF=-0.1); this is a
  valid-looking conversion but not a closure-level normalization.
- E772:0 is finite with ``A_pred/A_data=0.820`` for its tabulated
  xF=[0.1,0.3] interval and the isoscalar fallback target.

These mapped-Q results are materially different from the native ``dsdxf``
ratios (180 and 46) and establish a promising conversion route.  They remain
diagnostic until Q-slice refinement, target-composition checks, and a complete
row-level finite/positive audit are finished.  The 353-row W+Y promotion is
still unauthorized.

## Coefficient-exponentiation regularization audit (2026-08-19)

The E288 stress scan localized the inverse-transform pathology to the order-3
qg coefficient-exponentiation denominator in DYTurbo (``resum/gint.C``), where
``expcreg`` enters the regularized coefficient. The default value 1 can cross
a pole for low-energy, low-qT rows. This is not a Monte-Carlo precision issue:
the same cards can return finite lower-order values while order 3 produces
inverse-Bessel warnings, a timeout, or an astronomically large number.

An isolated row scan used 100k calls/component, explicit fixed-y Eq. (3.3)
cards, and no changes to DYTurbo or frozen production files:

* E288_200 representative rows Q=6.5--10.5 GeV: ``expcreg=0.5`` fails at
  Q=8.5 and 9.5, while ``expcreg=1.5`` is finite and positive on the five
  sampled rows (A ratios 0.81--2.11).
* E288_300 rows Q=6.5--11.5 GeV: ``expcreg=1.5`` is not safe globally. Four
  rows are finite, Q=7.5 returns about ``4.0e107`` (A ratio about ``4.5e106``),
  and Q=8.5 reaches the 600 s timeout. Repeating Q=7.5 alone shows a narrow
  pole near 1.5: values 0.5, 0.75, 1.0, 1.25, 1.75, and 2.0 are finite with
  A ratios about 1.51--1.52.
* ``expcreg=2.0`` is finite/positive on the three previously pathological
  representatives E288_200:57, E288_200:69, and E288_300:67, and on eight
  E288_400 low-qT rows spanning Q=6.5--13.5 GeV. This is a stability result,
  not yet a normalization or formal-scheme validation: sampled ratios range
  roughly 0.73--4.73.
* On the Tevatron CDF Run-2 sparse control (nine qT points, 1M
  calls/component), ``expcreg=0.5`` versus the default gives no warnings and a
  maximum relative shift 0.252% (RMS 0.168%), below the typical Monte-Carlo
  uncertainty of the control. Thus the regularization choice is numerically
  quiet in the Tevatron region but materially important for fixed-target rows.
* The mapped fixed-xF diagnostic remains finite with ``expcreg=2.0``:
  E605:0 (8 Q slices) has A_pred/A_data=1.916 and E772:0 has 0.824, compared
  with the default 4-slice values 1.889 and 0.814. The observable-conversion
  and target-composition audits are still open.

Artifacts are in ``reports/expcreg_e288_row_stress/``,
``reports/expcreg_tevatron_stress_parallel/``, and the suffixed mapped-Q
directories ``reports/fixed_xf_mapped_q2_probe/*_expcreg2p0/``. The new
``scripts/probe_fixed_xf_mapped_q2.py`` option is diagnostic-only. No
``expcreg`` value has been promoted, no fixed-target grid has been launched,
and the 353-row W+Y production run remains fail-closed until a complete
row-level convention and regularization audit is finished.

## Complete fixed-target expcreg=2.0 diagnostic grid (2026-08-19)

The isolated fixed-target runner was extended with an explicit ``--expcreg``
tag and run over all 243 fixed-target authority rows (E288_200, E288_300,
E288_400, E605, and E772) at 100k calls/component. Every row returned finite,
positive W+Y output with no timeout. The candidate conversion is the explicit
fixed-y Eq. (3.3) measure,
``A=I/[1000*A*pi*Delta(qT^2)*Delta(y)]``, followed by the row ``PreFactor``.

The resulting prediction/data ratios span 0.129--4.728, with median 0.460.
This closes the numerical finite/positive gate for this diagnostic setting,
but not the physics/convention gate: the Be and Cu nuclear targets are still
candidate approximations, E772 remains an unresolved isoscalar fallback, and
the fixed-y rectangular observable must not be called a production
normalization without the target/composition audit.

Artifacts:
``reports/fixed_target_expcreg2p0_full_100k/fixed_target_full_wy_grid.csv``
and ``fixed_target_grid_status.json``. The runner change is isolated in
``scripts/run_fixed_target_n3ll_nnlo_grid.py``; no frozen production file or
cache was modified.

## LHCb_7 full fiducial W+Y diagnostic (2026-08-19)

The six LHCb authority rows were evaluated with pp beams, y=2--4.25,
60<Q<120 GeV, and both-lepton pT>20 GeV and 2<eta<4.5 cuts. At 100k
calls/component the output is finite but not positive: four of six qT bins are
negative, including a robustly negative bin after accounting for its Monte
Carlo uncertainty. This is a genuine W--Y cancellation/numerical-precision
problem for the direct fiducial full-W+Y card, not permission to multiply a
positive W-only curve by the old acceptance factor and call it N3LL+NNLO.

The diagnostic status is in
``reports/lhcb7_full_n3ll_nnlo_fiducial_expcreg2p0_100k/lhcb7_grid_status.json``.
The 1M-call refinement is complete in the adjacent
``lhcb7_full_n3ll_nnlo_fiducial_expcreg2p0_1m`` directory: all six bins are
positive, but cancellation-driven relative Monte-Carlo uncertainty remains as
large as 1.23 (123%), with median prediction/data ratio 0.707 (range
0.557--1.391).
The direct fiducial result is therefore a diagnostic component only; it has
not passed a production-quality precision/normalization gate and no LHCb row
is promoted as validated production input.

## Assembled 353-row candidate (2026-08-19)

The scope components are now mechanically assembled into one auditable table:
104 Tevatron rows (80 authority rows plus 24 boundary rows), 243 fixed-target
rows, and the six LHCb authority rows. The resulting table has 353 unique row
IDs, all finite and positive at the component-grid level. Its overall
prediction/data ratio has median 0.746 and range 0.129--4.728.

This is a scope-complete diagnostic assembly, not a production result. The
fixed-target entries use the Eq. (3.3) rectangular conversion and candidate
Be/Cu/isoscalar targets; the LHCb entries use the 1M-call direct fiducial
W+Y diagnostic. The existing 96-start x 50-replica F_NP propagation has not
been refit against this assembled perturbative grid, so it cannot be claimed
as the final 353-row uncertainty propagation.

Artifacts:
``reports/tevatron_353_candidate_g1_1p017_expcreg2p0/candidate_353_full_wy.csv``
and ``candidate_353_status.json``. The assembly script is
``scripts/assemble_353_candidate.py``. Promotion remains unauthorized until
the fixed-target observable/target conventions, LHCb cancellation precision,
and a true full-scope F_NP/refit propagation are closed.

## Fixed-target conventional-Y term grid (2026-08-19)

The missing fixed-target finite-tail input was evaluated in a separate
candidate-only batch.  For each of the 243 authority rows, two cards were run
at 100k calls per VEGAS component with the same unprimed order-3 settings and
``expcreg=2.0``: the complete ``RES+CT+VJ`` card and the RES-only card.  The
row-level conventional term is formed from the stable difference
``Y=(RES+CT+VJ)-RES``.  This is algebraically ``FO_NNLO-ASY_NNLO`` when all
terms share the same card convention; it avoids treating DYTurbo's standalone
CT switch as an independent observable in fixed-target kinematics.

All 243 full and RES values are finite, and all full W+Y values are positive.
The resulting Y term is positive for 232 rows and negative for 11 rows.  The
fixed-target Y/RES table is:

``reports/fixed_target_y_fullminusres_expcreg2p0_100k/fixed_target_y_full_minus_res.csv``

with status in the adjacent ``fixed_target_y_full_minus_res_status.json``.
This closes the numerical fixed-target Y-input gate, but not the physics gate:
the Be/Cu/isoscalar target choices and rectangular fixed-y Eq. (3.3)
conversion remain diagnostic assumptions.  It is therefore not yet a
production F_NP fit input.

## Scope-complete b-space W coverage rebuild (2026-08-19)

The interface audit also found 46 fixed-target scope rows absent from the
historical 160-node b-space W cache.  They were absent because that cache was
created with an earlier row selection, not because the rows were physically
invalid.  The missing rows were recomputed with the same isolated internal
``n3llp``/``nloQ96`` W configuration (160 nodes, $b_T=10^{-4}$--$8\ \mathrm{GeV}^{-1}$, nuclear-isospin target mode) and written as a candidate
fragment:

``reports/missing_scope_bspace_w_n3llp_nloQ96_b160/missing_scope_bspace_w.csv``

All 46 rows and all 160 b nodes are finite.  Combining this fragment with the
existing Tevatron/LHCb cache gives complete b-space W coverage for all 353
scope rows.  The source cache itself was not edited.

## First coupled 353-row F_NP fit (2026-08-19)

The complete candidate-side W and Y tables were passed through the isolated
v19 local-bcurv trainer as a first coupled-F_NP test.  The run used a positive
monotone F_NP, no learned CS kernel, 353 rows, CUDA float32, and 5000 epochs;
the best epoch was 4931.  The loss reached a late plateau with
``chi2_like=33.125`` (0.0938 per row in this trainer's objective), and the
probe F_NP curves remained finite and monotone.  This is an optimizer and
interface check, not a production fit.

The output is
``reports/scope_353_coupled_fnp_fit_lambda1_candidate_s303/`` and its compact
record is ``reports/scope_353_coupled_fnp_fit_lambda1_candidate_summary.json``.
The result exposes three issues that must be resolved before starts or
replicas can be interpreted: (i) this candidate combines an internal n3llp /
nloQ96 W fragment with DYTurbo-derived Y inputs, rather than one common
perturbative engine; (ii) the six LHCb rows retain cancellation-driven Y Monte
Carlo uncertainties above 100%; and (iii) the generic v19 trainer's paper
normalization map has no CDF/D0 entries, so the CDF_RUN_2 normalization pull
is numerically pathological when dataset norms are enabled.  A corrected
candidate uncertainty convention is required before any like-for-like
comparison to the frozen lambda=1 baseline.

No frozen production file, cache, replica, or figure was modified.  The next
step is to repair the candidate normalization interface and rerun the central
fit, then test stationarity across independent starts before propagating
experimental replicas.

The repair is now complete in the candidate-only directory
``reports/scope_353_fnp_inputs/data_with_csv_uncertainties/``.  It parses the
published ``sysNorm`` and ``sysP2P`` fields for all nine datasets, including
CDF/D0.  The rerun
``reports/scope_353_coupled_fnp_fit_lambda1_candidate_s303_csvnorm/`` reaches
``chi2_like=16.095`` at epoch 4997, with no pathological CDF/D0 normalization
pull and monotone finite F_NP.  The LHCb six-row contribution remains about
270 in this trainer objective (median pull about 14), so this does not rescue
the unresolved LHCb fiducial/Y precision issue.  The corrected compact record
is ``reports/scope_353_coupled_fnp_fit_lambda1_candidate_s303_csvnorm_summary.json``.

## Corrected lambda-tail=1 scope campaign (in progress, 2026-08-19)

The first coupled 353-row propagation was not physically usable for Fig. 6:
its v19 monotone F_NP was unconstrained at large b and remained close to one
through $b_T=8\ \mathrm{GeV}^{-1}$.  The resulting Bessel transform produced artificial
multi-hundred-percent spikes.  Those no-tail outputs are retained only as a
diagnostic and are not accepted as final figures.

Three isolated central pilots tested the available large-b penalty with
``lambda_fnp_tail`` applied for $b_T\ge6\ \mathrm{GeV}^{-1}$ and target
$F_{\rm NP}=0.05$.  The lambda=1 pilot reaches objective/row=0.04532 and
$F_{\rm NP}(0.1,8)=0.00566$, close to
the frozen reference damping.  Lambda=10 gives a slightly worse objective
(0.04687) and stronger damping; lambda=100 over-damps already near b_T=2.
The selected corrected configuration is therefore lambda_tail=1.  This is a
new candidate constraint, not a modification of the frozen lambda=1 package.

As a sanity check, transforming the corrected seed-303 pilot with the frozen
regularized finite-b transform gives ftilde(k_T=0) approximately 4.36 for u
and 2.93 for d at x=0.1,Q=10 GeV, rather than the unphysical O(10^2) values
from the no-tail candidate.

The full corrected 24-start campaign is running under the isolated prefix
``scope_353_tail1_coupled_fnp_fit``.  A supervisor will automatically run the
50 replicas, long-revalidate robust objective outliers, form both the full
and fit-quality-accepted crossed ensembles, and render separate Fig. 2/Fig. 6
diagnostics.  No corrected result is promoted until this completes and the
W/Y-source and LHCb precision caveats are resolved.

## Completed lambda-tail=1 propagation and failure audit (2026-08-19)

The automated candidate campaign completed all requested numerical stages:

* 24 independent starts (seeds 303--326) and 50 experimental replicas
  (seeds 1001--1050), with no missing outputs;
* long revalidation of starts 315, 319, and 326 and replicas 1001, 1033,
  1040, and 1042 at 15,000 epochs/2,500-epoch patience;
* robust objective filtering, followed by 1,200 full crossed members and
  1,104 fit-quality-accepted crossed members;
* isolated Fig. 2 and Fig. 6 renders for both the full and accepted records.

The supervisor records seed 319 as a rejected start and seeds 1033 and 1040
as rejected replicas.  The accepted record therefore contains 23 starts and
48 replicas.  These exclusions are objective-based basin-quality filters;
they are not a statistical confidence prescription.

The result is **not physically usable as a final Fig. 6 uncertainty band**.
The accepted crossed F_NP ensemble still reaches ``F_NP(0.1,b=8)`` values of
about 0.26 for otherwise objective-accepted members.  In the accepted
propagation summary the active-region F_NP q16--q84 full width reaches about
17.4 times the median (median width about 1.24).  The corresponding
finite-b Bessel transform has an accepted u-quark q84 near 150 at k_T=0 and
member extrema of order 10^4, indicating tail-driven transform pathology
rather than a credible TMD uncertainty.  The Fig. 2/Fig. 6 files are
retained as explicit failure diagnostics, not production figures:

``reports/scope_353_tail1_final_fig2_fig6_full/``
``reports/scope_353_tail1_final_fig2_fig6_accepted/``

The root cause is now localized: the implemented term is a soft mean-square
penalty ``lambda_fnp_tail * mean(relu(F_NP-target)^2)`` over b_T>=6, not a
hard bound or a tail parameterization.  It can therefore leave high-b
plateaus in individual starts/replicas while preserving a good cross-section
objective.  The centered-log replica cross correctly propagates those curves
but cannot cure the pathology.  The completed campaign status is
``reports/scope_353_tail1_campaign_final_status.json`` and explicitly retains
``promotion_authorized=false``.

An isolated stronger-penalty comparison is now running under the independent
prefix ``scope_353_tail10_coupled_fnp_fit`` (lambda_tail=10, b_T>=6,
target=0.05).  It will first repeat all 24 starts; only if the resulting
F_NP tail and fit-quality gates are simultaneously credible will a full
50-replica propagation be launched.  This follow-up also remains diagnostic
and cannot modify frozen production files.

## Lambda-tail=10 start screen and continuation (2026-08-19)

All 24 lambda_tail=10 starts have completed.  The tail control is
substantially better than lambda_tail=1: at x=0.1 and b_T=8 the median F_NP
is ``6.3e-4``, the q84 is ``0.0153``, and the maximum over starts is ``0.0462``.
The median objective is ``0.04545`` per row, essentially in the same range as
the lambda_tail=1 starts.  One optimizer basin, seed 310, has objective/row
``0.13645`` and is the only reason the initial all-start screen failed its
strict maximum-objective gate; its F_NP tail itself is already strongly
damped.  The screen record is
``reports/scope_353_tail10_start_screen.json``.

Rather than reject the stronger-tail model on one recoverable optimizer basin,
the automated continuation long-revalidated seed 310 at 15,000 epochs with
2,500-epoch patience, then ran all 50 replicas, applied the robust objective
gate, and rendered full/accepted Fig. 2 and Fig. 6 diagnostics.  The completed
continuation was supervised by
``scripts/supervise_scope_353_tail10_campaign.py`` and remains explicitly
non-production; its final audit is recorded below.

## Completed lambda-tail=10 propagation and transform audit (2026-08-19)

The stronger tail-control candidate completed the full isolated propagation.
All 24 starts (seeds 303--326) and all 50 experimental replicas (seeds
1001--1050) produced initial fits. The robust gate selected replica 1033 for
long revalidation and rejected it after the long run; no start was rejected.
The long outputs used in the crossed ensemble are starts 303, 310, and 314,
and replicas 1016, 1021, 1033, and 1040. The long revalidations were run at
15,000 epochs with 2,500-epoch patience. The resulting full crossed ensemble
has 1,200 members; the fit-quality-accepted ensemble has 24*49=1,176
members.

The final records are:

``reports/scope_353_tail10_campaign_final_status.json``
``reports/scope_353_tail10_fit_quality_gate.json``
``reports/scope_353_tail10_start_replica_propagation_full/``
``reports/scope_353_tail10_start_replica_propagation_accepted/``
``reports/scope_353_tail10_final_fig2_fig6_full/``
``reports/scope_353_tail10_final_fig2_fig6_accepted/``

The candidate is materially better behaved than lambda_tail=1 but does not
close the non-uniqueness problem. In the accepted crossed F_NP ensemble the
active-region relative q16--q84 full width has maximum 20.10 and median 1.10.
For the transformed accepted Fig. 6 at x=0.1,Q=10 GeV, the q16--q84 band at
k_T=0 reaches 0.79--23.60 for u and 0.52--16.03 for d, while the central
curves are about 2.00 and 1.34. Negative lower excursions also occur at
some intermediate k_T because the empirical b-space ensemble is transformed
member-by-member. These values are substantially smaller than the
lambda_tail=1 accepted diagnostic (u q84 about 150 at k_T=0), but remain far
too broad and partly nonphysical to be a production uncertainty band.

The figures are therefore retained as explicit model-control diagnostics,
not promoted plots. The tail penalty is a soft mean-square penalty applied
only for b_T>=6 GeV^-1; increasing it from 1 to 10 damps the high-b tail but
does not identify the flexible FiLM solution in the data-sensitive region.
The full propagated band is an empirical q16--q84 envelope, not a Gaussian
one-sigma interval. ``promotion_authorized=false`` and
``frozen_production_modified=false`` remain in every final status record.

During completion, an audit found that the initial continuation script could
select early-stop start seeds for long revalidation without actually running
their long outputs. Starts 303 and 314 were run explicitly before the
crossing, and the supervisor was resumed only after all selected long files
existed. The supervisor source now enforces this requirement for future
runs. This sequencing repair changes no frozen production input or result.

**Decision:** lambda_tail=10 is a documented improvement over lambda_tail=1
for suppressing the most extreme transform spikes, but it is rejected as a
production replacement. The full unprimed 353-row W+Y candidate remains an
isolated perturbative/model diagnostic because of the hybrid W/Y provenance,
fixed-target convention assumptions, and unresolved LHCb cancellation
precision described above.

## Final verification record (2026-08-19)

The isolated scripts compile successfully.  The five unitary-transition
regression tests pass.  ``scripts/verify_frozen_baseline.py`` reports
``frozen_baseline_verified`` with 16/16 SHA-256 checks passing and
``writes_performed=false``.  No tail10 training, propagation, or handoff
operation wrote into the frozen lambda=1 baseline directory.

## W+Y FiLM interface diagnosis (2026-08-19)

The previously reported multi-hundred-percent W+Y uncertainty was not a
demonstration that FiLM cannot adapt to a deterministic Y term. A controlled
comparison found three independent candidate-side interface errors:

1. The scope data were renumbered densely after the historical W cache was
   built. The original assembler selected W curves by stale source row ID
   before applying the new row map, pairing some data bins with different
   kinematics. The repaired grid matches by dataset, qT, Q, x1, x2, and CS
   before assigning the dense trainer ID.
2. The first controls used raw table errors, whereas the frozen lambda=1
   baseline uses its Collins transition effective errors. The latter are
   larger by factors of roughly 5--50 in transition rows, so the raw-error
   control was not a like-for-like objective.
3. The external W cache already follows the fiducial convention used by the
   frozen baseline. The candidate assembler multiplied LHCb W curves by the
   acceptance factor a second time, suppressing them by about 0.45. The
   no-double-factor grid is now
   ``reports/scope_353_fnp_inputs/scope_353_bspace_w_kinematic_corrected_nofidfactor.csv``.

The exact-protocol matched controls use the repaired dense IDs, transferred
baseline-effective errors, the v22 Tevatron backend, and the frozen accepted
model state as a warm start. For the 329-row qT/Q<=0.2 core, three starts give
W-only objective/row 0.303--0.305, reproducing the frozen baseline scale. With
all candidate Y values the objective is 0.767--0.768 at 3000 epochs; a
10,000-epoch seed-303 run reaches 0.688 but still has not removed the LHCb
residual. Decomposing the Y input gives:

* all non-LHCb Y rows, with LHCb Y set to zero: 0.449/row;
* LHCb-only Y: 0.650/row, with the LHCb contribution about 19.8;
* all Y: 0.767/row, with LHCb contribution about 18.1.

Thus the corrected Tevatron/fixed-target W+Y fit is numerically close to the
baseline. The remaining degradation is localized to the six LHCb Y rows, not
to FiLM flexibility. Their full-minus-RES table has four negative and two
positive Y rows, |Y/data| up to 0.39, and a mean relative subtraction
uncertainty of 106% (maximum 234%). The first row, for example, has
Y=-0.825 pb/GeV and a 1.93 pb/GeV subtraction uncertainty. Treating those
central values as exact constraints is therefore not production-justified;
the table is explicitly marked diagnostic-only.

The complete machine-readable audit is
``reports/w_y_film_mismatch_audit.json``. The isolated decomposition launcher
and grid tools are ``scripts/run_scope_353_wy_controls.py``,
``scripts/repair_scope_353_w_grid_alignment.py``, and
``scripts/summarize_wy_film_mismatch.py``. No frozen production files,
replicas, or figures were modified. The current decision is to retain the
validated W+Y treatment for Tevatron/fixed-target rows, exclude the present
LHCb finite-Y table from any production propagation, and only revisit LHCb
after an independently converged finite-Y calculation with a defensible
uncertainty treatment.

## W+Y case-by-case Fig. 2/Fig. 6 diagnostics (2026-08-22)

For visual comparison, three nonzero-Y controls were rendered using the
corrected W/data interface and three fitted starts per case:

* ``reports/wy_control_fig2_fig6/all_y/``: all Y rows, including LHCb;
* ``reports/wy_control_fig2_fig6/non_lhcb_y/``: Tevatron/fixed-target Y with
  LHCb Y set to zero;
* ``reports/wy_control_fig2_fig6/lhcb_only_y/``: LHCb Y only.

Each directory contains ``fig2.png``/``fig2.pdf`` and
``fig6.png``/``fig6.pdf``. The bands are pointwise empirical q16--q84 spreads
over the three starts; they are not formal one-sigma intervals. The TMD curves
use the common frozen perturbative b-space reference solely as a visual
normalization, while the fitted F_NP comes from the corresponding W+Y control.
The generation manifest is ``reports/wy_control_fig2_fig6/manifest.json``.

## Corrected perturbed-start audit: prior three-start bands were artificial (2026-08-22)

The three-start case-by-case figures above were not adequate for uncertainty
interpretation: all starts were effectively warm starts from the same state.
They produced bands that were too narrow to see and should not be treated as a
complete non-uniqueness estimate.  A new isolated ensemble was generated by
perturbing every floating FiLM parameter tensor independently by 1% of its
tensor RMS around the same candidate state, then fitting seeds 303--310 for
3,000 epochs with the corrected W/Y interface.  These are controlled
sensitivity envelopes, not calibrated probability distributions.

The machine-readable record is
``reports/wy_perturbed_start_audit.json``.  The visible-band figures are under
``reports/wy_control_fig2_fig6_perturbed1pct/`` (no tail prior) and
``reports/wy_control_fig2_fig6_perturbed1pct_tail1/`` (the existing soft
``lambda_fnp_tail=1`` scaffold).  The plotting implementation is
``scripts/plot_wy_control_fig2_fig6.py`` and the start audit is
``scripts/summarize_perturbed_wy_start_audit.py``.

At x=0.1, the no-tail F_NP full ranges divided by their median are:

* all Y: 9.8% at b_T=1, 26.0% at b_T=2, and 26.9% at b_T=8;
* non-LHCb Y: 3.2% at b_T=1, 145.6% at b_T=2, and 156.3% at b_T=8;
* LHCb-only Y: 4.5% at b_T=1, 75.9% at b_T=2, and 105.6% at b_T=8.

The tail scaffold reduces the all-Y spread to 6.3% at b_T=2 and 11.6% at
b_T=8, and reduces the non-LHCb-Y spread to 81.8% and 52.3%, respectively.
It does not remove the core degeneracy below b_T=2.  Importantly, the
different Y cases are not interchangeable estimates of one uncertainty: the
all-Y and LHCb-only fits have LHCb_7 chi2-like contributions around 18 per row,
whereas the non-LHCb-Y control is around 0.21 per row.  This confirms that the
normalization/shape differences seen in the plots are partly a real failure of
the present six-row LHCb finite-Y input, not merely statistical broadening.

These plots remain diagnostic and are not production figures.  No frozen
production file, replica, or baseline figure was modified.

## Baseline-anchored W+Y normalization diagnostic (2026-08-22)

To make the W+Y study comparable to the promoted result, a separate diagnostic
now uses the production lambda=1 b-space central curve as the common
normalization.  For each W+Y start it applies only

``baseline ftilde(b) * [candidate F_NP(b) / baseline F_NP(b)]``.

The baseline F_NP is the median of the 96-start production ensemble at
``x=0.1,Q=10``.  This avoids the earlier practice of multiplying each new fit
by a frozen perturbative curve from a different interface.  The diagnostic
also writes a ``b_T=1`` shape-only version in which the candidate/baseline ratio
is forced to one at b_T=1; that version is explicitly not a physical
normalization prescription.

Outputs are under
``reports/wy_baseline_anchored_diagnostics/no_tail/`` and
``reports/wy_baseline_anchored_diagnostics/tail1/``.  Each case contains
baseline-anchored Fig. 2/Fig. 6 views, b_T=1 shape-only views, the candidate to
baseline F_NP ratio, and machine-readable member/band tables.  The generator
is ``scripts/plot_wy_baseline_anchored_diagnostics.py``.

This does not make the W+Y candidates agree with the baseline.  It shows why:
for the all-Y lambda-tail=1 case, the median candidate/baseline F_NP ratio is
approximately 0.97 at b_T=2, 1.65 at b_T=4, and 44 at b_T=8.  Thus the apparent
normalization jump is predominantly a large-b damping/shape mismatch, which
the Hankel transform amplifies at low k_T; it is not a free overall factor that
can be safely removed.  The baseline-anchored plots are therefore the useful
systematic comparison, while the raw W+Y curves remain interface diagnostics.

The extracted reference used by the diagnostic is also saved as
``reports/wy_baseline_anchored_diagnostics/{no_tail,tail1}/baseline_fnp_reference_x0p1.csv``
for reproducible follow-up reference-distance tests.

No frozen production file or baseline artifact was modified.

## Baseline-reference distance trials (2026-08-22, active)

The next controlled intervention is now implemented in the current isolated
W+Y trainer.  ``--lambda-fnp-reference-distance`` adds the same operational
relative pointwise distance used by the promoted historical lambda=1 baseline,

``lambda * mean[((F_NP-F_ref)/max(F_ref,0.10))^2]``

over a declared b_T interval and the eight fixed reference x knots.  The term
is optional and defaults to zero, so existing runs are unchanged.  The
reference loader requires a finite positive ``x,bT,F_NP`` CSV and interpolates
only onto the current 160-node external-W b grid.  The launcher records the
reference path and interval in every campaign status file.  The implementation
is in the repository-root trainer ``train_bt_dnn_v19_localbcurv.py`` and
``scripts/run_scope_353_wy_controls.py``; it is diagnostic-only and has not
modified any frozen production output.

The first preliminary eight-start test was the corrected non-LHCb-Y case with
the existing lambda_tail=1 scaffold, lambda_ref=1, and 0.1<=b_T<=2.  Its
3,000-epoch summary suggested a core-stability improvement, but it was not a
full-range or long-run decision and is superseded by the bracket completion
below.  The campaign is ``scope_329_perturbed1pct_non_lhcb_y_tail1_refdist1``.

A preliminary four-start scan with lambda_ref=3 on 0.1<=b_T<=2 improved the
median objective but did not solve the large-b tail.  A preliminary four-start
scan with lambda_ref=3 over 0.1<=b_T<=8 was visually promising; its final
eight-start, 10,000-epoch and promoted-reference results are recorded below.

The baseline-reference CSV used so far is the existing exact historical
reference ``dataset_identifiability_campaign_2026/summaries/exact_baseline_fnp_median/fnp_median.csv``;
it has the same x=0.1 curve as the promoted 96-start median to within a few
percent at the relevant b probes, but its provenance is the original 24-start
reference.  Before any promotion decision, the reference will be regenerated
with the promoted 96-start x=0.1 median explicitly substituted and the best
candidate rerun as a provenance check.  The current promising plots are under
``reports/wy_baseline_anchored_diagnostics/refdist3_b8/non_lhcb_y/``.

### Long-run and provenance-complete decision (2026-08-22)

The lambda_ref=3, full-range (0.1<=b_T<=8), eight-start campaign was run for
10,000 epochs per start, rather than accepting the visually promising 3,000-
epoch snapshot.  All eight starts reached a stationary endpoint or the fixed
epoch cap; the loss histories are flat at the endpoint and the LHCb_7
contribution no longer contains the earlier 3,000-epoch outliers.  Its final
objective range is 0.4069--0.4298 per row (median 0.4162).  At x=0.1 the
F_NP full-range/median spreads are 4.39% (b_T=1), 6.11% (b_T=2), 1.70%
(b_T=4), and 10.43% (b_T=8).  The baseline-anchored k-space q16--q84
half-width is below about 1% through k_T=0.5 GeV and remains finite at larger
k_T; the central u,d curves are visually close to the frozen baseline.  The
machine-readable trial record is
``reports/wy_reference_distance_trial_summary.json`` and the plots are under
``reports/wy_baseline_anchored_diagnostics/refdist3_b8_long10k/non_lhcb_y/``.

The same complete run was repeated with the promoted 96-start x=0.1 reference
explicitly substituted for the historical 24-start x=0.1 reference.  The
result is unchanged within run-to-run variation: objective 0.4066--0.4296
(median 0.4160), and F_NP spreads 4.53%, 7.87%, 1.79%, and 11.75% at
b_T=1,2,4,8.  This closes the reference-provenance concern.  Its plots are
under ``reports/wy_baseline_anchored_diagnostics/refdist3_b8_promoted96_long10k/non_lhcb_y/``.

The weaker lambda_ref=1 alternative was also completed with the same promoted
reference, eight starts, and 10,000 epochs.  It fits some starts slightly
better, but is not a defensible match: the objective range is 0.3472--0.4523
(median 0.3782), while the F_NP full-range/median spreads grow to 12.95% at
b_T=1, 36.30% at b_T=2, 72.85% at b_T=4, and 766.91% at b_T=8.  The resulting
baseline-anchored k-space half-width reaches about 13% at k_T=0, 20% near
k_T=1, and 42--51% near k_T=2.  It is rejected as under-regularized despite
its lower median objective.  The rejection plot is under
``reports/wy_baseline_anchored_diagnostics/refdist1_b8_promoted96_long10k/non_lhcb_y/``.

Decision: for the corrected non-LHCb-Y scope, lambda_ref=3 over the full
0.1<=b_T<=8 interval is the first isolated W+Y candidate with a defensible
baseline match after convergence and reference-provenance checks.  It remains
a candidate-side systematic diagnostic, not a production replacement: the
finite-Y LHCb rows remain excluded, and the candidate still needs the final
experimental-replica crossing and explicit combined-envelope plot before any
promotion discussion.  No frozen production file or baseline artifact was
modified.

### Lambda bracket completion (2026-08-22)

To check that the selected reference strength was not arbitrary, the same
promoted-reference, full-range, eight-start, long-run protocol was completed
for lambda_ref=1, 2, 3, and 4.  The consolidated values are in
``reports/wy_reference_distance_trial_summary.json``.  The F_NP full-range
spreads at x=0.1 (b_T=1,2,4,8) are:

| lambda_ref | median objective/row | full-range/median F_NP |
|---|---:|---|
| 1 | 0.350 | 4.3%, 31.7%, 13.1%, 99.8% |
| 2 | 0.338 | 2.9%, 12.0%, 13.3%, 101.7% |
| 3 | 0.416 | 4.5%, 7.9%, 1.8%, 11.8% |
| 4 | 0.344 | 1.9%, 13.0%, 11.7%, 52.6% |

The lower objective values at lambda_ref=1,2,4 do not indicate a better
extraction: they correspond to wider, poorly identified F_NP families.  Only
lambda_ref=3 controls both the core and large-b solution drift in this bracket
without the 50--100% tail excursions seen at the neighboring strengths.  The
lambda=2 and lambda=4 baseline-anchored plots are under
``reports/wy_baseline_anchored_diagnostics/refdist2_b8_promoted96_long10k/non_lhcb_y/``
and
``reports/wy_baseline_anchored_diagnostics/refdist4_b8_promoted96_long10k/non_lhcb_y/``.

The selected lambda_ref=3 candidate's final combined k-space diagnostic is
under
``reports/wy_final_candidate_envelope/refdist3_b8_promoted96_long10k/non_lhcb_y/``.
It transfers the existing Q=10, x=0.1 experimental replica excursions onto
the candidate central curve and adds them directionally to the eight-start
envelope.  Through k_T<=2 GeV its maximum relative half-width is 9.1%; the
experimental contribution is much smaller than the start contribution.  The
b-space output is intentionally labeled start-only because no like-for-like
Q=10, x=0.1 b-space replica table exists in this isolated W+Y scope.

This is a defendable **candidate-side** match for the corrected non-LHCb-Y
case, not a claim that the full 353-row or LHCb finite-Y production problem is
closed.  The LHCb finite-Y rows remain excluded pending their independent
observable/covariance closure, and frozen production files remain unchanged.

## Full 96-start reference-distance result (2026-08-24)

The isolated ``scope_329_refdist3_full96x50_long50k`` start batch completed all
96 independent 1%-perturbed starts (seeds 303--398), each run to the 50,000-
epoch ceiling.  This is the relevant start-only test of whether the
lambda_ref=3 constraint has actually controlled the model ambiguity; the
50-replica layer is not needed to answer that question.

At x=0.1, the empirical q16--q84 widths of F_NP are 3.5% at b_T=1,
9.5% at b_T=2, 2.4% at b_T=4, and 21.5% at b_T=8.  The full min--max
envelopes are wider: 9.4%, 30.1%, 9.2%, and 138.7%, respectively.  The
median objective is 0.4173 per row, with a range 0.3126--0.4495.

The large-b full envelope is not caused by failed optimization.  In
particular, seeds 329, 349, and 361 have among the *lowest* objectives
(0.3126, 0.3241, and 0.3404) but elevated F_NP(b_T=8), so they cannot be
discarded using a chi2/objective cut without an additional, explicitly chosen
basin-quality rule.  The start-prefix behavior shows that the first eight
starts understated the envelope: the b_T=8 full range grows from 13% at eight
starts to about 139% at all 96, while the central q16--q84 width settles near
21%.

Decision: lambda_ref=3 is a substantial improvement over the unconstrained and
tail-only diagnostics (which had order-100% to factor-of-tens excursions), but
the full start ensemble does not support calling the candidate uniquely
stable.  The start distribution is already sufficient to make that cautionary
conclusion; the remaining 50 replicas primarily propagate the smaller
experimental component.  A final promotion decision still requires the
member-by-member k_T transform of the full start ensemble and its comparison
with the frozen Fig. 6 baseline.  No starts were removed from this diagnostic,
and frozen production files remain unchanged.

## Why the refdist3 Fig. 6 band is narrower than production (2026-08-24)

The apparent narrowing was audited directly against the promoted lambda=1
96-start ensemble.  The audit is
``reports/scope_329_refdist3_full96x50_long50k_start_fig6/start_band_comparison_audit.json``
with probe values in
``reports/scope_329_refdist3_full96x50_long50k_start_fig6/start_band_comparison_probes.csv``.
Both tables contain 96 members per flavor, use the same finite-b transform
(``bmax=24``, 6001 b nodes, ``expb2`` tail, 401 k nodes, 0.92 taper), and use
the same pointwise q16/median/q84 calculation.  Thus the difference is not a
plotting, normalization, transform, or quantile implementation error.

The first-order cause is the declared prior.  Production uses the direct-FNP
reference distance with lambda=1 only on ``0.1<=b_T<=2``.  The candidate uses
lambda=3 over ``0.1<=b_T<=8`` (plus the separate lambda-tail=1 penalty above
``b_T=6``).  The extra full-range constraint suppresses the production
ensemble's large spread around ``b_T=4``; that is precisely the region to
which the low-k_T Hankel integral remains sensitive.  At x=0.1 the production
and candidate q16--q84 full F_NP widths are, respectively, 22.8% vs 2.5% at
``b_T=4`` and 22.6% vs 21.5% at ``b_T=8``.  The low-k_T TMD widths therefore
shrink even though the far-tail F_NP width does not.  At k_T=0 the u full
width is 12.2% in production versus 1.29% in the candidate; at k_T=1 it is
19.9% versus 5.62%.  The d values are 12.3% versus 1.29% and 20.9% versus
5.85%.

There is one convergence qualification.  The candidate starts all hit a
50,000-epoch ceiling and do not carry the production campaign's explicit
two-block F_NP stationarity gate; 26/96 have their best objective at or after
epoch 49,990.  This does not establish the direction of the eventual change:
continuing the fits could either make the current band narrower as starts
settle into the same basin or reveal additional spread.  The current band is
therefore provisional until the F_NP values themselves are compared after a
production-equivalent continuation.  There is, however, evidence that this
is not a large horizon artifact for the overlapping starts: comparing the same
seeds 303--310 at the earlier 10,000-epoch endpoint and the new 50,000-epoch
endpoint changes F_NP by at most 2.44% (six of eight are identical to the
stored grid precision).  This does not certify the additional 88 starts, and
it does not determine the direction of any residual correction.

The earlier wording about ``y_mode=zero`` was misleading.  That field is the
trainer's soft-evolution metadata, not a switch disabling an external finite-Y
grid.  The candidate ``predictions.csv`` records nonzero ``Y_CS_used`` for
321/329 fitted rows, with all six ``LHCb_7`` rows set to zero because the input
grid is intentionally ``scope_353_y_no_lhcb.csv``.  Thus this candidate did
use the finite-Y term for the non-LHCb scope; it is not W-only.  The six-row
LHCb finite-Y closure remains a separate unresolved issue.  The promoted
historical lambda=1 ensemble, by contrast, was generated by the W-only
production-control runner.  Consequently the current start-width comparison
also changes the fitted objective from W-only to non-LHCb finite-Y; that effect
has not yet been isolated from the stronger reference prior.

Conclusion: the smaller band is real conditional on the stronger full-range
reference prior, but it is not evidence that the data alone have reduced the
non-uniqueness.  It must not be promoted or compared as a replacement error
band until the candidate starts are continued to the production-equivalent
F_NP stationarity gate.  The baseline anchoring in the Fig. 6 script preserves
each member's F_NP ratio and cannot by itself create the narrowing.  Frozen
production files remain unchanged.

### Clarification of the epoch and finite-Y caveats (2026-08-24)

The epoch caveat is non-directional.  A 50,000-epoch endpoint ensemble that
has not passed the production block-stationarity gate could move either toward
a narrower band, a wider band, or essentially no change; it does not imply
that the displayed band is an overestimate.  For the eight starts available
at both 10,000 and 50,000 epochs (seeds 303--310), the maximum stored-grid
change in F_NP was 2.44%, with six of eight unchanged to the stored precision.
This argues against a large training-horizon artifact in that overlap, while
not replacing a formal continuation gate for all 96 starts.

The finite-Y term was in fact loaded in the refdist3 candidate.  Its
``predictions.csv`` has nonzero ``Y_CS_used`` for 321 of 329 fitted rows, and
``pred_match_CS_raw_before_dataset_norm - pred_W_CS_raw_before_dataset_norm``
closes to that value within $2.8\times10^{-6}$.  The six LHCb rows are zero-Y
by construction because the input is
``scope_353_y_no_lhcb.csv``.  The ``y_mode=zero`` field is legacy soft-evolution
metadata, not a switch disabling the external Y grid; reading it as such was
an earlier reporting error.

The promoted historical lambda=1 production ensemble is W-only, whereas this
candidate uses the repaired W grid and the defensible non-LHCb finite-Y grid.
Thus the current start-width comparison changes both the reference-distance
prior and the fitted W versus W+Y objective (and the W-grid row alignment),
so it is not yet a causal replacement of the production error band.  The
finite-Y term was not lost; the full 353-row/LHCb finite-Y production claim is
still blocked by the unresolved six-row closure and the remaining candidate
stationarity and propagation gates.

## LHCb finite-Y scope decision (2026-08-24)

For the current study, the six LHCb_7 core rows remain outside finite-Y
production propagation.  The available full-minus-RES table is not usable as
a fixed input: at one million calls its full W+Y integration has large
cancellation-driven MC uncertainty, while the derived Y values have four
negative and two positive rows with relative subtraction uncertainty up to
234%.  Fitting those values would turn numerical integration noise into an
artificial constraint on F_NP.

The active candidate scope therefore remains the corrected non-LHCb finite-Y
grid.  LHCb can re-enter only after its observable/acceptance/covariance
convention is verified and an independently stable, positive, variance-reduced
finite-Y calculation is available.  The machine-readable decision is
``reports/lhcb_finite_y_scope_decision.json``.  This is a scope decision for
the isolated candidate work; no frozen production file, replica, or figure was
modified.

## Lambda=3 candidate full propagation (2026-08-24)

The next candidate step is defined as a full empirical propagation, not a
replacement of the frozen lambda=1 production package.  It uses the repaired
external W grid, ``scope_353_y_no_lhcb.csv`` for finite Y, the six retained
LHCb_7 rows with Y=0 as W-only diagnostics, lambda_ref=3 over
0.1<=b_T<=8, lambda_fnp_tail=1 for b_T>=6, 96 perturbed starts, and 50
experimental replicas.  Each fit uses the 50,000-epoch ceiling; the final
crossed ensemble is 96x50 and will be summarized in b-space and k-space with
the empirical quantile/envelope convention.

The isolated supervisor is
``scripts/run_scope_329_refdist3_full96x50.py`` and writes under
``reports/scope_329_refdist3_full96x50_long50k_*``.  At this handoff edit,
all 96 starts and all 50 replica fits are complete with no failed members.
The crossed ensemble contains 4,800 members; its F_NP q16--q84 full-width
envelope has a maximum of 29.7% and median of 11.8% in the active b-space
region.  Candidate Fig. 2 and Fig. 6 are now rendered under
``reports/scope_329_refdist3_full96x50_long50k_final_fig2_fig6/``.  These are
candidate diagnostics: the final review still checks fit quality, finite
predictions, member counts, and production-equivalent F_NP/start diagnostics
before any candidate figure is called a new standard.  The frozen lambda=1
files and figures remain unchanged.

## Full lambda=3 versus frozen lambda=1 comparison (2026-08-24)

The complete 96-start x 50-replica candidate cross was compared with the
frozen lambda=1 96-start x 50-replica ensemble.  The reproducible audit is
``scripts/audit_refdist3_full_production_comparison.py``; its outputs are
``reports/scope_329_refdist3_full96x50_long50k_final_fig2_fig6/full_production_comparison.json``
and the accompanying probe CSV.

The lambda=3 candidate is not a uniformly improved uncertainty result.  Using
the pointwise empirical q16--q84 full width:

* at b_T=1, the width changes from 0.4% (lambda=1) to 8.3% (candidate);
* at b_T=2, from 3.1% to 27.6%;
* at b_T=4, it narrows from 24.0% to 10.5%;
* at k_T=0 for u, it narrows from 13.1% to 4.1%;
* at k_T=1, it changes from 20.3% to 22.9%;
* at k_T=1.5, from 6.5% to 14.2%;
* at k_T=2, from 20.1% to 63.4%.

Across the active k_T region, the maximum full width increases from 21.3% to
28.5% for u and from 22.5% to 27.7% for d.  The candidate central F_NP shape
also shifts by roughly -6--8% at b_T=1--2 and +8% near b_T=4.  Thus the
stronger full-range reference prior suppresses the b_T~3--4 ambiguity that
controls the very-low-k_T endpoint, but it relocates ambiguity into the
data-sensitive b_T<2 core and intermediate k_T region.

The cross-section fit remains numerically acceptable rather than failing: the
candidate start objective-like median is about 0.415 per row versus about
0.403 for the stationary lambda=1 starts.  However, the candidate comparison
is not strictly causal because it simultaneously changes the W grid, the
historical W-only objective to non-LHCb finite-Y, and the reference-prior
range.  Moreover, 26/96 candidate starts and 48/50 candidate replicas have
their best stored epoch at the 50,000-epoch ceiling, so the candidate lacks
the baseline's explicit F_NP stationarity certificate.

Decision: do not promote lambda_ref=3 as an improved replacement.  It is a
useful candidate diagnostic, but the complete propagation does not establish a
smaller total non-uniqueness band.  A fair attribution test would rerun
lambda=1 on the same repaired W and non-LHCb finite-Y inputs before drawing a
strong causal conclusion about the prior itself.  Frozen production remains
unchanged.

## Candidate surface figures (2026-08-25)

PRD-style surface diagnostics analogous to the earlier Fig. 7/8 were rendered
from the complete 96-start x 50-replica refdist-3 cross at Q=7.5 GeV. The
available candidate diagnostic grids contain x=0.1, 0.2, 0.3, and 0.5; the
surfaces use those four direct x knots for the numerical values, with a
display-only linear interpolation to a 41-row x mesh to avoid striping. At
each direct knot, all 4,800 crossed F_NP members were transformed with the
regularized finite-b settings used for Fig. 6. Surface color uses a robust
0--50% scale and smoothly fades in the low-signal tail; the CSV retains the
unclipped widths. This is not a Gaussian confidence surface and is not a
production promotion.

The renderer is ``scripts/plot_scope_329_refdist3_surface_fig7_fig8.py``.
Outputs are in
``reports/scope_329_refdist3_full96x50_long50k_final_fig2_fig6/``:

* ``fig7.png`` and ``fig7.pdf`` (u surface);
* ``fig8.png`` and ``fig8.pdf`` (d surface);
* ``wy_candidate_u_surface.png``/``.pdf`` and
  ``wy_candidate_d_surface.png``/``.pdf`` (neutral manuscript aliases);
* ``fig7_refdist3_full96x50_surface.csv`` and
  ``fig8_refdist3_full96x50_surface.csv`` (surface values and widths);
* ``surface_summary.json`` (scope and transform metadata).

Frozen production files remain unchanged.

## PRD integration handoff for the new W+Y section (2026-08-25)

The draft-specific section text, figure-placement guidance, neutral figure
labels, ready-to-paste subsection, numerical comparison table, and do-not-claim
checklist are recorded in
``reports/WplusY_new_section_PRD_handoff.md``.  It was prepared against
the author-supplied `b_space_PRD.pdf` draft.  The proposed new surface figures
are additional candidate diagnostics; they do not replace the nominal Figs.
7 and 8, and no frozen production output was modified.
