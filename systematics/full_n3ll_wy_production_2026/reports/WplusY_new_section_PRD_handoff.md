# Handoff for an additional W+Y section in the PRD draft

Date: 2026-08-25
Draft reviewed: author-supplied `b_space_PRD.pdf` (21 pages,
dated 2026-08-19 in the PDF)
Purpose: provide section-ready text, figure guidance, and an accuracy/provenance
record for adding the recent W+Y work without replacing the nominal production
figures or silently changing the paper's existing claims.

## Editorial decision

The new material should be presented as an **exploratory W+Y candidate and
non-uniqueness stress test**, not as a replacement of the nominal production
extraction.  The nominal 329-point result, its lambda=1 reference-distance
prescription, and the existing production Figures 2, 6, 7, and 8 remain the
primary result.  The two new surfaces are additional diagnostics and should
not be called “Fig. 7” or “Fig. 8” in the new section.

Recommended manuscript labels are descriptive labels such as
“candidate W+Y surface (u)” and “candidate W+Y surface (d)”.  In LaTeX, use
labels such as `fig:wy_candidate_u_surface` and
`fig:wy_candidate_d_surface`; let LaTeX assign the eventual figure numbers.
For a temporary handoff convention they may be called **Figure W1** and
**Figure W2**, but those are not replacements for the paper's current Figs. 7
and 8.

## Where the section belongs

The cleanest placement is immediately after the present Sec. VIII C
“Finite-Y transition at the Tevatron boundary” and before the present
subsection on pseudo-data and inverse-transform closure.  This makes the logic
explicit:

1. Sec. VIII C remains the controlled 24-point Tevatron-boundary robustness
   test already described in the draft.
2. The new subsection records the later, isolated attempt to connect a
   finite-Y grid to the flexible FiLM extraction over the 329-row core.
3. The existing nominal extraction and its production figures are not
   redefined.

Suggested subsection title:

> **Exploratory finite-$Y$ W+Y candidate and identifiability stress test**

If the journal style does not permit an extra subsection in Sec. VIII, the
same text can be placed as a short new section immediately before the
Discussion, with the same explicit “candidate diagnostic” status.

## What is already in the draft and should remain unchanged

The current draft correctly states that the nominal 329-point fit is W-only,
that the finite-Y transition is tested separately on 24 additional Tevatron
points, and that the current Figures 7 and 8 are the nominal regularized
momentum-space surfaces.  Those statements should remain true.  In
particular, do not replace the following nominal quantities with candidate
values:

* the lambda=1 reference-distance term acting over
  $0.1\le b_T\le2.0\ \mathrm{GeV}^{-1}$;
* the nominal $\chi^2_{\rm data}/N_{\rm acc}\simeq0.418$ statement;
* the nominal 96-start by 50-residual-field uncertainty construction;
* the existing production Figures 2, 6, 7, and 8;
* the statement that the paper's nominal sample is W-only.

The new section is an additional test of how the fitted nonperturbative
factor responds when a corrected finite-Y input is supplied.  It is not a
retroactive revision of the nominal fit.

## Perturbative W+Y work that can be stated separately

There are two distinct W+Y products in the recent work and they must not be
collapsed into one claim.

### Direct Tevatron perturbative grid

An isolated external DYTurbo calculation produced a directly evaluated,
unprimed N3LL+NNLO W+Y grid for 122 Tevatron bins:

* CDF Run I: 41 bins;
* CDF Run II: 61 bins;
* D0 Run I: 20 bins.

The convention is

$$
\begin{aligned}
W=\mathrm{RES},\qquad Y=\mathrm{FO}_{\rm NNLO}-\mathrm{ASY}_{\rm NNLO},\qquad W+Y=\mathrm{RES}+\mathrm{CT}+\mathrm{VJ},
\end{aligned}
$$

with the direct nonperturbative center $g_1=1.017\ \mathrm{GeV}^2$.  The
final isolated table is finite and positive in every row.  Its W+Y/data ratio
has median 1.002973 and range 0.775403--1.168256.  The mean estimated grid
integration uncertainty is 1.1773%, with a maximum of 16.5446% in
cancellation-dominated rows.  The explicit conventional-Y reconstruction has
median residual $2.947\times10^{-4}\ \mathrm{pb/GeV}$ and maximum residual
$9.070\times10^{-3}\ \mathrm{pb/GeV}$.  These integration estimates are
diagnostics, not extra statistical error bars on the TMD.

This 122-row table is the strongest recent perturbative W+Y validation and is
appropriate to cite as an isolated Tevatron benchmark.  It is not, by itself,
a refitted global TMD result.

### Coupled 329-row candidate input

The later DNN candidate used a repaired external W grid and an assembled
finite-Y correction on the 329-row core.  The repaired W grid removes the
earlier stale-row and double-fiducial-factor interface errors.  Rows are
matched by the dataset/bin identity and kinematics before being assigned to the
dense fit index.  The candidate inputs are:

* repaired W grid:
  `reports/scope_353_fnp_inputs/scope_353_bspace_w_kinematic_corrected_nofidfactor.csv`;
* finite-Y table:
  `reports/scope_353_fnp_inputs/scope_353_y_no_lhcb.csv`;
* fit-ready data and old effective errors:
  `reports/scope_353_fnp_inputs/data_with_old_effective_errors/`.

For the 329 fitted rows, 321 carry nonzero finite-Y input.  The six retained
`LHCb_7` core rows are deliberately assigned (Y=0) and are used only as
W-only diagnostics.  Thus this is **not** a 329-row finite-Y production
calculation and certainly not a fully matched 353-row W+Y result.

The present LHCb finite-Y subtraction is excluded because its full-minus-RES
calculation is cancellation- and integration-limited: the derived values have
both signs and relative subtraction uncertainty as high as 234%.  Treating
those noisy central values as exact F_NP constraints would convert numerical
integration noise into a fitted model feature.  The machine-readable scope
decision is
`reports/lhcb_finite_y_scope_decision.json`.

## Candidate fitting prescription

The candidate retains the same physics-informed monotone FiLM architecture as
the nominal extraction.  The network learns a positive damping rate and hence

$$
\begin{aligned}
F_{\rm NP}(x,b_T)=\exp\left[-\int_0^{b_T} d\beta\cdot 2\beta\cdot A_\theta(x,\beta)\right],
\end{aligned}
$$

so $F_{\rm NP}(x,0)=1$ and the factor is nonincreasing.  The perturbative
PDFs, evolution, matching factors, and transform remain outside the trainable
network.

Relative to the nominal production objective, the candidate adds the same
direct-$F_{\rm NP}$ reference-distance construction but changes its declared
scope and strength:

$$
\begin{aligned}
\Phi=\chi^2_{\rm data}+\chi^2_{\rm norm} +3 \chi^2_{\rm ref}+1 \chi^2_{\rm tail}.
\end{aligned}
$$

Here the reference distance acts over $0.1\le b_T\le8.0\ \mathrm{GeV}^{-1}$,
rather than only to $2.0\ \mathrm{GeV}^{-1}$, and the separate tail control
begins at $b_T=6.0\ \mathrm{GeV}^{-1}$ with target
factor 0.05.  The reference term is a direct pointwise distance in
$F_{\rm NP}$, with the same denominator floor used by the nominal implementation;
it is not a boundary condition at $b_T=8$ and does not mathematically make
the inverse problem unique.

The candidate protocol used 96 independent 1%-perturbed starts (seeds
303--398), 50 pseudo-data residual fits (seeds 1001--1050), a 50,000-epoch
ceiling, the monotone FiLM model, and the repaired external W/Y inputs.  The
4,800-member ensemble was constructed by adding centered replica residuals in
$\log(F_{\rm NP})$ to each start curve.  It therefore contains 96 fitted
networks and 50 residual fields; it does **not** contain 4,800 independent
neural-network fits.

The reported q16, q50, and q84 values are empirical quantiles of that crossed
procedure.  The experimental component is naturally associated with the
supplied pseudo-data errors.  The initialization and combined components are
model/optimization spreads and must not be described as calibrated Gaussian
one-standard-deviation intervals.

## Candidate result and interpretation

The candidate fit remains numerically usable, but it is not a uniformly better
uncertainty result than the frozen lambda=1 production ensemble.  The
candidate start objective-like value has median 0.4152 per row (range
0.3126--0.4447), compared with 0.4030 per row for the frozen stationary
lambda=1 starts.  The comparison is not causal because it simultaneously
changes the W grid, the W-only versus non-LHCb-finite-Y objective, and the
reference-distance interval.

Using the pointwise empirical q16--q84 **full width** in the active region:

| quantity | frozen lambda=1 | lambda=3 candidate |
|---|---:|---:|
| b-space u/d maximum | 24.0% | 29.8% |
| b-space u/d median | 3.1% | 11.7% |
| k-space u maximum | 21.3% | 28.5% |
| k-space u median | 11.1% | 13.9% |
| k-space d maximum | 22.5% | 27.7% |
| k-space d median | 11.5% | 14.3% |

Selected candidate full-width probes are 8.3% at $b_T\simeq1$, 27.6% at
$b_T\simeq2$, and 10.5% at $b_T\simeq4$. The stronger full-range prior
does suppress the $b_T\simeq3$--4 spread that controls the very-low-$k_T$
Hankel endpoint. It simultaneously broadens the data-sensitive
$b_T<2$ core and intermediate-$k_T$ region. The candidate central
$F_{\rm NP}$ shape is also lower by roughly 6--8% at $b_T=1$--2 and higher
by roughly 8% near $b_T=4$. The result is therefore a genuine shape change,
not merely a change in the plotted band.

The correct conclusion is that the stronger reference prior **relocates**
the non-uniqueness rather than removing it.  The candidate should not be
promoted as a replacement production model.  A fair causal comparison would
rerun lambda=1 on precisely the same repaired W and non-LHCb finite-Y inputs;
that comparison is separate from the current diagnostic.

There is also a convergence qualification.  All candidate runs reached the
50,000-epoch ceiling; 26/96 starts and 48/50 replicas have their best stored
epoch at that ceiling.  The candidate therefore does not carry the nominal
production campaign's same explicit F_NP stationarity certificate.  This is
why the new figures should be identified as candidate diagnostics even though
the full member count and transform are complete.

## Ready-to-paste subsection draft

The following text is written to fit the terminology and equation style of the
current paper.  It intentionally avoids calling the result a new production
fit.

> **Exploratory finite-$Y$ candidate.**  The nominal extraction described
> above is intentionally W-only.  As a separate diagnostic, we connected the
> repaired b-space W grid to an assembled finite-$Y$ correction on the
> 329-row core and refit only the nonperturbative factor.  The correction is
> nonzero on 321 of the 329 fitted rows.  The six LHCb rows were retained only
> as W-only diagnostics, with $Y=0$, because the available full-minus-RES
> LHCb subtraction has cancellation-driven integration uncertainty too large
> to be used as an exact fit input.  This scope is therefore not a fully
> matched 353-row W+Y production prediction and does not replace the nominal
> W-only result.
>
> The candidate uses the same positive-rate FiLM architecture as the nominal
> extraction, but applies the direct $F_{\rm NP}$ reference-distance term
> with coefficient $\lambda_{\rm ref}=3$ over $0.1\le b_T\le8.0\ \mathrm{GeV}^{-1}$,
> together with the separate large-$b_T$ tail
> control beginning at $b_T=6.0\ \mathrm{GeV}^{-1}$.  We performed 96
> independent 1%-perturbed starts and 50 pseudo-data residual fits to a
> 50,000-epoch ceiling.  The final ensemble contains 4,800 crossed members
> formed from 96 fitted networks and 50 residual fields; it is not a set of
> 4,800 independent fits.  The central curve and pointwise bands are the
> empirical q50 and q16--q84 quantiles of this procedure.
>
> The stronger full-range reference prior does not produce a uniformly smaller
> envelope.  It narrows the $b_T\simeq3$--4 region and the very-low-$k_T$
> endpoint, but broadens the data-sensitive $b_T<2$ core and the
> intermediate-$k_T$ region.  Relative to the frozen lambda=1
> ensemble, the candidate maximum active k-space q16--q84 full width changes
> from 21.3% to 28.5% for $u$ and from 22.5% to 27.7% for $d$.  The
> candidate is therefore retained as a model-identifiability diagnostic, not
> promoted as a replacement production extraction.  Its combined band is an
> empirical procedural envelope and is not assigned a calibrated Gaussian
> one-standard-deviation interpretation.

## New surface figures

The two additional surfaces are generated from the complete 4,800-member
candidate cross at $Q=7.5\ \mathrm{GeV}$.  They show $x f_1^q(x,k_T;Q)$ for
$k_T\in[0,3]$ GeV.  The numerical candidate grids are directly available at
$x=0.1,0.2,0.3,0.5$; the plotted surface uses a display-only 41-row
interpolation in $x$ to avoid a visually blocky four-row mesh.  The
surface height is the empirical q50.  The color encodes the relative
q16--q84 half-width, with a low-signal fade and a robust 0--50% display scale.
The underlying CSV files retain the unclipped numerical widths.

The color-bar text has been set to match the existing PRD surface convention:

> **Combined relative 68% half-width**

In the caption/prose this label must be qualified: the displayed quantity is
the empirical q16--q84 half-width of the crossed model/replica ensemble.  The
“68%” wording is retained for visual continuity with the nominal figures and
does not turn the candidate model spread into a calibrated 68% confidence
interval.

Suggested neutral figure references:

* **Candidate W+Y surface (u):** current artifact
  `reports/scope_329_refdist3_full96x50_long50k_final_fig2_fig6/wy_candidate_u_surface.png`;
  manuscript label `fig:wy_candidate_u_surface`.
* **Candidate W+Y surface (d):** current artifact
  `reports/scope_329_refdist3_full96x50_long50k_final_fig2_fig6/wy_candidate_d_surface.png`;
  manuscript label `fig:wy_candidate_d_surface`.

Suggested caption for the u panel:

> **Candidate W+Y surface for the $u$ flavor.**  The surface height is the
> empirical median of the 96-start by 50-residual-field crossed ensemble for
> the corrected non-LHCb finite-$Y$ candidate at $Q=7.5\ \mathrm{GeV}$.
> The color gives the relative q16--q84 half-width, shown with the standard
> “Combined relative 68% half-width” label; it is an empirical procedural
> envelope rather than a calibrated Gaussian 68% confidence band.  The direct
> numerical x knots are $0.1,0.2,0.3,0.5$; interpolation between them is for
> display only.  This candidate surface is an additional diagnostic and does
> not replace the nominal production surface.

Use the same caption with $u\to d$ for the d panel.  If a shorter caption is
required, retain at least the phrases “corrected non-LHCb finite-Y candidate,”
“96 starts x 50 residual fields,” “empirical q16--q84,” and “does not replace
the nominal production surface.”

## Files and reproducibility

Primary candidate and comparison artifacts:

* candidate runner:
  `scripts/run_scope_329_refdist3_full96x50.py`;
* surface renderer:
  `scripts/plot_scope_329_refdist3_surface_fig7_fig8.py`;
* candidate figure directory:
  `reports/scope_329_refdist3_full96x50_long50k_final_fig2_fig6/`;
* candidate b-space members and q16/q50/q84 bands:
  `fig2_bspace_crossed_members_long.csv` and
  `fig2_bspace_start_replica_bands.csv`;
* candidate k-space members and q16/q50/q84 bands:
  `fig6_kspace_crossed_members_long.csv` and
  `fig6_kspace_start_replica_bands.csv`;
* surface numerical tables:
  `fig7_refdist3_full96x50_surface.csv` and
  `fig8_refdist3_full96x50_surface.csv`;
* surface metadata:
  `surface_summary.json`;
* full candidate-versus-baseline audit:
  `full_production_comparison.json` and
  `full_production_comparison_probes.csv`;
* LHCb finite-Y scope decision:
  `reports/lhcb_finite_y_scope_decision.json`;
* full project chronology and earlier W+Y controls:
  `full_n3ll_wy_production_2026/HANDOFF.md`.

The plotting script has been syntax-checked and rerun after the color-scale
and label revisions.  The figures are isolated diagnostics.  Frozen production
files, the nominal lambda=1 ensemble, and the existing production figures were
not modified.

## Do-not-claim checklist

Do not write any of the following in the new section:

* “the 329-row data set has been replaced by a global finite-Y production fit”;
* “all six LHCb rows have a validated finite-Y term”;
* “the lambda=3 candidate has a smaller total uncertainty”;
* “the 4,800 members are independent fits”;
* “the combined candidate envelope is a calibrated 68% confidence band”;
* “the new surfaces replace Figs. 7 and 8”;
* “the finite-b Hankel transform is a fixed-order high-$k_T$ prediction.”

The defensible summary is narrower: the recent work demonstrates a usable
direct Tevatron W+Y perturbative grid and provides an isolated corrected
non-LHCb finite-Y candidate test.  In the flexible FiLM model, extending the
reference-distance control to the full b-space interval changes where the
non-uniqueness appears but does not establish a smaller overall model-form
envelope.  The nominal W-only production result therefore remains the paper's
primary result, while the two new surfaces document the W+Y identifiability
stress test.
