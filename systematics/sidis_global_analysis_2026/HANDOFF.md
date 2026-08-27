# SIDIS global-analysis campaign handoff

**Initialized:** 2026-08-26  
**Status:** first public-source harvest complete; discovery/provenance phase; no production fit authorized
**Scope:** shared unpolarized TMD-PDF plus SIDIS TMD-fragmentation global analysis

This campaign is intentionally separate from the completed DY identifiability
and production work. The current DY result remains frozen and is a read-only
comparison input. A future global fit may use it as a warm start, prior, or
baseline diagnostic only after the interface and data conventions are frozen.

## What is already established

- The fixed-target DY incumbent is the lambda=1 empirical-reference package
  with 96 stationary starts crossed with 50 conditional experimental replicas.
- Its non-uniqueness and operational q16--q84 bands are documented in
  `../dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50/`.
- The perturbative CSS/TMD, OPE, hard-factor, and transform source stack is
  documented in `../perturbative_provenance_completion/` and the `v22/`/`v23/`
  source trees.
- Finite-Y/W+Y work exists only in isolated Tevatron/collider scopes and is not
  a universal global-DY or SIDIS prescription.

None of those facts supplies a SIDIS fragmentation model or authorizes a joint
refit.

## 2026-08-26 public-source harvest

The first read-only public harvest is complete from version-1 HEPData CSV
submission archives. The source registry is `config/public_sources.json`; the
download hashes and record metadata are in `data/hepdata_download_manifest.json`.
The table profiler and source inventory are recorded in
`reports/hepdata_table_inventory.json` and
`reports/public_source_inventory.{json,md}`.
The scope comparison is summarized in `reports/candidate_scope_options.{json,md}`
and is intentionally a decision aid, not a row-selection manifest.
The search boundary and explicit exclusions (nuclear and spin-dependent
observables) are recorded in `reports/public_source_search_boundary.md` and
`config/excluded_public_sources.json`.

| Source | Scope | Tables | Raw rows | Primary rows | pT tables | Current use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| HERMES `46860` | charged/neutral pion multiplicities, hydrogen, z/x/Q2 projections | 4 | 103 | 103 | 0 | collinear complement; definition/overlap audit pending |
| COMPASS `ins1236358` | charged-hadron pT² distributions, 6LiD, x_B/Q²/z bins | 48 | 19,504 | 19,504 | 46 | historical transverse candidate; generic-error/overlap audit pending |
| HERMES `ins1208547` | identified pi+/K+, H/D, x_B/Q2/z/P_hperp | 64 | 1,136 | 1,136 | 16 | primary TMD candidate; row/covariance audit pending |
| COMPASS `ins1624692` | charged h+/h-, 6LiD, x/Q2/z/P_hT2 | 162 | 13,992 | 4,664 | 162 | primary TMD candidate; auxiliary correction blocks excluded explicitly |
| COMPASS `ins1444985` | pi+/- and h+/-, 6LiD, x/y/z | 4 | 6,220 | 1,244 | 0 | collinear complement; convention audit pending |
| COMPASS `ins1483098` | K+/-, 6LiD, x/y/z | 2 | 3,090 | 618 | 0 | collinear complement; convention audit pending |
| COMPASS `ins2840545` | pi+/K+/h+/-, H, x/y/z | 3 | 6,314 | 1,804 | 0 | recent collinear complement; full audit pending |

Total: 287 tables and 50,359 parsed rows (29,073 primary measurement rows and
21,286 explicitly marked auxiliary correction-factor rows); 224 tables expose a
transverse-momentum axis. The primary transverse candidate count is 24,088
rows. This is an inventory, not an approved fit selection. HEPData tables can
include correction factors, asymmetric statistical/systematic columns, and
projection-specific axes; the ingestion layer preserves them and intentionally
does not combine errors or infer a covariance matrix. The HERMES/COMPASS record
pages remain the provenance authorities for any converted table.

The harvest is reproducible with:

```text
python scripts/fetch_hepdata_records.py
python scripts/profile_hepdata_tables.py
python scripts/build_source_inventory.py
```

The raw archives are local campaign inputs. A public source release should
carry the registry, scripts, hashes, and inventory—not silently bundle all raw
data or claim that the rows are fit-ready.

The HERMES collaboration separately publishes a [multiplicity
database](https://hermesmults.appspot.com/) with multidimensional files and
statistical covariance matrices. The pointer and archive URL are recorded in
`config/public_sources.json`, but that supplemental archive has not yet been
mirrored because the DESY download endpoint was unreachable in this run. The
HEPData CSV remains a valid value-table candidate, not a covariance-complete
HERMES input.
`scripts/fetch_hermes_database_metadata.py` records the page hash, documented
binning/covariance warnings, and the proton/deuteron `zpt-3D` transverse-
momentum download links in `data/hermes_database_manifest.json`; the concise
interpretation is in `reports/hermes_database_provenance.md`.
The uncertainty/covariance boundary is summarized in
`config/covariance_manifest.json` and `reports/covariance_readiness.md`:
COMPASS HEPData archives provide component columns but no complete correlated
matrix, while the HERMES matrix archive remains pending.

## Software boundary (2026-08-26)

`FORMALISM.md` records the candidate leading-power W boundary and the
unresolved multiplicity/covariance interfaces. `sidis_data.py` is the
metadata-preserving CSV reader/profiler and strict plain-text/gzip covariance
reader. It handles duplicate labels, unprefixed HEPData description
continuations, and repeated target/charge headers while retaining raw columns
and row-level block metadata. `sidis_observables.py` supplies a radial
PDF-times-TMDFF Bessel convolution with explicit `qT=P_hT/z` and
`b db/(2 pi)` conventions, plus guarded multiplicity ratios. It returns the
SIDIS structure function only; experiment-specific prefactors, DIS denominators,
target/beam composition, radiative factors, and Y terms remain explicit and
unimplemented until the formalism lock. Sixteen local unit tests pass,
including compressed-covariance, repeated-header, auxiliary-block, strict
covariance, and explicit canonicalization regressions.
`scripts/audit_table_provenance.py` records the conservative table/row audit
without selecting rows, while `scripts/summarize_candidate_scopes.py` groups
audited rows into first-conversion, cross-check, combined, and collinear-
complement options.
`sidis_dataset.py` is the explicit row-to-observation adapter: it requires
caller-supplied column/block mappings, supports pT² and asymmetric errors,
target/data-block filters, and an explicit published-placeholder policy. It
preserves all provenance while rejecting ambiguous or uncertainty-free rows.
`sidis_covariance.py` provides strict Cholesky whitening and correlated
quadratic forms for a covariance supplied by the caller; no correlations are
invented and no stat/sys components are combined by the software.
`scripts/validate_candidate_mappings.py` validates the full HERMES and COMPASS
transverse candidate scopes in memory and records exact columns, block filters,
placeholder handling, and canonical-row counts in
`reports/candidate_mapping_validation.{json,md}`.

## Questions that must be answered before fitting

1. Which SIDIS observable is primary: multiplicities, absolute cross sections,
   or both? The numerator/denominator and radiative/acceptance treatment must
   be explicit.
2. Which experiments, targets, hadrons, and kinematic cuts have stable source
   tables and complete point-to-point plus correlated uncertainties?
3. Are bin-integrated theory predictions required, and what published bin
   convention is used for each experiment?
4. Which collinear fragmentation functions and flavor/hadron parameterization
   are allowed? How are charge conjugation, favored/unfavored channels,
   strange/heavy flavors, and hadron-mass effects handled?
5. Which perturbative order and evolution convention is shared with DY, and
   which SIDIS hard/OPE coefficients are actually available and tested?
6. Where is the SIDIS (Y) term required, and can its fixed-order observable
   and covariance treatment be closed independently?
7. Which parameters are common to DY and SIDIS, which are process-specific,
   and how are normalization and theory nuisances prevented from absorbing
   genuine TMD information?

## Locked rules for the discovery phase

- Do not copy or edit files inside the active DY production package.
- Do not select rows because they improve a joint chi-square before their
  observable and covariance provenance is complete.
- Do not call a SIDIS-only fit “global”; reserve that term for an approved joint
  DY+SIDIS scope with explicit dataset coverage.
- Do not identify a TMDFF by a narrow replica band alone. Use independent starts,
  pseudodata closure, held-out bins, and model-form comparisons.
- Keep experimental replicas, start non-uniqueness, FF parameterization, and
  perturbative/theory variations as separately auditable components.
- Every trial gets a unique tag and a machine-readable report under `reports/`.

## First deliverables

1. `config/scope_manifest.json` listing candidate SIDIS sources and their
   provenance state (seven HEPData records are harvested; row-level
   approval remains open).
2. A row-level data inventory with observable, target, hadron, units, cuts,
   covariance, normalization, and bin-integration fields.
3. `FORMALISM.md` plus a scalar reference implementation independent of the
   DNN; the candidate W convolution is documented, while the final observable
   convention remains open.
4. A DY interface regression showing that the frozen lambda=1 central is
   reproduced before any joint optimization.
5. A decision report selecting, rejecting, or deferring the first SIDIS scope.

## Restart instructions

Read this file, then `README.md`, then the parent
`../PROJECT_HANDOFF.md`. Inspect the lambda=1 production manifest/audit and the
perturbative/provenance handoff before importing any baseline code. Do not
start long fits until the first five deliverables and the data identity gate
are complete.

## Decision log

| Date | Decision | Evidence | Status |
| --- | --- | --- | --- |
| 2026-08-26 | Create separate SIDIS/global-analysis campaign | Existing DY production and systematics are frozen; SIDIS requires a TMDFF and new observable/covariance closure | Initialized |
| 2026-08-26 | Harvest seven version-1 public HEPData records | 287 tables and 50,359 parsed rows (29,073 primary, 21,286 auxiliary); 224 tables expose transverse momentum; no rows or covariance selected | Discovery only |
| 2026-08-26 | Add convention-explicit SIDIS software boundary | Metadata-preserving reader/profiler, radial PDF×TMDFF convolution, guarded ratio, strict covariance operations, and sixteen local scalar tests pass | Discovery only |
| 2026-08-26 | Correct repeated-block parsing before row audit | HERMES/COMPASS CSVs can repeat headers between target or charge blocks and append correction-factor blocks; the reader now retains block metadata, marks auxiliary headers, and skips only explicitly requested placeholders. Corrected inventory is 287 tables, 50,359 parsed rows, 224 transverse-momentum tables, and 24,088 primary transverse candidate rows; no rows are selected | Discovery only |
| 2026-08-26 | Add explicit canonical observation adapter | `sidis_dataset.py` converts only caller-mapped value/axis/error columns, supports comment-defined intervals, pT², target/data-block filters, explicit placeholder handling, preserves row provenance, and fails closed on ambiguity; sixteen local tests pass | Discovery only |
| 2026-08-26 | Validate source-specific candidate mappings | HERMES 16/16 tables -> 288 target-block observations; COMPASS 2013 46/46 -> 18,624 observations after explicit dash-placeholder handling; COMPASS 2018 162/162 -> 4,664 primary observations with 9,328 auxiliary correction rows excluded by an explicit block filter. No rows are approved and covariance remains unresolved | Discovery only |

## 2026-08-26 global-source expansion and staged-count benchmark

The initial seven-record HERMES/COMPASS harvest was not the complete public
unpolarized SIDIS data universe.  A separate registry now records 22 source
identities spanning HERMES, COMPASS, JLab Hall C/Hall A/CLAS, EMC, E665, H1,
and ZEUS.  The registry is `config/global_sources.json`; its machine-readable
inventory is `reports/global_source_inventory.json` and the review table is
`reports/global_source_inventory.md`.  The fetcher is
`scripts/fetch_global_sources.py` and writes only to the ignored
`data/raw/global/` cache plus `data/global_source_download_manifest.json`.

The public harvest currently contains 16 HEPData records (582 tables,
57,358 parsed rows, 35,963 primary and 21,395 explicitly auxiliary rows) and
three extracted arXiv source packages.  The public JLab packages are:

- CLAS 0809.1153, with machine-readable ancillary pi+ cross-section tables;
- Hall C E00-108 (1103.1649), with published H/D pi+/- cross-section tables in
  the TeX source;
- Hall A E06-010 (1610.02350), with published 3He pi+/- cross-section tables
  in the TeX source.

The CLAS and Hall C sources are stage-2 absolute/low-energy candidates, not
automatic additions to the multiplicity fit.  The Hall-A 3He source is a
nuclear diagnostic because a free-nucleon TMD fit would otherwise absorb
nuclear impulse-approximation and dilution effects.  The E12-09-017 proposal
and the CLAS Physics Database pointer remain deferred because no final
reproducible public table was located.  H1/ZEUS current-region and E665/EMC
forward or historical observables are retained as explicit diagnostic or
stage-2 candidates; their published axes are not silently relabelled as
standard SIDIS (P_{hT}) multiplicities.

The two external fit papers supplied for this campaign are recorded in
`config/external_fit_benchmarks.json`.  The 1,547 SIDIS-point number is a
post-cut HERMES/COMPASS benchmark, not a raw-table total.  It must be
reconstructed with its stated TMD-validity cuts, vector-meson-subtraction
choice, duplicate/projection policy, and uncertainty treatment before it is
used as a like-for-like comparison.  The benchmark papers are arXiv
[2206.07598](https://arxiv.org/abs/2206.07598) and
[2405.13833](https://arxiv.org/abs/2405.13833).

The staged rule is now locked: first reproduce the 1,547-point clean
HERMES/COMPASS scope; then add one experiment/hadron/target family at a time
with held-out and leave-one-family-out checks; finally test diagnostic
families only if their observable and nuclear/current-region formalism closes.
`approved_rows` remains zero and no DY production file was changed.

### Restart commands for the global inventory

```text
python scripts/fetch_global_sources.py --kind hepdata
python scripts/fetch_global_sources.py --kind arxiv_eprint_with_ancillary --kind arxiv_eprint_tex_tables
python scripts/build_global_source_inventory.py
```

Do not use the global raw cache as a fit dataset.  The next implementation
step is a source-specific converter plus an explicit cut-count report that
must reproduce the literature 1,547 count before any joint optimization.

## 2026-08-26 benchmark audit and arXiv/JLab inventory

The two supplied literature references have now been read from their public
arXiv sources and recorded in `config/external_fit_benchmarks.json`:

- arXiv:2206.07598 reports 1,547 SIDIS points, split into 344 HERMES and 1,203
  COMPASS points, after the explicit baseline cuts `Q > 1.4 GeV`,
  `0.2 < z < 0.7`, and
  `|P_hT| < min[min(0.2 Q, 0.5 z Q)+0.3 GeV, z Q]`.  It selects the HERMES
  vector-meson-subtracted `zxpt-3D` supplemental data and the COMPASS
  vector-boson-subtracted release.  Its abstract reports a combined DY+SIDIS
  N3LL fit with chi2/Ndat = 1.06 and warns that SIDIS multiplicities are
  normalized by factors associated with higher-order corrections.
- arXiv:2405.13833 reports a flavor-dependent unpolarized DY+SIDIS extraction
  at N3LL with chi2/Ndat = 1.08 and is used here as a formalism, dataset, and
  TMDFF benchmark, not as a raw-data source.  It states that its experimental
  dataset is identical to the 2022 analysis and applies a kinematics-dependent
  SIDIS normalization factor: a collinear SIDIS numerator through
  O(alpha_s^2) divided by the integrated W-term denominator.  This factor is
  a theory component to implement and validate explicitly; it is not a hidden
  rescaling of the data.

`scripts/reproduce_sidis_benchmark_count.py` applies the recorded cuts only
where source metadata supports them.  Its report is
`reports/sidis_1547_benchmark_audit.{json,md}` and its status is
`1547_benchmark_audited_not_reproduced`:

- the available HERMES HEPData `ins1208547` projection has 16 transverse
  tables and 288 primary rows, but no Q or x axes and no mirrored `zxpt-3D`
  value/covariance archive, so it cannot certify the literature's 344 rows;
- the available COMPASS HEPData `ins1624692` archive has 4,664 primary rows;
  deterministic choices of published Q2 value or Q2-bin midpoint and pT2
  center/edge produce 1,078--1,285 rows, bracketing but not uniquely
  reproducing 1,203.  The spread is evidence that a point-level selection
  manifest and bin-representative convention are required.

`scripts/inventory_arxiv_tables.py` is the source-only parser for the public
JLab arXiv packages.  The resulting
`reports/arxiv_table_inventory.{json,md}` records 668,799 CLAS ancillary
cross-section rows (four files), 25 Hall-C pT2 rows corresponding to 200
target/charge/rho-state entries, and 160 Hall-A TeX rows corresponding to 320
pi+/pi- entries.  CLAS rows carry statistical, systematic, and radiative
correction columns; Hall C contains before/after rho-subtraction columns; Hall
A is a 3He nuclear measurement.  None is fit-ready or approved.  The required
next conversion gates are, respectively, absolute-cross-section and
radiative/acceptance/covariance closure; Hall-C W/Mx and rho-subtraction
closure; and a validated 3He nuclear impulse-approximation/dilution model.
`scripts/convert_clas_ancillary.py` materializes the four CLAS files as
metadata-preserving CSVs in the ignored `data/derived/global` tree and writes
`conversion_manifest.json`; this is a reproducible conversion, not row
selection.

The staged global plan is therefore unchanged but now executable: (1) mirror
the HERMES `zxpt-3D` values and covariance, (2) freeze the COMPASS row-level
selection and bin rule, (3) run the clean HERMES/COMPASS closure and only then
add CLAS proton and Hall-C H/D families one at a time, and (4) test EMC/E665
and H1/ZEUS only after their observable and factorization boundaries close.
No rows have been approved, no SIDIS fit has been run, and no frozen DY file
has been modified.

## Progressive all-global-data policy (2026-08-26)

The campaign now explicitly separates **global candidate coverage** from a
single simultaneous fit. `config/staged_fit_plan.json` assigns all 22 registry
identities to a reproducible sequence: source inventory; the literature
HERMES/COMPASS 1,547-point checkpoint; clean HERMES/COMPASS extensions; JLab
CLAS and Hall-C absolute-cross-section families; recent/historical EMC, E665,
and COMPASS complements; nuclear/current-region diagnostics; and deferred or
access-restricted pointers. `reports/staged_fit_plan.md` is the human-readable
rendering and the plan validator is `scripts/build_staged_fit_plan.py`.

The fitting rule is one provenance-closed experiment/hadron/target family per
trial. A stage inherits the previous accepted model only after an independent
start check, and reports its incremental objective decomposition, central
prediction shift, held-out bins, leave-one-family-out behavior, covariance and
normalization nuisances, and start/model-form spread. Experimental replicas,
start non-uniqueness, TMDFF parameterization, dataset selection, and theory
variations remain separate uncertainty components. A narrow combined band is
never a promotion criterion by itself.

The supplied MAPTMD references anchor the first checkpoint. arXiv:2206.07598
reports the 344 HERMES + 1,203 COMPASS post-cut multiplicity scope, while
arXiv:2405.13833 states that it uses the same dataset and makes the SIDIS
normalization-factor construction explicit. These references do not authorize
silently adding JLab, EMC/E665, H1/ZEUS, or nuclear/current-region tables to
the 1,547-point fit; each additional family must pass the stage-specific gates.

## Corrected initial joint-fit validation (2026-08-26)

The campaign has now executed a real isolated joint fit: the frozen 329-row
lambda=1 DY W-only anchor plus a provisional 746-row identified COMPASS 2026
pi/K collinear scope. The corrected NNFF10 NNLO bin-average run is recorded in
`outputs/initial_joint_dy_compass_collinear_binavg_pilot_converged/` and gives
DY chi2/row = 0.3943 and SIDIS chi2/row = 17.13 for 745 fitted rows. One
signed fixed-order K- central prediction is negative and excluded explicitly;
no positivity clamp is used. A HAPS NNLO comparison reaches SIDIS chi2/row
= 2.94, but is circular because those FFs used modern COMPASS SIDIS data and
is not an independent closure candidate. The complete trial register is in
`reports/initial_fit_trials.{json,md}`.

All 101 NNFF10 members were profiled in midpoint and bin-average modes. The
raw lowest-objective member is invalid because it makes hundreds of rows
non-positive; the best member with every row positive still leaves a large
residual. The first run's all-scales-near-one normalization transient was
removed by initializing only the four SIDIS scalar nuisances near their
data/theory medians (the 10% priors remain). The DY anchor stays at about
0.394 chi2/row in every pilot. This demonstrates the software path and DY
non-regression, but does not close the physics gate.

The present work is explicitly **not** a production/global/TMD result. The
COMPASS addendum has no transverse axis or full covariance; HERMES zxpt-3D
identity/covariance remain unresolved; and the proper independently validated
NNLO SIDIS coefficient-function plus inclusive-DIS denominator/normalization
interface is still required. Do not use the HAPS comparison as a central fit,
do not promote the provisional rows, and do not modify frozen DY files. Local
source additions include `config/ff_sets.json`, `sidis_ff.py`,
`scripts/run_initial_joint_dy_sidis_fit.py`,
`scripts/profile_nnff10_replicas.py`,
`scripts/summarize_initial_fit_trials.py`, the dedicated tests, and the trial
reports; raw FF grids and fit outputs remain outside the public source tree.
