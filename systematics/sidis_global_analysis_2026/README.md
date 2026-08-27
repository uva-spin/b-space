# SIDIS global-analysis campaign (2026)

This is a new, isolated campaign for adding semi-inclusive deep-inelastic
scattering (SIDIS) to the existing unpolarized Drell--Yan analysis. Its target
is a complete global analysis with a shared TMD-PDF sector and a separately
documented TMD-fragmentation sector. It relies on the frozen DY result and
perturbative conventions as read-only inputs; it does not modify or replace
them.

**Status (2026-08-26):** source inventory and isolated interface validation
are complete enough for a first joint software pilot. A real 329-row frozen-DY
plus 746-row identified-COMPASS collinear SIDIS fit has been run, together
with a 101-member NNFF10 closure profile. It is explicitly **not** a
production or global result: scalar SIDIS closure remains poor with the
independent NNFF10 boundary, HERMES identity/covariance is unresolved, and
the proper perturbative SIDIS coefficient/denominator interface remains a
required gate. No frozen DY production file has changed.

## Dependency boundary

The campaign may read, but must not overwrite:

- `../dataset_identifiability_campaign_2026/production_lambda1_empirical_reference_full96x50/`
  (active lambda=1 fixed-target DY reference, 96 starts x 50 conditional
  replicas);
- `../perturbative_provenance_completion/` (CSS/TMD, OPE, hard-factor, and
  cache-provenance records);
- `../finite_y_completion_2026/` and `../full_n3ll_wy_production_2026/`
  (finite-Y and collider/W+Y scope evidence, not automatic SIDIS inputs);
- the source implementations in `../../v22/` and `../../v23/`.

The DY reference supplies a comparison baseline and an interface to the shared
PDF/evolution conventions. It is not a substitute for a SIDIS TMDFF model,
SIDIS hard-factor treatment, hadron-mass/target-mass conventions, or SIDIS
experimental covariance.

## Required phases

1. **Inventory and provenance.** Identify candidate SIDIS experiments, hadron
   species, targets, observables, bin definitions, radiative corrections,
   normalization correlations, covariance matrices, and publicly reproducible
   source tables. No rows enter a fit before this inventory is complete.
2. **Observable/formalism lock.** Write the exact SIDIS cross section or
   multiplicity definition, phase-space and Jacobian conventions, electroweak
   factors, target composition, perturbative order, evolution scales, OPE
   matching, and treatment of the (Y) term. Keep PDF and TMDFF conventions
   explicit and test units numerically.
3. **TMDFF model and interface.** Choose and document the fragmentation inputs,
   flavor/hadron parameterization, positivity/endpoint behavior, scale
   dependence, and any external collinear FF set. A SIDIS fit cannot be called
   global while the FF interface is implicit.
4. **Closure and identifiability.** Run pseudodata closure, held-out-bin,
   leave-one-experiment-out, start, model-form, and prior-sensitivity tests.
   Test shared-PDF directions against the frozen DY baseline before allowing a
   joint refit.
5. **Joint fit.** Fit the approved SIDIS scope jointly with the declared DY
   scope, retaining dataset-level normalization/covariance nuisances and
   recording the objective decomposition. Do not use SIDIS to silently alter
   the DY production package.
6. **Uncertainty propagation.** Cross experimental replicas with the complete
   stationary-start/model-form ensemble. Report PDF, TMDFF, shared-parameter,
   and dataset-selection components separately before any combined envelope.
7. **Promotion review.** A global candidate requires reproducible inputs,
   formalism closure, acceptable per-dataset and global fit quality, stationary
   shared/TMDFF parameters, start and model-form coverage, and an explicit
   comparison with the frozen DY result.

## Promotion gates

- **Data identity:** every row has a stable source ID, units, target/hadron
  metadata, cuts, bin integration rule, and uncertainty/covariance provenance.
- **Observable closure:** the implementation reproduces published definitions
  and agrees with an independent calculation or published benchmark in a
  controlled subset.
- **DY non-regression:** the joint interface reproduces the frozen lambda=1 DY
  central prediction before a joint refit; any intentional change is isolated
  and reported.
- **SIDIS closure:** pseudodata and held-out validation pass for each selected
  experiment and hadron/target class.
- **Stationarity:** independent starts reach the declared stopping rule for
  both shared PDF and TMDFF sectors; shallow directions are measured, not
  hidden by a narrow replica band.
- **Robustness:** experiment/hadron/target leave-one-out and reasonable
  parameterization/prior variations are quantified.
- **Uncertainty semantics:** statistical replica intervals, start
  non-uniqueness, model-form envelopes, normalization, and theory variations
  remain labelled separately. No operational q16--q84 band is silently called
  a calibrated one-sigma interval.
- **Scope:** any finite-(Y), high-(P_{hT}), heavy-quark, target-mass, or
  radiative-correction extension passes its own closure gate before entering a
  global production fit.

## Planned directory layout

```text
sidis_global_analysis_2026/
  HANDOFF.md                 # chronological decisions and restart notes
  README.md                  # scope and gates
  config/                    # frozen scope/formalism/data manifests
  data/                      # provenance manifests or small source tables only
  scripts/                   # inventory, conversion, closure, fit drivers
  reports/                   # machine-readable gate/decision records
  outputs/                   # isolated fits and diagnostics
  plots/                     # exploratory and candidate figures
```

Large raw datasets, checkpoints, external FF libraries, and private archives
remain outside the public source release and must be referenced by hashes and
portable manifests.

## Software extension already in place

`sidis_data.py` provides a metadata-preserving HEPData CSV reader/schema
profiler and a strict plain-text/gzip covariance-matrix reader for future
HERMES sidecars. It distinguishes primary measurement blocks from auxiliary
correction-factor blocks and preserves TeX-labelled kinematic intervals.
`sidis_observables.py` provides a convention-explicit radial
PDF-times-TMDFF Bessel convolution and a guarded multiplicity ratio. The
convolution returns only the structure-function piece: experiment-specific
electromagnetic prefactors, DIS denominators, radiative factors, target
composition, and finite-Y terms remain explicit inputs until their conventions
are locked. The isolated data/observable boundary has sixteen passing local
unit tests in `tests/`, including covariance, repeated-header, auxiliary-block,
and explicit canonicalization regressions. `sidis_covariance.py` evaluates a
caller-supplied positive-definite covariance by Cholesky whitening and fails
closed on shape, symmetry, singularity, or indefiniteness; it never invents
correlations or combines uncertainty components.

`sidis_dataset.py` is the next conversion boundary: a caller must provide an
explicit value/axis/error mapping (including comment-defined block intervals),
and each canonical observation retains source, table, row, block, bin, and
uncertainty provenance. `block_filters` and the explicit
`skip_missing_values` switch handle target blocks and published dash
placeholders; ambiguous or uncertainty-free mappings fail closed. The
source-specific `scripts/validate_candidate_mappings.py` driver validates all
224 transverse tables in memory and writes a provenance report without
approving rows.

The harvest/profiling drivers are `scripts/fetch_hepdata_records.py`,
`scripts/profile_hepdata_tables.py`, and `scripts/build_source_inventory.py`.
They create `data/hepdata_download_manifest.json`,
`reports/hepdata_table_inventory.json`, and
`reports/public_source_inventory.{json,md}`. The raw version-1 HEPData CSV
archives are local inputs under `data/raw/hepdata/` and are not silently mixed
into a fit.
The supplemental HERMES database metadata/link harvester is
`scripts/fetch_hermes_database_metadata.py`; its archive remains pending when
the DESY endpoint is unavailable.
`scripts/audit_table_provenance.py` produces the conservative table/row audit
in `reports/row_level_provenance_audit.json` without selecting rows.
`scripts/validate_candidate_mappings.py` validates explicit source-specific
mappings for the HERMES identified projection and the two COMPASS transverse
grids; its result is in `reports/candidate_mapping_validation.{json,md}`.

The candidate factorization boundary, bin-integration warning, and unresolved
physics interfaces are recorded in `FORMALISM.md`.

`config/global_sources.json` expands the initial seven-record harvest to a
22-entry global candidate registry covering the public HERMES/COMPASS core,
JLab CLAS and Hall A/Hall C sources, EMC, E665, H1, and ZEUS.  The registry
explicitly separates stage-1 multiplicity candidates from stage-2
absolute/low-energy candidates, nuclear or current-region diagnostics, and
deferred pointers.  `scripts/fetch_global_sources.py` downloads versioned
HEPData records and public arXiv source packages into the ignored
`data/raw/global/` cache; `scripts/build_global_source_inventory.py` writes
`reports/global_source_inventory.{json,md}` without selecting rows.  The
external literature benchmark and its reported 1,547 post-cut SIDIS points are
recorded in `config/external_fit_benchmarks.json`; the number is a staged
reproduction target, not a requirement to count incompatible observables.

The complete progressive fitting protocol is locked in
`config/staged_fit_plan.json`, with a rendered review table in
`reports/staged_fit_plan.md`. It assigns every registry identity to an inventory,
benchmark, extension, JLab, historical, diagnostic, or deferred stage. The rule
is to harvest the full candidate universe but fit one provenance-closed family at
a time; there is no all-at-once likelihood in this campaign.

## First actual joint DY+SIDIS pilot (2026-08-26)

The first joint software path has been exercised on real rows without
modifying the DY production package. `scripts/run_initial_joint_dy_sidis_fit.py`
combines the frozen 329-row lambda=1 W-only DY anchor with a provisional
746-row identified COMPASS 2026 pi/K collinear scope. The corrected NNFF10
NNLO bin-average pilot gives DY chi2/row = 0.3943 and SIDIS chi2/row = 17.13
for 745 fitted rows; one signed fixed-order K- central prediction is negative
and is excluded explicitly. The HAPS comparison gives 2.94 SIDIS chi2/row but
is circular because those FFs used modern COMPASS SIDIS information. Neither
run is production-authorized. See `reports/initial_fit_trials.{json,md}` and
`reports/initial_fit_decision.md`.

The all-member NNFF10 midpoint and bin-average closure profiles are retained
as external FF/theory diagnostics. The raw lowest-objective member becomes
non-positive for hundreds of rows, while the best all-rows-valid member still
does not close the data. The COMPASS addendum has no transverse axis, so this
pilot does not identify a TMDFF width; HERMES identity/covariance and the
validated NNLO coefficient-function plus inclusive-DIS denominator interface
remain open gates.

## Immediate next action

The first benchmark audit is now reproducible with
`scripts/reproduce_sidis_benchmark_count.py`; its result is in
`reports/sidis_1547_benchmark_audit.{json,md}`. It confirms that the literature
1,547 count is not recoverable by counting raw HEPData rows: the available
HERMES projection supplies 288 rows rather than the reported 344 and lacks the
required Q/x axes, while deterministic COMPASS bin-representative choices
give 1,078--1,285 selected rows around the reported 1,203. This is a
provenance/convention discrepancy, not permission to choose whichever count
fits.

The arXiv/JLab source parser is now in
`scripts/inventory_arxiv_tables.py`, with results in
`reports/arxiv_table_inventory.{json,md}`. It inventories 668,799 CLAS
ancillary cross-section rows, 25 Hall-C pT-squared rows (200 expanded
target/charge/rho-state entries), and 160 Hall-A TeX rows (320 expanded
pi+/pi- entries), all still outside the fit boundary.
`scripts/convert_clas_ancillary.py` can materialize the four CLAS files as
metadata-preserving CSVs under the ignored `data/derived/global` tree; its
manifest records 668,799 converted rows without selecting any for fitting.

The next gate is to mirror the HERMES `zxpt-3D` value/covariance files and
freeze the COMPASS point-level bin convention and row-selection manifest.
Only after that clean stage is closed will JLab CLAS/Hall-C families be added
one experiment/hadron/target family per trial with held-out and leave-one-
family-out checks. EMC/E665 and H1/ZEUS current-region or nuclear sources
remain explicit diagnostics until their observable, factorization, and
covariance conventions close. Until these gates are met, no rows may enter a
fit.
