# SIDIS global-analysis campaign (2026)

This is a new isolated campaign for adding semi-inclusive deep-inelastic
scattering (SIDIS) to the unpolarized Drell--Yan analysis. Its target is a
complete global analysis with a shared TMD-PDF sector and a separately
documented SIDIS TMD-fragmentation sector. It relies on the frozen DY result
and perturbative conventions as read-only inputs; it does not modify or replace
them.

**Status:** first public-source harvest complete; discovery/provenance phase.
No SIDIS row selection, fit, replica ensemble, or global-production claim is
approved.

## Dependency boundary

The campaign may read, but must not overwrite:

- `../../production/lambda1_empirical_reference_full96x50/` (active lambda=1
  fixed-target DY reference, 96 starts x 50 conditional replicas);
- `../perturbative_provenance_completion/` (CSS/TMD, OPE, hard-factor, and
  cache-provenance records);
- `../finite_y_completion_2026/` and `../full_n3ll_wy_production_2026/`
  (finite-Y and collider/W+Y scope evidence, not automatic SIDIS inputs);
- the `v22/` and `v23/` source implementations.

The DY reference supplies a comparison baseline and shared PDF/evolution
conventions. It is not a substitute for a SIDIS TMDFF model, SIDIS hard-factor
treatment, hadron/target-mass conventions, or SIDIS experimental covariance.

## Required phases

1. Inventory SIDIS experiments, hadrons, targets, observables, bins, cuts,
   radiative corrections, normalization correlations, covariance matrices, and
   reproducible source tables.
2. Lock the exact cross-section or multiplicity definition, Jacobians,
   electroweak factors, target composition, perturbative order, scales, OPE,
   and finite-Y treatment.
3. Specify the TMDFF and collinear-FF interface, flavor/hadron parameterization,
   endpoint/positivity behavior, and charge-conjugation conventions.
4. Run pseudodata closure, held-out-bin, leave-one-experiment-out, independent
   start, model-form, and prior-sensitivity tests before a joint refit.
5. Fit the approved SIDIS scope jointly with the declared DY scope while
   preserving the frozen DY package and decomposing all objectives/nuisances.
6. Cross experimental replicas with stationary-start/model-form ensembles and
   report PDF, TMDFF, shared-parameter, and dataset-selection components
   separately.
7. Promote only after input, observable, closure, stationarity, robustness, and
   uncertainty-semantic gates pass.

## First public-source harvest

The source registry in `config/public_sources.json` and the compact download
manifest in `data/hepdata_download_manifest.json` identify five version-1
HEPData CSV submissions from HERMES and COMPASS. They contain 235 tables and
31,178 rows in total; 178 tables expose a transverse-momentum axis. The
candidate records are HERMES `ins1208547` and COMPASS `ins1624692` for
identified/charged-hadron transverse-momentum multiplicities, with COMPASS
`ins1444985`, `ins1483098`, and `ins2840545` retained as collinear complements.
The inventory is recorded in `reports/public_source_inventory.md`.

These are provenance candidates, not fit-ready rows. The raw archives are kept
outside this public source checkout; reproduce them with the fetch/profiling
drivers and verify the SHA256 values in the manifest. Statistical and
systematic columns are preserved but are not silently combined into a
covariance matrix.

## Software boundary

`sidis_data.py` provides a metadata-preserving HEPData CSV reader and schema
profiler. `sidis_observables.py` provides a convention-explicit radial
PDF-times-TMDFF Bessel convolution using `qT=P_hT/z` and `b db/(2 pi)`, plus a
guarded multiplicity ratio. It returns the SIDIS structure-function piece
only; experiment-specific prefactors, DIS denominators, target composition,
radiative factors, and finite-Y terms remain explicit until the formalism is
locked. Five scalar unit tests pass. The source-only drivers are
`scripts/fetch_hepdata_records.py`, `scripts/profile_hepdata_tables.py`, and
`scripts/build_source_inventory.py`.
`fetch_hermes_database_metadata.py` records the supplemental HERMES database
page hash and covariance/download links; its large DESY archive is not bundled.
`audit_table_provenance.py` produces the conservative table/row audit without
selecting rows.

The candidate factorization boundary, bin-integration warning, and unresolved
physics interfaces are recorded in [`FORMALISM.md`](FORMALISM.md).

## Immediate next action

Use the harvested inventory to complete `HANDOFF.md`'s row-level
data/provenance audit before choosing a SIDIS dataset or architecture. Resolve
whether the first scope is multiplicities or absolute cross sections, which
experiments/hadrons have full covariance, and which collinear FF/TMDFF
conventions can be reproduced. Until then this is a discovery scaffold only.
