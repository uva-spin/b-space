# SIDIS global-analysis campaign handoff

**Initialized:** 2026-08-26  
**Status:** first public-source harvest complete; discovery/provenance phase; no production fit authorized
**Scope:** shared unpolarized TMD-PDF plus SIDIS TMD-fragmentation global analysis

This campaign is separate from the completed DY identifiability and production
work. The DY result remains frozen and is a read-only comparison input. A future
global fit may use it as a warm start, prior, or baseline diagnostic only after
the interface and data conventions are frozen.

## Established dependencies

- The fixed-target DY incumbent is the lambda=1 empirical-reference package
  with 96 stationary starts crossed with 50 conditional experimental replicas.
- Its non-uniqueness and q16--q84 ensemble bands are documented in
  `../../production/lambda1_empirical_reference_full96x50/`.
- The CSS/TMD, OPE, hard-factor, and transform source stack is documented in
  `../perturbative_provenance_completion/` and the `v22/`/`v23/` source trees.
- Finite-Y/W+Y work exists only in isolated Tevatron/collider scopes and is not
  a universal global-DY or SIDIS prescription.

None of these supplies a SIDIS fragmentation model or authorizes a joint refit.

## 2026-08-26 public-source harvest

The first read-only public harvest uses version-1 HEPData CSV submission
archives. The source registry is `config/public_sources.json`; archive URLs,
SHA256 values, and table lists are in `data/hepdata_download_manifest.json`.
The compact table inventory is in `reports/public_source_inventory.md`.

| Source | Scope | Tables | Rows | Tables with transverse momentum | Current use |
| --- | --- | ---: | ---: | ---: | --- |
| HERMES `ins1208547` | identified pi+/K+, H/D, x_B/Q2/z/P_hperp | 64 | 1,196 | 16 | primary TMD candidate; row/covariance audit pending |
| COMPASS `ins1624692` | charged h+/h-, 6LiD, x/Q2/z/P_hT2 | 162 | 14,316 | 162 | primary TMD candidate; row/covariance audit pending |
| COMPASS `ins1444985` | pi+/- and h+/-, 6LiD, x/y/z | 4 | 6,236 | 0 | collinear complement; convention audit pending |
| COMPASS `ins1483098` | K+/-, 6LiD, x/y/z | 2 | 3,098 | 0 | collinear complement; convention audit pending |
| COMPASS `ins2840545` | pi+/K+/h+/-, H, x/y/z | 3 | 6,332 | 0 | collinear complement; full audit pending |

Total: 235 tables and 31,178 rows; 178 tables expose a transverse-momentum
axis. No rows, errors, or covariance matrices are approved for a fit. The
public checkout carries manifests, scripts, hashes, and inventory only; raw
archives remain local inputs.

The HERMES collaboration separately publishes a [multiplicity
database](https://hermesmults.appspot.com/) with multidimensional files and
statistical covariance matrices. The pointer and archive URL are recorded in
`config/public_sources.json`; the supplemental archive has not yet been
mirrored, so the HEPData CSV remains a value-table candidate rather than a
covariance-complete HERMES input.
`scripts/fetch_hermes_database_metadata.py` records the page hash, documented
binning/covariance warnings, and proton/deuteron `zpt-3D` transverse-momentum
download links in `data/hermes_database_manifest.json`; the concise
interpretation is in `reports/hermes_database_provenance.md`.

Reproduce the source-only harvest with:

```text
python scripts/fetch_hepdata_records.py
python scripts/profile_hepdata_tables.py
python scripts/build_source_inventory.py
```

## Software boundary (2026-08-26)

`FORMALISM.md` records the candidate leading-power W boundary and the
unresolved multiplicity/covariance interfaces. `sidis_data.py` is a
metadata-preserving CSV reader/profiler and strict plain-text/gzip covariance
reader that handles
duplicate labels and HEPData description continuations. `sidis_observables.py`
implements a scalar-tested radial PDF-times-TMDFF Bessel convolution with
explicit `qT=P_hT/z` and `b db/(2 pi)` conventions and a guarded multiplicity
ratio. It returns the structure-function piece only; experiment-specific
prefactors, DIS denominators, target/beam composition, radiative factors, and
Y terms remain explicit until their conventions are locked. Five unit tests
pass.

## Questions before fitting

1. Is the primary observable a multiplicity, an absolute cross section, or
   both? Define numerator, denominator, radiative corrections, and acceptance.
2. Which experiments, targets, hadrons, and cuts have stable source tables and
   complete point-to-point plus correlated uncertainties?
3. What bin-integration convention is required for each experiment?
4. Which collinear FFs and TMDFF parameterization are allowed, including
   favored/unfavored, strange/heavy flavors, and charge conjugation?
5. Which perturbative order/evolution convention is shared with DY, and which
   SIDIS hard/OPE coefficients are available and independently tested?
6. Where is a SIDIS Y term required, and can its observable/covariance closure
   be established independently?
7. Which parameters are common to DY and SIDIS, and how are normalization and
   theory nuisances prevented from absorbing TMD information?

## Locked discovery rules

- Do not edit the active DY production package.
- Do not select rows because they improve chi-square before provenance is
  complete.
- Do not call a SIDIS-only fit global; reserve that term for an approved joint
  DY+SIDIS scope with explicit coverage.
- Do not identify a TMDFF from a narrow replica band alone. Use starts,
  pseudodata closure, held-out bins, and model-form comparisons.
- Give every trial a unique tag and a machine-readable report under `reports/`.

## First deliverables

1. A candidate-source and provenance inventory (the first five HEPData records
   are harvested; row-level approval remains open).
2. A row-level data table with observable, target, hadron, units, cuts,
   covariance, normalization, and bin-integration fields.
3. `FORMALISM.md` and a scalar reference implementation; the candidate W
   convolution is documented, while the final observable convention remains
   open.
4. A DY interface regression reproducing the frozen lambda=1 central before any
   joint optimization.
5. A decision report selecting, rejecting, or deferring the first SIDIS scope.

## Decision log

| Date | Decision | Evidence | Status |
| --- | --- | --- | --- |
| 2026-08-26 | Create separate SIDIS/global-analysis campaign | Existing DY production and systematics are frozen; SIDIS requires a TMDFF and new observable/covariance closure | Initialized |
| 2026-08-26 | Harvest five version-1 public HEPData records | 235 tables and 31,178 rows profiled; 178 tables expose transverse momentum; no rows or covariance selected | Discovery only |
| 2026-08-26 | Add convention-explicit SIDIS software boundary | Metadata-preserving reader/profiler, radial PDF×TMDFF convolution, guarded ratio, and five scalar tests pass | Discovery only |
