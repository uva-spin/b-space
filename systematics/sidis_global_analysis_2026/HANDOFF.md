# SIDIS global-analysis campaign handoff

**Initialized:** 2026-08-26  
**Status:** discovery/provenance phase; no production fit authorized  
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

1. A candidate-source and provenance inventory.
2. A row-level data table with observable, target, hadron, units, cuts,
   covariance, normalization, and bin-integration fields.
3. An exact SIDIS formalism note and scalar reference implementation.
4. A DY interface regression reproducing the frozen lambda=1 central before any
   joint optimization.
5. A decision report selecting, rejecting, or deferring the first SIDIS scope.

## Decision log

| Date | Decision | Evidence | Status |
| --- | --- | --- | --- |
| 2026-08-26 | Create separate SIDIS/global-analysis campaign | Existing DY production and systematics are frozen; SIDIS requires a TMDFF and new observable/covariance closure | Initialized |
