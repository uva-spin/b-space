# SIDIS global-analysis campaign (2026)

This is a new isolated campaign for adding semi-inclusive deep-inelastic
scattering (SIDIS) to the unpolarized Drell--Yan analysis. Its target is a
complete global analysis with a shared TMD-PDF sector and a separately
documented SIDIS TMD-fragmentation sector. It relies on the frozen DY result
and perturbative conventions as read-only inputs; it does not modify or replace
them.

**Status:** initialized discovery/provenance phase. No SIDIS dataset, fit,
replica ensemble, or global-production claim is approved.

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

## Immediate next action

Complete `HANDOFF.md`'s data/provenance inventory before choosing a SIDIS
dataset or architecture. Resolve whether the first scope is multiplicities or
absolute cross sections, which experiments/hadrons have full covariance, and
which collinear FF/TMDFF conventions can be reproduced. Until then this is a
scaffold only.
