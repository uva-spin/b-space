# Perturbative/provenance completion handoff

Last updated: 2026-08-26

## Current status pointer (2026-08-26)

The source-level W/OPE/hard stack and its coefficient/provenance inventory are
documented and the strict/multiplicative closure diagnostic is complete. The
promotion gate remains open: the historical cache lacks an explicit W
organization field, and no separate conventional full N3LL-prime implementation
has been established. Finite-Y/W+Y studies remain isolated under their own
directories. Read the repository `CODEX_HANDOFF.md` before changing scope or
labels.

## Scope

Finish the perturbative and cache-provenance work needed to support the
intended manuscript standard.  This is not permission to modify the active
lambda=1 production package, any frozen upstream production output, or the
paper.  All trials use new tags below this directory.

## Current baseline

- Active identifiability package:
  `../../production/lambda1_empirical_reference_full96x50/`
- Published source-release package: the same `production/` path in this
  checkout; its `PRODUCTION_MANIFEST.json` and `PRODUCTION_AUDIT.json` are the
  authoritative public records.
- Frozen v22 production cache and its metadata manifest: retained in the
  working archive and intentionally excluded from this source release.  Use
  the paths recorded by `production/.../paths.txt` and the input manifest when
  reconnecting the archive.

## Confirmed implementation

The repository contains the complete implemented fixed-target W stack:

- CSS2 one-loop OPE;
- general-scale OPE;
- one-loop Drell--Yan hard factor;
- scalar convolution and W references;
- v22 full W backend;
- smooth small-`b_T` perturbative profile;
- strict and multiplicative W reference forms;
- source-level and numerical bridge/audit programs.

The published source release contains the matching documentation at
`../../docs/MATCHING.md`.  Commit identifiers in older entries are historical
provenance only; use the current repository revision when reproducing a run.

## Open provenance questions

### W organization

The v22 full backend declares multiplicative NLO W as its default numerical
organization.  The strict form is implemented for one-loop expansion,
subtraction, and auditing.  The existing production cache metadata points to
the full backend but does not record an explicit `v22_w_organization` value.
The first completion deliverable is therefore a diagnostic cache with an
explicit organization field and a pointwise strict/multiplicative comparison.

### Perturbative accuracy

The active configuration uses `resum_order=n3llp`.  The legacy coefficient
dispatcher maps this to `n3ll_pilot`, documented as the implemented
`A1--A3`, `B1--B2` Sudakov content.  The fixed-target W stack is complete at
that implemented order, but no separate source for a conventional full
N3LL-prime coefficient/matching inventory has been located.

The completion task is to define the exact accuracy convention, inventory all
implemented coefficients, identify missing terms, and implement/test them if
they are required by the paper's intended claim.

### Finite-Y scope

The active low-`q_T` production fit uses `y_mode=zero`.  The finite-Y and
unitary W-to-fixed-order studies remain separate validation packages.  They
must not be mixed into the perturbative W completion unless the paper's scope
is explicitly expanded.

## Required gates

1. **Immutable input manifest:** hashes for backend, trainer, PDF/data inputs,
   profiles, perturbative flags, and cache files.
2. **Strict/multiplicative closure:** same-input scalar and grid comparison,
   with accepted-domain and audit-domain summaries.
3. **Coefficient inventory:** explicit table for cusp, noncusp, beta-function,
   hard, and OPE orders; no accuracy label inferred from a config string.
4. **Expansion closure:** strict W expansion, small-`b_T` profile, LO bridge,
   W bridge, and numerical finite-order checks.
5. **External observable closure:** only where the candidate includes finite-Y
   or high-`q_T` terms; retain the existing DYTurbo/MCFM and bin-integration
   gates.
6. **Candidate regeneration:** a new tagged cache and isolated central fit
   must pass before any production consideration.

## Existing evidence not to lose

- The hard-plus-OPE audit passes and finds the multiplicative/strict difference
  below roughly one percent in its benchmark points.
- The full-backend integration bridge closes at floating-point precision, but
  that is an internal closure and not an external observable benchmark.
- The existing finite-Y additive and unitary-transition studies are not
  production-approved.
- The lambda=1 96-start x 50-replica package remains the current identifiability
  baseline; its bands are operational q16--q84 ensembles, not calibrated
  Gaussian one-sigma intervals.

## Immediate next actions

1. Build the immutable input/provenance manifest.
2. Run a read-only cache metadata and backend-introspection audit.
3. Generate strict and multiplicative diagnostic W grids under unique tags.
4. Produce the coefficient/accuracy inventory and identify the minimum missing
   implementation needed for the intended paper claim.

The workspace is not promotion-authorized.  Frozen production files remain
unchanged.

## Initial read-only verification

On 2026-08-15 the existing source-level checks passed:

- 19 convolution checks;
- 13 canonical CSS2 NLO OPE checks;
- 15 general-scale CSS2 OPE checks;
- 8 Drell--Yan hard-factor checks;
- 6 standalone DY W-kernel checks;
- 8 small-`b_T` profile checks.

These verify the implemented baseline stack only.  They do not close the
strict/multiplicative cache provenance gap or establish a conventional full
N3LL-prime accuracy claim.

The first read-only provenance audit is recorded in
`reports/backend_provenance_audit.json`.  It confirms:

- the backend declares `multiplicative_nlo` and defaults to multiplicative W;
- a strict W branch is present for comparison;
- the frozen cache has no explicit W-organization metadata;
- the cache uses `resum_order=n3llp`, `match_order=nlo`, and `y_mode=zero`;
- the legacy coefficient dispatcher contains `A1`, `A2`, `A3`, `B1`, and
  `B2`, and maps `n3llp` to the documented pilot branch;
- the public backend copy is byte-identical to the working-tree backend.

## 2026-08-17 diagnostic regeneration result

The first attempted full recomputation was stopped after its progress report
showed approximately 22 seconds per accepted row; the 331-row strict and
multiplicative duplication would have required hours per branch. No output
was produced by that interrupted run and no production file was changed.

An isolated, deterministic diagnostic was then completed using two endpoint
rows per dataset (8 rows total) on the exact 160-point historical `b_T` grid.
The multiplicative values were extracted from the byte-identified full
reference cache, while the strict branch was recomputed from the same PDF,
cuts, profile, and perturbative configuration. Outputs are under:

`outputs/diagnostic_cache_n3llp_nloQ96_b160_qToQ05_20260817/`

The resulting comparison is:

- maximum pointwise strict--multiplicative difference: `3.73%` relative;
- median pointwise relative difference: `0.516%`;
- maximum integrated W difference: `2.79%` relative;
- multiplicative extraction versus the historical cache: exact by direct
  row-id extraction (`1280` values, zero numerical difference).

The run emitted only the known SciPy quadrature roundoff warnings from the
existing convolution module. Its status is diagnostic-only; it does not
authorize a cache or production replacement. The result shows that W
organization is a material convention choice in some points, not a
missing-data issue, and that the frozen cache's organization must be recorded
explicitly before any paper claim is finalized.

The downloaded PRD PDF currently describes the nominal W construction as
strict-NLO. This is not yet provenance-closed against the active backend:
`v22_full` defaults to multiplicative NLO and the historical cache metadata
does not record an override. The paper statement therefore remains a claim to
resolve by generator provenance or a new isolated strict cache; it must not be
silently assumed from the prose.

The reusable builder is
`scripts/build_strict_multiplicative_diagnostic.py`; it documents the
representative-row strategy and reuses the full historical multiplicative
cache rather than silently recomputing it.
The explicit diagnostic metadata is
`outputs/diagnostic_cache_n3llp_nloQ96_b160_qToQ05_20260817/diagnostic_metadata.json`.
The PDF/source claim comparison is recorded in
`reports/manuscript_claim_crosswalk.json`.
The generator inference is recorded in
`reports/historical_cache_organization_inference.json`: multiplicative is the
strong default inference, but the old cache remains metadata-incomplete.

## Next completion gates

1. Finish the coefficient/accuracy inventory with exact source references and
   a convention lock for the term `N3LL-prime`.
2. Add an explicit W-organization field to a new isolated metadata schema and
   run source-level strict/multiplicative expansion closure.
3. Decide whether the paper needs a finite-Y/high-`q_T` closure; the active
   low-`q_T` production fit remains `Y=0`.
4. Only after those gates, consider a full isolated candidate regeneration.

The source-level expansion closure was rerun on 2026-08-17 and passed all six
published checks (19 convolution, 13 canonical OPE, 15 general-scale OPE, 8
DY hard, 6 W-reference, and 8 small-`b_T` profile checks). The report is
`reports/expansion_closure.json`. This closes internal algebra/plumbing only;
it is explicitly not an external fixed-order observable validation.

The existing external-tail packages were also checked for scope: the finite-Y
benchmark reports 15 representative DYTurbo/MCFM pairs (13 passing the 5%
code-consistency gate), while the tier-1 boundary report records 25 complete
pairs and no direct-production approval. These artifacts support the current
separation of the active `Y=0` low-`q_T` extraction from high-`q_T` matching;
they do not close a new universal finite-Y production claim.
The compact scope record is
`reports/external_matching_scope.json`.

## Finite-Y completion handoff (2026-08-17)

The finite-Y question now has a separate isolated package at
`systematics/finite_y_completion_2026/`. The ordinary additive `FO-ASY`
construction remains rejected because the resummed W is not close to ASY near
the transition boundary. A unitary finite-Y correction,
`Y=p*(FO_NLO-W)`, was implemented and validated against the completed 24-row
Tevatron node/NLO artifacts. It passes algebraic, continuity, positivity,
bin-reconstruction, and numerical convergence gates for that scope.

This is a valid finite-Y transition construction, not a conventional
perturbative `Y=FO-ASY` claim and not yet a production replacement. New fit,
replica, and LHCb fiducial-acceptance gates remain open. See
`systematics/finite_y_completion_2026/reports/decision_status.json`.

## 2026-08-18 transition to full matched candidate

The current lambda=1 result is frozen and the next work has moved into the new
isolated workspace `../full_n3ll_wy_production_2026/`. That campaign targets
a genuine unprimed N3LL+NLO prediction with conventional
`Y=FO_NLO-ASY_NLO` for the Tevatron 353-row scope. This provenance workspace
remains the source of the accuracy inventory and strict/multiplicative audits;
its historical caches and the active production package are read-only inputs.
No production promotion is authorized until the new candidate passes the
coefficient, ASY, external fixed-order, fit-stationarity, and replica gates.

## 2026-08-18 accuracy-target correction

The new isolated campaign has now been raised to the genuine standard
**unprimed N3LL+NNLO** target for the Tevatron scope, rather than stopping at
an NLO-side label.  The external DYTurbo 1.4.2 engine contains the required
observable-level NNLO/N3LL ingredients and passed an isolated CDF row probe;
the fitted v22 W backend still lacks the corresponding imported higher-order
coefficients and conventional `Y=FO_NNLO-ASY_NNLO`.  Historical N3LL-prime/NLO
pilot caches remain diagnostics only and are not being relabeled.
