# Codex handoff: current b-space project state

**Reconciled:** 2026-08-26  
**Repository:** `uva-spin/b-space`  
**Branch:** `main`

This file is the portable entry point for an agent starting from the public
checkout. The working-tree companion, with archive paths and the longer
decision history, is `systematics/PROJECT_HANDOFF.md` in the original project
tree. Historical handoffs under `systematics/` retain dated evidence; an old
`in progress` paragraph is not evidence of a live job.

## Read first

1. `README.md` for scope and the public source release boundary.
2. `docs/SOURCE_MAP.md`, `docs/MATCHING.md`, and
   `docs/EXTERNAL_FIXED_ORDER.md` for the perturbative and external-engine
   implementation.
3. `production/lambda1_empirical_reference_full96x50/PRODUCTION_MANIFEST.json`
   and `PRODUCTION_AUDIT.json` for the active result.
4. `production/lambda1_empirical_reference_full96x50/README.md` and
   `systematics/dataset_identifiability_campaign_2026/README.md` for the
   identifiability interpretation.
5. `systematics/finite_y_completion_2026/HANDOFF.md`,
   `systematics/full_n3ll_wy_production_2026/HANDOFF.md`, and
   `systematics/perturbative_provenance_completion/HANDOFF.md` before making
   any finite-`Y`, W+Y, or accuracy claim.
6. `systematics/sidis_global_analysis_2026/README.md` and `HANDOFF.md` before
   selecting SIDIS data or proposing a joint global fit.

## Immutability and claim discipline

- The lambda=1 production package is frozen. Do not overwrite its artifacts,
  hashes, figures, manifests, or source inputs.
- New architecture, prior, dataset, start-family, PDF-through-refit, or Y-term
  work must be isolated under a new tag and must carry an input manifest,
  terminal status, and promotion audit.
- Fit quality alone is not a promotion gate: require stationarity, input
  identity, transform/observable closure, and uncertainty-semantic checks.
- q16--q84 ensemble bands are operational empirical bands. They are not
  automatically calibrated Gaussian 68% or one-sigma confidence intervals.
- Do not turn the fixed-target W-term result or the isolated Tevatron grid into
  a universal collider/global N3LL+Y claim.

The public repository is source-oriented. Large checkpoints, raw crossed
ensembles, external DYTurbo/MCFM installations, machine caches, and most
generated plots remain in the working/archive tree and are referenced by
portable manifests where possible.

## Current promoted result

The active package is
`production/lambda1_empirical_reference_full96x50/`. It is the lambda=1
empirical-reference objective on `x=0.1`, `0.1 <= b_T <= 2 GeV^-1`, with 96
stationary starts crossed with 50 conditional experimental-replica residuals
(4,800 members per flavor), promoted 2026-08-11.

The 96-start expansion is practically saturated for its declared perturbation
and optimization family: the 96/48 maximum-width ratios are 1.0144 in b-space,
1.0191 for `u` in k-space, and 1.0186 for `d`. The active k-space maximum full
relative q16--q84 widths are 21.257% (`u`) and 22.480% (`d`). These are
descriptive ensemble widths, not calibrated one-sigma confidence intervals.
The previous 24-start package is retained only as a rollback reference.

“Production” is limited to the declared fixed-target, W-dominated scope and
protocol. The result does not establish unique `F_NP`, PDF-through-refit
uncertainty, or a validated universal finite-`Y` observable.

## Evolution and present status

### Perturbative foundation

The `v22/` and `v23/` source stacks implement the convention locks, CSS2
evolution, small-`b_T` profiles, OPE coefficients, one-loop Drell--Yan hard
factor, scalar references, fixed-target W assembly, data/replica handling, and
regularized b-to-k transforms. The “N3LL W-term” label refers to this
implemented resummed backend and its recorded coefficient inventory; it is not
automatically a full N3LL-prime implementation. The provenance audit also
found that historical cache metadata does not explicitly record strict versus
multiplicative W organization, so that cache claim remains narrow.

### Identifiability and dataset selection

`systematics/dataset_identifiability_campaign_2026/` tested the dataset ladder,
FiLM/reduced models, tail grafts, derivative and roughness priors, localized
profiles, and multistart stationarity. The important evidence is:

- 24 lambda=1 starts underestimated non-uniqueness;
- 48 starts revealed the larger envelope;
- 96 starts changed the 48-start maximum width by only about 1.5--2% and all
  new starts passed the declared stationarity gate;
- lambda=600 failed the complete like-for-like challenger (three of 24 starts
  failed and the spread was larger/non-stationary);
- lambda=300 and C1-matched/low-dimensional tail controls failed the joint
  fit-quality/stationarity gates; none is promoted.

The active package is therefore lambda=1 full96x50. The long campaign handoff
contains all failed trials and must be read as chronology, not as a live queue.

### Finite-`Y` and W+Y

`systematics/finite_y_completion_2026/` validates a C2 unitary transition for
24 Tevatron boundary rows. It does not validate conventional additive `FO-ASY`
everywhere. Six LHCb rows remain outside finite-`Y` production because the
observable, acceptance, unit, subtraction, and covariance closure is not yet
sufficient.

`systematics/full_n3ll_wy_production_2026/` contains the isolated unprimed
N3LL+NNLO Tevatron grid (`Y = FO_NNLO - ASY_NNLO`), external DYTurbo/MCFM
comparisons, conventional/unitary diagnostics, corrected 329/353-row
candidates, start/replica trials, and the paper handoff. The 122-row Tevatron
external grid is complete and numerically checked as an observable candidate.
Coupled TMD candidates are diagnostics and are not promoted because they do not
close the broader fixed-target/LHCb conventions and can shift or broaden the
incumbent result.

`systematics/high_qt_direct_production_benchmark/` contains high-qT selection
and external-code diagnostics. Earlier Collins and pseudodata candidates are
archived evidence, not alternate production.

### New SIDIS/global-analysis campaign

`systematics/sidis_global_analysis_2026/` is a separate discovery workspace for
a complete shared-DY-plus-SIDIS unpolarized TMD analysis. Its registry now
contains 22 public-source identities spanning the HERMES/COMPASS core, JLab
CLAS/Hall A/Hall C, EMC, E665, H1, and ZEUS. The harvest contains 16 HEPData
records (582 tables, 57,358 parsed rows; 35,963 primary and 21,395 auxiliary
rows) and three extracted arXiv source packages. The source-only inventory
also records 668,799 CLAS ancillary cross-section rows, 25 Hall-C pT2 rows,
and 160 Hall-A TeX rows. These remain provenance candidates: no SIDIS rows,
observable convention, TMDFF model, fit, replica ensemble, or production claim
is approved. SIDIS requires its own data provenance, fragmentation-function
interface, bin/covariance treatment, scalar closure, and model/start
identifiability gates before any joint refit.

The literature benchmark audit is in
`systematics/sidis_global_analysis_2026/reports/sidis_1547_benchmark_audit.md`.
The published 1,547 HERMES/COMPASS points are a post-cut selection rather than
a raw-table total. The available HERMES projection has 288 rows and lacks the
required Q/x axes and covariance sidecars; deterministic COMPASS bin choices
give 1,078--1,285 rows around the reported 1,203. A row-level selection
manifest is therefore still required before fitting.

The machine-readable progressive fitting policy is
`systematics/sidis_global_analysis_2026/config/staged_fit_plan.json`, rendered
in `reports/staged_fit_plan.md`. All 22 registry identities are assigned to an
inventory, literature benchmark, clean extension, JLab, historical,
diagnostic, or deferred stage. The campaign deliberately fits one
provenance-closed experiment/hadron/target family at a time, with held-out and
leave-one-family-out checks, rather than optimizing all candidate rows at
once. No rows are approved and no production fit is authorized.

## Status table

| Workstream | Status | Interpretation |
| --- | --- | --- |
| Fixed-target lambda=1 TMD | **Active production** | W-dominated result with declared empirical ensemble band |
| 96-start non-uniqueness | **Complete for declared family** | Practical saturation, not proof over all models |
| Experimental replicas | **50 conditional replicas crossed** | Conditional replica contribution |
| Lambda 300/600 and smoothness trials | **Rejected/archived** | Failure-mode evidence only |
| Perturbative provenance | **Source/inventory audit complete; promotion gate open** | Implemented stack and explicit limitations |
| 24-row Tevatron unitary finite-Y | **Validated isolated scope** | Boundary diagnostic only |
| 122-row Tevatron N3LL+NNLO W+Y | **Complete observable candidate, not TMD production** | No global/collider-wide claim |
| LHCb finite-Y | **Fail-closed/out of scope** | Do not fit current subtraction table |
| SIDIS/global analysis | **Initial joint closure pilot; not production** | 329 frozen-DY + 746 COMPASS collinear rows fit in isolation; independent NNFF10 closure remains poor and HERMES/transverse gates are open |
| Paper methods/figures | **Lambda=1 dossier available** | Manuscript scope must remain explicit |

## Reproduction entry points

Run from the repository root, normally with Python 3.10+ and the project
environment containing PyTorch. Use `PYTHONPATH` pointing to the checkout for
Python drivers. Full fit reproduction also needs the archived inputs/caches,
PDF/LHAPDF support, and (for isolated fixed-order checks) external DYTurbo or
MCFM installations. The source release does not silently bundle those tools.

The convention regression suite is:

```text
python -m pytest -q v22/tests/test_conventions.py
```

If a minimal interpreter cannot import PyTorch, record that environment
dependency failure; do not modify source to bypass tests. Do not launch a long
campaign merely because an archival ledger says `in_progress`: verify process
state, timestamps, and terminal summaries first.

## Open decisions

No current decision authorizes replacing the lambda=1 package. A replacement
would require a new isolated protocol, complete input manifest, start/replica
evidence, fit and stationarity gates, uncertainty audit, and an explicit
promotion transaction. Open scientific work includes any stronger but
fit-preserving `F_NP` constraint, PDF-through-refit propagation, fixed-target
finite-`Y` closure, and a defensible scope for LHCb/global W+Y.

### SIDIS initial joint-fit update (2026-08-26)

The isolated SIDIS campaign now has a real joint software pilot, not just a
source inventory. The corrected NNFF10 NNLO bin-average run combines the
frozen 329-row lambda=1 DY W-only anchor with 746 identified COMPASS 2026 pi/K
collinear rows (745 fitted after one explicitly excluded negative signed FF
prediction). It gives DY chi2/row 0.3943 and SIDIS chi2/row 17.13. A HAPS
comparison gives 2.94 SIDIS chi2/row but is circular because its FFs used
modern COMPASS SIDIS information. All 101 NNFF10 members were profiled; the
raw lowest-objective member is invalid if it makes hundreds of rows
non-positive, and the best all-rows-valid member still leaves a large
residual. Details are in `systematics/sidis_global_analysis_2026/reports/`.

This is explicitly not a global/TMD production result. The COMPASS addendum
has no transverse axis or full covariance, HERMES zxpt-3D identity/covariance
remain unresolved, and the proper NNLO SIDIS coefficient plus inclusive-DIS
denominator/normalization interface still needs independent closure. Frozen
DY production files are unchanged.

The APFEL follow-up has now completed an isolated massless NLO SIDIS
coefficient/denominator diagnostic. `scripts/apfel_sidis_nlo_denominator_probe.cpp`
uses the SIDIS C20/C21/CL1 operators and APFEL's complete Observable path for
inclusive-DIS F2/FL, with NNPDF40 and NNFF10 NLO member zero. The midpoint fit
gives SIDIS chi2/row 12.9775 on 738 positive rows; a four-point x/y/z
bin-averaged run gives 12.7967, with DY unchanged at 0.3943/row. Eight
non-positive K- theory ratios are explicit exclusions, not positivity-clipped
values. These are interface-validation milestones, not production results:
independent bin-integral/order checks, heavy-quark/threshold and scale
choices, covariance, TMDFF transverse closure, and HERMES identity/covariance
remain open. See the campaign `HANDOFF.md`, `FORMALISM.md`, and
`reports/apfel_sidis_interface_probe.{json,md}`.
