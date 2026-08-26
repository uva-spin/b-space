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
a complete shared-DY-plus-SIDIS unpolarized TMD analysis. Its first
public-source harvest covers seven version-1 HEPData records (287 tables,
50,722 rows, 224 transverse-momentum tables), with source hashes, inventories,
and a metadata-preserving reader/profiler. The reader now skips repeated
target/charge headers and retains row-level block metadata; the corrected
inventory is 287 tables, 50,722 rows, and 33,740 transverse-momentum
candidate rows. These remain provenance candidates:
no SIDIS rows, observable convention, TMDFF model, fit, replica ensemble, or
production claim is approved. SIDIS requires its own data provenance,
fragmentation-function interface, bin/covariance treatment, scalar closure,
and model/start identifiability gates before any joint refit.

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
| SIDIS/global analysis | **Public-source harvest/discovery phase** | Five HEPData records profiled; no SIDIS or joint-global production claim yet |
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
