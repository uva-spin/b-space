# Source map and publication boundary

The public repository is a cleaned source release from the original working
repository. This table records the important mappings.

For the reconciled cross-workstream status and handoff order, start with the
repository-level [`CODEX_HANDOFF.md`](../CODEX_HANDOFF.md). It records which
artifacts are active, isolated, archived, or blocked and explains the boundary
between this source release and the numerical archive.

| Working-tree source | Public path | Included content |
| --- | --- | --- |
| `v22/src/` | `v22/src/` | OPE, hard factor, convolution, conventions, W assembly, and profiles. |
| `v22/backends/` | `v22/backends/` | Fixed-target full backend and numeric W-expansion scheme. |
| `bt_internal_css_backend_v19_smoothprofile.py` | repository root | Compatibility backend loaded by the scheme-Y v22 wrapper. |
| `v22/tests/` | `v22/tests/` | Unit and smoke tests for the perturbative stack. |
| `v22/tools/` | `v22/tools/` | Backend audits, grid construction, replica aggregation, and cache utilities. |
| `v23/backends/` | `v23/backends/` | Tevatron-aware wrapper and developmental W/Y modes. |
| `v23/tools/` | `v23/tools/` | v23a data, fit, PDF overlay, plotting, and audit utilities. |
| `v23/experimental/` | `v23/experimental/` | PDF-through-refit planning and execution scripts. |
| `v21_tail_release_amp0p019_candidate/train_*.py` | `v21_tail_release_amp0p019_candidate/` | Trainer sources required by the v22/v23 orchestration scripts. |
| `Data/` fixed-target tables | `Data/` | Small source tables, corrected row-99 variants, and uncertainty manifests. |
| `workflows/v22/{bootstrap,audits,construction,runs,replicas,utilities}/` | same public paths | Organized v22 orchestration and compatibility drivers from the working campaign. |
| `workflows/v23a/{audits,construction,plotting,runs,replicas,utilities}/` | same public paths | Organized v23a data, fit, replica, and publication drivers from the working campaign. |
| `workflows/compatibility/install_into_repo.sh` | same public path | Source-staging helper; not a physics calculation. |
| `systematics/dataset_identifiability_campaign_2026/scripts/` | same public path | Source-only identifiability, crossing, promotion, and integrity campaign. |
| `systematics/high_qt_direct_production_benchmark/` | same public path | Experimental W/Y and unitary-transition matching studies. |
| `systematics/finite_y_tail_benchmark/` | same public path | External finite-tail benchmark drivers and summary metadata. |
| `systematics/finite_y_completion_2026/` | same public path | Lambda=1 Tevatron unitary finite-Y completion and LHCb closure handoff. |
| `systematics/perturbative_provenance_completion/` | same public path | W organization, coefficient inventory, strict/multiplicative closure, and accuracy provenance. |
| `systematics/full_n3ll_wy_production_2026/` | same public path | Isolated Tevatron N³LL+NNLO W+Y source, diagnostics, candidate figures, and paper handoff. |
| `systematics/sidis_global_analysis_2026/` | same public path | Initialized SIDIS/global-analysis scope, dependency boundary, and provenance gates. |
| `systematics/prd_empirical_reference_lambda1_dossier/` | same public path | Paper-ready lambda=1 reconstruction figures and methods dossier. |
| `production/lambda1_empirical_reference_full96x50/` | same public path | Frozen active production outputs and integrity metadata. |

## Exclusions

The following working-tree material was intentionally excluded:

- `outputs/`, `plots/`, `logs/`, and replica directories;
- model checkpoints (`model_state.pt`), large ensemble CSVs, and backend caches;
- Python bytecode (`__pycache__`);
- the local `v22/external/artemide-v2.06` checkout;
- unrelated candidate campaigns and exploratory output trees;
- full external DYTurbo and MCFM installations.

These exclusions are not claims that the material is unimportant. They keep the
Git repository source-oriented and avoid silently redistributing third-party or
machine-specific artifacts. The active frozen numerical result remains in
`production/`, with hashes and provenance in its manifests.

## Entry-point rule

Run workflow shell entry points from the repository root:

```bash
cd /path/to/b-space
./workflows/v23a/runs/run_v23a_central_tmd_grid_and_audit.sh
```

Run Python modules with the repository root on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/b-space python /path/to/b-space/v22/tools/construct_v22_scheme_tmd_grid.py --help
```

When a fit or cache lives outside the checkout, pass its path explicitly with
`RUN=`, `CACHE_ENV=`, `INIT_STATE=`, `OUT=`, `DATA_DIR=`, `TRAIN=`, or
`BACKEND=`. Do not change a cache's data/PDF/backend identity by editing only
the path name.

The compatibility backend `bt_internal_css_backend_v19_smoothprofile.py`
remains at the root because the v22 scheme-Y wrapper and provenance manifests
load that exact path.  It is an implementation dependency, not an unorganized
workflow driver.
