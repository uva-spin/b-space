# Source map and publication boundary

The public repository is a cleaned source release from the working tree
`/home/dustin/work/project/bT-TMD`. This table records the important mappings.

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
| Root `run_*`, `audit_*`, `construct_*`, `bootstrap_*` | repository root | Compatibility entry points used by the working campaign. |
| `systematics/dataset_identifiability_campaign_2026/scripts/` | same public path | Source-only identifiability, crossing, promotion, and integrity campaign. |
| `systematics/high_qt_direct_production_benchmark/` | same public path | Experimental W/Y and unitary-transition matching studies. |
| `systematics/finite_y_tail_benchmark/` | same public path | External finite-tail benchmark drivers and summary metadata. |
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

Run root shell entry points from the repository root:

```bash
cd /path/to/b-space
./run_v23a_central_tmd_grid_and_audit.sh
```

Run Python modules with the repository root on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/b-space python /path/to/b-space/v22/tools/construct_v22_scheme_tmd_grid.py --help
```

When a fit or cache lives outside the checkout, pass its path explicitly with
`RUN=`, `CACHE_ENV=`, `INIT_STATE=`, `OUT=`, `DATA_DIR=`, `TRAIN=`, or
`BACKEND=`. Do not change a cache's data/PDF/backend identity by editing only
the path name.
