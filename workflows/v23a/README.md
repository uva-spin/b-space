# v23a workflow drivers

These drivers contain the data-selection, fixed-target fit, replica, and
publication-plot orchestration used after the v22 perturbative layer.  The
implementation modules remain in `v23/backends/`, `v23/tools/`, and
`v23/experimental/`; this directory contains the legacy campaign drivers that
used to clutter the repository root.

## 1. Select and audit the data

Start with the global intake and row-level audits.  They are read-only with
respect to `Data/` and should be run before building a cache:

```bash
export PYTHONPATH="$PWD"
PYTHONPATH=. python workflows/v23a/audits/audit_v23_global_dy_data_intake_v2.py --help
PYTHONPATH=. python workflows/v23a/audits/audit_v23a_norm_and_finite.py --help
PYTHONPATH=. python workflows/v23a/audits/audit_v23a_prefactor_triage.py --help
```

The corrected row-99 and normalization-prior variants are explicit directories
under `Data/v23a_fixed_target_lowQ_row99_variants/`.  Never mix their cache
metadata or nuisance conventions.

## 2. Build the check-only cache and central fit

The check-only stage verifies all W/Y rows without changing a fit.  The
central refit then consumes the cache and a staged trainer:

```bash
DATA_DIR="$PWD/Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99" \
DEVICE=cuda \
./workflows/v23a/runs/run_v23a_fixed_target_checkonly_cache.sh

TRAIN="$PWD/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py" \
BACKEND="$PWD/v22/backends/bt_internal_css_backend_v22_full.py" \
DATA_DIR="$PWD/Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99" \
OUT=outputs/v23a_central_refit \
./workflows/v23a/runs/run_v23a_fixed_target_corrected_central_refit_v2.sh
```

The CSV-normalization and explicit-normalization-prior drivers are controlled
comparisons, not interchangeable defaults.  Use a fresh `OUT` for every
variant and inspect the emitted fit status, row-level pulls, nuisance shifts,
and cache manifest.

## 3. Construct and audit b-space output

```bash
./workflows/v23a/runs/run_v23a_central_tmd_grid_and_audit.sh
PYTHONPATH=. python workflows/v23a/audits/audit_v23a_central_tmd_grid.py --help
```

For PDF-through-refit studies, first create the plan and then aggregate only
completed runs:

```bash
PYTHONPATH=. python workflows/v23a/construction/make_v23a_data_pdf_replica_plan.py --help
bash workflows/v23a/replicas/run_v23a_data_pdf_replicas.sh  # set its documented DATA_DIR/SEEDS/OUT variables first
PYTHONPATH=. python workflows/v23a/construction/construct_v23a_data_pdf_bspace_tmd_bands_v2.py --help
```

The b-space CSV is the primary numerical object.  Keep the PDF member,
experimental replica seed, data-table variant, backend commit, and trainer
configuration in the ensemble manifest so that the band can be reconstructed.

## 4. Regularized k-space and publication plots

The finite Hankel transform is a derived, explicitly regularized diagnostic:

```bash
PYTHONPATH=. python workflows/v23a/construction/construct_v23a_regularized_kspace_tmd_v2.py --help
PYTHONPATH=. python workflows/v23a/plotting/plot_v23a_traditional_kspace_tmd.py --help
PYTHONPATH=. python workflows/v23a/plotting/plot_v23a_paper_bspace_all_flavors_v2.py --help
PYTHONPATH=. python workflows/v23a/plotting/plot_v23a_paper_kspace_3d_tmd_v2.py --help
```

The standard first transform is `tail-mode=expb2`, `b-transform-max=24`,
`n-b-transform=6001`, `k-max=4`, `n-k=401`, with the last 8% of the integration
range smoothly tapered.  Compare `expb2`, `expb`, and `taper` before freezing
a k-space plot; do not treat finite-grid ringing as a fit failure.

## 5. Replica campaigns and boundaries

The scripts in `replicas/` are intentionally explicit about seed lists,
parallelism, and rescue/continuation blocks.  Run a pilot, verify plateau
status and output manifests, then append a disjoint seed block.  The
Tevatron-core validated campaign and the Tevatron-plus-LHCb candidate are
separate scopes; the latter must not be silently merged with the former.

All generated fit directories, GPU caches, and large tables belong outside the
source release.  The complete W+Y and finite-`Y` decision records under
`systematics/` explain which candidate boundaries are diagnostics only.
