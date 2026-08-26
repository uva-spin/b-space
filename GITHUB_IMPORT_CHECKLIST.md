# b-space GitHub import checklist

This file records the original public-release import checklist for
`https://github.com/uva-spin/b-space`.  The repository is now assembled on the
`main` branch; this is a historical staging record, not a command sequence to
run on an existing checkout.  For current documentation and reproduction
instructions, start with `README.md`, `docs/REPRODUCIBILITY.md`, and
`docs/SOURCE_MAP.md`.

## Historical copy step

From the unpacked bundle root:

```bash
rsync -av ./ /path/to/b-space/
cd /path/to/b-space
git status
```

## Code that should be committed

### Core public v23 tools

- `v23/tools/make_v23a_pdf_overlay_plan_from_runs.py`
- `v23/tools/construct_v23a_data_pdf_bspace_tmd_bands_v2.py`
- `v23/tools/audit_v23a_data_pdf_ensemble.py`
- `v23/tools/plot_v23a_data_pdf_bspace_bands_improved.py`
- `v23/tools/plot_v23a_paper_bspace_d_tmd.py`
- `v23/tools/construct_v23a_regularized_kspace_tmd.py`
- `v23/tools/compare_v23a_regularized_kspace_modes.py`
- `v23/tools/plot_v23a_traditional_kspace_tmd.py`
- `v23/tools/draw_v23a_tmd_dnn_node_architecture.py`
- `v23/tools/make_v23a_explicit_csv_norm_prior_dir.py`
- `v23/tools/audit_v23a_central_refit_v2.py`
- `v23/tools/audit_v23a_replica_fit_outliers.py`
- `v23/tools/audit_v23a_central_tmd_grid.py`

### Experimental/high-cost tools

- `v23/experimental/make_v23a_data_pdf_replica_plan.py`
- `v23/experimental/run_v23a_data_pdf_replicas.sh`
- `v23/experimental/construct_v23a_data_pdf_bspace_tmd_bands.py`

These are for true PDF-through-refit studies and are not the default public workflow.

## Freeze directories to commit

- `v23/freeze/v23a_fixed_target_DY_experimental_replica_fits_50rep/`
- `v23/freeze/v23a_fixed_target_DY_bspace_expPDF_overlay_50rep/`
- `v23/freeze/v23a_fixed_target_DY_kspace_regularized_expPDF_overlay/`

Commit README/manifest/path files.  Large CSVs, replica runs, and backend caches should be attached to a release or stored through the collaboration archive rather than committed directly unless the repo policy says otherwise.

## Large artifacts for release assets

Recommended release tarballs:

- `replica_pilot_v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda/`
- `replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/tmd_bspace_bands_expPDF_overlay/`
- `replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_expPDF_overlay_expb2/`
- `replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_comparison/`
- `plots/v23a_fixed_target_lowQ_normpriors15_p2p5_E772_E288400_central_exactx/`

## Historical commit sequence

```bash
git add README.md LICENSE CITATION.cff requirements.txt
git add v23/tools v23/experimental v23/freeze
git add docs figures scripts/examples
git commit -m "Add v23a fixed-target DY b-space TMD release artifacts"
```
