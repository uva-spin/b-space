# Reproducibility and execution guide

This is the practical guide for a clean checkout. Commands below are written
to be run from the repository root.

## 1. Checkout and environment

```bash
git clone https://github.com/uva-spin/b-space.git
cd b-space
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The code is tested with Python 3.10 or newer. PyTorch, NumPy, SciPy, pandas,
and matplotlib are Python dependencies. LHAPDF and the PDF grid
`NNPDF40_nnlo_as_01180` are required for real-PDF backend evaluations. The
repository does not vendor LHAPDF or PDF grids.

Check the environment and available PDF sets with:

```bash
bash Data/data_check.py
```

## 2. Public source layout

```text
v22/src/                         perturbative and convolution primitives
v22/backends/                   fixed-target W backend and W-expansion scheme
v22/tests/                      source-level smoke tests
v22/tools/                      backend, fit, grid, and audit utilities
v23/backends/                   Tevatron/collider-aware backend wrapper
v23/tools/                      v23a data, replica, plotting, and audit tools
v23/experimental/               experimental PDF-through-refit workflow
v21_tail_release_amp0p019_candidate/
                                publishable trainer sources used by v22/v23 scripts
Data/                            fixed-target tables and row-99 variants
production/                      frozen active production outputs and integrity files
systematics/full_n3ll_wy_production_2026/
                                isolated N³LL+NNLO W+Y source and handoff
systematics/finite_y_completion_2026/
                                unitary finite-Y completion records
systematics/perturbative_provenance_completion/
                                perturbative organization and accuracy audits
```

The top-level `run_*`, `audit_*`, `construct_*`, and `bootstrap_*` files are
compatibility entry points retained from the working campaign. They expect to
be launched from the repository root and write generated results under
`outputs/`, `plots/`, or `v23/outputs/`.

## 3. Data paths

The fixed-target datasets are:

```text
Data/E288_200.csv
Data/E288_300.csv
Data/E288_400.csv
Data/E605.csv
Data/E772.csv
```

The staged production table variants are:

```text
Data/v23a_fixed_target_lowQ_candidate/
Data/v23a_fixed_target_lowQ_verified/
Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99/
Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_explicit_normpriors_15pct/
Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_normpriors15_p2p5_E772_E288400/
```

The last directory is the default corrected row-99, 15% normalization-prior,
5% point-to-point-sensitivity table for the v23a fixed-target workflow. Its
`norm_prior_summary.csv` and `p2p5_summary.csv` files document the generated
uncertainty columns.

## 4. Run source-level tests first

These tests do not launch a fit and should run on a normal CPU-only machine:

```bash
python -m pytest -q v22/tests/test_conventions.py
python v22/tests/run_convolution_smoke.py
python v22/tests/run_css2_ope_nlo_smoke.py
python v22/tests/run_css2_ope_nlo_general_smoke.py
python v22/tests/run_dy_hard_nlo_smoke.py
python v22/tests/run_dy_w_nlo_reference_smoke.py
python v22/tests/run_small_b_profile_smoke.py
```

Run all pytest-discovered tests with:

```bash
python -m pytest -q
```

## 5. Build a W/Y backend cache

The cache is the boundary between the expensive perturbative calculation and
the DNN fit. It stores row-aligned W and Y grids, metadata, and the exact PDF
and backend configuration used to create them.

For the fixed-target backend, the primary entry point is:

```bash
TRAIN="$PWD/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py" \
BACKEND="$PWD/v22/backends/bt_internal_css_backend_v22_full.py" \
DATA_DIR="$PWD/Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_normpriors15_p2p5_E772_E288400" \
./run_v23a_fixed_target_checkonly_cache.sh
```

That command may require local adjustments to the output and device settings.
The lower-level backend cache export is:

```bash
./export_v22_backend_cache.sh
```

The expected cache handoff is a file such as:

```text
outputs/<cache-name>/backend_cache/cache_paths.env
```

It defines `W_GRID` and `Y_GRID`. Always inspect the generated metadata before
using the cache for a fit. A cache generated with a different PDF member,
dataset table, backend revision, or perturbative flags is not interchangeable.

## 6. Central fixed-target refit

The central workflow uses the staged trainer and the v22 full backend:

```bash
TRAIN="$PWD/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py" \
BACKEND="$PWD/v22/backends/bt_internal_css_backend_v22_full.py" \
DATA_DIR="$PWD/Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99" \
./run_v23a_fixed_target_corrected_central_refit_v2.sh
```

For the production-style 15% normalization-prior and 5% point-to-point
variant, override `DATA_DIR` with:

```text
Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_normpriors15_p2p5_E772_E288400
```

The script requires a compatible cache and initialization state. It refuses to
overwrite an existing output directory. Set `OUT=...` to choose a new output.

The older v22 stage-1 route is retained for backend closure checks:

```bash
FROZEN=/path/to/compatible/frozen/reference \
TRAIN="$PWD/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py" \
./run_v22_central_refit_stage1.sh
```

## 7. Construct the b-space TMD grid

After a successful fit, construct the exact-x central grid:

```bash
RUN="$PWD/outputs/<central-fit>" \
PYTHONPATH=. python v22/tools/construct_v22_scheme_tmd_grid.py \
  --run "$RUN" \
  --backend-script v22/backends/bt_internal_css_backend_v22_full.py \
  --pdf-set NNPDF40_nnlo_as_01180 \
  --pdf-member 0 \
  --resum-order n3llp \
  --pids 2 1 -2 -1 \
  --x-values 0.10 0.20 0.30 0.50 \
  --Q-values 5 10 \
  --b-min 0 --b-max 8 --n-b 321 \
  --out plots/central_bspace_grid
```

The main output is:

```text
plots/central_bspace_grid/v22_scheme_tmd_bspace_long.csv
```

The convenience wrapper `run_v23a_central_tmd_grid_and_audit.sh` performs the
same construction and then calls the v23a central-grid audit.

## 8. Experimental replicas and PDF overlays

The v23a experimental workflow is split into explicit stages:

```text
v23/experimental/make_v23a_data_pdf_replica_plan.py
v23/experimental/run_v23a_data_pdf_replicas.sh
v23/tools/construct_v23a_data_pdf_bspace_tmd_bands_v2.py
v23/tools/construct_v23a_regularized_kspace_tmd.py
v23/tools/compare_v23a_regularized_kspace_modes.py
```

The shell workflow builds or reuses a W/Y cache for each PDF member, fits the
assigned experimental pseudo-data replica, reconstructs the b-space TMD with
that member, and aggregates the ensemble. It is expensive and generally needs
CUDA or a long CPU run.

The final overlay is not PDF-through-refit uncertainty unless every PDF member
has been retrained. The standard published overlay varies the PDF member in
the reconstruction while retaining the experimental replica fits; see the
manifests in `v23/freeze/` and `production/`.

## 9. Regularized k-space companion

The b-space ensemble is primary. The k-space companion is generated afterward:

```bash
PYTHONPATH=. python v23/tools/construct_v23a_regularized_kspace_tmd.py --help
PYTHONPATH=. python v23/tools/compare_v23a_regularized_kspace_modes.py --help
```

The default prescription is `expb2`, with a finite transform range and an end
taper. Compare `expb2`, `expb`, and `taper` before treating the k-space range
as stable. Do not interpret finite-transform negative lobes as a positivity
constraint failure; they are retained diagnostics.

## 10. Production artifacts

The active production result is under:

```text
production/lambda1_empirical_reference_full96x50/
```

Read these files in this order:

1. `PRODUCTION_MANIFEST.json` — active status, artifact hashes, and limitations.
2. `PRODUCTION_AUDIT.json` — recomputed production gates and ensemble metrics.
3. `FREEZE_MANIFEST.json` — freeze transaction and interpretation.
4. `README.md` — human-readable description of the package.
5. `bspace_combined_bands.csv` and `kspace_combined_bands.csv` — numerical outputs.

The CSV files contain the active result; they are not a substitute for the
source code or for the audit records.

## 11. Isolated W+Y artifacts

The external Tevatron W+Y candidate and its paper-facing figures are under
`systematics/full_n3ll_wy_production_2026/`. Read `README.md` and `HANDOFF.md`
before using them. The 122-row grid is an isolated perturbative candidate; the
larger 329-row crossed ensemble is a diagnostic and is not production-
authorized. Reproduction requires the archived DYTurbo installation, PDF
inputs, backend/cache paths, and fit states named in the manifests. The public
checkout intentionally contains the scripts, compact decision records, and
figures but not those machine-specific or very large artifacts.

## 12. What is intentionally not committed

The following remain external or generated:

- PyTorch checkpoints and full replica-run directories;
- backend caches, which can be regenerated and are PDF-member-specific;
- logs and temporary output directories;
- LHAPDF itself and PDF grids;
- the vendored `artemide` checkout used during development;
- large external DYTurbo/MCFM installations and their full raw output trees;
- accelerator/global-DY production claims that have not passed the required
  covariance, units, electroweak, and bin-integration review.

Use `paths.txt`, manifests, and the documented absolute/relative path overrides
to reconnect archived artifacts without placing them in the Git repository.
