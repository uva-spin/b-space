# v22 workflow drivers

The v22 drivers reproduce the perturbative/backend development that precedes
the v23a production orchestration.  They are intentionally separated from the
implementation modules in `v22/src/`, `v22/backends/`, and `v22/tools/`.

## Stage A: convention and perturbative bootstrap

Run these from the repository root in this order:

```bash
export PYTHONPATH="$PWD"
./workflows/v22/bootstrap/bootstrap_v22_phaseA_conventions.sh
./workflows/v22/bootstrap/bootstrap_v22_convolution_reference.sh
./workflows/v22/bootstrap/bootstrap_v22_css2_nlo_canonical.sh
./workflows/v22/bootstrap/bootstrap_v22_css2_nlo_general_scale.sh
./workflows/v22/bootstrap/bootstrap_v22_dy_hard_nlo.sh
./workflows/v22/bootstrap/bootstrap_v22_small_b_profile.sh
./workflows/v22/bootstrap/bootstrap_v22_standalone_w_kernel.sh
```

These scripts write only tagged reports.  The source-level checks are also
available as CPU-only commands in `docs/MATCHING.md` and `docs/REPRODUCIBILITY.md`.
The canonical formulas are in `v22/CSS2_SCHEME.md`, `v22/GENERAL_SCALE_OPE.md`,
`v22/OPE_CONVOLUTION.md`, `v22/DY_HARD_FACTOR.md`, and `v22/SMALL_B_PROFILE.md`.

## Stage B: backend closure and grids

The v22 backend wrapper imports the frozen compatibility backend at the
repository root and adds the CSS2 OPE/hard-factor organization without editing
that file.  Build a cache and run the closure audits before fitting:

```bash
./workflows/v22/utilities/export_v22_backend_cache.sh
./workflows/v22/runs/run_v22_cache_cuda_smoketest.sh
PYTHONPATH=. python workflows/v22/audits/audit_v22_full_backend_integration.py --help
PYTHONPATH=. python workflows/v22/audits/audit_v22_full_profile_hard_ope.py --help
```

The actual cache command requires a staged trainer, LHAPDF, and the selected
fixed-target data directory; follow `docs/REPRODUCIBILITY.md` and pass
`TRAIN=`, `BACKEND=`, `DATA_DIR=`, `CACHE_ENV=`, and `OUT=` explicitly.  Cache
rows are keyed by the data row id and must be checked before a fit.

## Stage C: central fit, warm checks, and b-space construction

```bash
FROZEN=/path/to/frozen/reference \
TRAIN=/path/to/train_bt_dnn_v21_smoothedA_tail.py \
OUT=outputs/v22_central_refit \
./workflows/v22/runs/run_v22_central_refit_stage1.sh

FROZEN=/path/to/central_refit \
TRAIN=/path/to/train_bt_dnn_v21_smoothedA_tail.py \
OUT=outputs/v22_warmcheck \
./workflows/v22/runs/run_v22_full_warmcheck_v2.sh

RUN="$PWD/outputs/v22_central_refit" \
PYTHONPATH=. python v22/tools/construct_v22_scheme_tmd_grid.py \
  --run "$RUN" --backend-script v22/backends/bt_internal_css_backend_v22_full.py \
  --pdf-set NNPDF40_nnlo_as_01180 --pdf-member 0 --resum-order n3llp \
  --pids 2 1 -2 -1 --x-values 0.10 0.20 0.30 0.50 --Q-values 5 10 \
  --b-min 0 --b-max 8 --n-b 321 --out plots/v22_central_bspace

PYTHONPATH=. python workflows/v22/audits/audit_v22_scheme_tmd_grid.py --help
```

The `n3llp` label is retained for historical campaign compatibility; in this
source release it deliberately uses the available `A1--A3,B1--B2` Sudakov set
and is not a claim of complete N3LL-prime hard/OPE matching.  See the accuracy
inventory in `systematics/perturbative_provenance_completion/`.

## Stage D: replicas and freeze

Use a small pilot first, then increase the seed list only after all fits reach
the requested epoch/plateau gate:

```bash
SEEDS="1001 1002 1003" DEVICE=cuda MAX_PARALLEL=1 \
  ./workflows/v22/replicas/run_v22_three_replica_profiled.sh

SEEDS="1001 1002 1003 1004" DEVICE=cuda MAX_PARALLEL=1 \
  ./workflows/v22/replicas/run_v22_lambda3_cached_cuda_replicas.sh
./workflows/v22/replicas/append_v22_lambda3_cached_cuda_replicas.sh
./workflows/v22/replicas/freeze_v22_lambda3_50rep_bspace.sh
```

The freeze driver records hashes and status; it does not make an exploratory
campaign production by itself.  A production promotion requires the separate
identifiability and provenance gates documented in `production/` and
`systematics/dataset_identifiability_campaign_2026/`.

## Inputs and outputs

Required external inputs are a compatible trainer from
`v21_tail_release_amp0p019_candidate/`, LHAPDF plus
`NNPDF40_nnlo_as_01180`, one of the staged `Data/v23a_fixed_target_*` tables,
and (for fast fits) a row-aligned backend cache.  Typical generated paths are
`outputs/<tag>/`, `plots/<tag>/`, and external cache directories.  Do not place
checkpoints or large replica CSVs in this source repository.
