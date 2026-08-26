# Reproducible workflow drivers

This directory contains the executable drivers that were historically kept at
the repository top level.  They are now grouped by release and by function so
that a new checkout can be followed in the same order in which the calculation
was developed.  All commands below are run from the repository root; the
shell drivers deliberately use `ROOT="$(pwd)"` and therefore must not be
launched from inside their subdirectories.

The canonical implementation remains in `v22/` and `v23/`.  The scripts here
are orchestration, audit, plotting, bootstrap, and replica drivers.  They do
not contain generated output, fitted checkpoints, or external PDF/MCFM/
DYTurbo installations.  Pass explicit paths for those inputs and choose a new
output directory for every run.

## Progression

1. `v22/bootstrap/` establishes the convention lock, convolution reference,
   CSS2 canonical and general-scale OPE, hard factor, small-`b_T` profile, and
   standalone W-kernel smoke tests.
2. `v22/audits/` checks the backend against those source-level definitions;
   `v22/construction/` builds row-aligned b-space grids and replica bands.
3. `v22/runs/` performs warm checks and central refits; `v22/replicas/`
   launches/extends replica ensembles and freezes their source tables;
   `v22/utilities/` handles cache export and external-benchmark inventories.
4. `v23a/runs/` stages the corrected fixed-target data/cache and central fit.
   `v23a/audits/` checks data intake, normalization, fit outliers, and grid
   integrity.
5. `v23a/replicas/` runs the experimental/PDF replica campaigns;
   `v23a/construction/` aggregates b- and k-space tables; and
   `v23a/plotting/` produces publication-style figures.

The frozen lambda=1 result and the isolated finite-`Y`/W+Y studies are not
recreated by these drivers automatically.  Their exact source-only handoffs
are under `systematics/`; read the local `README.md` and `HANDOFF.md` before
using a candidate as a comparison.

## Reproducibility rules

- Set `PYTHONPATH="$PWD"` for Python tools.
- Keep the PDF set (`NNPDF40_nnlo_as_01180`, member 0 unless a manifest says
  otherwise), data-table variant, backend revision, and perturbative flags
  together in the cache metadata.  A path rename does not make a cache
  interchangeable.
- Never write into `production/` from an exploratory run.  Use a fresh
  `outputs/`, `plots/`, or systematics report directory.
- Inspect the emitted `*_status.json`, cache manifest, and audit JSON before
  using a numerical table in a fit or figure.

See `v22/README.md`, `v23/README.md`, and `docs/REPRODUCIBILITY.md` for the
physics and numerical settings behind these stages.
