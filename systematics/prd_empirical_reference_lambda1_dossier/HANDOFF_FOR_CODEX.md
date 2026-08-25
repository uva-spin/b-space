# Handoff to the Codex agent on UVA Afton/Rivanna

## Objective

Reproduce and extend the first PRD replacement candidate using UVA GPUs. The
candidate constrains the accepted monotone FiLM model with a pointwise distance
to the empirical median of 24 historical `F_NP` solutions:

- reference domain: `0.1 <= bT <= 2 GeV^-1`;
- strength: `lambda = 1`;
- data: accepted 329-row `qT/Q <= 0.20` selection;
- starts: fit seeds 303--326;
- experimental pseudo-data replicas: seeds 1001--1050;
- interval: pointwise q16--q84 of the declared combined ensemble.

The goal is not merely to run jobs. It is to obtain an auditable ensemble that
separately represents functional non-uniqueness and propagated experimental
uncertainty for the PRD TMD and cross-section figures.

## Scientific meaning of the ensemble

There are 24 independent-start fits and 50 experimental-replica fits, plus the
chosen central initializer. There are not 1,200 independent neural fits.

After all fits pass validation, combine whole positive curves in log space:

```text
log F_sr = log F_c
         + (log F_s - pointwise_median_s(log F_s))
         + (log F_r - pointwise_median_r(log F_r)).
```

Keep all 24x50 combinations, producing 1,200 operational members per flavor.
This preserves each curve's `bT` correlation and positivity. The q16--q84 band
is an operational hierarchical empirical 68% interval. Do not call it a
frequentist confidence interval or Bayesian credible interval.

The currently quoted provisional result reused a conditional 50-replica
residual ensemble and gave maximum active Fig. 6 full widths of 11.77% for `u`
and 12.49% for `d`. This UVA workflow refits all 50 pseudo-data replicas under
the lambda-1 objective. Therefore the new result is scientifically more
complete and is not required to reproduce those widths exactly. Any change
must be reported, not tuned away.

## Exact first-candidate fit prescription

- Accepted neural architecture is unchanged; do not add capacity or alter the
  model definition.
- `F_NP` reference penalty:

  `lambda * mean[((F_NP - F_ref) / max(F_ref, 0.10))**2]`

- `lambda=1`, `bmin=0.10`, `bmax=2.0`.
- Zero AdamW epochs.
- PyTorch L-BFGS, strong-Wolfe implementation supplied by the runner, maximum
  500 iterations.
- Float32, matching the registered first candidate. Do not pass `--float64`.
- Start initializers and normalizations must stay paired by seed.
- Experimental replicas use pseudo-data seeds 1001--1050 and deterministic
  optimizer seeds 2001--2050.

## UVA execution contract

- Allocation: `spinquest_standard`.
- Initial partition/GRES: `gpu`, `gpu:a6000:1`.
- Supported software pattern:

  ```bash
  module load apptainer/1.4.5 pytorch/2.11.0
  apptainer run --nv "$CONTAINERDIR/pytorch-2.11.0.sif" SCRIPT.py
  ```

- The included `container_python.sh` implements this pattern and prevents
  nested Apptainer execution by subprocesses.
- Start with concurrency 8 for each array. Change it only after the pilot and
  allocation/queue review.
- Do not use login nodes for computation.

## Required sequence

From `bT-TMD/systematics/prd_empirical_reference_lambda1_dossier/hpc`:

```bash
bash configure_rivanna.sh
bash preflight.sh
bash submit_workflow.sh --dry-run
bash submit_workflow.sh --submit-validation
```

Review the validation output. Then submit exactly one complete start fit:

```bash
bash submit_workflow.sh --submit-pilot
```

Review `fit_status.json`, `fnp_grid.csv`, logs, `sacct`, and `seff`. Confirm that
CUDA was used and compare the pilot endpoint with the staged historical result.
Only then may the production arrays be submitted:

```bash
bash submit_workflow.sh --submit-production
```

The production command submits start and replica arrays, a dependent audit,
and a dependent FNP combination job. Never bypass a failed dependency merely
to obtain downstream plots.

## Immutable inputs and write scope

Treat everything under `bT-TMD/staged_inputs` as immutable. Do not overwrite
the transfer bundle's code or inputs during fits. New fit outputs belong only
under:

```text
/scratch/dmk9m/prd_lambda1_complete/
```

The source workstation has another active campaign. Do not connect back to it
to alter, stop, or launch its processes. In particular, never modify the source
`dataset_identifiability_campaign_2026`, frozen production files, or existing
supervisor/finisher processes.

## Validation and failure policy

A member is complete only if the runner exits successfully and writes at least:

- `fit_status.json`;
- `fnp_grid.csv`;
- `dataset_norms.csv`;
- workflow identity metadata.

Do not silently accept or overwrite partial directories. Do not change seeds,
regularization, architecture, precision, or optimization settings to rescue a
failed member. Diagnose failures, preserve their logs, and report them before
changing the declared protocol.

Before accepting the full result, verify:

1. exact start identities 303--326, once each;
2. exact replica identities 1001--1050, once each;
3. no reported production-state modification;
4. finite, positive `F_NP` values on identical grids;
5. finite fit statistics and plausible replica chi-square distribution;
6. successful 24x50 whole-curve combination;
7. separate non-uniqueness-only, experimental-only, and combined bands;
8. convergence/stability diagnostics before claiming the ensemble is adequate.

## Deliverables to return

- Slurm job IDs and final states.
- Exact module, container, node, and GPU metadata.
- Pilot and production timing, memory, GPU utilization, and SU usage.
- Per-member audit table with seeds, status, fit statistics, and artifact paths.
- Combined FNP ensemble and q16/median/q84 bands.
- Separate start-only, replica-only, and combined uncertainty summaries.
- Any failed or anomalous members with untouched logs.
- A concise comparison with the provisional 11.77% (`u`) and 12.49% (`d`)
  Fig. 6 widths, clearly noting that the new 50 replicas were independently
  refitted under the lambda-1 objective.

Do not update paper text or promote a production result automatically. Return
the validated numerical products and diagnostics for review first.
