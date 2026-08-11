# Systematics and identifiability campaign source

The directory
`systematics/dataset_identifiability_campaign_2026/scripts/` contains the
source scripts used to study nonperturbative identifiability, reference-distance
objectives, multistart stationarity, experimental-replica crossing, and the
lambda=1 production promotion.

The source release includes the scripts and small configuration/manifest files,
but not the multi-gigabyte run directories, checkpoints, logs, or long
ensemble tables from the working machine.

## Main campaign families

| Family | Representative paths | Purpose |
| --- | --- | --- |
| Reference objective | `run_matched_baseline_reference_distance.py`, `build_exact_baseline_fnp_median.py` | Defines and evaluates the empirical-reference distance. |
| Multistart stationarity | `run_lambda1_start_expansion96.py`, `confirm_sensitive_reference_starts.py` | Extends and checks the stationary-start ensemble. |
| Experimental crossing | `cross_lambda1_96start_with_experimental.py`, `build_final_combined_tmd_ensemble.py` | Combines stationary starts with conditional experimental replicas. |
| b-space/k-space construction | `build_bspace_tmd_ensemble.py`, `transform_bspace_ensemble_to_kspace.py` | Reconstructs and transforms the ensemble. |
| Production promotion | `promote_lambda1_production_baseline.py`, `promote_lambda1_production_update96.py` | Writes or audits the active production package. |
| Integrity audits | `audit_lambda1_productionization.py`, `audit_final_combined_ensemble.py`, `audit_frozen_inputs.py` | Recomputes gates and verifies provenance. |
| Semantics/tests | `test_final_uncertainty_semantics.py`, `test_promotion_transaction.py` | Prevents operational bands from being reported as calibrated intervals. |

## Campaign execution boundary

These scripts are not generic command-line tools with all inputs bundled. Most
expect a campaign root containing run ledgers, summaries, checkpoints, and
input manifests. A clean checkout can inspect them and run their unit-style
tests, but reproducing the 96-start production ensemble requires the archived
campaign artifacts referenced by the production manifests.

Typical overrides used by the campaign are:

```bash
CAMPAIGN_ROOT=/path/to/dataset_identifiability_campaign_2026
PYTHONPATH=/path/to/b-space python \
  systematics/dataset_identifiability_campaign_2026/scripts/audit_lambda1_productionization.py \
  --help
```

Do not point a campaign script at a different summary directory merely to make
it run. The scripts bind input manifests, row identities, seed ancestry, and
artifact hashes to the outputs they audit.

## Active production interpretation

The active result is the package under
`production/lambda1_empirical_reference_full96x50/`. Its authoritative files
are `PRODUCTION_MANIFEST.json` and `PRODUCTION_AUDIT.json`. The campaign source
describes how the package was produced; the package itself records what was
promoted and which limitations remain.
