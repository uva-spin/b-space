# Progressive SIDIS fit plan

Status: **staged discovery only; no rows approved and no production fit authorized**.

Harvest the complete candidate universe, but fit one provenance-closed family at a time; no all-at-once global optimization is authorized.

## External benchmark

The literature checkpoint is **1,547 post-cut HERMES/COMPASS points** (344 HERMES + 1,203 COMPASS), not a raw-table count. It remains blocked until the source-specific row and covariance conventions are closed. See `reports/sidis_1547_benchmark_audit.md`.

## Stages

| ID | Scope | Sources | Status | Entry/exit rule |
| --- | --- | --- | --- | --- |
| `S0_inventory` | all registry records | all records in config/global_sources.json | complete | none; registry, source hashes, table inventory, and explicit stage/diagnostic/deferred classification |
| `S1_literature_benchmark` | HERMES-COMPASS-1547-benchmark | hepdata:ins1208547, hepdata:ins1624692 | blocked_provenance | HERMES zxpt-3D value/covariance files and COMPASS row-level selection convention are available. |
| `S1_core_extensions` | clean-HERMES-COMPASS-extensions | hepdata:46860, hepdata:ins1236358, hepdata:ins1444985 | pending | S1 promoted and each extension has a row-selection and covariance manifest. |
| `S2_jlab_clas` | CLAS-absolute-cross-section | arxiv:0809.1153 | inventory_complete_closure_pending | S1 interface reproduces DY and SIDIS scalar references; CLAS five-fold cross-section, radiative factor, bin integration, and covariance are closed. |
| `S2_jlab_hall_c` | Hall-C-low-energy | arxiv:1103.1649 | inventory_complete_closure_pending | CLAS or an equivalent absolute-cross-section closure passes and low-energy validity is documented. |
| `S2_recent_and_historical` | recent-and-historical-complements | hepdata:ins1483098, hepdata:ins2840545, hepdata:29288, hepdata:37889, hepdata:1432 | harvested_closure_pending | The source is shown to share the declared SIDIS observable and its target/energy corrections are explicit. |
| `S3_diagnostics` | nuclear-current-region-diagnostics | arxiv:1610.02350, hepdata:42505, hepdata:42540, hepdata:30476, hepdata:44930, hepdata:45525, hepdata:ins1217865, arxiv:hep-ex/9511010 | diagnostic_only | Dedicated nuclear/current-region/jet observable and covariance implementations pass independent closure. |
| `S4_deferred` | future-or-access-restricted | jlab:E12-09-017, jlab:clas_physics_database | deferred | A final, versioned, publicly reproducible table and uncertainty definition becomes available. |

## Source coverage check

Registry records: **22**. Explicitly assigned non-S0 records: **22**.

A source may be fit only after its stage entry gate passes. Sources in the diagnostic or deferred classes are never silently merged into the multiplicity likelihood.

## Required record for every trial

- `input_manifest_and_hashes`
- `row_selection_and_covariance_manifest`
- `observable_and_bin_integration_convention`
- `fit_objective_decomposition`
- `independent_start_and_plateau_evidence`
- `experimental_replica_protocol`
- `held_out_and_leave_one_family_out_metrics`
- `central_prediction_shift_relative_to_previous_stage`
- `model_form_and_prior_sensitivity`
- `promotion_or_rejection_decision`

Experimental replicas, start non-uniqueness, TMDFF parameterization, dataset selection, and perturbative/theory variations must remain separately labelled before any combined envelope is shown.

The machine-readable source is `config/staged_fit_plan.json`.
