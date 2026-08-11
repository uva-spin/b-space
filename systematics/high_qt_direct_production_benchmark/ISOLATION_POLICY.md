# Isolation policy

The accepted q020 production scheme is immutable during this study.

Protected references include:

- `systematics/production_candidate_v23a_collins_q020_tailbench/`
- `systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/`
- `systematics/collins_factorization_validity/replicas/rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep/`
- `v23/backends/bt_internal_css_backend_v22_tevatron.py`
- all existing q020 publication figures and manifests

Rules:

1. Benchmark cards, logs, outputs, summaries, and plots remain below
   `systematics/high_qt_direct_production_benchmark/`.
2. A replacement Y term must be introduced below a future
   `experimental_matched_y/` subtree with a new backend/module name.
3. Experimental fits must use new run tags and output directories. They may
   read frozen production artifacts as references but may not modify them.
4. Production and experimental predictions must be compared row by row with
   explicit manifests and hashes.
5. Promotion requires a reviewed manifest. Passing tests never silently
   changes the production default.
6. If promotion is eventually accepted, integration into production is a
   separate operation outside the benchmark campaign.
