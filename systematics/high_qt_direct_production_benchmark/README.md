# High-qT collider direct-production benchmark

This study is the promotion workspace for collider rows with `qT/Q > 0.20`.
It is intentionally separate from the accepted Collins-envelope production fit.
Nothing in this directory changes the current production candidate until the
promotion gates below are complete and explicitly reviewed.

## Baseline

- Production reference: `rowidfix_stageFT_all_qmax0p20_lam0p50`
- Source row gate:
  `systematics/finite_y_tail_benchmark/summaries/tail_benchmark_row_gate.csv`
- Candidate datasets: `CDF_RUN_1`, `CDF_RUN_2`, `D0_RUN_1`, `LHCb_7`
- Candidate definition: `qT/Q > 0.20`
- Current external-code convention: fixed-order V+jet comparison using the
  exact processed mass window and qT bin, with consistent beam and fiducial
  definitions.

Run the read-only baseline audit with:

```bash
/home/dustin/miniforge3/envs/pdf-fit/bin/python \
  systematics/high_qt_direct_production_benchmark/scripts/build_candidate_inventory.py
```

It writes the candidate inventory, missing-benchmark batches, and status
summary under `summaries/`. It does not launch DYTurbo, MCFM, or a fit.

## Current campaign result

The dense Tier-1 central campaign is complete as of 2026-07-21 local project
time. All 25 rows in `0.20 < qT/Q <= 0.30` have row-level DYTurbo/MCFM pairs.
The screening pass found one marginal failure, `CDF_RUN_2:48`, at 5.044%.
Its isolated high-statistics rerun gives 1.406%, so it passes the external-code
gate. No Tier-1 row is yet approved for direct production because the scale,
PDF, production matching, and fit-impact gates remain open.

Canonical results:

```text
summaries/tier1_boundary/central/external_pairs.csv
summaries/tier1_boundary/central/external_status.json
summaries/tier1_boundary/precision/external_pairs.csv
summaries/tier1_boundary/precision/external_status.json
```

The central campaign used 100,000 MCFM initialization calls as a screening
level. The marginal row was rerun with 1,000,000 calls. Raw screening and
precision artifacts are retained separately under `outputs/`.

The Tier-1 external uncertainty gates are also complete:

- alternate-seed reproducibility: 25/25 rows pass the 1% gate;
- dense seven-point scale comparison: 150/150 noncentral pairs pass the 5%
  code-agreement gate after six isolated high-statistics overlays;
- NNPDF40 members 1-50: propagated with DYTurbo for all 25 rows;
- selected member cross-code checks: 28/28 pass;
- PDF 68% relative half-width: median 0.81%, maximum 0.85%;
- LO scale range across the campaign: approximately -15% to +21% relative to
  the central predictions.

Summaries:

```text
summaries/tier1_boundary/seed/seed_reproducibility_status.json
summaries/tier1_boundary/scale_dense/scale_variation_status.json
summaries/tier1_boundary/pdf/pdf_variation_status.json
```

The scale envelope is a theory uncertainty diagnostic, not evidence that LO
is the final perturbative prescription. PDF propagation must be repeated for
the eventual experimental matched prediction.

Development of that matched prediction is isolated under
`experimental_matched_y/` and governed by `ISOLATION_POLICY.md`.

## Promotion gates

A row is approved for direct production only when every required gate passes:

1. **Observable identity**: exact mass, rapidity, qT-bin, beam, lepton cuts,
   arm convention, units, and bin-normalization conventions are documented.
2. **Independent fixed-order agreement**: DYTurbo and MCFM agree according to
   the symmetric relative difference in `config/promotion_policy.json`.
3. **Numerical precision**: Monte Carlo errors are sufficiently smaller than
   both the code difference and experimental uncertainty.
4. **Perturbative stability**: the agreed scale-variation prescription is
   evaluated; the prediction is not accepted merely because two central-scale
   calculations agree.
5. **PDF stability**: the production PDF prescription is propagated and any
   high-x sensitivity is recorded separately from scale uncertainty.
6. **Matched-tail continuity**: the production W+Y/matched prediction is
   continuous through the accepted-envelope boundary and agrees with the
   fixed-order limit in the row's active region.
7. **Fit impact**: staged refits show acceptable global and per-dataset pulls,
   normalization pulls, replica convergence, and stability of the extracted
   b-space TMD relative to the q020 reference.

The automated inventory reports only the state of gates 1-2 that can be
inferred from existing artifacts. Its `current_decision` column must not be
treated as final approval.

## Execution order

Use the batches in `summaries/benchmark_batch_plan.csv`:

- Tier 1: `0.20 < qT/Q <= 0.30`, the boundary extension.
- Tier 2: `0.30 < qT/Q <= 0.50`, the conventional high-qT validation range.
- Tier 3: `qT/Q > 0.50`, exceptional kinematics requiring an explicit physics
  review before compute is spent. This currently isolates the far-tail LHCb
  rows.

Within each tier, benchmark every row rather than interpolating approval from
representative points. Run the exact-bin central calculations first, then
precision reruns and scale/PDF variations only for rows that pass convention
checks.

## Layout

```text
config/       frozen thresholds and tool/convention metadata
scripts/      inventory and later orchestration/analysis scripts
summaries/    generated row-level plans and gate summaries
outputs/      external-code outputs, grouped by dataset and code
plots/        comparison, stability, matching, and fit-impact figures
cards/        reviewed card templates or generated-card snapshots
logs/         orchestration logs
```

Do not overwrite the representative artifacts in
`systematics/finite_y_tail_benchmark`. Dense runs belong in this directory and
should retain row IDs in every card, log, and summary record.
## Separately tagged unitary transition

The failed additive Y-term investigation is preserved under
`experimental_matched_y/`. A distinct convex W-to-fixed-order transition is
being tested under `experimental_unitary_transition/`. Neither experimental
subtree modifies the accepted q020 production scheme.
