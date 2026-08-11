# Finite-Y and fixed-order tail benchmark systematic

This directory tracks the external-code benchmark required before high-`qT`
collider rows are used as direct production constraints.

The production baseline avoids high-`qT` collider rows.  Rows outside the strict
TMD-core region are eligible only if one of the following is true:

- the row remains inside the accepted Collins-style factorization-validity
  envelope and is used with the added theory uncertainty;
- the row has a finite-tail benchmark against both DYTurbo and MCFM using the
  same processed mass window, rapidity interval, `qT` bin, beam type, and
  fiducial definition;
- otherwise the row is excluded from the production fit.

## Current gate

The first external-code gate is deliberately simple:

```text
abs(DYTurbo - MCFM) / (0.5 * (abs(DYTurbo) + abs(MCFM))) <= 0.05
```

This is an external-code consistency gate, not a data-agreement gate.  A row can
pass DYTurbo/MCFM agreement and still fail later phenomenological quality checks.

## Current artifacts

Scripts:

```text
systematics/finite_y_tail_benchmark/scripts/summarize_existing_tail_benchmarks.py
systematics/finite_y_tail_benchmark/scripts/plot_cdf_run2_tail_benchmark.py
systematics/finite_y_tail_benchmark/scripts/plot_external_tail_benchmarks.py
systematics/finite_y_tail_benchmark/scripts/run_lhcb7_dyturbo_benchmark.py
systematics/finite_y_tail_benchmark/scripts/run_lhcb7_mcfm_benchmark.py
```

Summaries:

```text
systematics/finite_y_tail_benchmark/summaries/tail_benchmark_summary.csv
systematics/finite_y_tail_benchmark/summaries/collider_row_inventory_by_region.csv
systematics/finite_y_tail_benchmark/summaries/tail_benchmark_row_gate.csv
systematics/finite_y_tail_benchmark/summaries/cdf_run2_dyturbo_mcfm_canonical.csv
systematics/finite_y_tail_benchmark/summaries/cdf_run1_dyturbo_mcfm_canonical.csv
systematics/finite_y_tail_benchmark/summaries/d0_run1_dyturbo_mcfm_canonical.csv
systematics/finite_y_tail_benchmark/summaries/lhcb7_dyturbo_mcfm_canonical.csv
systematics/finite_y_tail_benchmark/summaries/external_tail_benchmarks_canonical.csv
```

Figure:

```text
systematics/finite_y_tail_benchmark/plots/cdf_run2_dyturbo_mcfm_ratio.pdf
systematics/finite_y_tail_benchmark/plots/cdf_run2_dyturbo_mcfm_ratio.png
systematics/finite_y_tail_benchmark/plots/external_tail_benchmark_ratio.pdf
systematics/finite_y_tail_benchmark/plots/external_tail_benchmark_ratio.png
```

## Current status

The current processed collider inventory contains 136 rows:

| region | meaning | rows |
| --- | --- | ---: |
| strict core | `qT/Q <= 0.10` | 51 |
| Collins envelope | `0.10 < qT/Q <= 0.20` | 39 |
| high-`qT` candidate | `qT/Q > 0.20` | 46 |

Existing external-code artifacts now cover representative CDF Run I, CDF Run II,
D0 Run I, and LHCb 7 TeV rows.  DYTurbo outputs in these artifacts are stored as
fb/bin; the canonical summary converts them to pb/bin and pb/GeV before
comparing to MCFM.

The LHCb comparison requires one extra convention step.  The DYTurbo LHCb card
uses the positive forward LHCb arm, `2 < eta_mu < 4.5`, while MCFM's configured
`etaleptmin`/`etaleptmax` cuts are absolute-value cuts and therefore include
both symmetric `pp` forward arms.  The canonical LHCb comparison divides the MCFM
fiducial result by two before forming `DYTurbo/MCFM`.  With this explicit
forward-arm correction, all currently available DYTurbo/MCFM row pairs pass the
5% external-code gate.

The current gate summary is:

| quantity | value |
| --- | ---: |
| collider rows total | 136 |
| strict core | 51 |
| Collins envelope | 39 |
| high-`qT` candidates | 46 |
| external benchmarks available | 15 |
| external-pass non-strict rows | 13 |
| external-code agreement threshold | 5% |

## Pending work

The representative external-code benchmark is now complete for the current
collider families.  It validates the finite-tail implementation convention over
selected Tevatron and LHCb kinematics through `qT/Q ~ 0.30`.  It does not by
itself promote every unbenchmarked row to direct production use: rows with no
row-level external benchmark remain excluded from direct production use unless
they enter only through the Collins-style factorization-validity uncertainty
prescription.

The next optional extension would be dense row-by-row benchmarking for every
candidate collider row outside the strict core, but the present representative
benchmark resolves the LHCb `pp` forward-rapidity convention issue.
