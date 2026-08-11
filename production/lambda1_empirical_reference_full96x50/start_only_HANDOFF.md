# Lambda=1 96-start expansion handoff

Updated: 2026-08-11

## Scope and completion

This isolated campaign expanded the lambda=1 start-only diagnostic from 48 to
96 members by adding the paired perturbation starts `351--398`.  Every fresh
start was run with the unchanged direct-FNP reference-distance objective and
was continued until the declared FNP stationarity gate or the cumulative
horizon.  All 48 fresh starts passed the gate.  The ledger contains 144
terminal blocks (no running or failed rows).

Frozen production outputs and experimental replicas were not modified or
launched.  This is a start-only non-uniqueness study; its quantiles are not a
68% confidence interval and do not include replica uncertainty.

## Fixed protocol

- Objective: unchanged lambda=1 direct-FNP reference distance.
- Reference-distance region: `x=0.1`, `0.1 <= b_T <= 2.0`.
- Fresh initialization: two paired 1% seeded Gaussian perturbation sets.
- Blocks: 40,000 epochs; minimum 10,000; plateau patience 5,000.
- Learning rate: `2e-5`; per-fit L-BFGS closure ceiling: 20,000.
- Maximum cumulative exposure: 320,000 epochs.
- FNP gate: maximum relative block-to-block drift `<=2%`, after at least
  80,000 epochs and two consecutive quiet blocks.

## Width results

The reported widths are the maximum pointwise `q16--q84` full width divided by
the median, restricted to the accepted curve above 5% of its displayed peak.

| ensemble | max b-space width | max k-space width, u | max k-space width, d |
|---:|---:|---:|---:|
| 24 starts | 13.524% | 10.193% | 10.817% |
| 48 starts | 22.479% | 20.409% | 21.572% |
| 96 starts | 22.802% | 20.799% | 21.972% |

The 96-to-48 ratios are 1.0144 in b-space, 1.0191 for u in k-space, and
1.0186 for d in k-space.  Thus the dominant increase occurred when going from
24 to 48 starts (approximately a factor of two in the k-space widths); the
additional 48 starts produce only a roughly 1.5--2% increase.  The maximum
full start range, which is more sensitive to rare endpoints, changes from
43.315% to 50.041% to 50.695% in b-space for 24, 48, and 96 starts.

Interpretation: under this perturbation and optimization protocol, the
non-uniqueness distribution is substantially broader than the original
24-start estimate, but is close to saturated by 96 starts.  This does not
prove that other initialization families or model forms cannot reveal further
solutions.

## Authoritative artifacts

- `summary.json`: completion status, gate results, widths, ratios, hashes.
- `protocol.json`: frozen inputs and training/gate protocol.
- `runs.csv`: per-block terminal status and drift evidence.
- `bspace_24_start_only_bands.csv`, `bspace_48_start_only_bands.csv`,
  `bspace_96_start_only_bands.csv`.
- `kspace_24_start_only_members.csv`, `kspace_48_start_only_members.csv`,
  `kspace_96_start_only_members.csv` (quantile summaries retaining the u/d
  k-space projections).

The analysis implementation is
`dataset_identifiability_campaign_2026/scripts/run_lambda1_start_expansion96.py`.
The summarizer was byte-compiled and rerun after correcting its pandas column
access in the final audit.  `summary.json` reports
`status=complete`, `all_new_starts_pass_fnp_stationarity_gate=true`,
`replicas_launched=false`, and `production_sources_modified=false`.
