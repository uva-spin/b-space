# Isolated full N³LL+NNLO W+Y campaign (2026)

This directory is the public, source-oriented handoff for the 2026 finite-
`Y` and perturbative-accuracy campaign. It is deliberately separate from
`production/lambda1_empirical_reference_full96x50/`: the frozen lambda=1
fixed-target production result is unchanged, and none of the candidates here
is promoted as a replacement.

## What is included

- `HANDOFF.md`: chronological handoff, decisions, open gates, and exact
  working-tree provenance.
- `config/`, `manifests/`: target definitions, freeze records, and candidate
  metadata.
- `scripts/`, `tests/`: the complete source used for the external DYTurbo
  grids, conventional and unitary matching tests, fit/replica pilots, figure
  construction, and audits.
- `reports/`: compact JSON status and decision records, the Tevatron candidate
  figures, the 329-row diagnostic Fig. 2/Fig. 6 and surface figures, and the
  selected start/replica comparison plots.

Large raw DYTurbo tables, per-start/per-replica loss histories, checkpoints,
and crossed-member long tables remain in the working archive. They are not
needed to inspect the decisions and are intentionally not committed to git.
The paths and hashes needed to reconnect them are recorded in the manifests
and handoff.

## Scientific status

The direct Tevatron grid contains 122 published rows (CDF Run I/II and D0
Run I) evaluated with the nominal unprimed N³LL+NNLO identity

```text
W = RES,  ASY = -CT,  FO = VJ,  Y = VJ + CT,
W + Y = RES + CT + VJ.
```

The grid is finite, positive, numerically checked, and has an approximately
1.15% mean Monte-Carlo integration uncertainty. It is an isolated
perturbative candidate, not a global production fit.

The larger 329-row diagnostic retained the six LHCb rows as W-only (`Y=0`)
and used finite `Y` only for the non-LHCb rows. Its 96-start × 50-replica
ensemble is useful for understanding model dependence, but the candidate
widens rather than improves the frozen lambda=1 bands and changes the central
`F_NP` shape. It is therefore explicitly marked *not promoted*. The current
LHCb finite-`Y` subtraction is excluded because cancellation and integration
uncertainties are not yet compatible with the released experimental errors and
covariance convention.

The conventional additive `FO-ASY` construction remains rejected at the
tested transition boundary because the resummed `W` is not close to its strict
asymptotic expansion there. This is a scope/domain decision, not a rejection
of the Tevatron grid or of further finite-`Y` work.

## Reproducibility boundary

The scripts expect the larger working-tree artifacts (DYTurbo installation,
PDF grids, backend caches, fit states, and raw reports). From a checkout, a
dependency-free source check is:

```bash
PYTHONPATH=. python -m py_compile \
  systematics/full_n3ll_wy_production_2026/scripts/plot_scope_329_refdist3_surface_fig7_fig8.py
```

Reproducing the numerical campaign requires the archived inputs named in
`HANDOFF.md`; the plotting scripts intentionally use those explicit paths and
are not stand-alone commands in a clean checkout. The figures are empirical q16--q84
ensemble summaries. Labels such as “combined relative 68% half-width” refer
to the declared percentile construction and must not be read as calibrated
Gaussian confidence intervals.

For the paper-facing subsection and figure captions, see
`reports/WplusY_new_section_PRD_handoff.md`.
