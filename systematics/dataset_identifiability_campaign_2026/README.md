# Dataset and TMD identifiability campaign (2026)

This directory is the source record for the 2026 identifiability campaign.
The active result produced by that work is now the 96-start by 50-replica
lambda=1 package in
`../../production/lambda1_empirical_reference_full96x50/`, promoted on
2026-08-11.  The package README, `PRODUCTION_MANIFEST.json`, and
`PRODUCTION_AUDIT.json` are the authoritative current status and figure
records.

## Archived challenger protocol

The fixed-lambda-600 protocol in
`manifests/lambda600_fixed_challenger_protocol.json` is retained as an
archival preregistration.  It is not a live job queue or an authorization for
new runs.  `campaign_status.json` is a timestamped operational snapshot from
the early challenger stage (2026-08-06), not a current campaign status; do not
read its `in_progress` value as evidence that work is still running.

The challenger was never promoted over the lambda=1 production package.  No
other lambda, architecture, or smoothness prior is represented as production
by this directory.  Any future challenger must be separately authorized,
run under a new tag, and compared with the immutable package in `production/`.

## Completed lambda=1 result

- reference-distance domain: `0.1--2.0 GeV^-1`;
- reference-distance strength: `lambda=1`;
- stationary starts: 96;
- conditional experimental residual replicas: 50;
- crossed members: 4,800 per flavor;
- figures: the package's `champion_fig2space_*.png` and
  `champion_fig6_kT_*.png` files.

The propagated q16--q84 bands are operational empirical ensembles, not
calibrated Gaussian one-sigma confidence intervals.  Their completeness is
conditional on the declared perturbation family, objective, and replica
construction; it is not a proof that every possible model form has been
sampled.

## Isolation

- Existing production, benchmark, replica, registry, and candidate sources are
  read-only inputs.
- New campaign artifacts remain below this campaign directory.
- Fit quality alone cannot promote a candidate.
- Theory-nuisance variants remain distinct statistical models.
- Frozen production files remain unchanged.

See `HANDOFF.md` for the full scientific chronology, evidence, exact restart
logic, and detailed implementation notes.
