# Dataset and TMD identifiability campaign (2026)

This isolated campaign determines how much non-physics constraint is required
to obtain an identifiable `F_NP` while preserving the cross-section fit, then
propagates experimental and residual non-uniqueness effects to Fig. 2 and
Fig. 6. Historical dataset, architecture, and prior studies are retained in
`HANDOFF.md`.

## Current locked comparison

- The unchanged incumbent is the full-24-start empirical-reference result with
  reference-distance `lambda=1` and `b_T` domain `0.1--2.0 GeV^-1`.
- The only active challenger is the production-capacity FiLM model with
  reference-distance `lambda=600`, domain `0.1--4.0 GeV^-1`, and the fixed
  power-2, strength-100 fit-quality barrier.
- Lambda 600 is one fixed incumbent-replacement test. It is not another rung
  in a live lambda ladder and does not by itself establish the minimum prior.
- No other lambda, smoothness prior, or architecture is authorized before the
  complete lambda-600 verdict.

The immutable protocol is
`manifests/lambda600_fixed_challenger_protocol.json`. The machine-readable live
state is `campaign_status.json`; its progress block is explicitly timestamped
and source-hashed. A read-only five-minute observer publishes current
full-24 operational progress to
`summaries/lambda600_live_progress/summary.json`; that file is explicitly
provisional and non-promotable and is never used by a scientific decision
gate.

## Required order

1. Finish and provenance-audit exactly 24 long-horizon challenger starts.
2. Select one candidate central and train it through 300,000 requested
   capacity.
3. Finish and provenance-audit all 50 experimental replicas.
4. Form the 1,200-member centered-log-`F_NP` start-by-replica product ensemble.
5. Complete the stratified nested-interaction check and directional
   convergence envelopes.
6. Render the final candidate Fig. 2 and Fig. 6 and compare their complete
   propagated widths directly with the locked lambda-1 incumbent.
7. Promote or reject lambda 600; only then decide the next controlled study.

Intermediate start-only or replica-only widths cannot replace the incumbent.
Scientific failures are retained and do not truncate the requested evidence
collection.

## Final-result semantics

The plotted central is the audited 300,000-capacity trained central and its
paired transform, not a pointwise ensemble median. The final band contains the
24-by-50 product distribution plus directional convergence and nested-
interaction envelopes. It is not assigned a `1 sigma` or 68% confidence-level
interpretation.

The lambda-600 verdict is a checkpoint, not the end of the broader study. A
passing challenger becomes the new incumbent and systematic work continues
from it; a rejected challenger can motivate a separately authorized controlled
trial only after the complete verdict is recorded.

## Isolation

- Existing production, benchmark, replica, registry, and candidate sources are
  read-only inputs.
- New campaign artifacts remain below this campaign directory.
- Fit quality alone cannot promote a candidate.
- Theory-nuisance variants remain distinct statistical models.
- Frozen production files remain unchanged.

See `HANDOFF.md` for the full scientific chronology, evidence, exact restart
logic, and detailed implementation notes.
