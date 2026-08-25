# Isolated finite-Y completion campaign (2026)

This directory is an isolated decision package for a finite-
`Y`/high-`q_T` extension. It does not modify the accepted `q_T/Q <= 0.20`
production extraction, its frozen W cache, the lambda=1 identifiability
package, or the paper.

## Candidate under test

The ordinary additive construction

```text
W + (FO - ASY)
```

was tested previously and failed because the resummed W is not close to its
fixed-order asymptotic expansion in the transition region. The candidate
here is an explicit unitary finite-Y correction:

```text
Y_unitary = p(qT/Q) * (FO_NLO - W)
matched   = W + Y_unitary
          = (1-p) W + p FO_NLO
```

The C2 smootherstep profile is zero in the TMD core and one above the
transition window. This is a phenomenological unitary transition, not the
claim that the conventional perturbative `Y = FO - ASY` construction has
been rescued.

## Promotion gates

1. Algebraic endpoint and C2 continuity tests.
2. Exact row/bin and unit provenance.
3. Node-level W numerical convergence.
4. Genuine fixed-order NLO input with scale uncertainty.
5. Matched result tends exactly to W in the core and FO at high `qT/Q`.
6. Profile-window variation is recorded as a model uncertainty.
7. Tevatron central fit impact is assessed separately from construction
   validity; replicas and production promotion remain separate decisions.
8. LHCb/other fiducial datasets require their own acceptance closure.

The current target is a valid isolated Tevatron finite-Y construction. A
universal all-dataset production claim is not implied until fiducial
acceptance and fit-impact gates are separately closed.
