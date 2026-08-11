# Experimental externally anchored matched Y

Status: experimental; not production-approved.

This subtree develops a new additive matched prediction without modifying the
accepted q020 production scheme. The frozen production W result is a read-only
reference. All new modules, tests, grids, fits, and manifests remain here.

The target bin-level construction is

```text
Y_bin = profile(qT/Q) * (FO_external_bin - ASY_consistent_bin)
matched_bin = W_production_scheme_bin + Y_bin
```

where:

- `FO_external_bin` is externally benchmarked using exact processed
  kinematics and fiducial cuts;
- `ASY_consistent_bin` is the fixed-order expansion of the same W scheme and
  uses the same bin integration and observable normalization;
- `profile` is explicit, versioned, and varied as a matching uncertainty;
- `W_production_scheme_bin` is read from or evaluated against frozen
  production references, never edited in place.

## Mandatory gates

1. Algebraic additive-matching closure tests.
2. Bitwise/hash provenance for protected production inputs.
3. Exact collider mass, rapidity, lepton-fiducial, beam, bin, and unit closure.
4. Small-qT cancellation of `FO - ASY` within numerical uncertainty.
5. Large-qT convergence of `W + Y` toward external fixed order.
6. Continuity and profile-variation audit across `qT/Q = 0.20`.
7. PDF and scale propagation using the completed external campaign.
8. New-tag central and replica fits compared to, but separate from, q020.

The initial module implements only the algebraic composition and validation
layer. It deliberately refuses silent broadcasting, non-finite components,
or profiles outside `[0,1]`. It does not claim that an asymptotic component is
available yet.

## Exact-bin asymptotic pilot

An experimental collider-aware strict-one-loop extractor now combines the
v22 scheme expansion with the v23 p/pbar luminosity and electroweak weights at
runtime. It performs explicit qT-bin averaging and rapidity integration on row
copies; neither source backend is modified.

First pilot: `CDF_RUN_2:36`.

- 2x2 to 3x3 qT/y quadrature shift at `n_b=320`: 0.014%
- `n_b=320` to `n_b=640` shift: 1.47%
- O(h^2) Richardson estimate: -70.65 pb/GeV
- external DYTurbo/MCFM FO average: 2.057 pb/GeV
- formal `FO-ASY`: +72.71 pb/GeV

The numerical ASY pilot passes its provisional convergence thresholds, but the
large cancellation is not physically approved.

## Exact-bin resummed-W cancellation pilot

The next mandatory audit was run on the same `CDF_RUN_2:36` bin using the
accepted central NP state as a read-only input. Dataset normalization nuisances
and learned global normalization were not applied. The perturbative and fitted
transforms share the same cached W kernel.

- fitted W at `n_b=160,320,640`: 3.541, 5.911, 6.419 pb/GeV
- fitted-W `n_b=320` to `640` shift: 8.24% (numerical gate fails)
- O(h^2) fitted-W estimate: 6.588 pb/GeV
- formal `FO-ASY`: 72.706 pb/GeV
- O(h^2) fitted matched estimate: 79.294 pb/GeV
- external fixed-order average: 2.057 pb/GeV

This is a decisive physical cancellation failure, not an approval result. The
large discrepancy is far greater than the residual b-grid drift. Work stops
before defining a matching profile or launching new fits: the strict ASY and
resummed-W normalization/sign/order conventions must first be closed in a
controlled expansion-level test. The accepted q020 production scheme remains
unchanged.

Pilot summary:

```text
outputs/resummed_w_cancellation_pilot/cdf_run_2_36/convergence_status.json
```

## Controlled expansion closure

The strict ASY convention itself passes. At the central node of
`CDF_RUN_2:36`, scaling every alpha_s evaluation by epsilon gives:

- strict ASY: -42.7543 pb/GeV
- finite-difference W slope at epsilon=0.001: -42.7327 pb/GeV
- relative closure error: 0.0505%

This rules out an internal ASY sign or normalization error at the tested node.
The additive failure instead comes from using a fully resummed W that is not
close to its strict expansion in the high-qT region.

The audit also detected a separate pilot-level issue: explicit rapidity
integration retained the Tevatron backend's inclusive rapidity approximation
factor (3.6224 for this row), double weighting rapidity. Earlier exact-bin
component magnitudes must therefore be treated as superseded diagnostics. This
does not repair cancellation because the same factor multiplies W and ASY.

```text
outputs/expansion_closure_pilot/cdf_run_2_36/result_nb160.json
```

The next safe experiment is a unitary transition construction that switches
from W to externally validated FO at high qT, rather than adding `FO-ASY` where
`W~=ASY` is demonstrably false. It must remain a separately tagged alternative
and cannot replace the accepted q020 production scheme without all gates.

LHCb high-qT rows remain intentionally unavailable in this extractor: the
existing DYTurbo acceptance table ends at qT=8.7 GeV and cannot support the
high-qT fiducial bins. A node-level decay/acceptance evaluator or new dense
high-qT fiducial grid is required.

Pilot summary:

```text
outputs/asymptotic_pilot/cdf_run_2_36/convergence_status.json
outputs/asymptotic_pilot/cdf_run_2_36/convergence.csv
```

## Protected production references

Hashes are generated by:

```bash
/home/dustin/miniforge3/envs/pdf-fit/bin/python \
  systematics/high_qt_direct_production_benchmark/experimental_matched_y/scripts/freeze_production_references.py
```

The resulting manifest is `manifests/protected_production_references.json`.
