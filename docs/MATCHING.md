# Matching and perturbative backend

This document describes what “matching” means in this repository. There are
three related but distinct operations.

## 1. NLO hard/OPE matching in the b-space W term

The primary fixed-target object is the b-space `W` term. Its perturbative
construction is split into reusable layers:

| Layer | Public path | Role |
| --- | --- | --- |
| Conventions and Fourier/Hankel utilities | `v22/src/conventions.py` | Units, flavor labels, radial transforms, and low-order checks. |
| Scalar convolution reference | `v22/src/convolution.py` | Plus-distribution and regular convolution integrals. |
| NLO CSS2 OPE | `v22/src/css2_ope_nlo.py` | Canonical one-loop OPE coefficients and running-coupling factors. |
| General-scale OPE | `v22/src/css2_ope_nlo_general.py` | Scale/profile-dependent coefficient decomposition. |
| DY hard factor | `v22/src/dy_hard_nlo.py` | NLO Drell–Yan hard factor at the hard scale. |
| DY luminosity and W assembly | `v22/src/dy_w_nlo_reference.py` | NLO quark legs, luminosity decomposition, and W-term assembly. |
| Fixed-target backend | `v22/backends/bt_internal_css_backend_v22_full.py` | PDF evaluation, Sudakov evolution, OPE legs, hard factor, and row-level W grids. |
| Numeric W-expansion scheme | `v22/backends/bt_internal_css_backend_v22_scheme_y.py` | Alternate singular/W-expansion audit path. |
| Collider-aware wrapper | `v23/backends/bt_internal_css_backend_v22_tevatron.py` | pbar-p/pp luminosities, EW weights, bin integration, and developmental Y modes. |

For a single-hadron TMDPDF grid, the DY hard factor is intentionally not
included in the final one-hadron object. For a DY cross section, the hard
factor belongs to the two-hadron observable and is included by the DY W-term
backend.

The main data flow is:

```text
PDF member
   -> b* profile and perturbative scale
   -> NLO OPE convolution for each incoming leg
   -> NLO DY hard factor and flavor luminosity
   -> Sudakov evolution
   -> b-space W kernel
   -> finite-bT transform or fit prediction
```

The DNN does not learn this perturbative kernel. It learns only the constrained
nonperturbative factor `F_NP(x,bT)` and multiplies it into the perturbative
prediction during fitting.

## 2. “Matched” mode in the fit scripts

In the training scripts, `--mode matched` primarily selects the broader
kinematic selection used for the fixed-target workflow. The actual selection
is controlled by `--qT-max-over-Q`; the TMD-only selection uses
`--tmd-qT-max-over-Q`.

Therefore, `--mode matched` by itself does not prove that an audited high-qT
W+Y observable has been constructed.

The current fixed-target production commands use the W backend with
`--y-mode zero` for the accepted low-qT extraction. This is deliberate: the
published production claim is a fixed-target b-space extraction, not a
fully validated high-qT matched cross section.

## 3. Additive W+Y development route

The Tevatron-aware backend contains the developmental Y path in
`v23/backends/bt_internal_css_backend_v22_tevatron.py`:

```text
match_order=none / y_mode=zero
    Y = 0; W-only diagnostic or low-qT use

match_order=nlo_pilot or y_mode=nlo_pilot
    finite-tail pilot scaffold; not a production Y_NLO claim

match_order=nlo_dev
    developmental NLO finite-tail route with selectable singular component

y_mode=data_minus_w_debug
    Y = data - W debugging identity; not a prediction
```

The backend emits warnings for the pilot and developmental routes. Those
warnings are intentional and must not be removed when making plots. External
`DYTurbo`/`MCFM` comparisons and bin/covariance validation are required before
calling this route a production collider W+Y matching calculation.

## Matching audits

The source-level checks are:

```bash
PYTHONPATH=. python v22/tests/run_convolution_smoke.py
PYTHONPATH=. python v22/tests/run_css2_ope_nlo_smoke.py
PYTHONPATH=. python v22/tests/run_css2_ope_nlo_general_smoke.py
PYTHONPATH=. python v22/tests/run_dy_hard_nlo_smoke.py
PYTHONPATH=. python v22/tests/run_dy_w_nlo_reference_smoke.py
PYTHONPATH=. python v22/tools/audit_v22_full_backend_integration.py --help
PYTHONPATH=. python v22/tools/audit_v22_nlo_ope_insertion.py --help
PYTHONPATH=. python v22/tools/audit_v22_hard_plus_ope.py --help
```

The audit tools that require data, a PDF set, or a fitted model state are
intentionally not run by the lightweight source smoke test. They are described
in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## High-qT and unitary-transition research packages

The public source release also contains the experimental matching studies that
were previously only in the working tree:

```text
systematics/high_qt_direct_production_benchmark/experimental_matched_y/
systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/
systematics/finite_y_tail_benchmark/
```

These packages contain exact-bin/asymptotic Y prototypes, externally anchored
Y tests, W-cancellation pilots, differentiable boundary kernels, fixed-order
audits, and DYTurbo/MCFM comparison drivers. They are source and validation
packages, not part of the active fixed-target production claim. Their README
files define the gate status and external-program requirements.

The distinction is important:

```text
fixed-target production:
  NLO hard/OPE inserted in W; accepted low-qT extraction; Y set to zero

high-qT research:
  W/Y boundary and unitary transition prototypes; external validation pending
```
