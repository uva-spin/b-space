#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the convention-lock layer for v22.
# Run from the repository root after switching to the desired git branch.

ROOT="$(pwd)"
DEST="${ROOT}/v22"

if [[ -e "${DEST}/CONVENTIONS.md" ]]; then
  echo "Refusing to overwrite existing ${DEST}/CONVENTIONS.md" >&2
  exit 1
fi

mkdir -p \
  "${DEST}/src" \
  "${DEST}/tests" \
  "${DEST}/tools" \
  "${DEST}/outputs" \
  "${DEST}/configs"

touch "${DEST}/__init__.py"
touch "${DEST}/src/__init__.py"

cat > "${DEST}/conventions.json" <<'JSON'
{
  "schema_version": 1,
  "status": "phase_A_convention_lock",
  "published_object": {
    "name": "f_tilde_1_q_over_h",
    "arguments": ["x", "bT_GeV_inv", "mu_GeV", "zeta_GeV2"],
    "default_publication_point": {
      "mu": "Q",
      "zeta": "Q^2"
    },
    "b_space_dimension": "dimensionless",
    "k_space_dimension": "GeV^-2"
  },
  "fourier_transform": {
    "forward": "f_tilde(b)=integral d^2k exp(+i k.b) f(k)",
    "inverse": "f(k)=integral d^2b/(2pi)^2 exp(-i k.b) f_tilde(b)",
    "radial_inverse": "f(k)=1/(2pi) integral_0^infinity db b J0(k b) f_tilde(b)"
  },
  "dy_structure_function": {
    "radial_transform": "W(qT)=1/(2pi) integral_0^infinity db b J0(qT b) W_tilde(b)",
    "lo_flavor_sum": "sum_q e_q^2 [f_q/A f_qbar/B + f_qbar/A f_q/B]",
    "hard_factor_location": "multiplicative in W_tilde"
  },
  "collinear_pdf_input": {
    "lhapdf_method": "xfxQ",
    "conversion": "f_i(x,Q)=xfxQ(i,x,Q)/x"
  },
  "perturbative_series": {
    "internal_expansion_parameter": "a_s=alpha_s/(4*pi)",
    "coefficient_functions": "C=delta+sum_n a_s^n C^(n)",
    "hard_factor": "H=1+sum_n a_s^n H^(n)"
  },
  "collins_soper_convention": {
    "rapidity_equation": "d ln f_tilde / d ln sqrt(zeta) = K(b;mu)",
    "kernel_rg": "d K(b;mu) / d ln mu = -gamma_K(alpha_s)"
  },
  "nonperturbative_boundary": {
    "normalization": "F_NP(x,bT=0)=1",
    "small_b_behavior": "F_NP=1+O(bT^2 Lambda_QCD^2)",
    "kT_positivity_required": false
  },
  "external_validation": {
    "observable_benchmarks": ["MCFM", "DYTurbo"],
    "external_TMD_extractions_used_as_targets": false
  }
}
JSON

cat > "${DEST}/CONVENTIONS.md" <<'MD'
# v22 TMD conventions

## Scope

This file defines the object that v22 will calculate and eventually fit.
It is a convention lock, not a phenomenological fit and not a comparison
to another TMD extraction.

The external perturbative validation targets remain **MCFM** and
**DYTurbo** at the observable level. No published TMD extraction is used
as a calibration target.

## Published TMDPDF

The unpolarized quark TMDPDF is

\[
\widetilde f_{1,q/h}(x,b_T;\mu,\zeta).
\]

The default publication point is

\[
\mu=Q,\qquad \zeta=Q^2.
\]

The object is **not** multiplied by \(x\) or \(b_T\).

With the conventions below, the \(b_T\)-space TMD is dimensionless and
the \(k_T\)-space TMD has units of \({\rm GeV}^{-2}\).

## Fourier convention

\[
\widetilde f(x,\boldsymbol b_T)
=
\int d^2\boldsymbol k_T\,
e^{+i\boldsymbol k_T\cdot\boldsymbol b_T}
f(x,\boldsymbol k_T),
\]

\[
f(x,\boldsymbol k_T)
=
\int\frac{d^2\boldsymbol b_T}{(2\pi)^2}\,
e^{-i\boldsymbol k_T\cdot\boldsymbol b_T}
\widetilde f(x,\boldsymbol b_T).
\]

For azimuthally symmetric functions,

\[
f(x,k_T)
=
\frac{1}{2\pi}
\int_0^\infty db_T\,b_T
J_0(k_Tb_T)\widetilde f(x,b_T).
\]

Therefore,

\[
\int d^2\boldsymbol k_T\,f(x,k_T)
=
\widetilde f(x,b_T=0)
\]

when the integral exists in the regulated/perturbative sense relevant to
the chosen scheme.

## Drell--Yan structure function

The radial Fourier transform is

\[
W(q_T)
=
\frac{1}{2\pi}
\int_0^\infty db_T\,b_T J_0(q_Tb_T)
\widetilde W(b_T).
\]

At leading flavor structure,

\[
\widetilde W(b_T)
=
H_{\rm DY}
\sum_q e_q^2
\left[
\widetilde f_{q/A}(x_1,b_T)
\widetilde f_{\bar q/B}(x_2,b_T)
+
\widetilde f_{\bar q/A}(x_1,b_T)
\widetilde f_{q/B}(x_2,b_T)
\right].
\]

Electroweak prefactors, bin integration and the matched \(Y\) term are
observable-layer components and will be documented separately.

## Collinear PDF convention

LHAPDF returns `xfxQ(pid,x,Q) = x f_i(x,Q)`.  Every v22 OPE convolution
must explicitly divide by \(x\) before using a number as \(f_i(x,Q)\).

No variable named `pdf` may ambiguously hold `x f` in the v22 code.

## Perturbative expansion convention

Internally,

\[
a_s = \frac{\alpha_s}{4\pi}.
\]

Coefficient functions and the hard factor are expanded as

\[
C = \delta + \sum_{n\ge1}a_s^n C^{(n)},\qquad
H = 1 + \sum_{n\ge1}a_s^n H^{(n)}.
\]

Any formula imported from a convention using
\(\alpha_s/\pi\) or \(\alpha_s/(2\pi)\) must be converted explicitly and
covered by a unit test.

## Collins--Soper sign convention

\[
\frac{d\ln \widetilde f}{d\ln\sqrt{\zeta}}
=
K(b_T;\mu),
\]

\[
\frac{dK(b_T;\mu)}{d\ln\mu}
=
-\gamma_K(\alpha_s).
\]

An alternative \(D\)-kernel notation may be exposed only through an
explicit conversion function.  Silent sign or factor-of-two changes are
not allowed.

## Nonperturbative factor

For every fitted flavor structure,

\[
F_{\rm NP}(x,0)=1,\qquad
F_{\rm NP}(x,b_T)=1+\mathcal O(b_T^2\Lambda_{\rm QCD}^2).
\]

A momentum-space TMD or an auxiliary factor is not required to be
positive at every \(k_T\). Positivity is not used as a fit constraint.

## Phase-A validation order

1. Analytic Fourier and Gaussian tests.
2. LO OPE identity and explicit `xfx/x` conversion.
3. NLO OPE coefficient-function tests.
4. \(\mu\)- and \(\zeta\)-evolution path-independence tests.
5. Expansion of the resummed result against the validated singular
   fixed-order result.
6. Matched observable checks against MCFM and DYTurbo.
7. Experimental-bin integration checks.

The existing v21 `f_eff` object must not be relabeled as this v22 TMDPDF.
MD

cat > "${DEST}/OPEN_DECISIONS.md" <<'MD'
# v22 open decisions

These must be resolved before the first real-data fit.

- Exact TMD subtraction/renormalization scheme used for the NLO OPE coefficients.
- Heavy-flavor threshold prescription.
- Perturbative order of \(H_{\rm DY}\), anomalous dimensions and OPE coefficients.
- Small-\(b_T\) scale/profile prescription and its variation family.
- Large-\(b_T\) treatment of the Collins--Soper kernel.
- Exact experimental observable prefactors and bin integration.
- Matching prescription and \(q_T/Q\) fit cuts.
- Minimal identifiable DY-only flavor basis.
- Nuclear/isospin treatment and uncertainty.
MD

cat > "${DEST}/src/conventions.py" <<'PY'
"""Machine-readable v22 convention helpers.

This module contains no fit model.  It only locks normalization,
Fourier and leading-flavor conventions with analytically testable
functions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

from scipy.integrate import quad
from scipy.special import j0


TWO_PI = 2.0 * math.pi
FOUR_PI = 4.0 * math.pi

# Squared electric charges for light quarks, keyed by positive PDG id.
ELECTRIC_CHARGE_SQUARED: dict[int, float] = {
    1: 1.0 / 9.0,   # d
    2: 4.0 / 9.0,   # u
    3: 1.0 / 9.0,   # s
    4: 4.0 / 9.0,   # c
    5: 1.0 / 9.0,   # b
}


@dataclass(frozen=True)
class TMDArguments:
    """Arguments of the published b-space TMDPDF."""

    x: float
    bT_GeV_inv: float
    mu_GeV: float
    zeta_GeV2: float

    def validate(self) -> None:
        if not 0.0 < self.x < 1.0:
            raise ValueError("x must lie strictly between 0 and 1")
        if self.bT_GeV_inv < 0.0:
            raise ValueError("bT must be nonnegative")
        if self.mu_GeV <= 0.0:
            raise ValueError("mu must be positive")
        if self.zeta_GeV2 <= 0.0:
            raise ValueError("zeta must be positive")


def density_from_lhapdf_xfx(*, x: float, xfx: float) -> float:
    """Convert LHAPDF's x f(x,Q) return value into f(x,Q)."""

    if not 0.0 < x < 1.0:
        raise ValueError("x must lie strictly between 0 and 1")
    return float(xfx) / float(x)


def radial_inverse_hankel(
    f_tilde: Callable[[float], float],
    *,
    kT_GeV: float,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> float:
    """Compute f(k)=1/(2pi) int_0^inf db b J0(kb) f_tilde(b)."""

    if kT_GeV < 0.0:
        raise ValueError("kT must be nonnegative")

    value, _ = quad(
        lambda b: (
            b
            * j0(kT_GeV * b)
            * float(f_tilde(b))
            / TWO_PI
        ),
        0.0,
        math.inf,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=400,
    )
    return float(value)


def gaussian_b_space(*, width_GeV2: float, bT_GeV_inv: float) -> float:
    """Dimensionless exp(-width b^2) test TMD."""

    if width_GeV2 <= 0.0:
        raise ValueError("width must be positive")
    return math.exp(-width_GeV2 * bT_GeV_inv * bT_GeV_inv)


def gaussian_k_space_analytic(
    *,
    width_GeV2: float,
    kT_GeV: float,
) -> float:
    """Analytic inverse transform of exp(-width b^2)."""

    if width_GeV2 <= 0.0:
        raise ValueError("width must be positive")
    return (
        math.exp(
            -(kT_GeV * kT_GeV)
            / (4.0 * width_GeV2)
        )
        / (FOUR_PI * width_GeV2)
    )


def gaussian_dy_qt_analytic(
    *,
    width_a_GeV2: float,
    width_b_GeV2: float,
    qT_GeV: float,
) -> float:
    """Transform of the product of two Gaussian b-space TMDs."""

    return gaussian_k_space_analytic(
        width_GeV2=width_a_GeV2 + width_b_GeV2,
        kT_GeV=qT_GeV,
    )


def lo_dy_flavor_luminosity(
    pdf_a: Mapping[int, float],
    pdf_b: Mapping[int, float],
    *,
    flavors: tuple[int, ...] = (1, 2, 3),
) -> float:
    """LO charge-weighted q qbar luminosity.

    `pdf_a[pid]` and `pdf_b[pid]` are unweighted f_i(x,mu), not x f_i.
    Signed PDG ids denote quarks and antiquarks.
    """

    total = 0.0
    for pid in flavors:
        if pid <= 0:
            raise ValueError("flavors must contain positive quark PDG ids")
        charge2 = ELECTRIC_CHARGE_SQUARED[pid]
        total += charge2 * (
            float(pdf_a[pid]) * float(pdf_b[-pid])
            + float(pdf_a[-pid]) * float(pdf_b[pid])
        )
    return total
PY

cat > "${DEST}/tests/test_conventions.py" <<'PY'
from __future__ import annotations

import math

import pytest
from scipy.integrate import quad

from v22.src.conventions import (
    TWO_PI,
    TMDArguments,
    density_from_lhapdf_xfx,
    gaussian_b_space,
    gaussian_dy_qt_analytic,
    gaussian_k_space_analytic,
    lo_dy_flavor_luminosity,
    radial_inverse_hankel,
)


@pytest.mark.parametrize("kT", [0.0, 0.25, 1.0, 2.5, 5.0])
def test_single_gaussian_hankel_normalization(kT: float) -> None:
    width = 0.73

    numerical = radial_inverse_hankel(
        lambda b: gaussian_b_space(
            width_GeV2=width,
            bT_GeV_inv=b,
        ),
        kT_GeV=kT,
    )

    analytic = gaussian_k_space_analytic(
        width_GeV2=width,
        kT_GeV=kT,
    )

    assert numerical == pytest.approx(
        analytic,
        rel=2.0e-9,
        abs=2.0e-12,
    )


@pytest.mark.parametrize("qT", [0.0, 0.5, 2.0, 4.0])
def test_dy_product_of_two_gaussians(qT: float) -> None:
    width_a = 0.31
    width_b = 0.82
    total = width_a + width_b

    numerical = radial_inverse_hankel(
        lambda b: math.exp(-total * b * b),
        kT_GeV=qT,
    )

    analytic = gaussian_dy_qt_analytic(
        width_a_GeV2=width_a,
        width_b_GeV2=width_b,
        qT_GeV=qT,
    )

    assert numerical == pytest.approx(
        analytic,
        rel=2.0e-9,
        abs=2.0e-12,
    )


def test_gaussian_k_space_integrates_to_b_space_origin() -> None:
    width = 0.47

    value, _ = quad(
        lambda k: (
            TWO_PI
            * k
            * gaussian_k_space_analytic(
                width_GeV2=width,
                kT_GeV=k,
            )
        ),
        0.0,
        math.inf,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=300,
    )

    assert value == pytest.approx(1.0, rel=2.0e-11)


def test_lhapdf_xfx_conversion_is_explicit() -> None:
    x = 0.2
    f = 3.75
    xfx = x * f

    assert density_from_lhapdf_xfx(
        x=x,
        xfx=xfx,
    ) == pytest.approx(f)


def test_lo_dy_luminosity_is_symmetric_under_beam_exchange() -> None:
    a = {
        1: 1.7,
        -1: 0.21,
        2: 3.4,
        -2: 0.18,
        3: 0.12,
        -3: 0.11,
    }
    b = {
        1: 2.1,
        -1: 0.31,
        2: 2.8,
        -2: 0.27,
        3: 0.15,
        -3: 0.13,
    }

    ab = lo_dy_flavor_luminosity(a, b)
    ba = lo_dy_flavor_luminosity(b, a)

    assert ab == pytest.approx(ba, rel=1.0e-15)


def test_published_tmd_argument_validation() -> None:
    TMDArguments(
        x=0.1,
        bT_GeV_inv=1.2,
        mu_GeV=10.0,
        zeta_GeV2=100.0,
    ).validate()

    with pytest.raises(ValueError):
        TMDArguments(
            x=0.0,
            bT_GeV_inv=1.2,
            mu_GeV=10.0,
            zeta_GeV2=100.0,
        ).validate()
PY

cat > "${DEST}/tools/print_convention_summary.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "conventions.json"
with path.open() as handle:
    data = json.load(handle)

print(json.dumps(data, indent=2))
PY

chmod +x "${DEST}/tools/print_convention_summary.py"

cat > "${DEST}/README.md" <<'MD'
# v22 real-TMD development

The first milestone is a locked and tested convention layer.

Run:

```bash
python3 -m pytest -q v22/tests/test_conventions.py
python3 v22/tools/print_convention_summary.py
```

Do not begin a real-data fit until the OPE, evolution and observable
closure layers are implemented and independently tested.
MD

echo
echo "Created v22 Phase-A convention scaffold."
echo
echo "Run:"
echo "  python3 -m pytest -q v22/tests/test_conventions.py"
echo "  python3 v22/tools/print_convention_summary.py"
