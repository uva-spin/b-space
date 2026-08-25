#!/usr/bin/env bash
set -euo pipefail

# Install the canonical-scale one-loop CSS2 quark-TMD OPE layer.
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
SCHEME_DOC="${ROOT}/v22/CSS2_SCHEME.md"
SRC="${ROOT}/v22/src/css2_ope_nlo.py"
TEST="${ROOT}/v22/tests/run_css2_ope_nlo_smoke.py"

for required in \
  "${ROOT}/v22/CONVENTIONS.md" \
  "${ROOT}/v22/src/conventions.py" \
  "${ROOT}/v22/src/convolution.py"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required v22 file: ${required}" >&2
    exit 1
  fi
done

for target in "${SCHEME_DOC}" "${SRC}" "${TEST}"; do
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite existing ${target}" >&2
    exit 1
  fi
done

cat > "${SCHEME_DOC}" <<'MD'
# v22 one-loop CSS2 OPE scheme lock

## Purpose

This file locks the first explicit perturbative definition of the v22
quark TMDPDF. It is a theory convention, not a fit to another TMD
extraction.

The observable-level validation targets remain MCFM and DYTurbo.

## Expansion parameter

\[
a_s(\mu)=\frac{\alpha_s(\mu)}{4\pi}.
\]

## Canonical small-\(b_T\) scales

\[
b_0 = 2e^{-\gamma_E},\qquad
\mu_b=\frac{b_0}{b_T},\qquad
\zeta_b=\mu_b^2.
\]

A regulated/profiled version of \(\mu_b\) will be introduced only after
the canonical implementation passes its analytic checks.

## Small-\(b_T\) OPE

For a quark or antiquark flavor \(q\),

\[
\widetilde f_{q/H}(x,b_T;\mu_b,\zeta_b)
=
\sum_j
\left[
\widetilde C_{q/j}\otimes f_{j/H}
\right](x,\mu_b)
+\mathcal O(b_T^2\Lambda_{\rm QCD}^2).
\]

At one loop in the CSS2 convention and at the canonical scales,

\[
\widetilde C_{q/q}(z)
=
\delta(1-z)
+
a_s C_F
\left[
-\frac{\pi^2}{6}\delta(1-z)
+
2(1-z)
\right],
\]

\[
\widetilde C_{q/g}(z)
=
a_s\,2z(1-z),
\]

and the other flavor channels vanish at this order.

The convolution convention is

\[
(C\otimes f)(x)
=
\int_x^1\frac{dz}{z}\,
C(z)f(x/z).
\]

## Separation from the hard factor

The \(-\pi^2 C_F/6\) delta term belongs to the CSS2 TMD matching
coefficient in this scheme. It must not be silently moved into the
Drell--Yan hard factor.

The hard factor will be implemented and tested in a separate milestone.

## Scope of this milestone

This layer implements only:

- canonical-scale one-loop \(q\leftarrow q\) matching;
- canonical-scale one-loop \(q\leftarrow g\) matching;
- decomposition into Born, quark and gluon pieces;
- scalar high-accuracy reference evaluation.

It does not yet implement:

- general \((\mu,\zeta)\) logarithms;
- Collins--Soper evolution;
- the Drell--Yan hard factor;
- heavy-flavor matching;
- a \(b_*\) or scale profile;
- nonperturbative functions;
- a real-data fit.
MD

cat > "${SRC}" <<'PY'
r"""Canonical-scale one-loop CSS2 matching for an unpolarized quark TMD.

The implementation uses

    a_s = alpha_s / (4*pi),

and the canonical scales

    mu_b = b0 / bT,
    zeta_b = mu_b**2,
    b0 = 2 exp(-gamma_E).

At these scales,

    C_qq^(1)(z) =
        C_F[-pi^2/6 delta(1-z) + 2(1-z)],

    C_qg^(1)(z) =
        2 z (1-z).

The input callables return ordinary collinear densities f(x,mu), not
LHAPDF's x*f(x,mu).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from v22.src.convolution import convolve_regular


PDF = Callable[[float], float]

CF = 4.0 / 3.0
B0 = 2.0 * math.exp(-float(math.euler_gamma)) if hasattr(math, "euler_gamma") else 2.0 * math.exp(-0.5772156649015329)


@dataclass(frozen=True)
class CanonicalCSS2Scales:
    bT_GeV_inv: float
    mu_GeV: float
    zeta_GeV2: float


@dataclass(frozen=True)
class CanonicalOPEComponents:
    x: float
    alpha_s: float
    a_s: float
    born_quark: float
    one_loop_qq_delta: float
    one_loop_qq_regular: float
    one_loop_qg_regular: float
    one_loop_total: float
    matched_tmd: float


def _validate_x(x: float) -> float:
    x = float(x)
    if not 0.0 < x < 1.0:
        raise ValueError("x must lie strictly between 0 and 1")
    return x


def _checked_density(pdf: PDF, x: float) -> float:
    value = float(pdf(float(x)))
    if not math.isfinite(value):
        raise FloatingPointError(
            f"collinear density returned nonfinite value at x={x}"
        )
    return value


def a_s_from_alpha_s(alpha_s: float) -> float:
    alpha_s = float(alpha_s)
    if not math.isfinite(alpha_s) or alpha_s < 0.0:
        raise ValueError("alpha_s must be finite and nonnegative")
    return alpha_s / (4.0 * math.pi)


def canonical_css2_scales(bT_GeV_inv: float) -> CanonicalCSS2Scales:
    bT = float(bT_GeV_inv)
    if not math.isfinite(bT) or bT <= 0.0:
        raise ValueError("bT must be finite and strictly positive")

    mu = B0 / bT

    return CanonicalCSS2Scales(
        bT_GeV_inv=bT,
        mu_GeV=mu,
        zeta_GeV2=mu * mu,
    )


def c_qq_1_delta_coefficient() -> float:
    """Coefficient multiplying delta(1-z) in C_qq^(1)."""

    return -CF * math.pi * math.pi / 6.0


def c_qq_1_regular(z: float) -> float:
    """Ordinary regular part of C_qq^(1)."""

    z = float(z)
    if not 0.0 <= z <= 1.0:
        raise ValueError("z must lie in [0,1]")
    return 2.0 * CF * (1.0 - z)


def c_qg_1_regular(z: float) -> float:
    """Ordinary C_qg^(1) coefficient."""

    z = float(z)
    if not 0.0 <= z <= 1.0:
        raise ValueError("z must lie in [0,1]")
    return 2.0 * z * (1.0 - z)


def canonical_css2_quark_ope_nlo_components(
    *,
    x: float,
    alpha_s: float,
    quark_pdf: PDF,
    gluon_pdf: PDF,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> CanonicalOPEComponents:
    """Return the canonical-scale one-loop OPE decomposition.

    The result is

        f_q
        + a_s [
            C_qq^(1) tensor f_q
            + C_qg^(1) tensor f_g
          ].

    The same function applies to a quark or its antiquark; pass the
    corresponding signed-flavor collinear density as `quark_pdf`.
    """

    x = _validate_x(x)
    a_s = a_s_from_alpha_s(alpha_s)

    born = _checked_density(quark_pdf, x)

    qq_delta = (
        c_qq_1_delta_coefficient()
        * born
    )

    qq_regular = convolve_regular(
        quark_pdf,
        x=x,
        kernel=c_qq_1_regular,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    qg_regular = convolve_regular(
        gluon_pdf,
        x=x,
        kernel=c_qg_1_regular,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    one_loop_total = (
        qq_delta
        + qq_regular
        + qg_regular
    )

    matched = born + a_s * one_loop_total

    return CanonicalOPEComponents(
        x=x,
        alpha_s=float(alpha_s),
        a_s=a_s,
        born_quark=born,
        one_loop_qq_delta=qq_delta,
        one_loop_qq_regular=qq_regular,
        one_loop_qg_regular=qg_regular,
        one_loop_total=one_loop_total,
        matched_tmd=matched,
    )


def canonical_css2_quark_ope_nlo(
    *,
    x: float,
    alpha_s: float,
    quark_pdf: PDF,
    gluon_pdf: PDF,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> float:
    """Return the matched canonical-scale quark TMD OPE."""

    return canonical_css2_quark_ope_nlo_components(
        x=x,
        alpha_s=alpha_s,
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=epsabs,
        epsrel=epsrel,
    ).matched_tmd
PY

cat > "${TEST}" <<'PY'
from __future__ import annotations

import math

from scipy.integrate import quad

from v22.src.css2_ope_nlo import (
    B0,
    CF,
    a_s_from_alpha_s,
    c_qg_1_regular,
    c_qq_1_delta_coefficient,
    c_qq_1_regular,
    canonical_css2_quark_ope_nlo,
    canonical_css2_quark_ope_nlo_components,
    canonical_css2_scales,
)


def assert_close(
    actual: float,
    expected: float,
    *,
    rel: float = 2.0e-9,
    abs_tol: float = 2.0e-11,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=rel,
        abs_tol=abs_tol,
    ):
        raise AssertionError(
            f"actual={actual:.16e}, "
            f"expected={expected:.16e}"
        )


checks = 0


# 1-3: canonical scales.
for bT in [0.1, 1.0, 3.0]:
    scales = canonical_css2_scales(bT)

    assert_close(
        scales.mu_GeV,
        B0 / bT,
    )
    assert_close(
        scales.zeta_GeV2,
        scales.mu_GeV**2,
    )
    checks += 1


# 4: expansion parameter.
alpha_s = 0.24
assert_close(
    a_s_from_alpha_s(alpha_s),
    alpha_s / (4.0 * math.pi),
)
checks += 1


# 5-7: coefficient constants and endpoint behavior.
assert_close(
    c_qq_1_delta_coefficient(),
    -CF * math.pi**2 / 6.0,
)
checks += 1

assert_close(
    c_qq_1_regular(0.25),
    2.0 * CF * 0.75,
)
checks += 1

assert_close(
    c_qg_1_regular(0.25),
    2.0 * 0.25 * 0.75,
)
checks += 1


# 8: alpha_s=0 gives the Born collinear density.
x = 0.31
born = x**0.4 * (1.0 - x)**2

result = canonical_css2_quark_ope_nlo(
    x=x,
    alpha_s=0.0,
    quark_pdf=lambda y: y**0.4 * (1.0 - y)**2,
    gluon_pdf=lambda y: 7.0 * y**-0.1 * (1.0 - y)**5,
)

assert_close(result, born)
checks += 1


# 9: constant quark PDF and zero gluon PDF have a closed form.
x = 0.2
alpha_s = 0.27
a_s = alpha_s / (4.0 * math.pi)

expected_qq_regular = (
    2.0
    * CF
    * (
        -math.log(x)
        - 1.0
        + x
    )
)

expected_one_loop = (
    -CF * math.pi**2 / 6.0
    + expected_qq_regular
)

expected = 1.0 + a_s * expected_one_loop

components = canonical_css2_quark_ope_nlo_components(
    x=x,
    alpha_s=alpha_s,
    quark_pdf=lambda y: 1.0,
    gluon_pdf=lambda y: 0.0,
)

assert_close(
    components.one_loop_qq_regular,
    expected_qq_regular,
)
assert_close(
    components.matched_tmd,
    expected,
)
checks += 1


# 10: zero quark PDF and constant gluon PDF have qg=(1-x)^2.
x = 0.37
alpha_s = 0.19
a_s = alpha_s / (4.0 * math.pi)

components = canonical_css2_quark_ope_nlo_components(
    x=x,
    alpha_s=alpha_s,
    quark_pdf=lambda y: 0.0,
    gluon_pdf=lambda y: 1.0,
)

assert_close(
    components.one_loop_qg_regular,
    (1.0 - x)**2,
)
assert_close(
    components.matched_tmd,
    a_s * (1.0 - x)**2,
)
checks += 1


# 11: independent direct quadrature for nontrivial toy PDFs.
x = 0.43
alpha_s = 0.21
a_s = alpha_s / (4.0 * math.pi)

quark = lambda y: y**0.6 * (1.0 - y)**2
gluon = lambda y: 2.3 * y**-0.15 * (1.0 - y)**4

qq_regular_direct, _ = quad(
    lambda z: (
        2.0
        * CF
        * (1.0 - z)
        * quark(x / z)
        / z
    ),
    x,
    1.0,
    epsabs=1.0e-12,
    epsrel=1.0e-12,
    limit=400,
)

qg_direct, _ = quad(
    lambda z: (
        2.0
        * z
        * (1.0 - z)
        * gluon(x / z)
        / z
    ),
    x,
    1.0,
    epsabs=1.0e-12,
    epsrel=1.0e-12,
    limit=400,
)

direct = (
    quark(x)
    + a_s
    * (
        -CF
        * math.pi**2
        / 6.0
        * quark(x)
        + qq_regular_direct
        + qg_direct
    )
)

reference = canonical_css2_quark_ope_nlo(
    x=x,
    alpha_s=alpha_s,
    quark_pdf=quark,
    gluon_pdf=gluon,
)

assert_close(reference, direct)
checks += 1


# 12: decomposition sums exactly.
components = canonical_css2_quark_ope_nlo_components(
    x=0.28,
    alpha_s=0.22,
    quark_pdf=lambda y: y**0.3 * (1.0 - y)**3,
    gluon_pdf=lambda y: 1.8 * y**-0.2 * (1.0 - y)**5,
)

assert_close(
    components.one_loop_total,
    (
        components.one_loop_qq_delta
        + components.one_loop_qq_regular
        + components.one_loop_qg_regular
    ),
)
assert_close(
    components.matched_tmd,
    (
        components.born_quark
        + components.a_s
        * components.one_loop_total
    ),
)
checks += 1


# 13: invalid x is rejected.
try:
    canonical_css2_quark_ope_nlo(
        x=1.0,
        alpha_s=0.2,
        quark_pdf=lambda y: 1.0,
        gluon_pdf=lambda y: 1.0,
    )
except ValueError:
    pass
else:
    raise AssertionError("x=1 was not rejected")

checks += 1


print(f"{checks} canonical CSS2 NLO OPE checks passed")
PY

touch \
  "${ROOT}/v22/__init__.py" \
  "${ROOT}/v22/src/__init__.py"

python3 -m py_compile \
  "${SRC}" \
  "${TEST}"

cd "${ROOT}"

PYTHONPATH=. \
python3 \
  v22/tests/run_css2_ope_nlo_smoke.py

echo
echo "Created:"
echo "  v22/CSS2_SCHEME.md"
echo "  v22/src/css2_ope_nlo.py"
echo "  v22/tests/run_css2_ope_nlo_smoke.py"
