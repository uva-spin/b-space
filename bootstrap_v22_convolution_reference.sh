#!/usr/bin/env bash
set -euo pipefail

# Add the scheme-independent Mellin-convolution reference layer for v22.
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
SRC="${ROOT}/v22/src/convolution.py"
TEST="${ROOT}/v22/tests/run_convolution_smoke.py"
DOC="${ROOT}/v22/OPE_CONVOLUTION.md"

for required in \
  "${ROOT}/v22/CONVENTIONS.md" \
  "${ROOT}/v22/src/conventions.py"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required convention-layer file: ${required}" >&2
    exit 1
  fi
done

for target in "${SRC}" "${TEST}" "${DOC}"; do
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite existing ${target}" >&2
    exit 1
  fi
done

mkdir -p "${ROOT}/v22/src" "${ROOT}/v22/tests"

cat > "${SRC}" <<'PY'
r"""Reference Mellin-convolution machinery for v22 TMD matching.

The functions in this module are deliberately scalar and SciPy-based.
They are a high-accuracy reference implementation for validating the
later vectorized/differentiable training implementation.

Convention:

    (K \otimes f)(x) = \int_x^1 dz/z K(z) f(x/z).

For a plus distribution [g(z)]_+,

    \int_x^1 dz [g(z)]_+ phi(z)
      = \int_x^1 dz g(z) [phi(z)-phi(1)]
        - phi(1) \int_0^x dz g(z),

with phi(z)=f(x/z)/z.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from scipy.integrate import quad


PDF = Callable[[float], float]
Kernel = Callable[[float], float]
PrimitiveZeroToX = Callable[[float], float]


@dataclass(frozen=True)
class RegularTerm:
    coefficient: float
    kernel: Kernel
    name: str = "regular"


@dataclass(frozen=True)
class PlusTerm:
    coefficient: float
    kernel: Kernel
    integral_zero_to_x: PrimitiveZeroToX
    name: str = "plus"


@dataclass(frozen=True)
class DistributionKernel:
    delta_coefficient: float = 0.0
    regular_terms: tuple[RegularTerm, ...] = ()
    plus_terms: tuple[PlusTerm, ...] = ()


def _validate_x(x: float) -> float:
    x = float(x)

    if not 0.0 < x < 1.0:
        raise ValueError("x must lie strictly between 0 and 1")

    return x


def _checked_pdf(pdf: PDF, y: float) -> float:
    value = float(pdf(float(y)))

    if not math.isfinite(value):
        raise FloatingPointError(
            f"PDF returned nonfinite value at y={y}"
        )

    return value


def convolve_delta(
    pdf: PDF,
    *,
    x: float,
    coefficient: float = 1.0,
) -> float:
    """Convolution with coefficient times delta(1-z)."""

    x = _validate_x(x)

    return float(coefficient) * _checked_pdf(pdf, x)


def convolve_regular(
    pdf: PDF,
    *,
    x: float,
    kernel: Kernel,
    coefficient: float = 1.0,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
    limit: int = 400,
) -> float:
    """Convolution with an ordinary integrable kernel."""

    x = _validate_x(x)

    def integrand(z: float) -> float:
        kernel_value = float(kernel(z))

        if not math.isfinite(kernel_value):
            raise FloatingPointError(
                "regular kernel returned nonfinite value "
                f"at z={z}"
            )

        return (
            kernel_value
            * _checked_pdf(pdf, x / z)
            / z
        )

    value, _ = quad(
        integrand,
        x,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=int(limit),
    )

    return float(coefficient) * float(value)


def convolve_plus(
    pdf: PDF,
    *,
    x: float,
    kernel: Kernel,
    integral_zero_to_x: PrimitiveZeroToX,
    coefficient: float = 1.0,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
    limit: int = 400,
) -> float:
    r"""Convolution with a plus distribution [kernel(z)]_+.

    `integral_zero_to_x(x)` must return

        \int_0^x dz kernel(z)

    for x<1. The endpoint singularity is cancelled analytically before
    quadrature.
    """

    x = _validate_x(x)
    phi_at_one = _checked_pdf(pdf, x)

    def phi(z: float) -> float:
        return _checked_pdf(pdf, x / z) / z

    def subtracted_integrand(z: float) -> float:
        kernel_value = float(kernel(z))
        difference = phi(z) - phi_at_one
        value = kernel_value * difference

        if not math.isfinite(value):
            raise FloatingPointError(
                "subtracted plus-distribution integrand became "
                f"nonfinite at z={z}"
            )

        return value

    integral_subtracted, _ = quad(
        subtracted_integrand,
        x,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=int(limit),
    )

    subtraction = (
        phi_at_one
        * float(integral_zero_to_x(x))
    )

    return float(coefficient) * (
        float(integral_subtracted)
        - subtraction
    )


def convolve_distribution(
    pdf: PDF,
    *,
    x: float,
    distribution: DistributionKernel,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
    limit: int = 400,
) -> float:
    """Convolve a sum of delta, regular and plus-distribution terms."""

    total = convolve_delta(
        pdf,
        x=x,
        coefficient=distribution.delta_coefficient,
    )

    for term in distribution.regular_terms:
        total += convolve_regular(
            pdf,
            x=x,
            kernel=term.kernel,
            coefficient=term.coefficient,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )

    for term in distribution.plus_terms:
        total += convolve_plus(
            pdf,
            x=x,
            kernel=term.kernel,
            integral_zero_to_x=term.integral_zero_to_x,
            coefficient=term.coefficient,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )

    return float(total)


def kernel_inv_one_minus_z(z: float) -> float:
    return 1.0 / (1.0 - float(z))


def integral_inv_one_minus_z_zero_to_x(
    x: float,
) -> float:
    x = _validate_x(x)

    return -math.log1p(-x)


def kernel_log_one_minus_z_over_one_minus_z(
    z: float,
) -> float:
    z = float(z)

    return math.log1p(-z) / (1.0 - z)


def integral_log_one_minus_z_over_one_minus_z_zero_to_x(
    x: float,
) -> float:
    x = _validate_x(x)

    return -0.5 * math.log1p(-x) ** 2
PY

cat > "${TEST}" <<'PY'
from __future__ import annotations

import math

from v22.src.convolution import (
    DistributionKernel,
    PlusTerm,
    RegularTerm,
    convolve_delta,
    convolve_distribution,
    convolve_plus,
    convolve_regular,
    integral_inv_one_minus_z_zero_to_x,
    integral_log_one_minus_z_over_one_minus_z_zero_to_x,
    kernel_inv_one_minus_z,
    kernel_log_one_minus_z_over_one_minus_z,
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


# 1: delta(1-z) is the identity under Mellin convolution.
x = 0.27

assert_close(
    convolve_delta(
        lambda y: y**0.7,
        x=x,
    ),
    x**0.7,
)

checks += 1


# 2-5: regular kernel K(z)=1 with f(y)=y^p.
power = 0.7

for x in [0.05, 0.2, 0.6, 0.93]:
    numerical = convolve_regular(
        lambda y: y**power,
        x=x,
        kernel=lambda z: 1.0,
    )

    analytic = (
        1.0 - x**power
    ) / power

    assert_close(
        numerical,
        analytic,
    )

    checks += 1


# 6-9: [1/(1-z)]_+ with f(y)=1.
for x in [0.1, 0.3, 0.7, 0.95]:
    numerical = convolve_plus(
        lambda y: 1.0,
        x=x,
        kernel=kernel_inv_one_minus_z,
        integral_zero_to_x=(
            integral_inv_one_minus_z_zero_to_x
        ),
    )

    analytic = math.log(
        (1.0 - x) / x
    )

    assert_close(
        numerical,
        analytic,
    )

    checks += 1


# 10-13: [1/(1-z)]_+ with f(y)=y.
for x in [0.1, 0.3, 0.7, 0.95]:
    numerical = convolve_plus(
        lambda y: y,
        x=x,
        kernel=kernel_inv_one_minus_z,
        integral_zero_to_x=(
            integral_inv_one_minus_z_zero_to_x
        ),
    )

    analytic = (
        1.0
        - x
        + x
        * math.log(
            (1.0 - x) / x
        )
    )

    assert_close(
        numerical,
        analytic,
    )

    checks += 1


# 14-17: [ln(1-z)/(1-z)]_+ with f(y)=1/y.
#
# Here phi(z)=f(x/z)/z=1/x is constant, so the result is analytic.
for x in [0.1, 0.3, 0.7, 0.95]:
    numerical = convolve_plus(
        lambda y: 1.0 / y,
        x=x,
        kernel=(
            kernel_log_one_minus_z_over_one_minus_z
        ),
        integral_zero_to_x=(
            integral_log_one_minus_z_over_one_minus_z_zero_to_x
        ),
    )

    analytic = (
        0.5
        * math.log1p(-x) ** 2
        / x
    )

    assert_close(
        numerical,
        analytic,
    )

    checks += 1


# 18: a composite distribution equals the explicit sum of its pieces.
x = 0.42
pdf = lambda y: y**0.4 * (1.0 - y) ** 2

distribution = DistributionKernel(
    delta_coefficient=1.7,
    regular_terms=(
        RegularTerm(
            coefficient=-0.3,
            kernel=lambda z: 1.0 + z,
            name="1+z",
        ),
    ),
    plus_terms=(
        PlusTerm(
            coefficient=0.8,
            kernel=kernel_inv_one_minus_z,
            integral_zero_to_x=(
                integral_inv_one_minus_z_zero_to_x
            ),
            name="1/(1-z)+",
        ),
    ),
)

combined = convolve_distribution(
    pdf,
    x=x,
    distribution=distribution,
)

explicit = (
    convolve_delta(
        pdf,
        x=x,
        coefficient=1.7,
    )
    + convolve_regular(
        pdf,
        x=x,
        kernel=lambda z: 1.0 + z,
        coefficient=-0.3,
    )
    + convolve_plus(
        pdf,
        x=x,
        kernel=kernel_inv_one_minus_z,
        integral_zero_to_x=(
            integral_inv_one_minus_z_zero_to_x
        ),
        coefficient=0.8,
    )
)

assert_close(
    combined,
    explicit,
)

checks += 1


# 19: invalid x values are rejected.
try:
    convolve_delta(
        lambda y: y,
        x=1.0,
    )
except ValueError:
    pass
else:
    raise AssertionError(
        "x=1 was not rejected"
    )

checks += 1


print(
    f"{checks} convolution checks passed"
)
PY

cat > "${DOC}" <<'MD'
# v22 OPE convolution reference

The v22 matching convolution is

\[
(K\otimes f)(x)
=
\int_x^1 \frac{dz}{z}\,
K(z)\,f(x/z).
\]

For a plus distribution,

\[
\int_x^1 dz\,[g(z)]_+\phi(z)
=
\int_x^1 dz\,g(z)
[\phi(z)-\phi(1)]
-
\phi(1)\int_0^x dz\,g(z),
\]

where

\[
\phi(z)=\frac{f(x/z)}{z}.
\]

`v22/src/convolution.py` is a scalar high-accuracy reference
implementation. It is not the eventual training-time vectorized
implementation.

The smoke tests cover:

- delta-function identity;
- ordinary Mellin convolution;
- \( [1/(1-z)]_+ \);
- \( [\ln(1-z)/(1-z)]_+ \);
- a composite distribution;
- endpoint behavior up to \(x=0.95\).

The actual NLO TMD matching coefficients are intentionally not inserted
until the subtraction/renormalization scheme and perturbative convention
are explicitly locked.
MD

touch \
  "${ROOT}/v22/__init__.py" \
  "${ROOT}/v22/src/__init__.py"

python3 -m py_compile \
  "${SRC}" \
  "${TEST}"

cd "${ROOT}"

PYTHONPATH=. \
python3 \
  v22/tests/run_convolution_smoke.py

echo
echo "Created:"
echo "  v22/src/convolution.py"
echo "  v22/tests/run_convolution_smoke.py"
echo "  v22/OPE_CONVOLUTION.md"
