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
