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
