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
