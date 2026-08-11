r"""General-scale one-loop matching of a quark TMDPDF onto collinear PDFs.

Conventions:

    a_s = alpha_s/(4*pi)

    L_b = ln(mu^2 b_pert^2 / b0^2)
    l_zeta = ln(mu^2 / zeta)

The perturbative coordinate is `b_pert`, normally the backend's b_star,
not the physical large-b coordinate.

The input PDF callables return unweighted f(x,mu), not x*f.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from v22.src.convolution import (
    convolve_plus,
    convolve_regular,
)
from v22.src.css2_ope_nlo import (
    B0,
    CF,
    CanonicalOPEComponents,
    a_s_from_alpha_s,
    canonical_css2_quark_ope_nlo_components,
)


PDF = Callable[[float], float]

TR = 0.5


@dataclass(frozen=True)
class GeneralScaleOPEComponents:
    x: float
    alpha_s: float
    a_s: float
    b_pert_GeV_inv: float
    mu_GeV: float
    zeta_GeV2: float
    L_b: float
    l_zeta: float
    born_quark: float
    one_loop_qq_plus_log: float
    one_loop_qq_regular: float
    one_loop_qq_delta: float
    one_loop_qg_log: float
    one_loop_qg_regular: float
    one_loop_qq_total: float
    one_loop_qg_total: float
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


def matching_logs(
    *,
    b_pert_GeV_inv: float,
    mu_GeV: float,
    zeta_GeV2: float,
) -> tuple[float, float]:
    b = float(b_pert_GeV_inv)
    mu = float(mu_GeV)
    zeta = float(zeta_GeV2)

    if not math.isfinite(b) or b <= 0.0:
        raise ValueError("b_pert must be finite and positive")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    if not math.isfinite(zeta) or zeta <= 0.0:
        raise ValueError("zeta must be finite and positive")

    L_b = math.log((mu * b / B0) ** 2)
    l_zeta = math.log(mu * mu / zeta)

    return L_b, l_zeta


def p_qq_singular(z: float) -> float:
    """The ordinary function underlying [(1+z^2)/(1-z)]_+."""

    z = float(z)
    return (1.0 + z * z) / (1.0 - z)


def integral_p_qq_zero_to_x(x: float) -> float:
    """Integral from 0 to x of (1+z^2)/(1-z), for x<1."""

    x = _validate_x(x)

    return (
        -0.5 * x * x
        - x
        - 2.0 * math.log1p(-x)
    )


def p_qg(z: float) -> float:
    """Color-stripped g -> q DGLAP shape z^2+(1-z)^2."""

    z = float(z)
    if not 0.0 <= z <= 1.0:
        raise ValueError("z must lie in [0,1]")

    return z * z + (1.0 - z) ** 2


def c_qq_1_delta_general(
    *,
    L_b: float,
    l_zeta: float,
) -> float:
    return CF * (
        -L_b * L_b
        + 2.0 * L_b * l_zeta
        - math.pi * math.pi / 6.0
    )


def general_scale_quark_ope_nlo_components(
    *,
    x: float,
    alpha_s: float,
    b_pert_GeV_inv: float,
    mu_GeV: float,
    zeta_GeV2: float,
    quark_pdf: PDF,
    gluon_pdf: PDF,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> GeneralScaleOPEComponents:
    """Return the general-scale one-loop quark-TMD OPE decomposition."""

    x = _validate_x(x)
    a_s = a_s_from_alpha_s(alpha_s)

    L_b, l_zeta = matching_logs(
        b_pert_GeV_inv=b_pert_GeV_inv,
        mu_GeV=mu_GeV,
        zeta_GeV2=zeta_GeV2,
    )

    born = _checked_density(quark_pdf, x)

    qq_plus_log = (
        -2.0
        * L_b
        * CF
        * convolve_plus(
            quark_pdf,
            x=x,
            kernel=p_qq_singular,
            integral_zero_to_x=integral_p_qq_zero_to_x,
            epsabs=epsabs,
            epsrel=epsrel,
        )
    )

    qq_regular = convolve_regular(
        quark_pdf,
        x=x,
        kernel=lambda z: 2.0 * CF * (1.0 - z),
        epsabs=epsabs,
        epsrel=epsrel,
    )

    qq_delta = (
        c_qq_1_delta_general(
            L_b=L_b,
            l_zeta=l_zeta,
        )
        * born
    )

    qg_log = (
        -2.0
        * L_b
        * TR
        * convolve_regular(
            gluon_pdf,
            x=x,
            kernel=p_qg,
            epsabs=epsabs,
            epsrel=epsrel,
        )
    )

    qg_regular = (
        TR
        * convolve_regular(
            gluon_pdf,
            x=x,
            kernel=lambda z: 4.0 * z * (1.0 - z),
            epsabs=epsabs,
            epsrel=epsrel,
        )
    )

    qq_total = qq_plus_log + qq_regular + qq_delta
    qg_total = qg_log + qg_regular
    one_loop_total = qq_total + qg_total
    matched = born + a_s * one_loop_total

    return GeneralScaleOPEComponents(
        x=x,
        alpha_s=float(alpha_s),
        a_s=a_s,
        b_pert_GeV_inv=float(b_pert_GeV_inv),
        mu_GeV=float(mu_GeV),
        zeta_GeV2=float(zeta_GeV2),
        L_b=L_b,
        l_zeta=l_zeta,
        born_quark=born,
        one_loop_qq_plus_log=qq_plus_log,
        one_loop_qq_regular=qq_regular,
        one_loop_qq_delta=qq_delta,
        one_loop_qg_log=qg_log,
        one_loop_qg_regular=qg_regular,
        one_loop_qq_total=qq_total,
        one_loop_qg_total=qg_total,
        one_loop_total=one_loop_total,
        matched_tmd=matched,
    )


def general_scale_quark_ope_nlo(**kwargs) -> float:
    return general_scale_quark_ope_nlo_components(
        **kwargs
    ).matched_tmd


def canonical_recovery_difference(
    *,
    x: float,
    alpha_s: float,
    b_pert_GeV_inv: float,
    quark_pdf: PDF,
    gluon_pdf: PDF,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> tuple[GeneralScaleOPEComponents, CanonicalOPEComponents]:
    """Evaluate both modules at mu=b0/b and zeta=mu^2."""

    mu = B0 / float(b_pert_GeV_inv)

    general = general_scale_quark_ope_nlo_components(
        x=x,
        alpha_s=alpha_s,
        b_pert_GeV_inv=b_pert_GeV_inv,
        mu_GeV=mu,
        zeta_GeV2=mu * mu,
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    canonical = canonical_css2_quark_ope_nlo_components(
        x=x,
        alpha_s=alpha_s,
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    return general, canonical
