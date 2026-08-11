"""Standalone scalar reference for the v22 one-loop DY W kernel."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

from v22.src.css2_ope_nlo_general import (
    general_scale_quark_ope_nlo_components,
)


PDF = Callable[[float], float]

FOURIER_NORM = 1.0 / (2.0 * math.pi)


@dataclass(frozen=True)
class QuarkLegNLO:
    pid: int
    born: float
    delta_qq_coefficient: float
    delta_qg_coefficient: float
    a_s: float
    matched: float
    L_b: float
    l_zeta: float


@dataclass(frozen=True)
class DYLuminosityNLO:
    born: float
    delta_qq_coefficient: float
    delta_qg_coefficient: float
    a_s: float
    strict_ope: float
    naive_product: float


@dataclass(frozen=True)
class DYWKernelNLO:
    common_factor: float
    hard_factor: float
    hard_fraction: float
    born: float
    strict_nlo: float
    multiplicative_nlo: float
    strict_ratio_to_born: float
    multiplicative_ratio_to_born: float
    beyond_nlo_fraction_of_born: float


def build_quark_leg_nlo(
    *,
    pid: int,
    x: float,
    alpha_s_mu: float,
    b_pert_GeV_inv: float,
    mu_GeV: float,
    zeta_GeV2: float,
    quark_pdf: PDF,
    gluon_pdf: PDF,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> QuarkLegNLO:
    """Build one signed-flavor perturbative OPE leg."""

    components = general_scale_quark_ope_nlo_components(
        x=float(x),
        alpha_s=float(alpha_s_mu),
        b_pert_GeV_inv=float(b_pert_GeV_inv),
        mu_GeV=float(mu_GeV),
        zeta_GeV2=float(zeta_GeV2),
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
    )

    return QuarkLegNLO(
        pid=int(pid),
        born=float(components.born_quark),
        delta_qq_coefficient=float(
            components.one_loop_qq_total
        ),
        delta_qg_coefficient=float(
            components.one_loop_qg_total
        ),
        a_s=float(components.a_s),
        matched=float(components.matched_tmd),
        L_b=float(components.L_b),
        l_zeta=float(components.l_zeta),
    )


def build_dy_luminosity_nlo(
    *,
    legs_a: Mapping[int, QuarkLegNLO],
    legs_b: Mapping[int, QuarkLegNLO],
    charge_squared: Mapping[int, float],
    flavors: tuple[int, ...] = (1, 2, 3),
) -> DYLuminosityNLO:
    """Build the charge-weighted q qbar luminosity through one loop."""

    born = 0.0
    delta_qq = 0.0
    delta_qg = 0.0
    naive_product = 0.0
    a_s_values: list[float] = []

    for flavor in flavors:
        pid = abs(int(flavor))

        if pid not in legs_a or -pid not in legs_a:
            raise KeyError(f"beam-A legs missing flavor +/-{pid}")

        if pid not in legs_b or -pid not in legs_b:
            raise KeyError(f"beam-B legs missing flavor +/-{pid}")

        q_a = legs_a[pid]
        qb_a = legs_a[-pid]
        q_b = legs_b[pid]
        qb_b = legs_b[-pid]

        charge2 = float(charge_squared[pid])

        a_s_values.extend(
            [q_a.a_s, qb_a.a_s, q_b.a_s, qb_b.a_s]
        )

        born_channel = (
            q_a.born * qb_b.born
            + qb_a.born * q_b.born
        )

        delta_qq_channel = (
            q_a.delta_qq_coefficient * qb_b.born
            + q_a.born * qb_b.delta_qq_coefficient
            + qb_a.delta_qq_coefficient * q_b.born
            + qb_a.born * q_b.delta_qq_coefficient
        )

        delta_qg_channel = (
            q_a.delta_qg_coefficient * qb_b.born
            + q_a.born * qb_b.delta_qg_coefficient
            + qb_a.delta_qg_coefficient * q_b.born
            + qb_a.born * q_b.delta_qg_coefficient
        )

        naive_channel = (
            q_a.matched * qb_b.matched
            + qb_a.matched * q_b.matched
        )

        born += charge2 * born_channel
        delta_qq += charge2 * delta_qq_channel
        delta_qg += charge2 * delta_qg_channel
        naive_product += charge2 * naive_channel

    if not a_s_values:
        raise ValueError("no active flavor channels")

    a_s = float(a_s_values[0])

    if max(abs(value - a_s) for value in a_s_values) > 1.0e-13:
        raise ValueError("all legs must use the same a_s(mu_b)")

    strict_ope = born + a_s * (delta_qq + delta_qg)

    return DYLuminosityNLO(
        born=float(born),
        delta_qq_coefficient=float(delta_qq),
        delta_qg_coefficient=float(delta_qg),
        a_s=a_s,
        strict_ope=float(strict_ope),
        naive_product=float(naive_product),
    )


def assemble_dy_w_nlo(
    *,
    luminosity: DYLuminosityNLO,
    hard_factor: float,
    observable_prefactor: float,
    x1: float,
    x2: float,
    sudakov_pair_exponent: float,
    fourier_norm: float = FOURIER_NORM,
) -> DYWKernelNLO:
    """Assemble Born, strict-NLO and multiplicative-NLO W integrands."""

    hard = float(hard_factor)
    prefactor = float(observable_prefactor)
    x1 = float(x1)
    x2 = float(x2)
    sudakov = float(sudakov_pair_exponent)
    fourier = float(fourier_norm)

    values = [hard, prefactor, x1, x2, sudakov, fourier]

    if not all(math.isfinite(value) for value in values):
        raise ValueError("all scalar inputs must be finite")

    if x1 <= 0.0 or x2 <= 0.0:
        raise ValueError("x1 and x2 must be positive")

    common = (
        prefactor
        * fourier
        * x1
        * x2
        * math.exp(-sudakov)
    )

    hard_fraction = hard - 1.0

    born_lum = float(luminosity.born)

    strict_lum = (
        born_lum
        + luminosity.a_s
        * (
            luminosity.delta_qq_coefficient
            + luminosity.delta_qg_coefficient
        )
        + hard_fraction * born_lum
    )

    multiplicative_lum = (
        hard * luminosity.naive_product
    )

    born = common * born_lum
    strict = common * strict_lum
    multiplicative = common * multiplicative_lum

    scale = max(abs(born), 1.0e-300)

    return DYWKernelNLO(
        common_factor=float(common),
        hard_factor=hard,
        hard_fraction=hard_fraction,
        born=float(born),
        strict_nlo=float(strict),
        multiplicative_nlo=float(multiplicative),
        strict_ratio_to_born=float(strict / scale),
        multiplicative_ratio_to_born=float(
            multiplicative / scale
        ),
        beyond_nlo_fraction_of_born=float(
            (multiplicative - strict) / scale
        ),
    )
