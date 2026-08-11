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
