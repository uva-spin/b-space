from __future__ import annotations

import math

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


def assert_close(
    actual: float,
    expected: float,
    *,
    rel: float = 1.0e-9,
    abs_tol: float = 1.0e-12,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=rel,
        abs_tol=abs_tol,
    ):
        raise AssertionError(
            f"actual={actual:.16e}, expected={expected:.16e}"
        )


n_checks = 0


# 1-5: inverse Hankel transform of a Gaussian.
for kT in [0.0, 0.25, 1.0, 2.5, 5.0]:
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

    assert_close(
        numerical,
        analytic,
        rel=2.0e-8,
        abs_tol=2.0e-11,
    )

    n_checks += 1


# 6-9: DY transform of two Gaussian TMDs.
for qT in [0.0, 0.5, 2.0, 4.0]:
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

    assert_close(
        numerical,
        analytic,
        rel=2.0e-8,
        abs_tol=2.0e-11,
    )

    n_checks += 1


# 10: integral over kT returns the bT-space value at the origin.
width = 0.47

integral, _ = quad(
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

assert_close(integral, 1.0, rel=2.0e-10)
n_checks += 1


# 11: LHAPDF x*f conversion.
x = 0.2
f = 3.75

assert_close(
    density_from_lhapdf_xfx(
        x=x,
        xfx=x * f,
    ),
    f,
)

n_checks += 1


# 12: DY luminosity is symmetric under interchange of beams.
pdf_a = {
    1: 1.7,
    -1: 0.21,
    2: 3.4,
    -2: 0.18,
    3: 0.12,
    -3: 0.11,
}

pdf_b = {
    1: 2.1,
    -1: 0.31,
    2: 2.8,
    -2: 0.27,
    3: 0.15,
    -3: 0.13,
}

assert_close(
    lo_dy_flavor_luminosity(pdf_a, pdf_b),
    lo_dy_flavor_luminosity(pdf_b, pdf_a),
    rel=1.0e-14,
)

n_checks += 1


# 13: published TMD arguments validate correctly.
TMDArguments(
    x=0.1,
    bT_GeV_inv=1.2,
    mu_GeV=10.0,
    zeta_GeV2=100.0,
).validate()

try:
    TMDArguments(
        x=0.0,
        bT_GeV_inv=1.2,
        mu_GeV=10.0,
        zeta_GeV2=100.0,
    ).validate()
except ValueError:
    pass
else:
    raise AssertionError("Invalid x=0 was not rejected")

n_checks += 1


print(f"{n_checks} convention checks passed")
