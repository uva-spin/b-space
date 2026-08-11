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
