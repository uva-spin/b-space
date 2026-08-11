from __future__ import annotations

import math

from v22.src.css2_ope_nlo import B0
from v22.src.small_b_profile import (
    b_min_from_Q,
    b_ope_profile,
    matching_log_Lb,
    smooth_max_power,
)


def assert_close(
    actual: float,
    expected: float,
    *,
    rel: float = 2.0e-12,
    abs_tol: float = 2.0e-13,
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


# 1: b_min definition.
Q = 10.0
assert_close(
    b_min_from_Q(Q_GeV=Q),
    B0 / Q,
)
checks += 1


# 2: hard profile is an exact maximum.
b_min = B0 / Q
assert_close(
    b_ope_profile(
        b_star_GeV_inv=0.1 * b_min,
        Q_GeV=Q,
        kind="hard",
    ),
    b_min,
)
checks += 1


# 3: smooth profile reaches b_min at b_star=0.
assert_close(
    b_ope_profile(
        b_star_GeV_inv=0.0,
        Q_GeV=Q,
        power=16.0,
        kind="smooth",
    ),
    b_min,
)
checks += 1


# 4: the small-b cap gives L_b=0 when mu=Q.
assert_close(
    matching_log_Lb(
        b_pert_GeV_inv=b_min,
        mu_GeV=Q,
    ),
    0.0,
)
checks += 1


# 5: smooth maximum is monotonic.
values = [
    b_ope_profile(
        b_star_GeV_inv=value,
        Q_GeV=Q,
        power=16.0,
        kind="smooth",
    )
    for value in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
]

if not all(
    right >= left
    for left, right in zip(values[:-1], values[1:])
):
    raise AssertionError("smooth profile is not monotonic")

checks += 1


# 6: far above b_min, the profile approaches b_star.
b_star = 20.0 * b_min
profiled = b_ope_profile(
    b_star_GeV_inv=b_star,
    Q_GeV=Q,
    power=16.0,
    kind="smooth",
)

assert_close(
    profiled,
    b_star,
    rel=1.0e-12,
)
checks += 1


# 7: at equality, the p-norm enhancement is known analytically.
assert_close(
    smooth_max_power(
        b_min,
        b_min,
        power=16.0,
    ),
    b_min * 2.0 ** (1.0 / 16.0),
)
checks += 1


# 8: invalid profile mode is rejected.
try:
    b_ope_profile(
        b_star_GeV_inv=0.1,
        Q_GeV=Q,
        kind="unknown",
    )
except ValueError:
    pass
else:
    raise AssertionError("invalid profile kind was not rejected")

checks += 1


print(f"{checks} small-b profile checks passed")
