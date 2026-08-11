from __future__ import annotations

import math

from v22.src.css2_ope_nlo import CF
from v22.src.dy_hard_nlo import (
    dy_hard_coefficient_1,
    dy_hard_nlo,
    dy_hard_nlo_at_Q,
    hard_log_T,
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


# 1: T=0 at mu=Q.
assert_close(
    hard_log_T(Q_GeV=10.0, mu_GeV=10.0),
    0.0,
)
checks += 1


# 2: T=ln 4 at mu=Q/2.
assert_close(
    hard_log_T(Q_GeV=10.0, mu_GeV=5.0),
    math.log(4.0),
)
checks += 1


# 3: canonical hard coefficient.
expected_h1 = CF * (
    -16.0 + 7.0 * math.pi**2 / 3.0
)

assert_close(
    dy_hard_coefficient_1(
        Q_GeV=10.0,
        mu_GeV=10.0,
    ),
    expected_h1,
)
checks += 1


# 4: alpha_s=0 gives H=1.
assert_close(
    dy_hard_nlo(
        Q_GeV=10.0,
        mu_GeV=10.0,
        alpha_s=0.0,
    ),
    1.0,
)
checks += 1


# 5: direct one-loop formula at mu=Q.
alpha_s = 0.2
expected = (
    1.0
    + alpha_s
    / (4.0 * math.pi)
    * expected_h1
)

assert_close(
    dy_hard_nlo_at_Q(
        Q_GeV=10.0,
        alpha_s_at_Q=alpha_s,
    ),
    expected,
)
checks += 1


# 6: general-scale implementation.
Q = 12.0
mu = 7.0
alpha_s = 0.19
T = math.log(Q**2 / mu**2)
expected_general = (
    1.0
    + alpha_s
    / (4.0 * math.pi)
    * CF
    * (
        -16.0
        + 7.0 * math.pi**2 / 3.0
        + 6.0 * T
        - 2.0 * T**2
    )
)

assert_close(
    dy_hard_nlo(
        Q_GeV=Q,
        mu_GeV=mu,
        alpha_s=alpha_s,
    ),
    expected_general,
)
checks += 1


# 7-8: invalid scales are rejected.
for bad_Q, bad_mu in [
    (0.0, 1.0),
    (1.0, 0.0),
]:
    try:
        hard_log_T(
            Q_GeV=bad_Q,
            mu_GeV=bad_mu,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"invalid scales were not rejected: "
            f"Q={bad_Q}, mu={bad_mu}"
        )

    checks += 1


print(f"{checks} Drell--Yan hard-factor checks passed")
